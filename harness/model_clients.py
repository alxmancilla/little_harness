"""Standard-library model clients: streaming protocols, HTTP safety and provider selection.

Clients accept (system, messages, tools, on_text=None) and return content blocks.
This module does not import the agent loop, load .env or create workspace files.
"""

import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


def sse_events(lines):
    """Yield (event_name, data) for each server-sent event in an iterable of raw byte lines.
    Fields are "name: value" lines; a blank line ends the event; ":" lines are comments."""
    event, data = None, []
    for raw in lines:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield event, "\n".join(data)
            event, data = None, []
        elif not line.startswith(":"):
            field, _, value = line.partition(":")
            value = value.removeprefix(" ")
            if field == "event":
                event = value
            elif field == "data":
                data.append(value)
    # an event cut off before its blank line is dropped, not half-parsed


class ModelError(Exception):
    """An expected API, network or response-format failure, not a harness programming bug.

    category is a fixed, non-sensitive label for routing and traces; the message may
    be shown to users but is not traced unless content tracing is enabled.
    """
    CATEGORIES = frozenset({"unknown", "rate_limited", "server_error", "auth", "request_rejected",
                            "network", "malformed_response", "incomplete_response", "provider_error"})

    def __init__(self, message="", category="unknown"):
        if category not in self.CATEGORIES:
            raise ValueError(f"unknown ModelError category: {category}")
        super().__init__(message)
        self.category = category


def _http_category(code):
    if code == 429:
        return "rate_limited"
    if code in (401, 403):
        return "auth"
    return "server_error" if code >= 500 else "request_rejected"


class NoModelRedirects(urllib.request.HTTPRedirectHandler):
    """Fail closed: model credentials must never be forwarded by an HTTP redirect."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # urllib raises HTTPError; the transport closes it without reading its body


class HTTPModel:
    """Shared plumbing for HTTP model clients: bearer auth, JSON POST, retries with backoff.
    Subclasses build the payload and fold the streamed reply into a response dict."""
    BASE_URL = ""
    RETRYABLE = {429, 500, 502, 503, 529}

    def __init__(self, model, key=None, base_url=None, max_tokens=2048, key_header=None):
        self.model = model
        self.key = key                              # None is fine for local servers without auth
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        self.max_tokens = max_tokens
        self.key_header = key_header                # e.g. "x-api-key"; default is a bearer token
        self._opener = urllib.request.build_opener(NoModelRedirects())

    def _headers(self):
        headers = {"content-type": "application/json"}
        if self.key and self.key_header:
            headers[self.key_header] = self.key
        elif self.key:
            headers["authorization"] = f"Bearer {self.key}"
        return headers

    def _post(self, path, payload, on_text):
        body = json.dumps(payload).encode()
        headers = self._headers()
        for attempt in range(4):
            req = urllib.request.Request(self.base_url + path, data=body, headers=headers, method="POST")
            try:
                with self._opener.open(req, timeout=120) as resp:   # timeout applies per read
                    try:
                        return self._read_stream(resp, on_text)
                    except (http.client.IncompleteRead, OSError) as e:   # dropped connection or read timeout
                        raise ModelError("stream interrupted by a connection or read failure", "network") from e
                    except (ValueError, KeyError, TypeError, IndexError, AttributeError) as e:
                        # Provider data is untrusted; don't include malformed content in the error.
                        raise ModelError("malformed model response", "malformed_response") from e
            except urllib.error.HTTPError as e:
                e.close()                          # never read or retain provider diagnostic bodies
                if e.code in self.RETRYABLE and attempt < 3:
                    time.sleep(2 ** attempt)        # exponential backoff: 1s, 2s, 4s
                    continue
                raise ModelError(f"API error {e.code}", _http_category(e.code)) from None
            except urllib.error.URLError as e:
                if attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                raise ModelError("network error: could not connect to the model API", "network") from e
            except (OSError, http.client.HTTPException) as e:
                raise ModelError("connection failed before receiving a complete response", "network") from e


def _tool_result_text(block):
    """APIs without an is_error flag get the error marked in the text instead."""
    content = block["content"] if isinstance(block["content"], str) else json.dumps(block["content"])
    return f"Error: {content}" if block.get("is_error") else content


class AnthropicModel(HTTPModel):
    """Anthropic Messages API. The internal message shape is this API's shape, so no translation."""
    BASE_URL = "https://api.anthropic.com/v1"

    def _headers(self):
        headers = super()._headers()                # bearer auth, or "x-api-key" via key_header
        headers["anthropic-version"] = "2023-06-01"
        # Only needed when the key is NOT scoped to a single workspace
        workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID")
        if workspace:
            headers["anthropic-workspace-id"] = workspace
        return headers

    def __call__(self, system, messages, tools, on_text=None):
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": messages,
            "stream": True,
        }
        if tools:                                   # the summarizer calls with no tools
            payload["tools"] = tools
        return self._post("/messages", payload, on_text)

    def _read_stream(self, lines, on_text=None):
        """Fold a stream of events back into the same dict a non-streaming call returns."""
        msg, blocks, partial_json, done, truncated = {}, {}, {}, False, []
        for _, data in sse_events(lines):
            ev = json.loads(data)
            kind = ev["type"]
            if kind == "message_start":
                msg = ev["message"]
            elif kind == "content_block_start":
                blocks[ev["index"]] = ev["content_block"]
                partial_json[ev["index"]] = ""
            elif kind == "content_block_delta":
                i, delta = ev["index"], ev["delta"]
                if delta["type"] == "text_delta":
                    blocks[i]["text"] += delta["text"]
                    if on_text:
                        on_text(delta["text"])
                elif delta["type"] == "input_json_delta":       # tool arguments arrive as JSON fragments
                    partial_json[i] += delta["partial_json"]
                elif delta["type"] == "thinking_delta":
                    blocks[i]["thinking"] += delta["thinking"]
                elif delta["type"] == "signature_delta":
                    blocks[i]["signature"] = delta["signature"]
            elif kind == "content_block_stop":
                block = blocks[ev["index"]]
                if block["type"] == "tool_use":
                    try:
                        block["input"] = json.loads(partial_json[ev["index"]] or "{}")
                    except json.JSONDecodeError:                # arguments cut off, e.g. by max_tokens
                        block["input"] = {}
                        truncated.append(block["id"])
            elif kind == "message_delta":
                msg["stop_reason"] = ev["delta"].get("stop_reason")
                msg.setdefault("usage", {}).update(ev.get("usage", {}))
            elif kind == "message_stop":
                done = True
            elif kind == "error":                               # e.g. overloaded_error mid-stream
                raise ModelError("model API stream reported an error", "provider_error")
        if not done:
            raise ModelError("stream ended before message_stop", "incomplete_response")
        return {"stop_reason": msg.get("stop_reason"),
                "content": [blocks[i] for i in sorted(blocks)],
                "usage": msg.get("usage"),
                "truncated_tool_ids": truncated}


class OpenAIResponsesModel(HTTPModel):
    """OpenAI Responses API (POST /responses). Grove's AI Gateway serves it for GPT models
    at https://ai-gateway.corp.mongodb.com/openai/v1/responses."""
    BASE_URL = "https://api.openai.com/v1"

    def __call__(self, system, messages, tools, on_text=None):
        payload = {
            "model": self.model,
            "instructions": system,
            "input": self.to_input(messages),
            "max_output_tokens": self.max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = [{"type": "function", "name": t["name"], "description": t["description"],
                                 "parameters": t["input_schema"]} for t in tools]
        return self._post("/responses", payload, on_text)

    @staticmethod
    def to_input(messages):
        """Content blocks -> Responses input items. Anthropic thinking blocks are not sent."""
        items = []
        for m in messages:
            if isinstance(m["content"], str):
                items.append({"role": m["role"], "content": m["content"]})
                continue
            for b in m["content"]:
                if b["type"] == "text" and b["text"]:
                    items.append({"role": m["role"], "content": b["text"]})
                elif b["type"] == "tool_use":
                    items.append({"type": "function_call", "call_id": b["id"], "name": b["name"],
                                  "arguments": json.dumps(b.get("input", {}))})
                elif b["type"] == "tool_result":
                    items.append({"type": "function_call_output", "call_id": b["tool_use_id"],
                                  "output": _tool_result_text(b)})
        return items

    def _read_stream(self, lines, on_text=None):
        """Print text deltas as they arrive, then parse the full response sent with the final event."""
        final = None
        for _, data in sse_events(lines):
            ev = json.loads(data)
            kind = ev.get("type")
            if kind == "response.output_text.delta":
                if on_text:
                    on_text(ev["delta"])
            elif kind in ("response.completed", "response.incomplete"):
                final = ev["response"]
            elif kind == "response.failed":
                raise ModelError("model API response failed", "provider_error")
            elif kind == "error":
                raise ModelError("model API stream reported an error", "provider_error")
        if final is None:
            raise ModelError("stream ended before response.completed", "incomplete_response")
        return self.parse(final)

    @staticmethod
    def parse(response):
        """Responses output items -> the internal response dict. Reasoning items are dropped."""
        content, truncated, refused = [], [], False
        for item in response.get("output", []):
            if item["type"] == "message":
                for part in item.get("content", []):
                    if part["type"] == "output_text":
                        content.append({"type": "text", "text": part["text"]})
                    elif part["type"] == "refusal":
                        content.append({"type": "text", "text": part["refusal"]})
                        refused = True
            elif item["type"] == "function_call":
                try:
                    args = json.loads(item.get("arguments") or "{}")
                except json.JSONDecodeError:                    # arguments cut off, e.g. by max tokens
                    args = {}
                    truncated.append(item["call_id"])
                content.append({"type": "tool_use", "id": item["call_id"], "name": item["name"], "input": args})
        incomplete = (response.get("incomplete_details") or {}).get("reason")
        if incomplete == "max_output_tokens":
            stop = "max_tokens"
        elif refused or incomplete == "content_filter":
            stop = "refusal"
        elif any(b["type"] == "tool_use" for b in content):
            stop = "tool_use"
        else:
            stop = "end_turn"
        usage = response.get("usage") or {}
        return {"stop_reason": stop, "content": content,
                "usage": {"input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")},
                "truncated_tool_ids": truncated}


class OpenAIChatModel(HTTPModel):
    """OpenAI-compatible Chat Completions (POST /chat/completions). Most providers and local
    servers speak this format: OpenAI, Gemini's OpenAI endpoint, Groq, OpenRouter, vLLM, Ollama."""
    BASE_URL = "https://api.openai.com/v1"
    STOP_REASONS = {"tool_calls": "tool_use", "length": "max_tokens", "content_filter": "refusal"}

    def __call__(self, system, messages, tools, on_text=None):
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + self.to_messages(messages),
            "max_tokens": self.max_tokens,          # the most widely accepted name across providers
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": {
                "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
                for t in tools]
        return self._post("/chat/completions", payload, on_text)

    @staticmethod
    def to_messages(messages):
        """Content blocks -> chat messages. Tool results become role "tool" messages, which must
        directly follow the assistant message that made the calls, so they go before any user text."""
        out = []
        for m in messages:
            if isinstance(m["content"], str):
                out.append({"role": m["role"], "content": m["content"]})
                continue
            text = "\n\n".join(b["text"] for b in m["content"] if b["type"] == "text" and b["text"])
            if m["role"] == "assistant":
                msg = {"role": "assistant", "content": text or None}
                calls = [{"id": b["id"], "type": "function",
                          "function": {"name": b["name"], "arguments": json.dumps(b.get("input", {}))}}
                         for b in m["content"] if b["type"] == "tool_use"]
                if calls:
                    msg["tool_calls"] = calls
                out.append(msg)
                continue
            for b in m["content"]:
                if b["type"] == "tool_result":
                    out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": _tool_result_text(b)})
            if text:
                out.append({"role": "user", "content": text})
        return out

    def _read_stream(self, lines, on_text=None):
        """Fold chat completion chunks into the internal response dict. Tool calls arrive as
        fragments keyed by index: the id and name once, the arguments in pieces."""
        text, calls, finish, usage, done = [], {}, None, {}, False
        for _, data in sse_events(lines):
            if data.strip() == "[DONE]":
                done = True
                break
            ev = json.loads(data)
            if ev.get("error"):
                raise ModelError("model API stream reported an error", "provider_error")
            usage = ev.get("usage") or usage
            for choice in ev.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    text.append(delta["content"])
                    if on_text:
                        on_text(delta["content"])
                for tc in delta.get("tool_calls") or []:
                    call = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                    fn = tc.get("function") or {}
                    call["id"] = tc.get("id") or call["id"]
                    call["name"] = fn.get("name") or call["name"]
                    call["arguments"] += fn.get("arguments") or ""
                finish = choice.get("finish_reason") or finish
        if not done:
            raise ModelError("stream ended before [DONE]", "incomplete_response")
        content = [{"type": "text", "text": "".join(text)}] if text else []
        truncated = []
        for i in sorted(calls):
            call_id = calls[i]["id"] or f"call_{i}"
            try:
                args = json.loads(calls[i]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
                truncated.append(call_id)
            content.append({"type": "tool_use", "id": call_id, "name": calls[i]["name"], "input": args})
        stop = self.STOP_REASONS.get(finish, "end_turn")
        if calls and stop == "end_turn":            # some servers report "stop" alongside tool calls
            stop = "tool_use"
        return {"stop_reason": stop, "content": content,
                "usage": {"input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens")},
                "truncated_tool_ids": truncated}


GATEWAY = "https://ai-gateway.corp.mongodb.com"

# provider name: (client class, default base URL, env var holding the API key, default model, key header)
PROVIDERS = {
    "anthropic": (AnthropicModel, AnthropicModel.BASE_URL, "ANTHROPIC_API_KEY", "claude-sonnet-5-5", None),
    "openai": (OpenAIResponsesModel, OpenAIResponsesModel.BASE_URL, "OPENAI_API_KEY", None, None),
    "openai-chat": (OpenAIChatModel, OpenAIChatModel.BASE_URL, "OPENAI_API_KEY", None, None),
    # Grove's AI Gateway has one route per vendor API; "gateway" picks the route from the model name
    "gateway-openai": (OpenAIResponsesModel, f"{GATEWAY}/openai/v1", "AI_GATEWAY_API_KEY", "gpt-5.6-sol", None),
    "gateway-anthropic": (AnthropicModel, f"{GATEWAY}/anthropic/v1", "AI_GATEWAY_API_KEY", "claude-opus-5",
                          "x-api-key"),
}


def make_model(env=os.environ):
    """Build the model client from HARNESS_PROVIDER, HARNESS_MODEL, HARNESS_BASE_URL and
    HARNESS_API_KEY_ENV (the *name* of the variable holding the key, so the key never sits in config)."""
    name = env.get("HARNESS_PROVIDER", "anthropic")
    if name == "gateway":                                    # claude-* -> Anthropic route, anything else -> OpenAI
        model = env.get("HARNESS_MODEL") or PROVIDERS["gateway-openai"][3]
        name = "gateway-anthropic" if model.startswith("claude") else "gateway-openai"
    if name not in PROVIDERS:
        sys.exit(f"Unknown HARNESS_PROVIDER '{name}'. Choose one of: gateway, {', '.join(PROVIDERS)}")
    cls, base_url, key_var, model, key_header = PROVIDERS[name]
    key_var = env.get("HARNESS_API_KEY_ENV") or key_var
    key = env.get(key_var)
    model = env.get("HARNESS_MODEL") or model
    if not key and not env.get("HARNESS_BASE_URL"):         # a custom URL may be a local server without auth
        sys.exit(f"Set {key_var} for provider '{name}', or run: python3 -m agents.teaching.cli mock")
    if not model:
        sys.exit(f"Set HARNESS_MODEL: provider '{name}' has no default model")
    client = cls(model, key=key, base_url=env.get("HARNESS_BASE_URL") or base_url, key_header=key_header)
    client.provider = name                           # resolved route, not just the "gateway" alias
    return client


def print_model_config(model):
    """Show the selected route without credentials, URL user info, query or fragment."""
    url = urllib.parse.urlsplit(model.base_url)
    endpoint = urllib.parse.urlunsplit((url.scheme, url.netloc.rsplit("@", 1)[-1], url.path, "", ""))
    fields = {"provider": model.provider, "url": endpoint, "model": model.model}
    # JSON quoting also escapes terminal control characters in configured values.
    summary = json.dumps(fields)
    if model.key:
        summary = summary.replace(json.dumps(model.key)[1:-1], "[redacted]")
    print("Model configuration: " + summary)