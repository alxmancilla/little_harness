#!/usr/bin/env python3
"""
mini_harness.py
A minimal but functional agent harness built only on the Python 3 standard library.

The five pieces of every agent harness:
  1. Tool registry  : a decorator turns plain Python functions into tools the model can call
  2. Model client   : streams the Anthropic Messages API (SSE) with urllib (plus a scripted fake model)
  3. Agent loop     : model -> tool calls -> tool results -> model ... until the model stops
  4. Guardrails     : step budget, sandboxed paths, error capture, human approval for risky tools
  5. Tracing        : every event appended to trace.jsonl so you can inspect or replay a run
  Plus context management: old turns are replaced by a model-written summary when history grows large
  Plus persistence: interactive mode saves the conversation to session.json and resumes it on restart

Usage:
  python3 mini_harness.py mock                 offline demo, no API key needed
  python3 mini_harness.py "your task here"     one task against the real API
  python3 mini_harness.py                      interactive chat (state persists across turns and restarts)

Real API runs need:  export ANTHROPIC_API_KEY=...
Optional:            export HARNESS_MODEL=claude-sonnet-5-5
                     export HARNESS_COMPACT_CHARS=40000         (history size that triggers compaction)
                     export HARNESS_SESSION=session.json        (interactive mode: where the conversation is saved)
                     export ANTHROPIC_WORKSPACE_ID=wrkspc_...   (only if your key is not workspace scoped)
"""

import ast
import http.client
import inspect
import json
import operator
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

MAX_TOOL_OUTPUT = 8000          # characters; keeps one tool from flooding the context window
COMPACT_AT = int(os.environ.get("HARNESS_COMPACT_CHARS", "40000"))   # compact history above this size
WORKSPACE = Path(os.environ.get("HARNESS_WORKSPACE", "workspace")).resolve()
WORKSPACE.mkdir(exist_ok=True)


# =============================================================================
# 1. TOOL REGISTRY
# =============================================================================
# The model never runs code. It only *asks* for a tool by name with JSON arguments.
# The harness owns the registry, validates the request, executes it, and reports back.

TOOLS = {}
PY_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean"}


def tool(risky=False):
    """Register a function as a tool. Its JSON schema is derived from the signature,
    and its docstring becomes the description the model reads to decide when to use it."""
    def register(fn):
        props, required = {}, []
        for name, param in inspect.signature(fn).parameters.items():
            props[name] = {"type": PY_TO_JSON.get(param.annotation, "string")}
            if param.default is inspect.Parameter.empty:
                required.append(name)
        TOOLS[fn.__name__] = {
            "fn": fn,
            "risky": risky,
            "schema": {
                "name": fn.__name__,
                "description": inspect.getdoc(fn) or "",
                "input_schema": {"type": "object", "properties": props, "required": required},
            },
        }
        return fn
    return register


def _safe_path(rel: str) -> Path:
    """Resolve a path and refuse anything that escapes the workspace sandbox."""
    p = (WORKSPACE / rel).resolve()
    if not p.is_relative_to(WORKSPACE):
        raise ValueError(f"path escapes workspace: {rel}")
    return p


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval(node):
    """Tiny arithmetic interpreter. Never use eval() on model output."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("only numbers and + - * / ** % are allowed")


@tool()
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as '(3 + 4) * 2 ** 3'.
    Supports + - * / ** % and parentheses. Use it instead of doing math in your head."""
    return str(_eval(ast.parse(expression, mode="eval").body))


@tool()
def current_time() -> str:
    """Return the current UTC date and time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@tool()
def list_files(subdir: str = ".") -> str:
    """List the files in the agent workspace, or in a subdirectory of it."""
    entries = sorted(p.name + ("/" if p.is_dir() else "") for p in _safe_path(subdir).iterdir())
    return "\n".join(entries) or "(empty)"


@tool()
def read_file(path: str) -> str:
    """Read a UTF-8 text file from the agent workspace. Paths are relative to the workspace."""
    return _safe_path(path).read_text(encoding="utf-8")


@tool(risky=True)
def write_file(path: str, content: str) -> str:
    """Create or overwrite a UTF-8 text file in the agent workspace. Requires human approval."""
    p = _safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} characters to {path}"


@tool(risky=True)
def http_get(url: str) -> str:
    """Fetch an http(s) URL with a GET request and return the response body as text.
    Requires human approval."""
    if not url.startswith(("http://", "https://")):    # urlopen also accepts file: and ftp:
        raise ValueError("only http:// and https:// URLs are allowed")
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read(MAX_TOOL_OUTPUT + 1).decode("utf-8", errors="replace")



# =============================================================================
# 2. MODEL CLIENTS
# =============================================================================
# Any callable with the signature (system, messages, tools, on_text=None) -> response dict works.
# That single seam is what lets you swap providers or test with a fake model.
# on_text(delta) is called with each piece of text as it arrives; models that cannot stream ignore it.


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

class AnthropicModel:
    URL = "https://api.anthropic.com/v1/messages"
    RETRYABLE = {429, 500, 502, 503, 529}

    def __init__(self, model=None, max_tokens=2048):
        self.key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.key:
            sys.exit("Set ANTHROPIC_API_KEY, or run:  python3 mini_harness.py mock")
        self.model = model or os.environ.get("HARNESS_MODEL", "claude-sonnet-5-5")
        self.max_tokens = max_tokens

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
        body = json.dumps(payload).encode()
        headers = {
            "authorization": f"Bearer {self.key}",   # legacy alternative: "x-api-key": self.key
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        # Only needed when the key is NOT scoped to a single workspace
        workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID")
        if workspace:
            headers["anthropic-workspace-id"] = workspace
        for attempt in range(4):
            req = urllib.request.Request(self.URL, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:   # timeout applies per read
                    try:
                        return self._read_stream(resp, on_text)
                    except (http.client.IncompleteRead, OSError) as e:   # dropped connection or read timeout
                        raise RuntimeError(f"stream interrupted: {e!r}") from e
            except urllib.error.HTTPError as e:
                if e.code in self.RETRYABLE and attempt < 3:
                    time.sleep(2 ** attempt)        # exponential backoff: 1s, 2s, 4s
                    continue
                raise RuntimeError(f"API error {e.code}: {e.read().decode()}") from e
            except urllib.error.URLError as e:
                if attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"network error: {e.reason}") from e


    def _read_stream(self, lines, on_text=None):
        """Fold a stream of events back into the same dict a non-streaming call returns."""
        msg, blocks, partial_json, done = {}, {}, {}, False
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
                    block["input"] = json.loads(partial_json[ev["index"]] or "{}")
            elif kind == "message_delta":
                msg["stop_reason"] = ev["delta"].get("stop_reason")
                msg.setdefault("usage", {}).update(ev.get("usage", {}))
            elif kind == "message_stop":
                done = True
            elif kind == "error":                               # e.g. overloaded_error mid-stream
                raise RuntimeError(f"API stream error: {ev['error']}")
        if not done:
            raise RuntimeError("stream ended before message_stop")
        return {"stop_reason": msg.get("stop_reason"),
                "content": [blocks[i] for i in sorted(blocks)],
                "usage": msg.get("usage")}


class ScriptedModel:
    """Replays canned responses shaped exactly like real API responses.
    Use it to test the loop, the tools and the guardrails without spending tokens."""

    def __init__(self, script):
        self.script = list(script)

    def __call__(self, system, messages, tools, on_text=None):
        if not self.script:
            return {"stop_reason": "end_turn", "content": [{"type": "text", "text": "(script finished)"}]}
        return self.script.pop(0)


DEMO_SCRIPT = [
    # Turn 1: the model asks for two tools at once (parallel tool calls)
    {"stop_reason": "tool_use", "content": [
        {"type": "text", "text": "I'll compute the total and grab a timestamp."},
        {"type": "tool_use", "id": "call_1", "name": "calculator", "input": {"expression": "17 * 23 + 4"}},
        {"type": "tool_use", "id": "call_2", "name": "current_time", "input": {}},
    ]},
    # Turn 2: a tool call that fails, to show error capture
    {"stop_reason": "tool_use", "content": [
        {"type": "tool_use", "id": "call_3", "name": "read_file", "input": {"path": "../etc/passwd"}},
    ]},
    # Turn 3: a risky tool that needs human approval
    {"stop_reason": "tool_use", "content": [
        {"type": "tool_use", "id": "call_4", "name": "write_file",
         "input": {"path": "report.txt", "content": "17 * 23 + 4 = 395\n"}},
    ]},
    # Turn 4: no tool calls, so the loop ends
    {"stop_reason": "end_turn", "content": [
        {"type": "text", "text": "Done. The result is 395 and I saved it to report.txt."},
    ]},
]


# =============================================================================
# 3 + 4 + 5. AGENT LOOP, GUARDRAILS, TRACING
# =============================================================================

def ask_human(name, args):
    """Default approval gate for risky tools."""
    print(f"\n  ⚠  The agent wants to run {name}({json.dumps(args)[:200]})")
    return input("     Allow? [y/N] ").strip().lower() == "y"


SUMMARY_SYSTEM = (
    "You compress the earlier part of an agent conversation into a summary that replaces it. "
    "The agent will continue the work from your summary alone, so keep: the user's goal and any "
    "constraints, facts and results learned, files read or written (with paths), decisions made, "
    "errors hit and how they were resolved, and what is still unfinished. Be concise and factual. "
    "Reply with the summary only."
)


def _size(obj):
    return len(json.dumps(obj))


def render_transcript(messages, clip=1000):
    """Flatten messages into plain text for the summarizer. Tool inputs and results are clipped
    so the summarizer request stays far smaller than the history it replaces."""
    lines = []
    for m in messages:
        if isinstance(m["content"], str):
            lines.append(f"{m['role']}: {m['content']}")
            continue
        for b in m["content"]:
            if b["type"] == "text":
                lines.append(f"{m['role']}: {b['text']}")
            elif b["type"] == "tool_use":
                lines.append(f"tool call: {b['name']}({json.dumps(b.get('input', {}))[:clip]})")
            elif b["type"] == "tool_result":
                label = "tool error" if b.get("is_error") else "tool result"
                lines.append(f"{label}: {str(b['content'])[:clip]}")
    return "\n".join(lines)


class Agent:
    def __init__(self, model, system, max_steps=12, approve=ask_human, trace_path="trace.jsonl",
                 compact_at=COMPACT_AT, session_path=None):
        self.model = model
        self.system = system
        self.max_steps = max_steps
        self.approve = approve
        self.trace_path = Path(trace_path)
        self.compact_at = compact_at
        self.session_path = Path(session_path) if session_path else None
        self.messages = self._load()            # the whole conversation is the agent's state
        self._size_after_compact = 0            # history size right after the last compaction

    def _load(self):
        if not self.session_path or not self.session_path.exists():
            return []
        try:
            messages = json.loads(self.session_path.read_text(encoding="utf-8"))
            if not isinstance(messages, list):
                raise ValueError("expected a list of messages")
        except ValueError as e:
            sys.exit(f"{self.session_path} is not a valid session ({e}); fix or delete it to start fresh")
        print(f"Resumed {len(messages)} messages from {self.session_path}")
        return messages

    def _save(self):
        """Write the conversation atomically: a crash mid-write leaves the previous file intact."""
        if not self.session_path:
            return
        tmp = self.session_path.with_name(self.session_path.name + ".tmp")
        tmp.write_text(json.dumps(self.messages), encoding="utf-8")
        os.replace(tmp, self.session_path)

    def _add_user_message(self, text):
        """Append a user turn. A session killed mid-run ends on a user message (the task or tool
        results), and the API wants alternating roles, so fold the new text into that message."""
        last = self.messages[-1] if self.messages else None
        if last and last["role"] == "user":
            if isinstance(last["content"], str):
                last["content"] = [{"type": "text", "text": last["content"]}]
            last["content"].append({"type": "text", "text": text})
        else:
            self.messages.append({"role": "user", "content": text})

    def _compact(self, step):
        """Replace old turns with a model-written summary once history exceeds compact_at.
        The kept tail starts at an assistant message, so no tool_use is separated from its
        tool_result; the summary is the new first (user) message."""
        size = _size(self.messages)
        keep = self.compact_at // 3
        # require real growth since the last compaction, or a big tail would recompact every step
        if size <= self.compact_at or size < self._size_after_compact + keep:
            return
        sizes = [_size(m) for m in self.messages]
        assistants = [i for i, m in enumerate(self.messages) if m["role"] == "assistant"]
        cut, tail = None, 0
        for i in range(len(self.messages) - 1, 1, -1):         # newest first, within the size budget
            tail += sizes[i]
            if tail > keep:
                break
            if self.messages[i]["role"] == "assistant":
                cut = i
        if cut is None:                                         # newest turn alone exceeds the budget
            cut = next((i for i in reversed(assistants) if 2 <= i < len(self.messages)), None)
        if cut is None:
            return
        summary = self.model(SUMMARY_SYSTEM,
                             [{"role": "user", "content": render_transcript(self.messages[:cut])}], [])
        text = "".join(b["text"] for b in summary["content"] if b["type"] == "text").strip()
        if not text:                                            # never replace history with nothing
            return
        self.messages[:cut] = [{"role": "user", "content":
                                "[Summary of the earlier conversation, compacted to save context. "
                                "Not a new request.]\n" + text}]
        self._size_after_compact = _size(self.messages)
        print(f"[step {step}] 🗜  compacted {cut} messages into a summary "
              f"({size} -> {self._size_after_compact} chars)")
        self._trace("compact", step=step, dropped=cut, chars_before=size,
                    chars_after=self._size_after_compact, summary=text, usage=summary.get("usage"))

    def _trace(self, kind, **data):
        record = {"ts": datetime.now(timezone.utc).isoformat(), "kind": kind, **data}
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _run_tool(self, call):
        """Execute one tool call. Returns (output_text, is_error). Never raises:
        errors go back to the model as data so it can recover."""
        name, args = call["name"], call.get("input", {})
        entry = TOOLS.get(name)
        if entry is None:
            return f"unknown tool '{name}'. Available: {', '.join(TOOLS)}", True
        if entry["risky"] and not self.approve(name, args):
            return "the user denied this action", True
        try:
            out = str(entry["fn"](**args))
        except Exception as e:
            return f"{type(e).__name__}: {e}", True
        if len(out) > MAX_TOOL_OUTPUT:
            out = out[:MAX_TOOL_OUTPUT] + f"\n[truncated, {len(out)} characters total]"
        return out, False

    def run(self, task):
        self._add_user_message(task)
        self._save()
        self._trace("task", text=task)
        schemas = [t["schema"] for t in TOOLS.values()]

        for step in range(1, self.max_steps + 1):
            self._compact(step)
            streamed = []

            def on_text(delta):
                if not streamed:
                    print(f"\n[step {step}] 💬 ", end="", flush=True)
                streamed.append(delta)
                print(delta, end="", flush=True)

            resp = self.model(self.system, self.messages, schemas, on_text=on_text)
            if streamed:
                print()
            content = resp["content"]
            self.messages.append({"role": "assistant", "content": content})
            self._trace("model", step=step, stop_reason=resp.get("stop_reason"),
                        usage=resp.get("usage"), content=content)

            for block in content:
                if block["type"] == "text" and block["text"].strip() and not streamed:
                    print(f"\n[step {step}] 💬 {block['text']}")

            calls = [b for b in content if b["type"] == "tool_use"]
            if not calls:                       # the model is done
                text = "".join(b["text"] for b in content if b["type"] == "text")
                if resp.get("stop_reason") == "max_tokens":
                    text += "\n[warning: response hit max_tokens and may be cut off]"
                self._save()
                return text

            results = []
            for call in calls:
                out, is_error = self._run_tool(call)
                mark = "✗" if is_error else "✓"
                print(f"[step {step}] 🔧 {call['name']}({json.dumps(call.get('input', {}))[:120]}) {mark} {out[:200]}")
                self._trace("tool", step=step, name=call["name"], input=call.get("input"),
                            output=out, is_error=is_error)
                results.append({"type": "tool_result", "tool_use_id": call["id"],
                                "content": out, "is_error": is_error})

            # All results for one assistant turn go back together in a single user message
            self.messages.append({"role": "user", "content": results})
            self._save()                        # only at points where every tool_use has its result

        self._trace("stopped", reason="max_steps")
        return f"[stopped: hit the {self.max_steps} step budget]"


SYSTEM_PROMPT = (
    "You are a careful, capable assistant running inside a tool harness. "
    "Use tools whenever they make your answer more reliable, especially for arithmetic and files. "
    "All file paths are relative to a sandboxed workspace. "
    "If a tool returns an error, read it and adjust instead of repeating the same call. "
    "When the task is complete, reply with a short summary and no further tool calls."
)


def main():
    args = sys.argv[1:]
    if args and args[0] == "mock":
        agent = Agent(ScriptedModel(DEMO_SCRIPT), SYSTEM_PROMPT)
        print("\n=== FINAL ===\n" + agent.run("Compute 17*23+4, timestamp it, and save a report."))
        return

    if args:
        agent = Agent(AnthropicModel(), SYSTEM_PROMPT)
        print("\n=== FINAL ===\n" + agent.run(" ".join(args)))
        return

    agent = Agent(AnthropicModel(), SYSTEM_PROMPT,
                  session_path=os.environ.get("HARNESS_SESSION", "session.json"))

    print("Interactive mode. The agent remembers the conversation. Ctrl+C or 'exit' to quit.")
    while True:
        try:
            task = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if task.lower() in {"exit", "quit"}:
            break
        if task:
            print("\nagent> " + agent.run(task))


if __name__ == "__main__":
    main()
