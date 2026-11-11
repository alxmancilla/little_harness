"""Reusable execution loop. Application prompts, tools and paths are explicit."""

import json
import sys
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .config import ContextConfig, ObservabilityConfig, ToolConfig, merge_group
from .compaction import (CHARS_PER_TOKEN, SUMMARY_PREFIX, SUMMARY_SYSTEM, CompactionMixin, _size,
                         render_transcript)
from .dispatch import ToolCallOutcome, ToolDispatchMixin
from .journal import ActionJournal, JournalError
from .events import RunEvent
from .runtime import BudgetExceeded, ExecutionBudget, RunLimits, RunResult
from .model_clients import ModelError
from .tool_execution import ToolExecutionStopped, validate_tool_timeout, validate_worker_env
from .policy import ToolPolicy
from .session import SessionFile, SessionStore

# Re-exported for existing imports from harness.agent; the implementations moved.
__all__ = ["Agent", "ask_human", "render_transcript", "ToolCallOutcome", "SUMMARY_SYSTEM", "SUMMARY_PREFIX",
           "CHARS_PER_TOKEN"]


def ask_human(name, args, preview=500):
    """Default terminal approval; deny when input is unavailable."""
    print(f"\n  ⚠  The agent wants to run {name} with:")
    for key, value in args.items():
        text = value if isinstance(value, str) else json.dumps(value)
        if len(text) > preview:
            text = text[:preview] + f"\n... [preview only, {len(text)} characters total]"
        print(f"     {key}: " + text.replace("\n", "\n       "))
    try:
        return input("     Allow? [y/N] ").strip().lower() == "y"
    except EOFError:
        print("\n     (no input available, denied)")
        return False


def _check_response(response):
    """Reject malformed model responses before history changes; other block types pass through.

    A tool_use input of the wrong shape is not checked here: argument validation returns it
    to the model as a recoverable tool error.
    """
    def bad(problem):
        raise ModelError(f"malformed model response: {problem}", "malformed_response")

    if not isinstance(response, dict):
        bad("expected a dict")
    content = response.get("content")
    if not isinstance(content, list):
        bad("content must be a list")
    if response.get("stop_reason") is not None and type(response["stop_reason"]) is not str:
        bad("stop_reason must be a string or None")
    truncated = response.get("truncated_tool_ids")
    if truncated is not None and not isinstance(truncated, (list, tuple)):
        bad("truncated_tool_ids must be a list")
    ids = set()
    for block in content:
        if not isinstance(block, dict) or type(block.get("type")) is not str:
            bad("each content block must be a dict with a string type")
        if block["type"] == "text" and type(block.get("text")) is not str:
            bad("text blocks need string text")
        if block["type"] == "tool_use":
            if type(block.get("id")) is not str or not block["id"] or block["id"] in ids:
                bad("tool_use blocks need a unique nonempty string id")
            if type(block.get("name")) is not str:
                bad("tool_use blocks need a string name")
            ids.add(block["id"])


_OBSERVABILITY_DEFAULTS = dict(trace_path="trace.jsonl", trace_context=None, trace_include_content=False,
                               on_event=None, event_include_content=False, quiet=False)
_TOOL_DEFAULTS = dict(approve=ask_human, tool_policy=None, tool_admission=None, max_tool_output=8000,
                      tool_timeout_seconds=None, writable_roots=(), worker_env=None)
_CONTEXT_DEFAULTS = dict(compact_at=40000, compact_at_tokens=None, session_path=None, session_store=None)


class Agent(CompactionMixin, ToolDispatchMixin):
    """Synchronous execution with explicit application tools.

    tool_admission(name, available) may reject a non-truncated attempt before input
    validation: return None to continue, or a nonempty error string to reject.
    It cannot grant authority; invalid results/errors propagate without dispatch.
    trace_context snapshots JSON annotations; runtime fields take precedence.
    compact_at_tokens, when set, replaces the compact_at character threshold with an
    estimated token threshold: the last reported input_tokens plus characters added since,
    or characters alone, at CHARS_PER_TOKEN characters per token.
    get_steering(), when set, is called after each saved tool batch; nonempty returned
    text is added to that user turn before the next model call. It cannot authorize tools.
    worker_env, when set, is the complete environment of timed-tool workers instead of a
    copy of os.environ. Inline tools still see the parent environment. It is not a sandbox.
    session_store and journal accept any SessionStore or ActionJournal; session_store
    is an alternative to session_path.
    observability, tool_config and context optionally group the matching keyword
    arguments (ObservabilityConfig, ToolConfig, ContextConfig); a group cannot be
    combined with any of its keyword arguments set to a non-default value.
    """

    def __init__(self, model, system, max_steps=12, approve=ask_human, trace_path="trace.jsonl",
                 compact_at=40000, session_path=None, *, tools, limits=None, tool_timeout_seconds=None,
                 on_event=None, event_include_content=False, quiet=False, tool_policy=None,
                 writable_roots=(), max_tool_output=8000, tool_admission=None, trace_context=None,
                 trace_include_content=False, compact_at_tokens=None, get_steering=None, worker_env=None,
                 session_store=None, journal=None, observability=None, tool_config=None, context=None):
        self.model, self.system, self.max_steps = model, system, max_steps
        obs = merge_group("observability", observability, ObservabilityConfig, dict(
            trace_path=trace_path, trace_context=trace_context, trace_include_content=trace_include_content,
            on_event=on_event, event_include_content=event_include_content, quiet=quiet), _OBSERVABILITY_DEFAULTS)
        tc = merge_group("tool_config", tool_config, ToolConfig, dict(
            approve=approve, tool_policy=tool_policy, tool_admission=tool_admission,
            max_tool_output=max_tool_output, tool_timeout_seconds=tool_timeout_seconds,
            writable_roots=writable_roots, worker_env=worker_env), _TOOL_DEFAULTS)
        ctx = merge_group("context", context, ContextConfig, dict(
            compact_at=compact_at, compact_at_tokens=compact_at_tokens,
            session_path=session_path, session_store=session_store), _CONTEXT_DEFAULTS)
        self._configure_observability(obs["trace_path"], obs["trace_context"], obs["trace_include_content"],
                                      obs["on_event"], obs["event_include_content"], obs["quiet"])
        self._configure_tools(tools, tc["tool_policy"], tc["tool_admission"], tc["approve"],
                              tc["max_tool_output"], tc["tool_timeout_seconds"], tc["writable_roots"],
                              tc["worker_env"])
        self._configure_context(ctx["compact_at"], ctx["compact_at_tokens"], ctx["session_path"],
                                ctx["session_store"])
        self._configure_controls(limits, get_steering, journal)
        # Per-run state, reset by run_result().
        self._budget = None
        self._event_run_id, self._event_step, self._event_sequence = None, None, 0
        # Journal action in flight, read only when an exception interrupts dispatch.
        self._active_action_id = self._journal_run_id = None
        self.messages = self._load()

    def _configure_observability(self, trace_path, trace_context, trace_include_content,
                                 on_event, event_include_content, quiet):
        """Trace file, event callback and console output; none can authorize execution."""
        if trace_context is not None and (not isinstance(trace_context, Mapping)
                or any(type(key) is not str for key in trace_context)
                or {"ts", "kind", "run_id"} & trace_context.keys()):
            raise ValueError("trace_context must be a mapping with non-reserved string keys")
        if type(trace_include_content) is not bool:
            raise TypeError("trace_include_content must be a boolean")
        if on_event is not None and not callable(on_event):
            raise TypeError("on_event must be callable or None")
        if type(event_include_content) is not bool or type(quiet) is not bool:
            raise TypeError("event_include_content and quiet must be booleans")
        self.trace_path = Path(trace_path)
        self.trace_context = json.loads(json.dumps(dict(trace_context or {}), allow_nan=False))
        self.trace_include_content = trace_include_content
        self.on_event, self.event_include_content, self.quiet = on_event, event_include_content, quiet
        self.event_callback_failed = False

    def _configure_tools(self, tools, tool_policy, tool_admission, approve,
                         max_tool_output, tool_timeout_seconds, writable_roots, worker_env):
        """The tool mapping and every gate that decides whether a request may run."""
        if not isinstance(tools, Mapping):
            raise TypeError("tools must be an explicit mapping; use {} for no tools")
        if tool_policy is not None and not isinstance(tool_policy, ToolPolicy):
            raise TypeError("tool_policy must be ToolPolicy or None")
        if tool_admission is not None and not callable(tool_admission):
            raise TypeError("tool_admission must be callable or None")
        if type(max_tool_output) is not int or max_tool_output <= 0:
            raise ValueError("max_tool_output must be a positive integer")
        # Snapshot membership; functions and trusted entry metadata remain application-owned.
        self.tools = dict(tools)
        if tool_policy is not None:
            tool_policy.validate_tools(self.tools)
        self.tool_policy, self.tool_admission = tool_policy, tool_admission
        self.approve = (lambda *_: False) if self.quiet and approve is ask_human else approve
        self.max_tool_output = max_tool_output
        self.tool_timeout_seconds = validate_tool_timeout(tool_timeout_seconds)
        self.worker_env = validate_worker_env(worker_env)
        self.writable_roots = tuple(Path(root).resolve() for root in writable_roots)

    def _configure_context(self, compact_at, compact_at_tokens, session_path, session_store):
        """Conversation persistence and compaction thresholds."""
        if compact_at_tokens is not None and (type(compact_at_tokens) is not int or compact_at_tokens <= 0):
            raise ValueError("compact_at_tokens must be a positive integer or None")
        if session_store is not None and (session_path or not isinstance(session_store, SessionStore)):
            raise TypeError("session_store must be a SessionStore, and not combined with session_path")
        self.compact_at, self.compact_at_tokens = compact_at, compact_at_tokens
        self._size_after_compact = 0
        # (reported input_tokens, message characters sent) from the latest main model call.
        self._context_usage = None
        self.session_path = Path(session_path) if session_path else None
        self._session = SessionFile(self.session_path) if self.session_path is not None else session_store
        # The model sees the summary, then history from _context_start; history itself is never shortened.
        self._context_start, self._context_summary = 0, None

    def _configure_controls(self, limits, get_steering, journal):
        """Per-task budgets, optional steering and the crash-recovery journal, opened last."""
        if limits is not None and not isinstance(limits, RunLimits):
            raise TypeError("limits must be RunLimits or None")
        if get_steering is not None and not callable(get_steering):
            raise TypeError("get_steering must be callable or None")
        if journal is not None and not isinstance(journal, ActionJournal):
            raise TypeError("journal must be an ActionJournal")
        self.get_steering = get_steering
        self.limits = limits if limits is not None else RunLimits()
        self.journal = journal

    def _print(self, *args, **kwargs):
        if not self.quiet:
            print(*args, **kwargs)

    def _emit(self, kind, *, content=None, **data):
        """Observers cannot mutate payloads; ordinary failures disable this run's observer."""
        if self._event_run_id is None or self.on_event is None or self.event_callback_failed:
            return
        if self._event_step is not None:
            data = {"step": self._event_step, **data}
        if self.event_include_content and content:
            data.update(content)
        self._event_sequence += 1
        try:
            event = RunEvent(self._event_run_id, self._event_sequence, kind,
                             datetime.now(timezone.utc).isoformat(), data)
            self.on_event(event)
        except Exception:
            # No exception details are printed, traced or added to model context.
            self.event_callback_failed = True
        except BaseException:
            self.event_callback_failed = True
            raise

    def _call_model(self, *args, event_purpose="main", **kwargs):
        if self._budget is not None:
            self._budget.start_model_call()
        metadata = {"purpose": event_purpose,
                    "model_call": self._budget.model_calls if self._budget is not None else None}
        try:
            self._emit("model_started", **metadata)
            if self._budget is not None and self.on_event is not None:
                self._budget.check("work")
            response = self.model(*args, **kwargs)
            _check_response(response)
        except BaseException as error:
            if self._budget is not None:
                self._budget.record_model_usage(None)
            status = ("interrupted" if isinstance(error, KeyboardInterrupt) else
                      "not_run" if isinstance(error, BudgetExceeded) else "failed")
            if isinstance(error, ModelError):
                metadata["error_category"] = getattr(error, "category", "unknown")
            self._emit("model_finished", **metadata, status=status)
            raise
        if self._budget is not None:
            self._budget.record_model_usage(response.get("usage"))
        raw_usage = response.get("usage")
        raw_usage = raw_usage if isinstance(raw_usage, dict) else {}
        usage = {key: raw_usage[key] if type(raw_usage.get(key)) is int and raw_usage[key] >= 0 else None
                 for key in ("input_tokens", "output_tokens")}
        self._emit("model_finished", **metadata, status="completed",
                   stop_reason=response.get("stop_reason"), usage=usage)
        return response

    def _load(self):
        """Restore history and the compaction point from the session store, if a session exists."""
        loaded = self._session.load() if self._session is not None else None
        if loaded is None:
            return []
        messages, compaction = loaded
        if compaction is not None:
            self._context_start, self._context_summary = compaction["start"], compaction["summary"]
        self._print(f"Resumed {len(messages)} messages from {self._session}")
        return messages

    def _save(self):
        """Write the session atomically; journal-enabled runs make it durable."""
        if self._session is None:
            return
        compaction = (None if self._context_summary is None else
                      {"start": self._context_start, "summary": self._context_summary})
        self._session.save(self.messages, compaction, durable=self.journal is not None)

    def _add_user_message(self, text):
        """Fold a resumed task into the last user turn when necessary."""
        last = self.messages[-1] if self.messages else None
        if last and last["role"] == "user":
            if isinstance(last["content"], str):
                last["content"] = [{"type": "text", "text": last["content"]}]
            last["content"].append({"type": "text", "text": text})
        else:
            self.messages.append({"role": "user", "content": text})

    def _trace(self, kind, *, content_fields=None, **data):
        """Separate runtime metadata from opt-in conversation and error content."""
        if self.trace_include_content:
            data.update(content_fields or {})
        elif "usage" in data:
            usage = data["usage"] if isinstance(data["usage"], dict) else {}
            data["usage"] = {key: usage[key] if type(usage.get(key)) is int and usage[key] >= 0 else None
                             for key in ("input_tokens", "output_tokens")}
        if self._journal_run_id is not None:
            data = {"run_id": self._journal_run_id, **data}
        record = {"ts": datetime.now(timezone.utc).isoformat(), "kind": kind, **self.trace_context, **data}
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _steer(self, step):
        """Ask for between-step guidance; add nonempty text to the saved tool-result turn."""
        if self.get_steering is None:
            return
        text = self.get_steering()
        if text is None or text == "":
            return
        if type(text) is not str:
            raise ValueError("get_steering must return None or a string")
        if not text.strip():
            return
        self.messages[-1]["content"].append({"type": "text", "text": text})
        self._save()
        self._print(f"[step {step}] 🧭 steering: {text[:200]}")
        self._trace("steering", step=step, characters=len(text), content_fields={"text": text})
        self._emit("steering", characters=len(text), content={"text": text})

    def run(self, task):
        """Compatibility API: return the same display text used by the CLI."""
        return self.run_result(task).text

    def run_result(self, task):
        """Run with fresh accounting. In-flight operations may overshoot cooperative caps."""
        if self._budget is not None:
            raise RuntimeError("Agent runs must not overlap or re-enter")
        self._budget = ExecutionBudget(self.limits)
        self._status, self._stop_reason, self._answer = "completed", "end_turn", ""
        self._event_run_id, self._event_sequence, self._event_step = uuid4().hex, 0, None
        self.event_callback_failed = False
        self._journal_run_id = self._event_run_id if self.journal is not None else None
        journal_ready, task_started, text = False, False, None
        try:
            self._emit("run_started", content={"task": task})
            if self.journal is not None:
                self.journal.assert_ready()
                journal_ready = True
            task_started = True
            text = self._run(task)
        except JournalError as e:
            self._status, self._stop_reason = "stopped", e.reason
            if task_started:
                self._save()
            self._trace("stopped", reason=e.reason)
            text = f"[stopped: {e.reason}; inspect the execution journal before continuing]"
        except BudgetExceeded as e:
            self._status, self._stop_reason = "stopped", e.reason
            self._save()
            self._trace("stopped", reason=e.reason)
            text = f"[stopped: run budget exhausted: {e.reason}]"
        except ToolExecutionStopped as e:
            self._status, self._stop_reason = "stopped", e.reason
            self._save()
            self._trace("stopped", reason=e.reason)
            detail = ("inspect the execution journal before continuing" if e.reason == "journal_error" else
                      "tool policy check failed; action was not run" if e.reason == "tool_policy_error" else
                      "tool outcome unknown; inspect effects before retrying")
            text = f"[stopped: {e.reason}; {detail}]"
        except KeyboardInterrupt:
            last = self.messages[-1] if self.messages else None
            pending = [b for b in last["content"] if b["type"] == "tool_use"] \
                if last and last["role"] == "assistant" else []
            if pending:
                self.messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": b["id"],
                     "content": "not run: cancelled by the user", "is_error": True} for b in pending]})
            self._save()
            self._trace("stopped", reason="interrupted")
            self._print()
            self._status, self._stop_reason = "interrupted", "interrupted"
            text = "[stopped: interrupted by the user]"
        finally:
            try:
                if (journal_ready and text is not None and self._status != "interrupted"
                        and self._stop_reason not in ("journal_error", "journal_recovery_required")):
                    try:
                        self.journal.close_run(self._journal_run_id)
                    except JournalError:
                        self._status, self._stop_reason = "stopped", "journal_error"
                        self._trace("stopped", reason="journal_error")
                        text = "[stopped: journal_error; inspect the execution journal before continuing]"
            finally:
                budget = self._budget
                try:
                    failed = sys.exc_info()[0] is not None
                    status, reason = (("error", "harness_error") if failed else (self._status, self._stop_reason))
                    self._event_step = None
                    if status != "completed":
                        self._emit("run_stopped", status=status, stop_reason=reason)
                    self._emit("run_finished", status=status, stop_reason=reason,
                               usage=asdict(budget.snapshot()), elapsed_seconds=budget.elapsed_seconds,
                               content={"answer": self._answer, "text": text})
                finally:
                    self._budget = None
                    self._journal_run_id = self._event_run_id = None
        return RunResult(self._status, self._answer, self._stop_reason,
                         budget.snapshot(), budget.elapsed_seconds, text)

    def _run(self, task):
        self._add_user_message(task)
        self._save()
        self._trace("task", content_fields={"text": task})
        schemas = [t["schema"] for t in self.tools.values()
                   if self.tool_policy is None or not self.tool_policy.read_only or t.get("read_only") is True]

        for step in range(1, self.max_steps + 1):
            self._event_step = step
            self._compact(step)
            streamed = []

            def on_text(delta):
                if not streamed:
                    self._print(f"\n[step {step}] 💬 ", end="", flush=True)
                streamed.append(delta)
                self._print(delta, end="", flush=True)
                self._emit("model_text", model_call=self._budget.model_calls, phase="delta",
                           characters=len(delta), content={"text": delta})

            context = self._model_context()
            try:
                resp = self._call_model(self.system, context, schemas, on_text=on_text)
            except ModelError as e:
                if streamed:
                    self._print()
                self._trace("error", step=step, category=getattr(e, "category", "unknown"),
                            content_fields={"error": str(e)})
                self._save()
                self._status, self._stop_reason = "model_error", "model_error"
                return f"[stopped: model call failed: {e}]"
            if streamed:
                self._print()
            if self.compact_at_tokens is not None:
                self._record_context_usage(resp.get("usage"), _size(context))
            content = resp["content"]
            self.messages.append({"role": "assistant", "content": content})
            self._trace("model", step=step, stop_reason=resp.get("stop_reason"),
                        usage=resp.get("usage"), content_fields={"content": content})

            for block in content:
                if block["type"] == "text" and block["text"].strip() and not streamed:
                    self._print(f"\n[step {step}] 💬 {block['text']}")
                    self._emit("model_text", model_call=self._budget.model_calls, phase="complete",
                               characters=len(block["text"]), content={"text": block["text"]})

            calls = [b for b in content if b["type"] == "tool_use"]
            self._budget.tool_calls_requested += len(calls)
            if not calls:
                text = "".join(b["text"] for b in content if b["type"] == "text")
                self._answer = text
                self._budget.check()
                self._stop_reason = resp.get("stop_reason") or "end_turn"
                if resp.get("stop_reason") == "max_tokens":
                    self._status = "max_tokens"
                    text += "\n[warning: response hit max_tokens and may be cut off]"
                elif resp.get("stop_reason") == "refusal":
                    self._status = "refusal"
                    text += "\n[warning: the model declined to continue (stop_reason: refusal)]"
                self._save()
                return text

            results = []
            truncated = resp.get("truncated_tool_ids") or ()
            active_id = None
            try:
                for call in calls:
                    if call["id"] in truncated:
                        result = ToolCallOutcome("the arguments for this call were cut off (max_tokens), so it was "
                                                 "not run. Retry with smaller inputs, e.g. split a large file into "
                                                 "several writes.", True)
                    else:
                        self._budget.start_tool_call()
                        active_id = call["id"]
                        result = self._run_tool(call)
                    out, is_error = result.output, result.is_error
                    # Retain results before notifications so cancellation cannot erase an action.
                    results.append({"type": "tool_result", "tool_use_id": call["id"],
                                    "content": out, "is_error": is_error})
                    active_id = None
                    outcome = result.execution
                    status = outcome.status if outcome else ("failed" if is_error else "completed")
                    if not result.dispatched:
                        status = "not_run"
                    self._emit_tool_finished(call, out, is_error, status=status,
                                             outcome_unknown=bool(outcome and outcome.outcome_unknown),
                                             action_id=result.action_id)
                    mark = "✗" if is_error else "✓"
                    self._print(f"[step {step}] 🔧 {call['name']}({json.dumps(call.get('input', {}))[:120]}) {mark} {out[:200]}")
                    execution = ({"execution_status": outcome.status, "outcome_unknown": outcome.outcome_unknown}
                                 if outcome is not None else {})
                    if result.action_id is not None:
                        execution["action_id"] = result.action_id
                    self._trace("tool", step=step, name=call["name"],
                                content_fields={"input": call.get("input"), "output": out},
                                is_error=is_error, **execution)
                    if result.stop_reason is not None:
                        raise ToolExecutionStopped(result.stop_reason)
            except (KeyboardInterrupt, BudgetExceeded, ToolExecutionStopped, JournalError) as e:
                pending_events = []
                for call in calls[len(results):]:
                    if isinstance(e, BudgetExceeded):
                        status = f"not run: run budget exhausted: {e.reason}"
                    elif isinstance(e, JournalError):
                        status = "not run: execution journal blocked dispatch; inspect it before continuing"
                    elif isinstance(e, ToolExecutionStopped):
                        status = ("not run: tool policy check failed; fix the policy before continuing"
                                  if e.reason == "tool_policy_error" else
                                  "not run: execution stopped; inspect the journal and tool effects before continuing")
                    else:
                        status = ("interrupted: outcome unknown; inspect effects before retrying"
                                  if call["id"] == active_id else "not run: cancelled by the user")
                    results.append({"type": "tool_result", "tool_use_id": call["id"],
                                    "content": status, "is_error": True})
                    unknown = isinstance(e, KeyboardInterrupt) and call["id"] == active_id
                    pending_events.append((call, status, unknown,
                                           self._active_action_id if call["id"] == active_id else None))
                self._budget.record_tool_results(results)
                self.messages.append({"role": "user", "content": results})
                for call, status, unknown, action_id in pending_events:
                    self._emit_tool_finished(call, status, True,
                                             status="outcome_unknown" if unknown else "not_run",
                                             outcome_unknown=unknown, action_id=action_id)
                raise

            self.messages.append({"role": "user", "content": results})
            self._budget.record_tool_results(results)
            self._save()
            self._budget.check()
            self._steer(step)

        self._trace("stopped", reason="max_steps")
        self._status, self._stop_reason = "stopped", "max_steps"
        return f"[stopped: hit the {self.max_steps} step budget]"