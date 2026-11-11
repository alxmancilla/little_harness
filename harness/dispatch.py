"""Tool dispatch: ordered gates, journaled invocation and isolated execution."""

from dataclasses import dataclass
from typing import Optional

from .journal import JournalError
from .memory import _tool_context
from .policy import PolicyDecision
from .runtime import BudgetExceeded
from .tool_execution import ToolOutcome, run_isolated_tool, validate_tool_timeout
from .tools import validate_tool_input


@dataclass(frozen=True)
class ToolCallOutcome:
    """Result of one tool request; execution is recorded for timed/journaled or denied actions."""

    output: str
    is_error: bool
    execution: Optional[ToolOutcome] = None
    dispatched: bool = False
    action_id: Optional[str] = None
    stop_reason: Optional[str] = None


class ToolDispatchMixin:
    """Agent methods that pass one tool request through each gate and run it once."""

    def _run_tool(self, call):
        """Pass one request through each gate in order; return a ToolCallOutcome.

        A gate returns a rejection to stop before dispatch, or None to continue.
        Budget/cancellation exceptions propagate.
        """
        self._active_action_id = None
        name, args = call["name"], call.get("input", {})
        entry = self.tools.get(name)
        rejection = (self._gate_admission(name, entry)
                     or self._gate_known_tool(name, entry)
                     or self._gate_validation(entry, args))
        if rejection:
            return rejection
        args, rejection = self._gate_policy(call, entry, args)
        if rejection:
            return rejection
        timeout, rejection = self._gate_timeout(entry)
        rejection = rejection or self._gate_approval(call, entry, args)
        if rejection:
            return rejection
        if self._budget is not None:
            self._budget.check("work")
        return self._dispatch(call, entry, args, timeout)

    def _gate_admission(self, name, entry):
        """Optional application admission; it learns only whether the tool exists."""
        if self.tool_admission is None:
            return None
        rejection = self.tool_admission(name, entry is not None)
        if rejection is None:
            return None
        if type(rejection) is not str or not rejection:
            raise ValueError("tool_admission must return None or a nonempty rejection string")
        return ToolCallOutcome(rejection, True)

    def _gate_known_tool(self, name, entry):
        """Only tools in this agent's mapping can run."""
        if entry is None:
            return ToolCallOutcome(f"unknown tool '{name}'. Available: {', '.join(self.tools)}", True)
        return None

    def _gate_validation(self, entry, args):
        """Arguments must match the advertised schema."""
        try:
            validate_tool_input(args, entry["schema"].get("input_schema", {}))
        except ValueError as e:
            return ToolCallOutcome(str(e), True)
        return None

    def _gate_policy(self, call, entry, args):
        """Return (args, rejection); an enabled policy snapshots args before checking them."""
        if self.tool_policy is None:
            return args, None
        name = call["name"]
        try:
            args = self.tool_policy.snapshot(args)
            decision = self.tool_policy.check(name, args, declared_read_only=entry.get("read_only"))
        except Exception as e:
            decision = PolicyDecision(False, "policy_error", type(e).__name__)
        self._trace("tool_policy", name=name, tool_use_id=call.get("id"),
                    allowed=decision.allowed, reason=decision.reason,
                    content_fields=None if decision.error is None else {"error": decision.error})
        self._emit("tool_policy", name=name, tool_call_id=call.get("id"),
                   allowed=decision.allowed, reason=decision.reason)
        if not decision.allowed:
            stop_reason = None
            if decision.reason == "policy_error":
                stop_reason = "tool_policy_error"
                output = "tool policy check failed; action was not run; fix the policy before continuing"
            else:
                output = f"tool policy denied this action ({decision.reason}); action was not run"
            return args, ToolCallOutcome(output, True, ToolOutcome("not_run", output, True), stop_reason=stop_reason)
        if self._budget is not None:
            self._budget.check("work")
        return args, None

    def _gate_timeout(self, entry):
        """Return (timeout, rejection); a per-tool timeout overrides the agent default."""
        timeout = entry.get("timeout_seconds", self.tool_timeout_seconds)
        try:
            validate_tool_timeout(timeout)
        except ValueError:
            return timeout, ToolCallOutcome("invalid tool timeout configuration; action was not run", True)
        return timeout, None

    def _gate_approval(self, call, entry, args):
        """Risky tools need approval from trusted application code."""
        if not entry["risky"]:
            return None
        name = call["name"]
        self._emit("approval_requested", name=name, tool_call_id=call.get("id"), content={"input": args})
        try:
            approval_args = self.tool_policy.snapshot(args) if self.tool_policy is not None else args
            allowed = bool(self.approve(name, approval_args))
        except BaseException:
            self._emit("approval_decided", name=name, tool_call_id=call.get("id"), decision="aborted")
            raise
        self._emit("approval_decided", name=name, tool_call_id=call.get("id"),
                   decision="allowed" if allowed else "denied")
        if not allowed:
            return ToolCallOutcome("the user denied this action", True)
        return None

    def _dispatch(self, call, entry, args, timeout):
        """Commit the journal action, invoke once, then record the outcome."""
        name = call["name"]
        action_id = None
        if self.journal is not None:
            if self._journal_run_id is None:
                raise JournalError()
            action_id = self.journal.begin(self._journal_run_id, call["id"], name, entry["risky"])
            self._active_action_id = action_id
        try:
            if action_id is not None:
                self._trace("tool_started", name=name, tool_use_id=call["id"], action_id=action_id)
            if self._budget is not None:
                self._budget.check("work")
            self._emit("tool_started", name=name, tool_call_id=call.get("id"), risky=bool(entry["risky"]),
                       action_id=action_id, content={"input": args})
            if self._budget is not None and self.on_event is not None:
                self._budget.check("work")
            token = _tool_context.set((self._event_run_id, self.max_tool_output))
            try:
                outcome, stop_reason = self._invoke_tool(entry, args, timeout)
            finally:
                _tool_context.reset(token)
        except BaseException as error:
            if action_id is not None:
                try:
                    known_not_run = isinstance(error, BudgetExceeded)
                    self.journal.finish(action_id, "not_run" if known_not_run else "outcome_unknown",
                                        not known_not_run)
                except JournalError:
                    pass
            raise
        if outcome.outcome_unknown and stop_reason is None:
            stop_reason = "tool_outcome_unknown"
        if action_id is not None:
            try:
                self.journal.finish(action_id, outcome.status, outcome.outcome_unknown)
            except JournalError:
                stop_reason = "journal_error"
        execution = outcome if timeout is not None or self.journal is not None else None
        return ToolCallOutcome(outcome.output, outcome.is_error, execution, dispatched=True,
                               action_id=action_id, stop_reason=stop_reason)

    def _emit_tool_finished(self, call, output, is_error, *, status, outcome_unknown=False, action_id=None):
        self._emit("tool_finished", name=call["name"], tool_call_id=call["id"],
                   status=status, is_error=is_error, outcome_unknown=outcome_unknown, action_id=action_id,
                   content={"input": call.get("input", {}), "output": output})

    def _execute_isolated(self, entry, args, timeout):
        """One worker seam; worker_env=None inherits the parent environment."""
        return run_isolated_tool(entry["fn"], args, timeout, self.max_tool_output,
                                 risky=entry["risky"], env=self.worker_env,
                                 memory_context=_tool_context.get())

    def _invoke_tool(self, entry, args, timeout):
        """Invoke once; return (ToolOutcome, stop_reason). With journaling, a risky inline failure is uncertain."""
        if timeout is not None:
            timeout_reason = "tool_timeout"
            remaining = self._budget.remaining_seconds if self._budget is not None else None
            if remaining is not None and remaining <= timeout:
                self._budget.check("work")
                timeout, timeout_reason = remaining, "max_seconds"
            outcome = self._execute_isolated(entry, args, timeout)
            stop_reason = None
            if outcome.outcome_unknown:
                stop_reason = timeout_reason if outcome.status == "timed_out" else "tool_outcome_unknown"
            return outcome, stop_reason
        try:
            out = str(entry["fn"](**args))
        except Exception as e:
            if self.journal is not None and entry["risky"]:
                return ToolOutcome("failed", "tool execution failed; outcome unknown; inspect effects before retrying. "
                                   "Do not retry automatically.", True, True), None
            return ToolOutcome("failed", f"{type(e).__name__}: {e}", True), None
        if len(out) > self.max_tool_output:
            out = out[:self.max_tool_output] + f"\n[truncated, {len(out)} characters total]"
        return ToolOutcome("completed", out, False), None
