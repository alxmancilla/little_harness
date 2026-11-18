"""Shared scorer interface: small named functions that return a value and a short rationale.

A scorer is any callable decorated with @scorer. It receives keyword context (for example
case, result, messages or trace) and returns a bool or (bool, rationale). Scorers are
deterministic code checks, never LLM judges, and never see expectations they don't need.
"""

import json
from pathlib import Path


def scorer(name=None):
    """Mark a function as a scorer; the name labels its column in results and reports."""
    def mark(function):
        function.scorer_name = name or function.__name__
        return function
    return mark


def run_scorers(scorers, **context):
    """Return {name: {"value", "rationale"}}; a crashing scorer fails closed with its error type."""
    scores = {}
    for function in scorers:
        name = function.scorer_name
        try:
            outcome = function(**context)
        except Exception as error:  # A broken scorer must not hide the remaining scores.
            scores[name] = {"value": False, "rationale": f"scorer error: {type(error).__name__}"}
            continue
        value, rationale = outcome if isinstance(outcome, tuple) else (outcome, "")
        if type(value) is not bool:
            raise TypeError(f"scorer {name} must return a bool")
        scores[name] = {"value": value, "rationale": str(rationale)}
    return scores


def values(scores):
    return {name: score["value"] for name, score in scores.items()}


def rationales(scores):
    return {name: score["rationale"] for name, score in scores.items() if score["rationale"]}


def read_trace(path):
    """Read a metadata trace; a missing file is an empty trace, not an error."""
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# Trace scorers read only the recorded events, so they can rescore old runs without a model.

@scorer()
def trace_present(trace, **_):
    return bool(trace), f"{len(trace)} events" if trace else "no trace events recorded"


@scorer()
def tools_before_answer(trace, **_):
    """The final answer followed at least one successful tool result (grounding happened)."""
    ok = [event["name"] for event in trace if event.get("kind") == "tool" and not event.get("is_error")]
    return bool(ok), f"successful tools: {', '.join(ok)}" if ok else "answered without a successful tool call"


@scorer()
def no_tool_errors(trace, **_):
    failed = [event["name"] for event in trace if event.get("kind") == "tool" and event.get("is_error")]
    return not failed, f"failed tools: {', '.join(failed)}" if failed else ""


@scorer()
def no_policy_denials(trace, **_):
    denied = [f"{event['name']} ({event.get('reason')})" for event in trace
              if event.get("kind") == "tool_policy" and not event.get("allowed")]
    return not denied, f"denied: {', '.join(denied)}" if denied else ""


@scorer()
def ended_normally(trace, **_):
    models = [event for event in trace if event.get("kind") == "model"]
    last = models[-1].get("stop_reason") if models else None
    return last == "end_turn", f"last stop reason: {last}"


TRACE_SCORERS = (trace_present, tools_before_answer, no_tool_errors, no_policy_denials, ended_normally)
