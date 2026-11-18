"""Structural checks against draft expectations, never an answer-quality judge."""

from examples.support_agent import parse_recommendation
from evaluation.scorers import rationales, run_scorers, scorer, values


@scorer()
def escalation_matches_draft(value, case, **_):
    expected, actual = case["expected"]["needs_human"], value["needs_human"]
    return actual is expected, f"needs_human={actual}, draft expects {expected}"


@scorer()
def required_sources_cited(value, case, **_):
    missing = sorted(set(case["expected"]["required_sources"]) - set(value["sources"]))
    return not missing, f"missing: {', '.join(missing)}" if missing else ""


ANSWER_SCORERS = (escalation_matches_draft, required_sources_cited)


def score_result(result, messages, case):
    """Return (recommendation, checks, rationales); invalid outputs stay in the denominator.

    Rationales are short scorer explanations and never include parser error text.
    """
    checks = {"valid_output": False, "escalation_matches_draft": False, "required_sources_cited": False}
    try:
        value = parse_recommendation(result, messages, case["input"]["sku"])
    except (ValueError, KeyError, TypeError, RecursionError):
        return None, checks, {"valid_output": f"no valid recommendation (status: {result.status})"}
    scores = run_scorers(ANSWER_SCORERS, value=value, case=case)
    checks.update(valid_output=True, **values(scores))
    return value, checks, rationales(scores)


def summarize(rows, live):
    """Resource totals are unknown if any attempted case lacks that measurement."""
    checks = ("valid_output", "escalation_matches_draft", "required_sources_cited")
    usage_keys = ("model_calls", "tool_calls_requested", "tool_calls_attempted", "tool_errors",
                  "input_tokens", "output_tokens")
    usage = {}
    for key in usage_keys:
        values = [row["usage"].get(key) if row["usage"] is not None else None for row in rows]
        usage[key] = sum(values) if all(value is not None for value in values) else None
    return {"mode": "live" if live else "offline-smoke-not-quality",
            "cases": len(rows), "attempted": sum(row["status"] != "not_run" for row in rows),
            "automated_passes": sum(all(row["checks"].values()) for row in rows),
            "checks_passed": {key: sum(row["checks"][key] for row in rows) for key in checks},
            "trace_checks_passed": {key: sum(row.get("trace_checks", {}).get(key, False) for row in rows)
                                    for key in sorted({key for row in rows for key in row.get("trace_checks", {})})},
            "statuses": {status: sum(row["status"] == status for row in rows)
                         for status in sorted({row["status"] for row in rows})},
            "usage": usage, "elapsed_seconds": sum(row["elapsed_seconds"] for row in rows),
            "answer_quality": None, "human_review_pending": len(rows),
            "expectations": "unreviewed-synthetic-draft"}