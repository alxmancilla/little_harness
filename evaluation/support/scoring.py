"""Structural checks against draft expectations, never an answer-quality judge."""

from examples.support_agent import parse_recommendation


def score_result(result, messages, case):
    """Keep failed executions/invalid outputs in the denominator; disclose no error text."""
    checks = {"valid_output": False, "escalation_matches_draft": False, "required_sources_cited": False}
    try:
        value = parse_recommendation(result, messages, case["input"]["sku"])
    except (ValueError, KeyError, TypeError, RecursionError):
        return None, checks
    checks.update(valid_output=True,
                  escalation_matches_draft=value["needs_human"] is case["expected"]["needs_human"],
                  required_sources_cited=set(case["expected"]["required_sources"]) <= set(value["sources"]))
    return value, checks


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
            "statuses": {status: sum(row["status"] == status for row in rows)
                         for status in sorted({row["status"] for row in rows})},
            "usage": usage, "elapsed_seconds": sum(row["elapsed_seconds"] for row in rows),
            "answer_quality": None, "human_review_pending": len(rows),
            "expectations": "unreviewed-synthetic-draft"}