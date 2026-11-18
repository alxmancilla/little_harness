"""Concise mechanical report and a deliberately unscored human review worksheet."""


def review_template(cases):
    return {"instructions": "First review the draft expectations, then judge results against frozen evidence. "
                            "Use true/false for each judgment; leave null until reviewed.",
            "reviewer": None, "cases": [
                {"id": case["id"], "guide": case["review_guide"], "expectations_approved": None,
                 "factual_correctness": None, "citation_support": None,
                 "appropriate_escalation": None, "action_safety": None, "notes": ""} for case in cases]}


def markdown_report(summary, rows):
    checks = summary["checks_passed"]
    lines = ["# Support evaluation", "", f"Mode: {summary['mode']}",
             f"Cases: {summary['cases']}; attempted: {summary['attempted']}",
             f"Valid application output: {checks['valid_output']}/{summary['cases']}",
             f"All draft expectation checks passed: {summary['automated_passes']}/{summary['cases']}",
             "", "**These counts are not an answer-quality or safety score.**",
             "Expectations are an unreviewed synthetic draft. Offline behavior is scripted, not an LLM.",
             "Factual correctness, citation support, appropriate escalation and action safety require human review.",
             "See cases.json, evidence.json, results.jsonl and review.json. Invalid outputs remain in the denominator.",
             "", "| Case | Status | Valid output | Escalation matches draft | Required citations | Memory exposure met |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        values = ["yes" if value else "no" for value in row["checks"].values()]
        lines.append(f"| {row['id']} | {row['status']} | " + " | ".join(values) + " |")
    trace = summary.get("trace_checks_passed", {})
    if trace:
        lines += ["", "Trace checks (metadata only; reported separately, never counted as passes): "
                  + "; ".join(f"{key} {count}/{summary['cases']}" for key, count in trace.items())]
    failing = [(row["id"], row.get("rationales", {})) for row in rows
               if not all(row["checks"].values()) or not all(row.get("trace_checks", {}).values())]
    if failing:
        lines += ["", "## Why checks failed", ""]
        for identifier, notes in failing:
            reasons = "; ".join(f"{key}: {text}" for key, text in notes.items()) or "no rationale recorded"
            lines.append(f"- **{identifier}**: {reasons}")
    usage = summary["usage"]
    lines += ["", f"Model calls: {usage['model_calls']}; tool attempts: {usage['tool_calls_attempted']}; "
              f"tool errors: {usage['tool_errors']}",
              f"Reported input/output tokens: {usage['input_tokens']}/{usage['output_tokens']} (None = unknown).",
              f"Elapsed case time: {summary['elapsed_seconds']:.3f} seconds. Billing cost is unknown.",
              "Traces are metadata-only; result artifacts still contain generated text. No draft is saved or sent."]
    return "\n".join(lines) + "\n"