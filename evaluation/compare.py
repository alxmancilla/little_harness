"""Compare two saved evaluation runs case by case, or rescore one run's saved traces offline.

    python3 -m evaluation.compare runs/A runs/B
    python3 -m evaluation.compare runs/A runs/B --output comparison.md
    python3 -m evaluation.compare runs/A --traces

Reads results.jsonl and per-case trace.jsonl only; never calls a model, never edits a run.
Works with support, multi-turn and HotpotQA runs. Deltas are descriptive, not significance
claims, and are only meaningful when both runs used the same frozen cases.
"""

import argparse
import json
from pathlib import Path

from evaluation.scorers import TRACE_SCORERS, rationales, read_trace, run_scorers, values

HOTPOT_CHECKS = ("em", "sp_em", "joint_em")


def _known_sum(numbers):
    return sum(numbers) if numbers and all(number is not None for number in numbers) else None


def normalize(row):
    """Map any supported result row to {id, folder, passed, checks, tokens, tool_calls}."""
    if "turns" in row:  # multi-turn support scenario
        checks = {f"turn{number}:{key}": value for number, turn in enumerate(row["turns"], 1)
                  for key, value in turn["checks"].items()}
        checks.update(row["trajectory"])
        usages = [turn.get("usage") for turn in row["turns"]]
        tokens = _known_sum([None if usage is None else _known_sum([usage.get("input_tokens"),
                                                                     usage.get("output_tokens")])
                             for usage in usages])
        tools = _known_sum([None if usage is None else usage.get("tool_calls_attempted") for usage in usages])
        return {"id": row["id"], "folder": row["id"], "passed": row["passed"], "checks": checks,
                "tokens": tokens, "tool_calls": tools}
    if "scores" in row:  # HotpotQA question; exact-match scores become pass/fail checks
        scores = row["scores"] or {}
        checks = {key: scores.get(key) == 1.0 for key in HOTPOT_CHECKS}
        tokens = row.get("tokens") or {}
        number = row.get("question_number")
        return {"id": row["id"], "folder": f"q{number:02d}" if type(number) is int else row["id"],
                "passed": all(checks.values()), "checks": checks,
                "tokens": _known_sum([tokens.get("input_tokens"), tokens.get("output_tokens")]),
                "tool_calls": row.get("tool_calls_executed")}
    usage = row.get("usage") or {}
    return {"id": row["id"], "folder": row["id"], "passed": all(row["checks"].values()),
            "checks": dict(row["checks"]), "rationales": row.get("rationales", {}),
            "tokens": None if not usage else _known_sum([usage.get("input_tokens"), usage.get("output_tokens")]),
            "tool_calls": usage.get("tool_calls_attempted")}


def load_run(path):
    path = Path(path)
    lines = (path / "results.jsonl").read_text(encoding="utf-8").splitlines()
    return {row["id"]: row for row in (normalize(json.loads(line)) for line in lines if line.strip())}


def trace_scores(run, row):
    return run_scorers(TRACE_SCORERS, trace=read_trace(Path(run) / row["folder"] / "trace.jsonl"))


def compare(before, after):
    left, right = load_run(before), load_run(after)
    shared = [identifier for identifier in left if identifier in right]
    cases = []
    for identifier in shared:
        a, b = left[identifier], right[identifier]
        flips = {key: (a["checks"].get(key), b["checks"].get(key))
                 for key in sorted(set(a["checks"]) | set(b["checks"]))
                 if a["checks"].get(key) != b["checks"].get(key)}
        trace_a, trace_b = values(trace_scores(before, a)), values(trace_scores(after, b))
        trace_flips = {key: (trace_a[key], trace_b[key]) for key in trace_a if trace_a[key] != trace_b[key]}
        delta = (lambda key: b[key] - a[key] if a[key] is not None and b[key] is not None else None)
        change = ("fixed" if b["passed"] and not a["passed"] else "regressed" if a["passed"] and not b["passed"]
                  else "changed" if flips or trace_flips else "same")
        cases.append({"id": identifier, "change": change, "passed": (a["passed"], b["passed"]),
                      "check_flips": flips, "trace_flips": trace_flips,
                      "tokens_delta": delta("tokens"), "tool_calls_delta": delta("tool_calls"),
                      "traces": [str(Path(before) / a["folder"] / "trace.jsonl"),
                                 str(Path(after) / b["folder"] / "trace.jsonl")]})
    return {"before": str(before), "after": str(after), "shared_cases": len(shared),
            "only_before": sorted(set(left) - set(right)), "only_after": sorted(set(right) - set(left)),
            "passed": (sum(left[i]["passed"] for i in shared), sum(right[i]["passed"] for i in shared)),
            "counts": {kind: sum(case["change"] == kind for case in cases)
                       for kind in ("fixed", "regressed", "changed", "same")},
            "cases": cases}


def _flip(pair):
    return " → ".join("pass" if value else "fail" if value is not None else "n/a" for value in pair)


def markdown(result):
    passed, counts = result["passed"], result["counts"]
    lines = ["# Run comparison", "", f"Before: `{result['before']}`", f"After: `{result['after']}`", "",
             f"Shared cases: {result['shared_cases']}; automated passes {passed[0]} → {passed[1]}",
             f"Fixed: {counts['fixed']}; regressed: {counts['regressed']}; other changes: {counts['changed']}; "
             f"unchanged: {counts['same']}", "",
             "**Descriptive only.** Check the runs used the same frozen cases and settings (run.json) before",
             "attributing a change to a prompt, model or code change. Automated checks are not quality scores."]
    if result["only_before"] or result["only_after"]:
        lines += ["", f"Not compared — only before: {result['only_before']}; only after: {result['only_after']}"]
    changed = [case for case in result["cases"] if case["change"] != "same"]
    if changed:
        lines += ["", "| Case | Change | Checks that flipped | Trace checks that flipped | Δ tokens | Δ tool calls |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for case in changed:
            flips = ", ".join(f"{key} {_flip(pair)}" for key, pair in case["check_flips"].items()) or "—"
            trace = ", ".join(f"{key} {_flip(pair)}" for key, pair in case["trace_flips"].items()) or "—"
            lines.append(f"| {case['id']} | {case['change']} | {flips} | {trace} | "
                         f"{case['tokens_delta']} | {case['tool_calls_delta']} |")
        lines += ["", "Traces to inspect:", ""]
        lines += [f"- {case['id']}: `{case['traces'][0]}` vs `{case['traces'][1]}`" for case in changed]
    return "\n".join(lines) + "\n"


def trace_report(run):
    rows = load_run(run)
    lines = ["# Trace checks", "", f"Run: `{run}`", "",
             "Recomputed from saved metadata traces; no model was called. Not counted as passes.", "",
             "| Case | " + " | ".join(scorer.scorer_name for scorer in TRACE_SCORERS) + " | Notes |",
             "| --- |" + " --- |" * (len(TRACE_SCORERS) + 1)]
    for row in rows.values():
        scores = trace_scores(run, row)
        notes = "; ".join(f"{key}: {text}" for key, text in rationales(scores).items()
                          if not scores[key]["value"]) or "—"
        lines.append(f"| {row['id']} | " + " | ".join("yes" if score["value"] else "no"
                                                     for score in scores.values()) + f" | {notes} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="one run with --traces, or two runs to compare")
    parser.add_argument("--traces", action="store_true", help="rescore one run's saved traces")
    parser.add_argument("--output", type=Path, help="write the Markdown report to a new file")
    args = parser.parse_args()
    if len(args.runs) != (1 if args.traces else 2):
        parser.error("give one run with --traces or exactly two runs to compare")
    try:
        text = trace_report(args.runs[0]) if args.traces else markdown(compare(*args.runs))
        if args.output:
            with args.output.open("x", encoding="utf-8") as target:
                target.write(text)
        else:
            print(text, end="")
    except (OSError, ValueError, KeyError, TypeError):
        parser.exit(1, "Comparison failed; check both run directories contain results.jsonl and the output is new.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
