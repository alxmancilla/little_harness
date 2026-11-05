"""Evaluate the support example offline; --live explicitly permits paid calls. Never loads .env."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from examples import support_agent as app
from harness import ModelConfigError, RunLimits, make_model

from .dataset import load_cases
from .report import markdown_report, review_template
from .scoring import score_result, summarize


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def write_json(path, value):
    with path.open("xb") as target:
        target.write(json_bytes(value))


def model_fingerprint(model):
    """Hash known client settings rather than persist endpoints, credentials or raw configuration."""
    settings = {name: getattr(model, name, None) for name in ("provider", "model", "base_url", "max_tokens")}
    return hashlib.sha256(json_bytes(settings)).hexdigest()


def skipped_result(case, status="not_run"):
    return {"id": case["id"], "status": status, "execution_status": status,
            "stop_reason": status, "answer": None, "recommendation": None,
            "checks": {"valid_output": False, "escalation_matches_draft": False, "required_sources_cited": False},
            "usage": {key: 0 for key in ("model_calls", "tool_calls_requested", "tool_calls_attempted",
                                        "tool_errors", "input_tokens", "output_tokens")},
            "elapsed_seconds": 0, "tools_requested": [], "model_config_sha256": None}


def evaluate_case(case, model, evidence, output, limits):
    """Only the input is shown to a fresh agent; expectations never enter the prompt or tools."""
    output.mkdir()
    agent = app.create_agent(model, evidence["documents"], evidence["products"], trace_path=output / "trace.jsonl",
                             limits=limits, trace_include_content=False)
    result = agent.run_result(json.dumps(case["input"]))
    value, checks = score_result(result, agent.messages, case)
    requested = [block["name"] for message in agent.messages
                 if message["role"] == "assistant" and isinstance(message["content"], list)
                 for block in message["content"] if block["type"] == "tool_use"]
    return {"id": case["id"], "status": "invalid_output" if result.status == "completed" and value is None
            else result.status, "execution_status": result.status, "stop_reason": result.stop_reason,
            "answer": result.answer, "recommendation": value, "checks": checks,
            "usage": asdict(result.usage), "elapsed_seconds": result.elapsed_seconds,
            "tools_requested": requested, "model_config_sha256": model_fingerprint(model)}


def run_evaluation(output, *, live=False, limit=None, limits=None, model_factory=None):
    """Create a new run only; no retries, automatic judging, draft saving or previous-run mutation."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("choose a new output directory")
    if type(live) is not bool:
        raise ValueError("live must be a boolean")
    dataset = load_cases()
    if limit is not None and (type(limit) is not int or not 1 <= limit <= len(dataset["cases"])):
        raise ValueError("invalid case limit")
    limits = limits if limits is not None else RunLimits(max_model_calls=6, max_tool_calls=6, max_seconds=30)
    if not isinstance(limits, RunLimits):
        raise ValueError("limits must be RunLimits")
    cases = dataset["cases"][:limit]
    evidence = json.loads(json_bytes({"documents": app.load_documents(), "products": app.PRODUCTS}))
    factory = model_factory if model_factory is not None else (make_model if live else app.OfflineModel)
    try:
        first_model = factory()
    except ModelConfigError:
        raise ValueError("invalid live model configuration") from None
    repo = Path(__file__).resolve().parents[2]
    sources = [*sorted((repo / "harness").glob("*.py")), repo / "examples/support_agent.py",
               *sorted(Path(__file__).parent.glob("*.py"))]
    contract = {"schema_version": 1, "mode": "live" if live else "offline-smoke-not-quality",
                "started_at": datetime.now(timezone.utc).isoformat(), "limits_per_case": asdict(limits),
                "dataset_sha256": hashlib.sha256(json_bytes(dataset)).hexdigest(),
                "evidence_sha256": hashlib.sha256(json_bytes(evidence)).hexdigest(),
                "source_sha256": {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in sources}, "case_ids": [case["id"] for case in cases],
                "expectations": dataset["review_status"], "trace_content": False}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run.json", contract)
    write_json(output / "cases.json", dataset)
    write_json(output / "inputs.json", [case["input"] for case in cases])
    write_json(output / "evidence.json", evidence)
    write_json(output / "review.json", review_template(cases))
    rows, halted, interrupted = [], False, False
    with (output / "results.jsonl").open("x", encoding="utf-8") as target:
        for index, case in enumerate(cases):
            if halted:
                row = skipped_result(case)
            else:
                started = time.monotonic()
                try:
                    model = first_model if index == 0 else factory()
                    row = evaluate_case(case, model, evidence, output / case["id"], limits)
                except KeyboardInterrupt:
                    row = skipped_result(case, "interrupted")
                    row["usage"] = None  # The interrupted operation may already have incurred usage.
                    row["elapsed_seconds"] = time.monotonic() - started
                interrupted |= row["status"] == "interrupted"
                halted = interrupted or row["status"] == "model_error"
            rows.append(row)
            target.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            target.flush()
    summary = {**summarize(rows, live), "interrupted": interrupted}
    write_json(output / "summary.json", summary)
    with (output / "report.md").open("x", encoding="utf-8") as target:
        target.write(markdown_report(summary, rows))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new directory; never overwrite a previous run")
    parser.add_argument("--live", action="store_true", help="use shell provider settings; may incur costs")
    parser.add_argument("--limit", type=int, help="run the first N cases; start live evaluation with 1")
    parser.add_argument("--max-model-calls", type=int, default=6)
    parser.add_argument("--max-tool-calls", type=int, default=6)
    parser.add_argument("--max-seconds", type=float, default=30, help="cooperative per-case limit, not a hard timeout")
    args = parser.parse_args()
    try:
        limits = RunLimits(max_model_calls=args.max_model_calls, max_tool_calls=args.max_tool_calls,
                           max_seconds=args.max_seconds)
        summary = run_evaluation(args.output, live=args.live, limit=args.limit, limits=limits)
    except (OSError, ValueError, TypeError, KeyError):
        parser.exit(1, "Support evaluation failed; check local files, limits, model configuration and new output path.\n")
    except KeyboardInterrupt:
        return 130
    print(f"{summary['mode']}: {summary['checks_passed']['valid_output']}/{summary['cases']} valid outputs; "
          f"{summary['automated_passes']}/{summary['cases']} pass draft expectations. "
          "Answer quality is unscored; human review is required.")
    print("Saved report.md, results.jsonl and review.json in the selected output directory.")
    if summary["interrupted"]:
        return 130
    passes = summary["automated_passes"] if args.live else summary["checks_passed"]["valid_output"]
    return 0 if passes == summary["cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())