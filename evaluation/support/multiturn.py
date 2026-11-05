"""Evaluate whole support conversations offline; --live explicitly permits paid calls. Never loads .env.

Each scenario keeps one agent and its history across turns. Every turn is scored with
the single-turn checks, and the scenario passes only if every turn passes.
"""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from examples import support_agent as app
from harness import RunLimits, make_model

from .dataset import load_scenarios
from .evaluate import json_bytes, model_fingerprint, write_json
from .scoring import score_result

TURN_CHECKS = ("valid_output", "escalation_matches_draft", "required_sources_cited")


def history_retained(messages, turns):
    """Runtime check: every earlier task is still in history, in order, when later turns run."""
    tasks = [json.loads(m["content"]) for m in messages if m["role"] == "user" and isinstance(m["content"], str)]
    return tasks == [turn["input"] for turn in turns]


def evaluate_scenario(scenario, model, evidence, output, limits):
    """Expectations never enter the prompt, tools or history; only each turn's input does."""
    output.mkdir()
    agent = app.create_agent(model, evidence["documents"], evidence["products"], trace_path=output / "trace.jsonl",
                             limits=limits, trace_include_content=False)
    turns, ran, stopped = [], [], False
    for turn in scenario["turns"]:
        # After an incomplete turn the next task would join a tool-result turn and blur provenance.
        if stopped:
            turns.append({"status": "not_run", "checks": dict.fromkeys(TURN_CHECKS, False), "usage": None})
            continue
        result = agent.run_result(json.dumps(turn["input"]))
        value, checks = score_result(result, agent.messages, turn)
        ran.append(turn)
        stopped = result.status != "completed"
        turns.append({"status": "invalid_output" if result.status == "completed" and value is None
                      else result.status, "stop_reason": result.stop_reason, "answer": result.answer,
                      "recommendation": value, "checks": checks, "usage": asdict(result.usage)})
    trajectory = {"all_turns_completed": all(t["status"] == "completed" for t in turns),
                  "history_retained": history_retained(agent.messages, ran)}
    return {"id": scenario["id"], "turns": turns, "trajectory": trajectory,
            "passed": all(trajectory.values()) and all(all(t["checks"].values()) for t in turns),
            "model_config_sha256": model_fingerprint(model)}


def summarize(rows, live):
    turns = [turn for row in rows for turn in row["turns"]]
    return {"mode": "live" if live else "offline-smoke-not-quality", "scenarios": len(rows), "turns": len(turns),
            "scenarios_passed": sum(row["passed"] for row in rows),
            "turn_checks_passed": {key: sum(turn["checks"][key] for turn in turns) for key in TURN_CHECKS},
            "history_retained": sum(row["trajectory"]["history_retained"] for row in rows),
            "statuses": {status: sum(t["status"] == status for t in turns) for status in sorted({t["status"] for t in turns})},
            "answer_quality": None, "human_review_pending": len(rows), "expectations": "unreviewed-synthetic-draft"}


def markdown_report(summary, rows):
    lines = ["# Multi-turn support evaluation", "", f"Mode: {summary['mode']}",
             f"Scenarios passing every turn check: {summary['scenarios_passed']}/{summary['scenarios']} "
             f"({summary['turns']} turns)", f"History retained across turns: {summary['history_retained']}/{summary['scenarios']}",
             "", "**These counts are not an answer-quality or safety score.** Expectations are an unreviewed synthetic",
             "draft, and offline behavior is scripted, not an LLM. Review each whole conversation by hand.",
             "", "| Scenario | Turn | Status | Valid output | Escalation matches draft | Required citations |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        for number, turn in enumerate(row["turns"], 1):
            marks = ["yes" if turn["checks"][key] else "no" for key in TURN_CHECKS]
            lines.append(f"| {row['id']} | {number} | {turn['status']} | " + " | ".join(marks) + " |")
    return "\n".join(lines) + "\n"


def run_evaluation(output, *, live=False, limits=None, model_factory=None):
    """Create a new run only. A model error or interruption stops the remaining scenarios."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("choose a new output directory")
    if type(live) is not bool:
        raise ValueError("live must be a boolean")
    limits = limits if limits is not None else RunLimits(max_model_calls=6, max_tool_calls=6, max_seconds=30)
    if not isinstance(limits, RunLimits):
        raise ValueError("limits must be RunLimits")
    dataset = load_scenarios()
    evidence = json.loads(json_bytes({"documents": app.load_documents(), "products": app.PRODUCTS}))
    factory = model_factory if model_factory is not None else (make_model if live else app.OfflineModel)
    try:
        first_model = factory()
    except SystemExit:
        raise ValueError("invalid live model configuration") from None
    repo = Path(__file__).resolve().parents[2]
    sources = [*sorted((repo / "harness").glob("*.py")), repo / "examples/support_agent.py",
               *sorted(Path(__file__).parent.glob("*.py"))]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run.json", {
        "schema_version": 1, "mode": "live" if live else "offline-smoke-not-quality",
        "started_at": datetime.now(timezone.utc).isoformat(), "limits_per_turn": asdict(limits),
        "dataset_sha256": hashlib.sha256(json_bytes(dataset)).hexdigest(),
        "evidence_sha256": hashlib.sha256(json_bytes(evidence)).hexdigest(),
        "source_sha256": {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "trace_content": False})
    write_json(output / "scenarios.json", dataset)
    write_json(output / "evidence.json", evidence)
    write_json(output / "review.json", {"instructions": "Judge each whole conversation against frozen evidence; "
                                                        "leave null until reviewed.", "reviewer": None, "scenarios": [
        {"id": s["id"], "guide": s["review_guide"], "expectations_approved": None, "context_carried_correctly": None,
         "factual_correctness": None, "appropriate_escalation": None, "notes": ""} for s in dataset["scenarios"]]})
    rows, halted = [], False
    with (output / "results.jsonl").open("x", encoding="utf-8") as target:
        for index, scenario in enumerate(dataset["scenarios"]):
            if halted:
                row = {"id": scenario["id"], "passed": False, "model_config_sha256": None,
                       "trajectory": {"all_turns_completed": False, "history_retained": False},
                       "turns": [{"status": "not_run", "checks": dict.fromkeys(TURN_CHECKS, False), "usage": None}
                                 for _ in scenario["turns"]]}
            else:
                started = time.monotonic()
                model = first_model if index == 0 else factory()
                row = evaluate_scenario(scenario, model, evidence, output / scenario["id"], limits)
                row["elapsed_seconds"] = time.monotonic() - started
                halted = any(turn["status"] in ("model_error", "interrupted") for turn in row["turns"])
            rows.append(row)
            target.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            target.flush()
    summary = {**summarize(rows, live), "halted": halted}
    write_json(output / "summary.json", summary)
    with (output / "report.md").open("x", encoding="utf-8") as target:
        target.write(markdown_report(summary, rows))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new directory; never overwrite a previous run")
    parser.add_argument("--live", action="store_true", help="use shell provider settings; may incur costs")
    parser.add_argument("--max-model-calls", type=int, default=6, help="per turn")
    parser.add_argument("--max-tool-calls", type=int, default=6, help="per turn")
    parser.add_argument("--max-seconds", type=float, default=30, help="cooperative per-turn limit, not a hard timeout")
    args = parser.parse_args()
    try:
        limits = RunLimits(max_model_calls=args.max_model_calls, max_tool_calls=args.max_tool_calls,
                           max_seconds=args.max_seconds)
        summary = run_evaluation(args.output, live=args.live, limits=limits)
    except (OSError, ValueError, TypeError, KeyError):
        parser.exit(1, "Multi-turn evaluation failed; check local files, limits, model configuration and new output path.\n")
    print(f"{summary['mode']}: {summary['scenarios_passed']}/{summary['scenarios']} scenarios pass every draft turn "
          "check. Answer quality is unscored; human review is required.")
    print("Saved report.md, results.jsonl and review.json in the selected output directory.")
    if "interrupted" in summary["statuses"]:
        return 130
    passed = (summary["scenarios_passed"] == summary["scenarios"] if args.live
              else summary["turn_checks_passed"]["valid_output"] == summary["turns"])
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
