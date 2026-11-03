"""Measured single/multi-agent HotpotQA experiments; Python standard library only.

Default: offline protocol smoke, NOT an accuracy baseline. --live explicitly opts
into the configured model. Uses the original ten teaching IDs with per-question
candidate sets, gold-only host scoring and fresh agent processes/workspaces.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import prepare as p
from .comparison import HOST_CODE_FILES, SHARED_CODE_FILES, compare_runs, load_reference
from .dev import DEFAULT_DEV_OUTPUT, load_records, read_bounded
from .scoring import parse_prediction, score_prediction

REPO = Path(__file__).resolve().parents[2]
TEACHING_SUBSET = p.DEFAULT_OUTPUT
EMPTY_PREDICTION = {"answer": "", "supporting_facts": [], "abstain": True}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    with Path(path).open("xb") as target:
        target.write(p.json_bytes(value))


def select_examples(records, limit):
    """Pin the original balanced ten IDs, not the first ten development records."""
    with (TEACHING_SUBSET / "source.json").open("rb") as source:
        data = read_bounded(source)
    manifest = json.loads((TEACHING_SUBSET / "manifest.json").read_bytes())
    if digest(data) != manifest["snapshot_sha256"]:
        raise ValueError("teaching subset checksum mismatch")
    subset = p.select_questions(json.loads(data))
    by_id = {record["_id"]: (number, record) for number, record in enumerate(records, 1)}
    selected = []
    for example in subset[:limit]:
        number, record = by_id[example["_id"]]
        if any(record[field] != example[field] for field in ("question", "context")):
            raise ValueError("development inputs differ from the pinned teaching examples")
        selected.append((number, record))
    return selected


def agent_input(number, record):
    files, question, _, _, _ = p.build_example(record, number, full_dev=True)
    payload = {"id": question["id"], "question": question["question"], "documents": {
        name: {"title": title, "sentences": sentences}
        for name, (title, sentences) in zip(question["candidates"], record["context"])}}
    return payload, files


def failure_result(identifier, status, runtime=0.0):
    return {"id": identifier, "status": status, "prediction": None, "raw_answer": None,
            "runtime_seconds": runtime, "model_calls": None, "tool_calls_requested": None,
            "tool_calls_executed": None, "tool_calls_rejected": None, "tool_errors": None,
            "tokens": {"input_tokens": None, "output_tokens": None, "complete": False},
            "cost_usd": None, "citation_validity": None, "read_evidence_fraction": None}


def run_worker(payload, files, output, *, live, max_steps, max_tool_calls, max_output_tokens, timeout,
               worker_module="agents.hotpotqa.single"):
    if worker_module not in ("agents.hotpotqa.single", "agents.hotpotqa.multi"):
        raise ValueError("unsupported worker")
    with tempfile.TemporaryDirectory(prefix="hotpotqa-baseline-") as temporary:
        root = Path(temporary)
        (root / "knowledge").mkdir()
        for name, content in files.items():
            with (root / name).open("xb") as target:
                target.write(content)
        write_json(root / "input.json", payload)  # exactly one question; no source/gold or other examples
        env = dict(os.environ)
        env["HARNESS_KNOWLEDGE"] = str(root / "knowledge")
        env["HARNESS_WORKSPACE"] = str(root / "workspace")
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(REPO), env.get("PYTHONPATH"))))
        command = [sys.executable, "-m", worker_module, "--input", str(root / "input.json"),
                   "--output", str(output / "result.json"), "--trace", str(output / "trace.jsonl"),
                   "--max-steps", str(max_steps), "--max-tool-calls", str(max_tool_calls),
                   "--max-output-tokens", str(max_output_tokens)]
        if live:
            command.append("--live")
        started = time.perf_counter()
        try:
            child = subprocess.run(command, env=env, capture_output=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return failure_result(payload["id"], "timeout", time.perf_counter() - started)
        # Do not persist stderr: unexpected provider/client exceptions may contain credentials.
        with (output / "agent.log").open("xb") as target:
            target.write(child.stdout)
        if child.returncode != 0:
            return failure_result(payload["id"], "worker_error", time.perf_counter() - started)
        try:
            result = json.loads((output / "result.json").read_bytes())
            if result["id"] != payload["id"]:
                raise ValueError("worker returned a different question")
            if result["prediction"] is not None:
                parse_prediction(json.dumps(result["prediction"]))
            return result
        except (OSError, ValueError, KeyError, TypeError):
            return failure_result(payload["id"], "worker_error", time.perf_counter() - started)


def sum_known(values):
    return sum(values) if all(value is not None for value in values) else None


def summarize(results, live):
    scores = None
    denominators = None
    if live:
        names = results[0]["scores"]
        denominators = {name: sum(row["scores"][name] is not None for row in results) for name in names}
        scores = {name: sum(row["scores"][name] or 0.0 for row in results) / denominators[name]
                  if denominators[name] else None for name in names}
    return {"mode": "live-baseline" if live else "offline-smoke-not-an-accuracy-baseline",
            "architecture": "single-agent", "questions": len(results),
            "statuses": dict(Counter(row["status"] for row in results)),
            "valid_predictions": sum(row["prediction"] is not None for row in results),
            "abstentions": sum(bool(row["prediction"] and row["prediction"]["abstain"]) for row in results),
            "scores": scores, "score_denominators": denominators,
            "runtime_seconds": sum(row["runtime_seconds"] for row in results),
            **{field: sum_known([row[field] for row in results]) for field in (
                "model_calls", "tool_calls_requested", "tool_calls_executed", "tool_calls_rejected", "tool_errors")},
            "tokens": {field: sum_known([row["tokens"][field] for row in results])
                       for field in ("input_tokens", "output_tokens")}, "cost_usd": None}


def run_baseline(output, *, dataset=DEFAULT_DEV_OUTPUT, limit=10, live=False, max_steps=12,
                 max_tool_calls=12, max_output_tokens=2048, timeout=180,
                 architecture="single-agent", reference=None):
    if any(type(value) is not int or value < 1 for value in (limit, max_steps, max_tool_calls,
                                                           max_output_tokens, timeout)):
        raise ValueError("limits must be positive integers")
    if limit > 10 or max_steps > 50 or max_tool_calls > 100 or max_output_tokens > 8192 or timeout > 1800:
        raise ValueError("baseline limits exceed supported bounds")
    if architecture not in ("single-agent", "multi-agent") or (architecture == "multi-agent" and max_steps < 2):
        raise ValueError("unsupported architecture or insufficient total model calls")
    if reference is not None and architecture != "multi-agent":
        raise ValueError("reference comparison requires the multi-agent architecture")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("choose a new output directory; previous results are never overwritten")
    output = output.resolve()
    started = time.perf_counter()
    manifest, records = load_records(dataset)
    selected = select_examples(records, limit)
    prepared = [agent_input(number, record) for number, record in selected]
    payloads = [payload for payload, _ in prepared]
    contract = {
        "version": 2, "experiment_version": "hardened-v1",
        "architecture": architecture, "mode": "live" if live else "offline-smoke",
        "selection": "original ten teaching IDs, in pinned source order (optionally a prefix)",
        "question_ids": [payload["id"] for payload in payloads],
        "dataset_snapshot_sha256": manifest["snapshot_sha256"],
        "input_fingerprint": digest(p.json_bytes(payloads)),
        "budgets": {"max_model_calls": max_steps, "max_tool_executions": max_tool_calls,
                    "max_output_tokens_per_call": max_output_tokens, "timeout_seconds_per_question": timeout},
        "tools": ["search_documents", "read_document", "calculator"], "compaction": False,
        "code_sha256": {name: digest((REPO / name).read_bytes()) for name in SHARED_CODE_FILES + HOST_CODE_FILES},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if architecture == "multi-agent":
        contract["code_sha256"]["agents/hotpotqa/multi.py"] = digest((REPO / "agents/hotpotqa/multi.py").read_bytes())
        contract["roles"] = {"order": ["researcher", "reviewer"],
                             "researcher_max_model_calls": max_steps // 2,
                             "researcher_max_tool_executions": max_tool_calls // 2,
                             "reviewer_budget": "all unused calls and tool executions", "handoff_max_chars": 16000}
    reference_rows = load_reference(reference, contract, p.json_bytes(payloads), REPO) if reference is not None else None
    if reference is not None:
        contract["reference"] = str(Path(reference).resolve())
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run.json", contract)
    write_json(output / "inputs.json", payloads)  # frozen, label-free inputs for the multi-agent comparison
    results, interrupted = [], False
    predictions = {"answer": {}, "sp": {}}
    for ordinal, ((number, record), (payload, files)) in enumerate(zip(selected, prepared), 1):
        question_output = output / f"q{ordinal:02d}"
        question_output.mkdir()
        if interrupted:
            result = failure_result(payload["id"], "not_run")
        else:
            print(f"[{ordinal}/{limit}] {'Live baseline' if live else 'Offline smoke'} ({architecture}): {payload['id']}", flush=True)
            question_started = time.perf_counter()
            try:
                worker_options = {"worker_module": "agents.hotpotqa.multi"} if architecture == "multi-agent" else {}
                result = run_worker(payload, files, question_output, live=live, max_steps=max_steps,
                                    max_tool_calls=max_tool_calls, max_output_tokens=max_output_tokens, timeout=timeout,
                                    **worker_options)
            except KeyboardInterrupt:
                result = failure_result(payload["id"], "interrupted", time.perf_counter() - question_started)
                interrupted = True
        if result["status"] == "interrupted":
            interrupted = True
        prediction = result["prediction"] or EMPTY_PREDICTION
        result["question_number"] = number
        result["input_sha256"] = digest(p.json_bytes(payload))
        result["scores"] = score_prediction(prediction, record) if live else None
        if live and result["prediction"] is None:
            result["scores"] = {name: 0.0 if value is not None else None
                                for name, value in result["scores"].items()}
        results.append(result)
        predictions["answer"][payload["id"]] = prediction["answer"]
        predictions["sp"][payload["id"]] = prediction["supporting_facts"]
        with (output / "results.jsonl").open("a", encoding="utf-8") as target:
            target.write(json.dumps(result, ensure_ascii=False) + "\n")
    write_json(output / "predictions.json", predictions)
    summary = summarize(results, live)
    summary.update({"architecture": architecture, "input_fingerprint": contract["input_fingerprint"], "interrupted": interrupted,
                    "wall_seconds": time.perf_counter() - started})
    if architecture == "multi-agent":
        summary["handoffs"] = sum_known([row.get("handoffs") for row in results])
    if reference_rows is not None:
        single = summarize(reference_rows, live)
        comparison = compare_runs(single, summary, reference_rows, results)
        write_json(output / "comparison.json", comparison)
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DEV_OUTPUT)
    parser.add_argument("--output", type=Path, default=Path("runs") / ("hotpotqa-single-" + uuid.uuid4().hex[:12]))
    parser.add_argument("--limit", type=int, default=10, help="prefix of the original ten IDs, 1–10")
    parser.add_argument("--live", action="store_true", help="explicitly opt into paid-model baseline calls")
    parser.add_argument("--architecture", choices=("single-agent", "multi-agent"), default="single-agent")
    parser.add_argument("--reference", type=Path, help="single-agent run with identical inputs/settings; writes comparison.json")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-tool-calls", type=int, default=12)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=180, help="wall-clock seconds per question")
    args = parser.parse_args()
    try:
        summary = run_baseline(args.output, dataset=args.dataset, limit=args.limit, live=args.live,
                               max_steps=args.max_steps, max_tool_calls=args.max_tool_calls,
                               max_output_tokens=args.max_output_tokens, timeout=args.timeout,
                               architecture=args.architecture, reference=args.reference)
    except (OSError, ValueError, KeyError, TypeError):
        parser.exit(1, "Evaluation setup failed; check dataset integrity, limits and the new output directory.\n")
    print(f"Saved {summary['mode']} results to {args.output}.")
    print(json.dumps({"statuses": summary["statuses"], "scores": summary["scores"]}))
    if summary["interrupted"]:
        return 130
    return 0 if all(status == "completed" for status in summary["statuses"]) else 1


if __name__ == "__main__":
    sys.exit(main())