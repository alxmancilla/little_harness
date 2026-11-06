"""Read-only reference checks and descriptive architecture comparisons; no model calls."""

import hashlib
import json
import re
from pathlib import Path

# Every shared module affecting model requests, agent execution, tools or scoring.
# Keep this list in one place so extraction cannot silently bypass reference checks.
SHARED_CODE_FILES = ("harness/__init__.py", "harness/__main__.py", "harness/agent.py", "harness/tools.py",
                     "harness/runtime.py", "harness/events.py", "harness/model_clients.py",
                     "harness/tool_execution.py", "harness/journal.py", "harness/policy.py", "harness/session.py",
                     "harness/memory.py",
                     "agents/__init__.py", "agents/teaching/__init__.py", "agents/teaching/config.py",
                     "agents/teaching/tools.py", "agents/teaching/factory.py",
                     "agents/teaching/demos.py", "agents/teaching/cli.py",
                     "agents/hotpotqa/__init__.py", "agents/hotpotqa/common.py", "agents/hotpotqa/single.py",
                     "agents/hotpotqa/prediction.py", "agents/hotpotqa/worker.py",
                     "evaluation/__init__.py", "evaluation/hotpotqa/__init__.py",
                     "evaluation/hotpotqa/scoring.py")

# Host-only source provenance; paired preflight compares the shared boundary above.
HOST_CODE_FILES = ("evaluation/hotpotqa/evaluate.py", "evaluation/hotpotqa/comparison.py",
                   "evaluation/hotpotqa/prepare.py", "evaluation/hotpotqa/dev.py",
                   "evaluation/hotpotqa/run_question.py")


def load_reference(path, contract, inputs_bytes, repo):
    """Reject mismatched experiment conditions before any worker is started."""
    path = Path(path)
    reference = json.loads((path / "run.json").read_bytes())
    frozen = (path / "inputs.json").read_bytes()
    if (reference["architecture"] != "single-agent" or reference["mode"] != contract["mode"]
            or frozen != inputs_bytes
            or hashlib.sha256(frozen).hexdigest() != reference["input_fingerprint"]):
        raise ValueError("reference architecture, mode or frozen inputs differ")
    for field in ("budgets", "tools", "compaction", "question_ids", "dataset_snapshot_sha256"):
        if reference[field] != contract[field]:
            raise ValueError("reference experiment settings differ")
    for name in SHARED_CODE_FILES:
        if reference.get("code_sha256", {}).get(name) != hashlib.sha256((Path(repo) / name).read_bytes()).hexdigest():
            raise ValueError("reference code differs; create a fresh single-agent reference with the current code")
    rows = [json.loads(line) for line in (path / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    payloads = json.loads(inputs_bytes)
    if [row["id"] for row in rows] != contract["question_ids"]:
        raise ValueError("reference results do not cover the selected cohort in order")
    for row, payload in zip(rows, payloads):
        encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        if row["input_sha256"] != hashlib.sha256(encoded).hexdigest():
            raise ValueError("reference per-question fingerprint differs")
    return rows


def known_model_config(config):
    """Legacy provider/model-only metadata is insufficient to establish a match."""
    if (not isinstance(config, dict) or config.get("version") != 2 or config.get("complete") is not True
            or not all(isinstance(config.get(name), str) and config[name] for name in ("provider", "model"))):
        return False
    generation = config.get("generation")
    if (not isinstance(generation, dict) or type(generation.get("max_output_tokens")) is not int
            or generation["max_output_tokens"] < 1):
        return False
    if config["provider"] == "offline-smoke":
        return (config["model"] == "protocol-only" and config.get("endpoint_sha256") is None
                and generation.get("sampling") == "not-applicable")
    return (isinstance(config.get("endpoint_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", config["endpoint_sha256"]) is not None
            and generation.get("sampling") == "provider-defaults")


def compare_runs(single, multi, single_rows, multi_rows):
    """Deltas are descriptive, not a significance claim or a statement about causality."""
    configs = [row.get("model_config") for row in single_rows + multi_rows]
    same_model = bool(single_rows and len(single_rows) == len(multi_rows)
                      and all(known_model_config(config) for config in configs)
                      and all(config == configs[0] for config in configs))

    def difference(left, right):
        return {"single_agent": left, "multi_agent": right,
                "delta": right - left if left is not None and right is not None else None}

    quality = None
    paired = None
    if single["scores"] is not None and multi["scores"] is not None:
        quality = {name: difference(single["scores"][name], multi["scores"][name]) for name in single["scores"]}
        paired = {}
        for metric in ("em", "f1", "sp_f1", "joint_f1"):
            pairs = [(left["scores"][metric], right["scores"][metric])
                     for left, right in zip(single_rows, multi_rows)
                     if left["scores"][metric] is not None and right["scores"][metric] is not None]
            paired[metric] = {"wins": sum(b > a for a, b in pairs), "ties": sum(b == a for a, b in pairs),
                              "losses": sum(b < a for a, b in pairs), "questions": len(pairs)}
    resources = {name: difference(single[name], multi[name]) for name in (
        "model_calls", "tool_calls_executed", "tool_errors", "runtime_seconds")}
    resources.update({name: difference(single["tokens"][name], multi["tokens"][name])
                      for name in ("input_tokens", "output_tokens")})
    return {"input_fingerprint": multi["input_fingerprint"], "questions": multi["questions"],
            "conditions_verified_before_run": ["inputs", "budgets", "tools", "scorer", "mode"],
            "model_configs_match": same_model, "recorded_settings_match": same_model,
            "comparable": same_model, "comparison_scope": "recorded-settings-only",
            "warnings": [] if same_model else [
                "recorded model/endpoint/generation settings differ or are incomplete; do not attribute deltas to architecture"],
            "limitations": ["Matching recorded settings do not verify provider-side model revisions, routing or sampling defaults."],
            "quality": quality, "paired": paired, "resources": resources,
            "score_denominators": {"single_agent": single["score_denominators"],
                                   "multi_agent": multi["score_denominators"]},
            "statuses": {"single_agent": single["statuses"], "multi_agent": multi["statuses"]},
            "cost_usd": None, "interpretation": "Descriptive development-cohort comparison, not a general quality estimate."}