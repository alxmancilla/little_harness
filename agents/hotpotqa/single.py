"""Isolated single-agent worker. Receives candidate text and questions, NEVER gold.

Invoked by evaluation.hotpotqa.evaluate. The only exposed tools are keyword search,
bounded sentence reading and arithmetic. No arbitrary file reads or network tools.
"""

import json
import time
from pathlib import Path

import harness as h
from .common import (BASELINE_EVIDENCE_NOTICE as BASELINE_EVIDENCE_NOTICE, BASELINE_PROMPT,
                     HotpotQAConfig, MeteredModel, classify_run, create_agent, make_tools, model_config, smoke_model)
from .prediction import parse_prediction as parse_prediction
from .worker import classify_prediction, worker_main


def run_agent(payload, model, trace_path, *, max_steps=12, max_tool_calls=12,
              config=None, runtime_factory=h.Agent, tool_factory=make_tools):
    observed = set()
    metered = MeteredModel(model)
    agent, budget = create_agent(metered, BASELINE_PROMPT, payload["documents"], observed, trace_path,
                                max_steps=max_steps, max_tool_calls=max_tool_calls, config=config,
                                runtime_factory=runtime_factory, tool_factory=tool_factory)
    started = time.perf_counter()
    raw = agent.run(payload["question"])
    elapsed = time.perf_counter() - started
    events = [json.loads(line) for line in Path(trace_path).read_text(encoding="utf-8").splitlines()]
    status, prediction = classify_prediction(raw, events)
    facts = set(map(tuple, prediction["supporting_facts"])) if prediction else set()
    valid = {(entry["title"], index) for entry in payload["documents"].values()
             for index in range(len(entry["sentences"]))}
    return {"id": payload["id"], "status": status, "raw_answer": raw, "prediction": prediction,
            "runtime_seconds": elapsed, "model_calls": metered.calls,
            "tool_calls_requested": metered.requested_tools, "tool_calls_executed": budget.executed_tools,
            "tool_calls_rejected": budget.rejected_tools, "tool_errors": budget.tool_errors,
            "tokens": metered.tokens(), "cost_usd": None,
            "citation_validity": len(facts & valid) / len(facts) if facts else None,
            "read_evidence_fraction": len(facts & observed) / len(facts) if facts else None}


def main():
    worker_main(__doc__, run_agent, smoke_model, BASELINE_PROMPT)


if __name__ == "__main__":
    main()