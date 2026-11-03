"""Shared worker mechanics; role orchestration and budgets stay in each application."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import harness as h
from agents.teaching.config import load_env
from harness.model_clients import print_model_config
from .common import HotpotQAConfig, classify_run, model_config
from .prediction import parse_prediction


def classify_prediction(raw, events):
    """Parse only a completed answer; preserve terminal failure/cancellation priority."""
    status = classify_run(events)
    try:
        prediction = parse_prediction(raw) if status == "completed" else None
    except (ValueError, TypeError):
        status, prediction = "invalid_output", None
    return status, prediction


def worker_main(description, runner, smoke_model, prompts, *, argv=None):
    """Keep both worker CLIs identical except their runner, smoke model and prompts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-tool-calls", type=int, default=12)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    args = parser.parse_args(argv)
    load_env()
    config = HotpotQAConfig.from_env(os.environ)
    model = h.make_model() if args.live else smoke_model
    if args.live:
        model.max_tokens = args.max_output_tokens
        print_model_config(model)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = runner(payload, model, args.trace, max_steps=args.max_steps,
                    max_tool_calls=args.max_tool_calls, config=config)
    result["model_config"] = model_config(model if args.live else None, max_output_tokens=args.max_output_tokens)
    if isinstance(prompts, str):
        result["prompt_sha256"] = hashlib.sha256(prompts.encode()).hexdigest()
    else:
        result["prompt_sha256"] = {role: hashlib.sha256(prompt.encode()).hexdigest()
                                   for role, prompt in prompts.items()}
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(result, target, ensure_ascii=False, indent=2)
        target.write("\n")