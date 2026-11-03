"""Lab-only execution/display helpers; no provider configuration or persistent files."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory


def run_offline(model, tools, text, *, limits=None):
    """Return (result, messages, events); delete the temporary runtime trace on exit."""
    from harness import Agent, RunLimits

    events = []
    with TemporaryDirectory(prefix="little-harness-lab-") as directory:
        agent = Agent(model, "Count only by using the supplied tool.", tools=tools,
                      quiet=True, trace_path=Path(directory) / "trace.jsonl",
                      limits=limits if limits is not None else RunLimits(
                          max_model_calls=3, max_tool_calls=3), on_event=events.append)
        result = agent.run_result(text)
    return result, agent.messages, events


def options(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--check", action="store_true", help="check this lab's exercise")
    parser.add_argument("--solution", action="store_true", help="use the reference solution")
    parser.add_argument("--messages", action="store_true", help="print synthetic conversation messages")
    parser.add_argument("--events", action="store_true", help="print ordered metadata event names")
    args = parser.parse_args()
    show_mode(args.solution)
    return args


def show_mode(use_solution):
    if use_solution:
        print("Reference mode: using solutions without changing your exercise file.")
    else:
        print("Exercise mode: unfinished starters are expected to fail checks; this is not a setup failure.")


def exercise_module(use_solution):
    if use_solution:
        from . import solutions
        return solutions
    from . import exercises
    return exercises


def show_run(run, args):
    result, messages, events = run
    executions = sum(event.kind == "tool_started" for event in events)
    print(f"status: {result.status} ({result.stop_reason})")
    print(f"answer: {result.answer or '(none)'}")
    print(f"model calls: {result.usage.model_calls}; tool attempts: {result.usage.tool_calls_attempted}; "
          f"tool errors: {result.usage.tool_errors}")
    print(f"tool executions: {executions}")
    print("tokens: unknown (scripted models report no usage)")
    if args.messages:
        print("\nSynthetic conversation (not private model reasoning):")
        print(json.dumps(messages, ensure_ascii=False, indent=2))
    if args.events:
        print("\nMetadata events (content omitted):")
        for event in events:
            print(f"{event.sequence:02d} {event.kind}")


def check_cases(cases):
    """Give concise feedback and a nonzero exit status without depending on assert."""
    passed = True
    for description, observe, expected in cases:
        try:
            okay = observe() == expected
        except Exception:
            okay = False
        print(f"{'PASS' if okay else 'FAIL'}: {description}")
        passed = passed and okay
    return 0 if passed else 1