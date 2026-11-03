"""Lab 0: functions, workflows and a learner-built loop. No harness required."""

import argparse
import json

from .support import exercise_module, show_mode


TEXT = "little harness little"


def word_count(text: str) -> str:
    """Count whitespace-separated words, including repeats."""
    return str(len(text.split()))


def fixed_workflow(text):
    """Application code chooses both steps: count, then format."""
    count = word_count(text)
    return f"Word count: {count}"


def counting_model(messages):
    """A one-block scripted proposer, not an LLM or the harness model interface."""
    content = messages[-1]["content"]
    if isinstance(content, str):
        return {"type": "tool_use", "id": "count-1", "name": "word_count",
                "input": {"text": content}}
    result = content[0]
    answer = "Unable to count." if result["is_error"] else result["content"]
    return {"type": "text", "text": answer}


def invalid_then_repair_model(messages):
    """Request the wrong argument type first, then repair after seeing an error."""
    content = messages[-1]["content"]
    if isinstance(content, str):
        return {"type": "tool_use", "id": "bad-1", "name": "word_count",
                "input": {"text": 42}}
    if content[0]["is_error"]:
        return {"type": "tool_use", "id": "repair-1", "name": "word_count",
                "input": {"text": messages[0]["content"]}}
    return counting_model(messages)


def repeating_model(messages):
    """Keep requesting a valid tool; the loop, not the script, must stop the run."""
    return {"type": "tool_use", "id": f"repeat-{len(messages)}", "name": "word_count",
            "input": {"text": messages[0]["content"]}}


def unchecked_failure():
    """Demonstrate unsafe dispatch on a pure function, catching only its known error."""
    try:
        word_count(**{"text": 42})
    except AttributeError:
        return "AttributeError: unchecked input reached the function."
    return "Unexpected success."


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution", action="store_true", help="use the reference loop")
    parser.add_argument("--check", action="store_true", help="assess your loop across scenarios")
    parser.add_argument("--stage", type=int, choices=range(1, 6),
                        help="with --check, assess checkpoints 1 through this stage")
    parser.add_argument("--messages", action="store_true", help="print the synthetic conversation")
    parser.add_argument("--scenario", choices=("count", "invalid", "repeat"), default="count")
    parser.add_argument("--max-model-calls", type=positive_integer, default=3)
    args = parser.parse_args()
    if args.stage is not None and not args.check:
        parser.error("--stage requires --check")
    if args.check and (args.scenario != "count" or args.max_model_calls != 3):
        parser.error("--check uses fixed inputs; run custom scenarios/limits separately")
    show_mode(args.solution)
    exercise = exercise_module(args.solution)
    if args.check:
        from .checks import check_basics
        return check_basics(exercise.run_loop, stage=args.stage, show_messages=args.messages)
    print(f"direct function: {word_count(TEXT)}")
    print(f"fixed workflow: {fixed_workflow(TEXT)}")
    if args.scenario == "invalid":
        print("Deliberate failure demonstration (caught; no external actions):")
        print(f"unchecked dispatch: {unchecked_failure()}")
    model = {"count": counting_model, "invalid": invalid_then_repair_model,
             "repeat": repeating_model}[args.scenario]
    try:
        report = exercise.run_loop(model, {"word_count": word_count}, TEXT,
                                   max_model_calls=args.max_model_calls)
        print(f"status: {report['status']}")
        print(f"answer: {report['answer'] or '(none)'}")
        print(f"model calls: {report['model_calls']}")
        if report["status"] == "not_implemented":
            print("Your loop starter is ready to edit; nothing is broken.")
            print("Start here: python3 -m lessons.lab00_basics --check --stage 1")
        if args.messages:
            print("\nSynthetic conversation (not private model reasoning):")
            print(json.dumps(report["messages"], indent=2))
    except Exception:
        print("Loop exercise failed; compare its return shape and steps with the walkthrough.")
        print("Use --check --stage 1 for focused feedback; exception details are omitted.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())