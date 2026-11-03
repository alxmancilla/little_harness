"""Lab 2: implement a tool. Start at lessons/02-build-a-tool.md."""

from .lab01_loop import CountingModel, TEXT, create_tools
from .support import exercise_module, options, run_offline, show_run


def run(counter):
    tools = create_tools(counter)
    return run_offline(CountingModel(counter.__name__), tools, TEXT)


def main():
    args = options(__doc__)
    exercise = exercise_module(args.solution)
    observation = run(exercise.unique_word_count)
    show_run(observation, args)
    if args.check:
        from .checks import check_tool
        return check_tool(exercise.unique_word_count, observation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())