"""Lab 3: repair a malformed tool call. Start at lessons/03-diagnose-a-failure.md."""

from harness import RunLimits

from .lab01_loop import TEXT, create_tools
from .support import exercise_module, options, run_offline, show_run


class RecoveryModel:
    """Scripted error handling for a pure tool; no model reasoning or automatic retries."""

    def __init__(self, next_action):
        self.next_action = next_action
        self.calls = 0

    def __call__(self, system, messages, tools, on_text=None):
        self.calls += 1
        content = messages[-1]["content"]
        action = "repeat" if isinstance(content, str) else self.next_action(content[0])
        if action == "finish":
            return {"stop_reason": "end_turn", "content": [{"type": "text", "text": content[0]["content"]}]}
        # 42 intentionally violates the string schema. Repair sends the original text.
        arguments = {"text": TEXT if action == "repair" else 42}
        return {"stop_reason": "tool_use", "content": [{
            "type": "tool_use", "id": f"count-{self.calls}", "name": "word_count", "input": arguments}]}


def run(next_action):
    return run_offline(RecoveryModel(next_action), create_tools(), TEXT,
                       limits=RunLimits(max_model_calls=3, max_tool_calls=3))


def main():
    args = options(__doc__)
    exercise = exercise_module(args.solution)
    observation = run(exercise.next_action)
    show_run(observation, args)
    if args.check:
        from .checks import check_recovery
        return check_recovery(exercise.next_action, observation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())