"""Lab 1: follow one model/tool loop. Start at lessons/01-follow-the-loop.md."""

from harness import ToolRegistry

from .support import exercise_module, options, run_offline, show_run


TEXT = "little harness little"


def word_count(text: str) -> str:
    """Count whitespace-separated words, including repeated words."""
    return str(len(text.split()))


def create_tools(counter=word_count):
    """Register the supplied function in a fresh, application-owned registry."""
    tools = ToolRegistry()
    tools.tool(read_only=True)(counter)
    return tools


class CountingModel:
    """A fixed two-turn protocol demonstration, NOT an LLM or a quality evaluation."""

    def __init__(self, tool_name="word_count"):
        self.tool_name = tool_name

    def __call__(self, system, messages, tools, on_text=None):
        content = messages[-1]["content"]
        if isinstance(content, str):
            return {"stop_reason": "tool_use", "content": [{
                "type": "tool_use", "id": "count-1", "name": self.tool_name,
                "input": {"text": content}}]}
        tool_result = content[0]
        text = "Unable to count." if tool_result["is_error"] else tool_result["content"]
        return {"stop_reason": "end_turn", "content": [{"type": "text", "text": text}]}


def run():
    return run_offline(CountingModel(), create_tools(), TEXT)


def main():
    args = options(__doc__)
    exercise = exercise_module(args.solution)
    observation = run()
    show_run(observation, args)
    if args.check:
        from .checks import check_loop
        return check_loop(exercise.count_successful_tools, observation[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())