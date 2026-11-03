"""Edit these four starters. Read each lab before looking at solutions.py."""


def run_loop(model, tools, text, *, max_model_calls=3):
    """Lab 0: build the bounded loop described in 00-from-functions-to-agents.md."""
    messages = [{"role": "user", "content": text}]
    # TODO: ask the model, append its block, validate/execute tools, return results.
    # Return status 'completed' on text or 'stopped' when the call limit is reached.
    return {"status": "not_implemented", "answer": "", "messages": messages, "model_calls": 0}


def count_successful_tools(messages):
    """Lab 1: count tool_result blocks whose is_error flag is False."""
    # TODO: inspect message content lists, not ordinary text messages.
    return 0


def unique_word_count(text: str) -> str:
    """Lab 2: count distinct whitespace-separated words, preserving case/punctuation."""
    # TODO: repeated words should count only once. Keep the string return type.
    return str(len(text.split()))


def next_action(tool_result):
    """Lab 3: return 'repair' on a tool error and 'finish' on success."""
    # TODO: use is_error; do not keep sending the same malformed arguments.
    return "repeat"