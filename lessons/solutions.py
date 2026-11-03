"""Reference solutions; compare with your own work after attempting each exercise."""


def run_loop(model, tools, text, *, max_model_calls=3):
    """Lab 0 teaching code only: one block per response and text-only tools."""
    if type(max_model_calls) is not int or max_model_calls < 1:
        raise ValueError("max_model_calls must be a positive integer")
    messages = [{"role": "user", "content": text}]
    report = {"status": "stopped", "answer": "", "messages": messages, "model_calls": 0}
    for _ in range(max_model_calls):
        block = model(messages)
        report["model_calls"] += 1
        messages.append({"role": "assistant", "content": [block]})
        if block["type"] == "text":
            report.update(status="completed", answer=block["text"])
            return report
        name, arguments = block["name"], block["input"]
        if not isinstance(name, str) or name not in tools:
            output, is_error = "Unknown tool.", True
        elif (not isinstance(arguments, dict) or set(arguments) != {"text"}
              or not isinstance(arguments["text"], str)):
            output, is_error = "Expected exactly one string argument: text.", True
        else:
            output, is_error = tools[name](**arguments), False
        messages.append({"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": block["id"],
            "content": output, "is_error": is_error}]})
    return report


def count_successful_tools(messages):
    total = 0
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            for block in content:
                if block["type"] == "tool_result" and block.get("is_error") is False:
                    total += 1
    return total


def unique_word_count(text: str) -> str:
    """Count distinct whitespace-separated words, preserving case and punctuation."""
    return str(len(set(text.split())))


def next_action(tool_result):
    return "repair" if tool_result["is_error"] else "finish"