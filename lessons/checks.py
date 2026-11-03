"""Exercise feedback, kept separate so learners can focus on their four functions."""

import json

from .support import check_cases


BASIC_STAGES = (
    ("Finish with text", "Call the model once, append its text block and return a completed report."),
    ("Execute one valid request", "Invoke the registered function with the requested text, then return."),
    ("Return the result to the model", "Append a paired tool_result, then ask for the final answer."),
    ("Reject invalid requests", "Check the name and argument shape before execution; return errors for repair."),
    ("Stop at the call limit", "Bound every model call, including the final answer, and reject invalid limits."),
)


def check_basics(loop, *, stage=None, show_messages=False):
    from .lab00_basics import counting_model, invalid_then_repair_model, repeating_model, word_count

    if stage is not None and (type(stage) is not int or not 1 <= stage <= len(BASIC_STAGES)):
        raise ValueError("stage must be an integer from 1 through 5")

    def observe(model, *, limit=3, text="one two", proposal=None, execution_only=False):
        calls, executions = [], []

        def guarded(messages):
            calls.append(None)
            # Turn common missing-budget bugs into feedback instead of endless calls.
            if len(calls) > limit:
                raise AssertionError("model call limit exceeded")
            if proposal is not None:
                return proposal if len(calls) == 1 else {"type": "text", "text": "handled"}
            return model(messages)

        def counter(text):
            executions.append(text)
            return word_count(text)

        report = loop(guarded, {"word_count": counter}, text, max_model_calls=limit)
        if execution_only:
            return len(calls), executions
        messages = report["messages"]
        if show_messages:
            print("Synthetic checkpoint conversation (not private model reasoning):")
            print(json.dumps(messages, indent=2))
        results = [block for message in messages if isinstance(message["content"], list)
                   for block in message["content"] if block["type"] == "tool_result"]
        requests = [block for message in messages if isinstance(message["content"], list)
                    for block in message["content"] if block["type"] == "tool_use"]
        return (report["status"], report["answer"], report["model_calls"], len(calls),
                executions, [block["is_error"] for block in results],
                [block["tool_use_id"] for block in results] == [block["id"] for block in requests],
                [message["role"] for message in messages])

    cases = [[
        ("finish immediately without executing a tool",
         lambda: observe(lambda _: {"type": "text", "text": "done"}),
         ("completed", "done", 1, 1, [], [], True, ["user", "assistant"])),
    ], [
        ("execute the requested function once with the supplied text",
         lambda: observe(counting_model, limit=1, execution_only=True), (1, ["one two"])),
    ], [
        ("count new input, return the result to the model and pair call IDs",
         lambda: observe(counting_model),
         ("completed", "2", 2, 2, ["one two"], [False], True,
          ["user", "assistant", "user", "assistant"])),
    ], [
        ("return validation errors to the model, then execute only the repaired call",
         lambda: observe(invalid_then_repair_model),
         ("completed", "2", 3, 3, ["one two"], [True, False], True,
          ["user", "assistant", "user", "assistant", "user", "assistant"])),
    ], [
        ("stop repeated requests at the call limit without inventing an answer",
         lambda: observe(repeating_model, limit=2),
         ("stopped", "", 2, 2, ["one two", "one two"], [False, False], True,
          ["user", "assistant", "user", "assistant", "user"])),
        ("include the final-answer call in the budget",
         lambda: observe(counting_model, limit=1),
         ("stopped", "", 1, 1, ["one two"], [False], True, ["user", "assistant", "user"])),
    ]]
    for label, name, arguments in (
        ("unknown tool", "unregistered_tool", {"text": "one"}),
        ("wrong argument type", "word_count", {"text": 42}),
        ("missing argument", "word_count", {}),
        ("extra argument", "word_count", {"text": "one", "extra": True}),
        ("non-object arguments", "word_count", ["one"]),
    ):
        proposal = {"type": "tool_use", "id": "invalid-1", "name": name, "input": arguments}
        cases[3].append((f"reject {label} before executing a function",
                         lambda proposal=proposal: observe(None, proposal=proposal),
                         ("completed", "handled", 2, 2, [], [True], True,
                          ["user", "assistant", "user", "assistant"])))

    def rejects_invalid_limits():
        for limit in (0, -1, True, 1.5, "2", None):
            def forbidden(*args, **kwargs):
                raise AssertionError("invalid limits must be rejected before calling the model or tools")
            try:
                loop(forbidden, {"word_count": forbidden}, "one two", max_model_calls=limit)
            except ValueError:
                continue
            return False
        return True

    cases[4].append(("reject invalid limits before calling the model or tools", rejects_invalid_limits, True))
    target = len(BASIC_STAGES) if stage is None else stage
    for index, ((title, hint), group) in enumerate(zip(BASIC_STAGES, cases), 1):
        if index > target:
            break
        print(f"\nCheckpoint {index}/5: {title}")
        if check_cases(group):
            print(f"Focus on checkpoint {index}; later checkpoints have not been assessed.")
            print(f"Hint: {hint}")
            print("Read lessons/00-from-functions-to-agents.md; edit only run_loop in lessons/exercises.py.")
            print(f"Try again: python3 -m lessons.lab00_basics --check --stage {index}")
            return 1
    if target < len(BASIC_STAGES):
        print(f"\nCheckpoints 1-{target} passed for the selected implementation.")
        print(f"Next: python3 -m lessons.lab00_basics --check --stage {target + 1}")
    else:
        print("\nAll five checkpoints complete. Next: lessons/01-follow-the-loop.md")
    return 0


def check_loop(count, messages):
    mixed = [{"role": "user", "content": [
        {"type": "text", "text": "not a tool result"},
        {"type": "tool_result", "is_error": True, "content": "invalid"},
        {"type": "tool_result", "is_error": False, "content": "3"},
        {"type": "tool_result", "is_error": False, "content": "2"}]}]
    return check_cases([
        ("count one successful tool result in the real run", lambda: count(messages), 1),
        ("handle an empty conversation", lambda: count([]), 0),
        ("ignore ordinary message text", lambda: count([{"role": "user", "content": "hello"}]), 0),
        ("count multiple successes, not errors or text blocks", lambda: count(mixed), 2),
    ])


def check_tool(counter, observation):
    result = observation[0]
    cases = []
    for text, expected in (("", "0"), (" \t\n", "0"), ("one two one", "2"),
                           ("One one", "2"), ("one, one", "2"), ("one\ttwo\none", "2")):
        cases.append((f"unique count for {text!r} (return a string)",
                      lambda text=text: counter(text), expected))
    cases.extend([
        ("the agent uses the new tool and answers 2", lambda: result.answer, "2"),
        ("the run completes without tool errors",
         lambda: (result.status, result.usage.tool_errors), ("completed", 0)),
    ])
    return check_cases(cases)


def check_recovery(next_action, observation):
    result, messages, events = observation
    results = [block for message in messages if isinstance(message["content"], list)
               for block in message["content"] if block["type"] == "tool_result"]
    return check_cases([
        ("an error selects repair", lambda: next_action({"is_error": True}), "repair"),
        ("a successful result selects finish", lambda: next_action({"is_error": False}), "finish"),
        ("the repaired run completes with answer 3", lambda: (result.status, result.answer), ("completed", "3")),
        ("one invalid call followed by one successful result",
         lambda: [block["is_error"] for block in results], [True, False]),
        ("validation prevents the malformed call from executing",
         lambda: sum(event.kind == "tool_started" for event in events), 1),
        ("repair takes three model calls, two attempts and one tool error",
         lambda: (result.usage.model_calls, result.usage.tool_calls_attempted, result.usage.tool_errors), (3, 2, 1)),
    ])