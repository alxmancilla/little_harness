# Lab 2 — build a tool

[Learning path](../LEARN.md) · Previous: [the loop](01-follow-the-loop.md) ·
Next: [diagnose a failure](03-diagnose-a-failure.md)

**Outcome:** implement a schema-backed tool and test behavior, not just execution.
**Prerequisite:** Lab 1. About 20–30 minutes.

## A completed run can still be wrong

The new task counts **distinct** words in `little harness little`. Predict the
correct answer, then run the intentionally unfinished implementation:

````sh
python3 -m lessons.lab02_tools --messages
````

The starter answers `3`, completes normally and reports zero tool errors.
But the correct distinct count is **2**. The runtime validated execution—not
the meaning of your algorithm. A model repeating a wrong tool result is still wrong.

## Define the contract first

Implement `unique_word_count(text: str) -> str` in [exercises.py](exercises.py):

- Split on any whitespace, not only a single space.
- Count repeated words once.
- Preserve case and punctuation: `One` and `one` are different; so are `one,` and `one`.
- Return the count as a **string**, including `"0"` for empty or whitespace-only input.

This deliberately simple contract avoids implying that whitespace splitting is
linguistic tokenization. These word counts are also **not model token counts**.

## How a function becomes a tool

[lab02_tools.py](lab02_tools.py) reuses Lab 1's loop with your function. The registry
derives the tool name from the function name and its description from the docstring.
The `text: str` parameter becomes a required string argument; extra arguments are
not allowed. The runtime supports a limited flat schema, not every JSON Schema feature.
Two optional annotations add detail: `Literal["asc", "desc"]` limits an argument to
fixed choices that the runtime enforces, and `Annotated[str, "words to count"]`
gives the model a description of that argument. This lab does not need either.

Registration in [lab01_loop.py](lab01_loop.py) is equivalent to:

````python
tools = ToolRegistry()
tools.tool(read_only=True)(unique_word_count)
````

`read_only=True` is trusted application metadata, not automatic verification of
the function. Keep the tool pure: do not add filesystem/network access or side effects.
Annotations validate inputs; they do not prove the function returns the right answer.

## Check your implementation

````sh
python3 -m lessons.lab02_tools --check
````

The starter should fail the repeated-word cases and the agent's answer check.
After your fix, expect eight passing checks, answer `2`, two model calls and one
successful execution. The checks cover empty input, repeated words, tabs/newlines,
case sensitivity, punctuation and the actual model/tool round trip.

### Hints

1. First split the string into words.
2. A Python `set` keeps distinct values.
3. Count those values, then convert the count to a string.

### Reference solution

Read `unique_word_count` in [solutions.py](solutions.py), or compare its behavior:

````sh
python3 -m lessons.lab02_tools --solution --check
````

## Explain it back

- Why did the incorrect starter still report `completed` with no tool errors?
- What do the schema, docstring and function implementation each contribute?
- If you change case handling, have you improved the tool or changed its contract?
- Add one new test input and explain its expected result before running it.

**Checkpoint:** all checks pass, and you can explain why that establishes this
tool's behavior on these cases—not a live model's ability to select the right tool.