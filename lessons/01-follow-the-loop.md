# Lab 1 — follow the loop

[Learning path](../LEARN.md) · Previous: [build your own loop](00-from-functions-to-agents.md)
· Next: [build a tool](02-build-a-tool.md)

**Outcome:** explain how a model request becomes an executed tool and a final answer.
**Prerequisites:** Python functions, lists and dictionaries. About 20–30 minutes.
Complete Lab 0 first to understand the loop before using the runtime API.

## Predict, then run

The task is the literal text `little harness little`. The registered `word_count`
function counts whitespace-separated words, including repeats. Predict how many
model calls and tool executions are required to request the tool and return its result.

````sh
python3 -m lessons.lab01_loop --messages --events
````

Expect answer `3`, **two model calls**, **one tool attempt**, **one execution**,
zero tool errors and status `completed`. Token counts are unknown, not zero:
the scripted model does not report usage.

## Follow the code

Read [lab01_loop.py](lab01_loop.py), then the [annotated trace](annotated-trace.md).
Compare it with your Lab 0 loop: the task and message blocks are the same, but
`Agent` now owns the loop and `ToolRegistry` supplies schema-backed validation.

1. `word_count` is an ordinary pure Python function; it has no network or file access.
2. `create_tools` registers it in a fresh `ToolRegistry`. The signature becomes an
   argument schema; the docstring becomes the tool description.
3. `CountingModel` first proposes `word_count` with the task in its `text` argument.
4. `Agent` checks that the tool exists and the arguments are valid before execution.
5. The function returns the string `3`. The harness appends a `tool_result` paired
   with the original call ID. Tool results use the `user` message role internally.
6. On the second call, the scripted model copies that result into a final text response.

[support.py](support.py) handles temporary trace cleanup, explicit call budgets,
quiet output and observation. There is no `.env` loading or real provider here.
The prompt does not make the script choose a tool: its Python branches do that.

## Exercise: count successful results

Implement `count_successful_tools(messages)` in [exercises.py](exercises.py).
Count `tool_result` blocks only when `is_error` is exactly `False`.

- Message `content` can be a string or a list of blocks.
- A list can contain text, calls or results. Do not count `tool_use` requests.
- One message may contain several results; a failed attempt is not a success.

````sh
python3 -m lessons.lab01_loop --check
````

The starter returns zero, so it should fail until you implement it. A complete
solution passes four checks, including an empty conversation and mixed blocks.

### Hints

1. Iterate over messages, skipping content that is not a list.
2. Inspect each block's `type` and `is_error`, not its text or numeric output.
3. Add one for every matching block, not one for every matching message.

### Reference solution

Read `count_successful_tools` in [solutions.py](solutions.py) after trying it.
To verify the reference without overwriting your edits:

````sh
python3 -m lessons.lab01_loop --solution --check
````

## Explain it back

- Why are there two model calls but only one tool execution?
- What joins a tool request to its result? What would break if the ID changed?
- Why is a final answer not proof of correctness?
- Would a direct function call be better for this task? Why?

**Checkpoint:** point to the request, result and final answer in `--messages`.
Explain what the application controls versus what the model proposes.