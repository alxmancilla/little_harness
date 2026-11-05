# Lab 0 — from a function to an agent loop

[Learning path](../LEARN.md) · Next: [follow the harness loop](01-follow-the-loop.md)

**Outcome:** build a small loop yourself and explain why a harness exists.
**Prerequisites:** Python functions, loops, lists and dictionaries. About 30–45 minutes.
No model account, third-party package or runtime API is required.

If you have not seen a working run yet, start with `python3 -m lessons.preview`.
It shows the protocol without loading your exercise or revealing its solution.
See the [setup checklist](../LEARN.md#before-you-start) if you have not opened the project yet.

This is deliberately limited **teaching code, not a second supported runtime**.
Use synthetic inputs and pure functions only. Do not connect it to a live model,
files, shell commands or external actions.

## 1. Start with a function

Read `word_count` in [lab00_basics.py](lab00_basics.py):

````python
def word_count(text: str) -> str:
    """Count whitespace-separated words, including repeats."""
    return str(len(text.split()))
````

Predict the result for `little harness little`. Does solving this problem require
a model? No: a direct function call is simpler, cheaper and more predictable.

## 2. Choose a fixed workflow

`fixed_workflow` calls `word_count`, then formats its output as `Word count: 3`.
Application code chooses both steps in advance. This is a **fixed workflow**, not
a model choosing its next action. More steps alone do not make it an agent.

````sh
python3 -m lessons.lab00_basics
````

The first lines show direct counting and the fixed workflow. The CLI labels this
**exercise mode**; the untouched loop reports `not_implemented` and tells you where
to begin. This is expected, not a setup failure. The deliberate unchecked-input
failure is reserved for the explicit invalid scenario later in this lab.

## 3. Represent a proposal, not execution

Inspect `counting_model(messages)`. On its first call it returns this dictionary:

````python
{"type": "tool_use", "id": "count-1", "name": "word_count",
 "input": {"text": "little harness little"}}
````

That dictionary does **not** run Python. Trusted application code must decide
whether the name and arguments are allowed, then call the function explicitly.
The local `tools` dictionary is an allowlist: the model cannot register functions.

After receiving a tool result, the script returns a final block:
`{"type": "text", "text": "3"}`. These Python branches demonstrate the
**model-driven loop protocol**, not intelligence or learned planning. A real model
can make different or invalid choices; scripted success says nothing about its quality.

This toy model accepts only `messages` and returns **one block**. It is intentionally
simpler than the harness model interface, which also accepts system instructions,
tool schemas and a callback, and returns a response containing a list of blocks.

## 4. Build the loop yourself

Implement `run_loop` in [exercises.py](exercises.py). Do not import `Agent` or copy
the reference yet. The caller supplies the model function, tool dictionary, input
text and `max_model_calls` (a positive integer).

You will grow the **same function** through five checkpoints, not write five loops.
Keep a finite loop from the start; later checkpoints test its boundaries. Until
checkpoint 3, a tool request may return early with an unfinished report. Do not
run invalid or repeating scenarios until their corresponding checkpoints.

The completed function returns a dictionary with these fields:

| Field | Contract |
| --- | --- |
| `status` | `completed` after a final text block; `stopped` at the call limit |
| `answer` | Final text, or the empty string when stopped |
| `messages` | The complete conversation, including rejected calls and their errors |
| `model_calls` | Number of model invocations, including the final-answer call |

### Checkpoint 1: finish with text

Begin with `[{"role": "user", "content": text}]`. Call `model(messages)`, count
the call and append its block as an `assistant` message whose `content` is a
one-element list. If the block's type is `text`, return `completed` and its text
as the answer. You do not need tool execution yet: this check's model finishes
immediately. An unfinished tool branch can return a `stopped` report for now.

````sh
python3 -m lessons.lab00_basics --check --stage 1
````

An untouched starter fails **one** check and prints a hint. Once it passes, the
CLI gives you the next command. That is your first implemented piece of the loop.

### Checkpoint 2: execute one valid request

For a `tool_use` block, find the named function in `tools` and call it with the
block's `input` dictionary. This checkpoint supplies a valid request and a one-call
budget. It checks the model call and the actual function invocation, not result
delivery yet. You can temporarily return a `stopped` report after execution.
Keep using only these supplied pure functions and valid fixtures at this stage.

````sh
python3 -m lessons.lab00_basics --check --stage 2
````

Checkpoint 1 runs again too: adding a tool branch must not break text completion.

### Checkpoint 3: return the result to the model

Remove the temporary early return after execution. Append a `user` message whose
`content` is a one-element list containing this result (`block` is the proposal):

````python
{"type": "tool_result", "tool_use_id": block["id"],
 "content": output, "is_error": is_error}
````

Copy the original ID unchanged so the model can match result and request.
For now, `output` is the function's returned string and `is_error` is `False`.
Repeat the loop so the model sees the result and can propose its final text.
A tool's output is not automatically a final answer: the model must finish.

````sh
python3 -m lessons.lab00_basics --check --stage 3
````

The checks use alternate input (`one two`, answer `2`) so copying the demo's `3`
cannot pass. All three checkpoints so far should now work.

### Checkpoint 4: reject invalid requests and allow repair

Before invoking a function, check that the name is in `tools`, `input` is a
dictionary with exactly the key `text`, and its value is a string. Type annotations
do not enforce this. On rejection, do **not** call a function: set `output` to a
short error string and `is_error` to `True`, then append the paired result as usual.
The model gets a chance to propose a repaired request on the next call.

````sh
python3 -m lessons.lab00_basics --check --stage 4
````

This stage checks repair, unknown names, wrong types, missing/extra keys and
non-dictionary arguments. Earlier successful stages are still checked.

### Checkpoint 5: stop at the call limit

Check the limit before **every** model call, including a final-answer request.
A finite `for` loop over `range(max_model_calls)` is sufficient. If exhausted,
return `stopped` with an empty answer even if a tool returned successfully.
Reject zero, negative or non-integer limits with `ValueError` before calling
either the model or a tool; `True` is not a valid limit either.

````sh
python3 -m lessons.lab00_basics --check --stage 5
python3 -m lessons.lab00_basics --check
python3 -m lessons.lab00_basics --messages
````

There are twelve checks across five **cumulative** checkpoints. Both commands
with `--check` above assess all five. Assessment stops at the first unfinished
checkpoint, prints a hint and exits 1; completed selected checkpoints exit 0.
`--check --messages` shows their synthetic conversations, not the demo scenario.
Run custom scenarios/limits separately from assessment.

A guarded test model turns common missing-limit bugs into failed feedback. It is
**not a sandbox** and cannot stop arbitrary learner code stuck outside a model call;
interrupt such code with Ctrl-C. Do not intentionally write an endless loop.

### Hints before the solution

1. Keep the messages and call counter local to each invocation.
2. Check the tool name before dictionary lookup; check the argument shape before `**`.
3. Both errors and successes need a result message. Only final text completes a run.
4. Reject zero, negative or non-integer call limits with `ValueError`.

If a check fails, consult [troubleshooting](../LEARN.md#troubleshooting) and work on
that checkpoint alone. Your progress is the function you save in `exercises.py`;
the checker does not store or reset it.

## 5. Break the assumptions safely

`unchecked_failure` passes `42` to the pure counting function and catches the
resulting `AttributeError`. This shows that unchecked dispatch fails **inside the
function**. For a tool with side effects, executing first and validating afterward
could already be too late. No external action is used in this demonstration.

After completing your loop, predict and run these scenarios:

````sh
python3 -m lessons.lab00_basics --scenario invalid --messages
python3 -m lessons.lab00_basics --scenario repeat --max-model-calls 2 --messages
python3 -m lessons.lab00_basics --max-model-calls 1 --messages
````

- **Invalid, then repair:** three model calls, one error result, one successful
  execution and answer `3`. The integer input never reaches `word_count`.
- **Repeated valid requests:** two calls and two executions, then `stopped` with no
  answer. Valid input does not guarantee progress or termination.
- **Only one call available:** one successful tool result but no final answer.
  Finishing also needs a model call. No extra model request is allowed.

Read `run_loop` in [solutions.py](solutions.py) only after trying it. You can select
it without overwriting your exercise; `--solution` works with every scenario:

````sh
python3 -m lessons.lab00_basics --solution --check --messages
python3 -m lessons.lab00_basics --solution --scenario invalid --messages
python3 -m lessons.lab00_basics --solution --scenario repeat --max-model-calls 2
````

## 6. Replace the toy with Little Harness

| What you just wrote | Where the supported runtime handles it |
| --- | --- |
| Local tool dictionary and hand-written argument checks | `ToolRegistry`, signature-derived schemas and validation in [tools.py](../harness/tools.py) |
| Ask, append, execute, repeat | `Agent` in [agent.py](../harness/agent.py) |
| Finite model-call counter | `RunLimits` in [runtime.py](../harness/runtime.py) |
| Paired result blocks | Tool execution handling in [tool_execution.py](../harness/tool_execution.py) |

The toy assumes well-formed `text`/`tool_use` envelopes, one block per response,
and synchronous, non-raising tools taking one string and returning one string.
It has no provider adapter, schema generation, exception recovery, approval,
tool-call budget, timeout, cancellation, isolation or persistent trace. A call
counter cannot interrupt a model or function that never returns.

Do not grow it into another runtime. In [Lab 1](01-follow-the-loop.md), use `Agent`
to run the same task and compare its messages and execution controls with yours.
Lab 0 itself imports no `harness` code and performs no file or network I/O.

## Explain it back

- Who chooses the next proposed action? Who authorizes and executes it?
- What distinguishes the fixed workflow from the model-driven loop?
- Why return a validation error to the model instead of executing bad input?
- Why does a valid tool call still need a stopping limit?
- When is the direct function the better design?

**Checkpoint:** draw request → validation → execution → result → next request,
then show both a rejected call with no execution and a bounded repeating run.