# Lab 3 — diagnose a failure

[Learning path](../LEARN.md) · Previous: [build a tool](02-build-a-tool.md)

**Outcome:** distinguish a request, a rejected attempt and a real execution;
repair an invalid call and understand why limits matter.
**Prerequisite:** Labs 1–2. About 20–30 minutes.

## Run the failure safely

This lab returns to Lab 1's ordinary `word_count`, not the distinct-count tool.
The scripted model deliberately sends `{"text": 42}` instead of a string.

<augment_code_snippet mode="EXCERPT">
````sh
python3 -m lessons.lab03_failures --messages --events
````
</augment_code_snippet>

With the untouched starter, expect status **`stopped`**, reason **`max_model_calls`**,
no final answer, three model calls, three tool attempts/errors and **zero executions**.
The demo command exits 0 because the demonstration ran; `--check` assesses the exercise.

The same invalid input was requested repeatedly. A three-call budget stops further
work; it does not turn the failed attempts into useful work. `tool_finished` events
can describe rejected calls—look for the absence of `tool_started` to see that
the function was never invoked.

## Find the cause, not just the symptom

Read [lab03_failures.py](lab03_failures.py). `RecoveryModel` first requests the bad
argument, then delegates its next move to your `next_action` function.

1. The schema requires `text` to be a string.
2. The harness rejects the integer **before** entering the tool function.
3. The model receives a paired `tool_result` with `is_error=True` and an explanation.
4. The starter always chooses `repeat`, so it never changes the malformed input.

This is not an unknown-word problem, a provider outage or a tool function crash.
Do not remove validation, increase budgets indefinitely or catch and ignore the error.

## Exercise: choose repair or finish

Implement `next_action(tool_result)` in [exercises.py](exercises.py):

- Return `"repair"` if `is_error` is true. The lab model will propose a **new call**
  with the original synthetic text and a new ID.
- Return `"finish"` on success. The lab model will return the tool's value as its answer.

<augment_code_snippet mode="EXCERPT">
````sh
python3 -m lessons.lab03_failures --check
````
</augment_code_snippet>

The starter should fail. After your fix, expect six passing checks and:

| Observation | Expected |
| --- | --- |
| Final result | `completed`, answer `3` |
| Model calls | 3: malformed request, corrected request, final answer |
| Tool attempts | 2 |
| Tool errors | 1: the initial invalid request is still recorded |
| Actual executions | 1: only the corrected request |

Successful recovery does not erase earlier errors. The budget must allow enough
calls to repair the input **and** obtain a final answer.

### Hints and reference solution

1. Read the `is_error` flag rather than searching the error message text.
2. Choose one branch for error and one for success.
3. Compare `next_action` in [solutions.py](solutions.py) after attempting it.

<augment_code_snippet mode="EXCERPT">
````sh
python3 -m lessons.lab03_failures --solution --check --events
````
</augment_code_snippet>

## Do not generalize this into blind retries

This correction is safe because validation proves the first call was **not run**,
and the tool is a pure counting function. The scripted model makes the correction;
the harness does not secretly retry the tool or repair model output.

A timed-out payment or interrupted file operation may already have taken effect.
Retrying it could duplicate the action. Verify effects and use the
[human recovery procedure](../README.md#recover-without-replaying) when appropriate.
Neither budgets nor subprocess timeouts provide a security sandbox.

## Explain it back

- Why are failed attempts counted even though nothing executed?
- Which observation proves the malformed request did not enter the function?
- Why does the repaired run need three model calls instead of two?
- Why would this same retry rule be dangerous for a tool with side effects?

**Checkpoint:** explain both the stopped starter and the completed solution without
claiming that the model reasoned, that errors disappeared, or that retries are always safe.

Next: [build your own application](04-build-an-application.md), with two read-only
tools, application-owned data, explicit budgets and a test you write yourself.