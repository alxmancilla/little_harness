# An annotated execution trace

[Learning path](../LEARN.md) · [Lab 1](01-follow-the-loop.md)

This is the **synthetic Lab 1 conversation**, not a historical user run or hidden
model reasoning. Reproduce it from the repository root:

<augment_code_snippet mode="EXCERPT">
````sh
python3 -m lessons.lab01_loop --messages --events
````
</augment_code_snippet>

## 1. Task enters the conversation

The application supplies this user message. The model is also given a separate
system prompt and the registered tool schemas; those are not extra conversation
messages in this display.

<augment_code_snippet mode="EXCERPT">
````json
{"role": "user", "content": "little harness little"}
````
</augment_code_snippet>

## 2. First model call proposes a tool

`CountingModel` sees a string and returns a normalized response with
`stop_reason="tool_use"`. The harness appends its content as an assistant message:

<augment_code_snippet mode="EXCERPT">
````json
{"role": "assistant", "content": [
  {"type": "tool_use", "id": "count-1", "name": "word_count",
   "input": {"text": "little harness little"}}
]}
````
</augment_code_snippet>

**A proposal is not execution.** The harness checks the tool allowlist, call budget
and argument schema first. The schema requires a `text` string and forbids extra
arguments. This pure tool is not risky, so no approval prompt is needed. This lab
does not configure an additional tool policy, worker timeout or action journal.

## 3. The function runs and returns a result

`word_count` splits the text and returns the string `3`. The harness pairs the
result with the request using `tool_use_id="count-1"`:

<augment_code_snippet mode="EXCERPT">
````json
{"role": "user", "content": [
  {"type": "tool_result", "tool_use_id": "count-1", "content": "3", "is_error": false}
]}
````
</augment_code_snippet>

The `user` role here is an internal tool-result convention, not a second human
typing. Real provider adapters translate this normalized representation.
Only now has the actual counting function executed.

## 4. Second model call returns final text

The scripted model reads that result and returns `stop_reason="end_turn"`:

<augment_code_snippet mode="EXCERPT">
````json
{"role": "assistant", "content": [{"type": "text", "text": "3"}]}
````
</augment_code_snippet>

The final answer comes from this response, not automatically from the tool.
A live model might misread the result; this script simply copies it.

## 5. Read the outcome

The command prints this stable summary before the conversation and events:

<augment_code_snippet mode="EXCERPT">
````text
status: completed (end_turn)
answer: 3
model calls: 2; tool attempts: 1; tool errors: 0
tool executions: 1
tokens: unknown (scripted models report no usage)
````
</augment_code_snippet>

`completed` describes the loop ending normally. Lab 2 will show why that alone
does not establish correctness. Unknown token usage is not free live-model usage;
these Python scripts simply make no API requests and have no token report.

## 6. Connect metadata events to the conversation

The printed event names are ordered as follows. Run IDs and timestamps are omitted
from this display, and event content is disabled.

| Sequence | Event | What it tells you |
| --- | --- | --- |
| 1 | `run_started` | A task begins |
| 2 | `model_started` | First model call begins |
| 3 | `model_finished` | Tool-request response received |
| 4 | `tool_started` | Validated tool is about to execute |
| 5 | `tool_finished` | Tool result is available |
| 6 | `model_started` | Second model call begins |
| 7 | `model_finished` | Final response received |
| 8 | `model_text` | Non-streaming answer text observed |
| 9 | `run_finished` | The run's result is finalized |

Events observe execution; callbacks do not grant approval. Full runtime traces
can contain sensitive inputs/outputs even when events omit content. These labs use
synthetic inputs and delete their temporary JSONL traces on exit. `--messages`
prints the synthetic in-memory messages, not a persisted trace file.

**Next:** complete [Lab 1's exercise](01-follow-the-loop.md#exercise-count-successful-results),
then compare valid execution with semantic correctness in [Lab 2](02-build-a-tool.md).