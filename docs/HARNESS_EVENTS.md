# Little Harness — event callbacks and embedding

`Agent(..., on_event=callback, event_include_content=False, quiet=False)` observes
execution without parsing console text or trace files. All options are library-only;
event formats and CLI/evaluation displays are unchanged. HotpotQA applications
use metadata-only finish events for tool-error accounting. No dependencies or
environment settings are needed. Import `Agent` and `RunEvent` from `harness`
and supply an explicit `tools` mapping when constructing an agent.

## Envelope and delivery

Each immutable event contains `version` (currently 1), `run_id`, `sequence`, `kind`,
`ts` (UTC ISO timestamp) and deeply immutable `data`. `event.to_dict()` returns a
detached JSON-serializable copy. Nested mappings cannot be changed and lists become
tuples inside events. The callback cannot mutate model arguments through the event.

- Delivery is synchronous in the parent, on the calling thread, in sequence order.
  Sequences start at 1 and IDs are new for every task, including recovery blocks.
  When journaling is enabled, event and journal run IDs match. Tool action IDs are
  present when a journal record exists; otherwise `action_id` is `None`.
- `run_started` is first. `run_finished` follows session saving and journal
  finalization, and is last on normal return. No later tool/model work follows it.
  Non-completed runs also emit `run_stopped` immediately before `run_finished`.
  A recovery block emits these run events without model or tool activity.
- Model/tool/approval/compaction events include the main-loop `step` when known.
  `model_call` numbers include compaction and reset each task. HTTP retries inside
  a provider client do not create separate logical model events.
- Observations are not a durable queue, replay log, execution authority or proof
  of external effects. Use a PostgreSQL or MongoDB journal for crash recovery. Existing trace
  kinds and saved-session formats are unchanged; callbacks are not persisted.

## Event kinds

| Kind | Principal metadata in `data` |
| --- | --- |
| `run_started` | No task content by default |
| `model_started` | `model_call`, `purpose`: `main` or `compaction` |
| `model_text` | `model_call`, `phase`: `delta` or `complete`, `characters` |
| `model_finished` | Call/purpose, `status`; successful responses add `stop_reason`, allowlisted `usage` |
| `tool_policy` | Opt-in only: `name`, `tool_call_id`, `allowed`, `reason`; after validation, before approval/dispatch |
| `approval_requested` | `name`, `tool_call_id` |
| `approval_decided` | Name/ID, `decision`: `allowed`, `denied` or `aborted` |
| `tool_started` | Name/ID, `risky`, `action_id`; validated, approved dispatch is about to start |
| `tool_finished` | Name/ID/action ID, `status`, `is_error`, `outcome_unknown` |
| `compaction_started` | `chars_before`; with `compact_at_tokens`, also `estimated_tokens_before` |
| `compaction_finished` | `status`; successful compaction adds `dropped` (messages summarized out of the model's view), `context_start` (history index where the view now resumes), before/after character counts and, with `compact_at_tokens`, estimated token counts |
| `steering` | `characters`; after a saved tool batch, when `get_steering()` returned nonempty text |
| `run_stopped` | `status`, `stop_reason` |
| `run_finished` | Final `status`, `stop_reason`, `usage`, `elapsed_seconds` |

`model_finished.status` is `completed`, `failed`, `interrupted` or `not_run`.
A `failed` call caused by `ModelError` adds `error_category` (for example
`rate_limited` or `network`); the error message is never included.
`completed` means a response arrived, not that the task succeeded. Compaction
statuses are `completed`, `failed`, `empty` or `aborted`. Unknown token usage stays
`None`, never zero.

`tool_finished.status` is `completed`, `failed`, `timed_out`, `outcome_unknown` or
`not_run`. Invalid, denied, unknown and truncated requests have no `tool_started`.
Budget/cancellation/uncertain-outcome batch stops also emit `tool_finished` for
unstarted requests, after preserving results already obtained. Interrupted active
requests are conservatively uncertain, even if execution might not have begun.
Starting a dispatch is not proof that a worker imported or ran the function.

Opt-in [tool policies](TOOL_POLICIES.md) emit one `tool_policy` decision for each
validated request to a known tool. Reasons are `allowed`, `read_only_required`,
`argument_rejected` or `policy_error`. These events never include arguments or
exception details, even with content enabled. Denials have a not-run tool finish
without a tool start; policy errors also stop the remaining batch and task.

Streaming deltas arrive between `model_started` and `model_finished`. For models
that do not stream, nonempty text blocks emit `phase=complete` after the response.
Streaming responses do not also emit complete text. Deltas are provisional: a
request may fail after partial text. Use `RunResult.answer` for the final answer;
do not concatenate text from multiple model calls into a final answer. Compaction
does not stream text through these callbacks.

## Privacy and quiet mode

By default, callbacks omit task text, model text, tool arguments/results, summaries
and exception messages. Metadata still includes tool names, caller/provider IDs,
stop reasons and sizes; these may themselves be sensitive. This is data minimization,
**not general-purpose secret detection or redaction**.

`event_include_content=True` adds `task` on run start, `text` on model text events,
`input` on approval requests/tool starts, `input` and `output` on tool finishes,
`summary` on successful compaction, `text` on steering, and `answer`/display `text`
on run finish.
Only opt in for trusted consumers. JSONL traces independently default to metadata
only; `trace_include_content=True` opts into task/model/tool/summary/error content.
The teaching CLI uses `HARNESS_TRACE_CONTENT=1`; controlled HotpotQA evaluation
explicitly opts in. Sessions still contain conversation content regardless of
observation settings. Quiet mode does not redact returned results or sessions.

`quiet=True` suppresses harness console output, including session-resume messages.
It replaces the default `ask_human` gate with denial, **never automatic approval**.
Provide your own `approve(name, args)` for UI/service decisions; its arguments contain
the full request. Events only observe decisions; callback return values are ignored.
Quiet mode cannot suppress printing by your own model, inline tool or callbacks.

## Failures, cancellation and UI integration

- Any ordinary `Exception` from event construction or the callback disables
  observation for the rest of that run and sets `agent.event_callback_failed=True`.
  The flag resets and delivery is retried on the next task. The trace records an
  `event_callback_failed` entry naming the failing event kind (plus the exception
  type name with trace content enabled); exception text never enters the
  conversation, trace or console. Tool execution is never retried because
  of observer failure. Read the flag after `run_result()` to detect lost events.
- `KeyboardInterrupt` and other `BaseException` subclasses disable the observer and
  propagate. Ctrl+C uses the harness's existing interruption cleanup and preserves
  completed tool results. A cancelling/failed callback will not receive cleanup or
  terminal events. An interrupt during final notification can prevent returning an
  already-finalized result; it does not undo that run or any tool effects.
- Unexpected harness exceptions still propagate; if delivery is available, the
  terminal event reports generic `status=error`, `stop_reason=harness_error`.
- Callbacks run trusted application code. Keep them short and do not mutate the
  agent, registry or execution settings. Reentrant/overlapping calls to the same
  agent are unsupported and rejected; it is not a thread-safe service object.
- Callback time counts toward the cooperative deadline; callbacks cannot be forcibly
  timed out. Model/tool dispatch rechecks the deadline after start observers.
  Completion notification time can make total caller latency exceed a deadline.
  For a UI, enqueue `event.to_dict()` into your own bounded queue and consume it on
  the UI thread. Define a queue-full policy explicitly: a raised queue error disables
  that run's observer. Do not perform blocking UI/network work inside callbacks.