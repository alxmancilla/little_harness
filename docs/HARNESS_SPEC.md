# Little Harness specification

**Status:** as-built specification, revision 5 (separation Phase 4 complete). Describes the current implementation,
not a promise of unimplemented features. **Runtime:** Python 3.11+, standard library only.

This document uses specification-driven structure: scope, contracts, numbered
requirements, invariants and acceptance scenarios. **MUST** denotes a required
current behavior; **MAY** denotes an optional capability. Changes to these contracts
should update the implementation, mapped tests and this specification together.
For implementation structure and diagrams, see [Architecture](ARCHITECTURE.md).
Ownership changes and completed checkpoints are specified in the
[separation contract](HARNESS_SEPARATION_SPEC.md) and [migration plan](HARNESS_MIGRATION_PLAN.md).

## 1. Purpose and scope

Provide a small, understandable runtime that lets a model request registered tools,
while the application retains control over execution, approval, limits and recovery.
Support CLI teaching demos, live providers and embedding in another Python application.

### Goals

- Keep the model/tool loop readable and provider-independent.
- Preserve conversation structure and completed results during supported stops.
- Offer opt-in policies, budgets, isolated timeouts, durable action records and events.
- Make uncertainty visible instead of silently retrying potentially completed writes.
- Support reproducible, explicitly cost-opted-in single/multi-agent evaluations.

### Non-goals

- OS-level sandboxing, tenant isolation, secret detection or network confinement.
- Exactly-once external effects, automatic action recovery, rollback or tool retries.
- Distributed scheduling, parallel tool execution or overlapping runs on one agent.
- Full JSON Schema validation, provider reasoning-state portability or guaranteed accuracy.
- Hard task-wide billing, CPU, memory or wall-clock limits.

## 2. Actors, boundaries and vocabulary

| Term | Meaning |
| --- | --- |
| Application/operator | Trusted owner of model configuration, tools, policies and approval decisions |
| Model | Untrusted source of proposed text and tool requests; not execution authority |
| Agent | Stateful `Agent` instance holding conversation and orchestrating synchronous work |
| Task/run | One `run()` or `run_result()` invocation; limits and observation IDs reset each time |
| Step | One main-loop iteration; optional compaction may add a model call before it |
| Tool attempt | A non-truncated request admitted by the attempt budget, including invalid/denied calls |
| Action | A tool dispatch; with journaling, identified by a random stable `action_id` |
| Outcome unknown | Execution may have had effects, but no trustworthy known outcome is available |

Tools, predicates, approval callbacks and observers are trusted Python code. Model
output, retrieved documents and remote responses are untrusted data. A worker process
is an execution boundary for timeouts, **not** a security boundary.

## 3. Public interfaces and data contracts

### 3.1 Agent and extension interfaces

| Interface | Contract |
| --- | --- |
| `Agent(model, system, ...)` | Configure conversation, tools and optional controls; no model call at construction |
| `Agent.run(task)` | Return display text, including harness warnings/stop messages |
| `Agent.run_result(task)` | Return immutable structured result for supported outcomes; unexpected programming/I/O failures can propagate |
| `model(system, messages, tools, on_text=None)` | Return normalized response; optionally deliver provisional text deltas |
| `ToolRegistry.tool(risky=False, timeout_seconds=None, read_only=False)` | Decorator registering a function and signature-derived tool schema in that registry |
| `approve(name, args)` | Application decision for risky calls; truthiness determines permission |
| `ToolPolicy(read_only=False, argument_checks={})` | Optional explicit read-only gate and per-tool predicates |
| `on_event(event)` | Optional synchronous observation; return value cannot authorize execution |
| `tool_admission(name, available)` | Optional trusted per-attempt admission; `None` continues normal checks, nonempty string rejects only that call |
| `get_steering()` | Optional; called after each saved tool batch that passed the budget check. Nonempty text is appended as a text block to that user turn and saved before the next model call; `None`/blank adds nothing; other values raise `ValueError`. It is guidance, not authorization |
| `trace_context={...}` | Optional JSON trace annotations, copied at construction; reserved `ts`, `kind`, `run_id` keys rejected |
| `trace_include_content=False` | Metadata-only JSONL by default; `True` includes tasks/model/tool/summary/error content independently of events |
| `ActionJournal.inspect()` / `resolve(...)` | Inspection and explicitly confirmed human outcome recording |
| `session_store=` (a `SessionStore`) | Optional replacement for `session_path`, never combined with it: `load()` returns `(messages, compaction)` or `None`; `save(messages, compaction, *, durable)` is all-or-nothing. `SessionFile(path)` is the file implementation |
| `journal=` (an `ActionJournal`) | Optional journal with `prepare`, `assert_ready`, `begin`, `finish`, `close_run`, `inspect`, `resolve`, with its errors and guarantees (FR-08) |
| `MemoryStore` | Explicit namespace-scoped long-term memory; database implementations are in `stores/` |
| `memory_tools(store, namespace, allow_write=True)` | Bound `recall`, `remember`, `forget` registry; writes are risky and may be omitted |

`harness.Agent` requires an explicit keyword-only `tools` mapping; `None` is invalid.
`ToolRegistry.tool(...)` registers into that application-owned mapping. An explicit
mapping restricts both advertised and executable tools; `{}` disables tools.
One agent supports sequential tasks, not reentrant/overlapping runs; it is not a
thread-safe service object.

Admission runs after allowlist lookup and the core attempt-budget check, before
argument validation; it receives only the name and whether the tool is available.
Unknown names can never be authorized. Truncated calls skip admission. A rejection
returns an ordinary not-run tool error with no policy/approval/journal/worker action;
the model can recover. Invalid callback returns or ordinary exceptions propagate
as programming errors without dispatch, unlike isolated observer failures. A
`KeyboardInterrupt` uses normal cancellation handling. Admission is trusted,
synchronous code, not a replacement for policy, authorization or timeouts.

`trace_context` must contain JSON-serializable values (no NaN/infinity); nested
values are detached from the supplied mapping. Actual runtime trace fields take
precedence over annotations. These annotations are not added to `RunEvent` payloads
and are not automatically redacted. Do not put secrets in them.

Supported embedding surface (validated by teaching, HotpotQA and inventory clients):

- Import `Agent`, `SessionLoadError`, `ToolRegistry`, `RunLimits`, `RunResult`,
  `RunUsage`, `RunEvent` and `ToolPolicy` from `harness`; use the documented
  constructor options and hooks.
- Public provider exports are `HTTPModel`, `ModelError`, `ModelConfigError`,
  `AnthropicModel`, `OpenAIChatModel`, `OpenAIResponsesModel` and `make_model`.
  Provider selection is explicit; importing the package does not load `.env` or
  select a provider.
- The runtime never terminates the host process. Configuration and session problems
  raise `ModelConfigError` or `SessionLoadError`, both subclasses of `ValueError`;
  command-line applications catch them and choose the exit message and code.
- Recovery exports are `ActionJournal`, `JournalError` and
  `JournalRecoveryRequired`; `validate_tool_input` is also public.
- Session exports are `SessionStore`, `SessionFile`, `validate_session`,
  `SessionStoreError` (an `OSError`, so existing file-error handlers still catch it)
  and its subclass `SessionConflictError`. Stores must check stored data with
  `validate_session`, and their `str()` must not reveal connection details, because
  it appears in the resume message.
- Private methods and internal worker descriptors are implementation details,
  not application extension points. The runtime has no built-in prompts, tools,
  workspace setup, dataset loading or evaluation stages.
- Root compatibility imports/commands are removed. Applications use `harness`
  and package module commands, not mutable globals. The copied runtime
  directory is self-contained; the optional standard-library wheel builder in
  `tools/package_runtime.py` packages only runtime source and installation metadata.

The independent [inventory example](../examples/inventory_agent.py) exercises this
surface without any `agents/`, evaluation or root application files. Changes to
these contracts require matching documentation and focused regression tests.

### 3.2 Model and conversation representation

Messages use Anthropic-style `role`/`content` structures internally. Text blocks have
`type="text"` and `text`. Requests have `type="tool_use"`, `id`, `name` and `input`.
Results have `type="tool_result"`, `tool_use_id`, string `content` and `is_error`.
Provider adapters translate this common shape to their API's wire format.

A normalized response contains `content`, `stop_reason`, optional `usage` with
`input_tokens`/`output_tokens`, and optional `truncated_tool_ids`. Missing usage is
unknown, not zero. Stream deltas are not themselves a final answer.

Every response from any model callable, including scripted ones, MUST pass one shape
check before history changes: a dict whose `content` is a list of dicts with a string
`type`; `text` blocks with string `text`; `tool_use` blocks with a unique nonempty
string `id` and a string `name`; a string or missing `stop_reason`; and a list for
`truncated_tool_ids` when present. A failure raises `ModelError` with category
`malformed_response`: a main call stops as `model_error`, a summary call fails
compaction, and nothing is appended. Other block types (for example provider
`thinking` blocks) pass through unchanged. A `tool_use` input of the wrong shape is
not a malformed response; argument validation returns it to the model as a tool error.

### 3.3 Results, observations and outcomes

| Record | Fields |
| --- | --- |
| `RunResult` | `status`, `answer`, `stop_reason`, `usage`, `elapsed_seconds`, `text`; JSON snapshot via `to_dict()` |
| `RunUsage` | `model_calls`, `tool_calls_requested`, `tool_calls_attempted`, `tool_errors`, nullable input/output tokens; computed `total_tokens` |
| `ToolOutcome` | `status`, `output`, `is_error`, `outcome_unknown` |
| `RunEvent` | `version`, `run_id`, `sequence`, `kind`, UTC `ts`, deeply immutable `data`; detached `to_dict()` |

`RunResult.answer` is final model text without harness-added warnings. It may retain
an answer received after a limit overshoot even though the task did not complete.
`text` is the user-facing display representation. `total_tokens` is a computed
usage property, not an extra serialized field in `RunResult.to_dict()`.

## 4. Functional requirements

### FR-01 — Orchestrate sequential model/tool turns

The agent MUST append the task, optionally compact history, request a model response,
then execute its tool calls sequentially in response order. Results for one assistant
turn MUST return together in one user message. A response without tool calls ends
the task; the default main-loop budget is 12 steps.

### FR-02 — Enforce tool registration and input validation

Only registered/allowlisted names MAY dispatch. Signature-derived schemas support
`str`, `int`, `float` and `bool`, required fields and rejection of unknown arguments.
`typing.Literal` choices that are all strings or all integers become `enum`;
`typing.Annotated[T, "text"]` with exactly one nonempty string adds a parameter
`description`, where `T` is a supported type or `Literal`.
Registration MUST reject missing/unsupported annotations, incompatible defaults,
duplicate names, positional-only/variadic parameters and coroutine/generator tools
before modifying the registry. Builtin postponed annotation strings are supported
without evaluating expressions; postponed `Literal`/`Annotated` strings are rejected.
Keyword-only parameters work. Return annotations
do not define output validation. Deliberate replacement requires explicit removal.
Before approval or execution, the harness MUST validate object shape, required
fields, supported primitive types, `enum` choices (matching type and value) and
finite floating-point numbers. Tools MUST retain their own semantic checks;
the generic validator does not enforce all schema keywords or numeric/string bounds.
Truncated argument requests MUST return a not-run error without dispatch.

### FR-03 — Enforce policy before approval or action creation

With a policy configured, read-only mode MUST require literal registry
`read_only=True`; non-risky is not equivalent. Argument predicates MUST receive
detached JSON snapshots and return literal `True` to allow. Rejection returns a
not-run tool error; ordinary predicate/snapshot exceptions stop with
`tool_policy_error` without exposing exception text. Approved execution arguments
MUST be detached from predicate and approval mutation. See [policy details](TOOL_POLICIES.md).

### FR-04 — Gate risky actions

Risky calls MUST obtain approval after validation/policy and before journaling or
execution. Denial MUST not invoke the tool. Quiet mode with the default interactive
gate MUST deny rather than prompt or auto-approve. The offline `mock` CLI explicitly
auto-approves risky calls, but cannot bypass an enabled read-only policy.

### FR-05 — Apply one per-task execution budget

`RunLimits` MAY cap logical model calls, tool attempts, reported total tokens and
monotonic elapsed seconds. Main calls and compaction MUST share these counters.
Failed model calls count; internal HTTP retries do not add logical calls. Invalid
and denied tool requests count as attempts; truncated/unstarted requests do not.
All error results, including not-run results, contribute to `tool_errors`.

Limits default to unset; zero forbids corresponding further work. A token cap with
unknown usage MUST stop with `usage_unavailable`. Token/time checks are cooperative;
an in-flight operation may overshoot. Exactly meeting the token cap allows an already
received answer to finish, but forbids further model/tool work.

### FR-06 — Execute tools inline or in disposable workers

Inline execution MUST remain the default. Positive agent/per-tool timeouts MAY opt
into fresh Python worker processes; a per-tool timeout takes precedence. The worker
wait MUST use the smaller of tool timeout and remaining task deadline. Unsupported
callables MUST fail rather than silently run inline. Timed functions MUST be importable
module-level functions or standard-library `functools.partial` bindings of them with
JSON-only values; arguments/bindings travel over stdin, not command arguments.
Workers MUST inherit a copy of the parent environment unless the application passes
`worker_env`, a string-to-string mapping that then becomes their entire environment
(on Windows, `SYSTEMROOT` is kept so Python can start). Inline tools are unaffected.

Timeout, loss of a trustworthy result or uncertain risky failure MUST stop subsequent
work without retry. Cleanup is bounded/best-effort; parent-memory mutation does not
propagate, and process/remote effects cannot be undone. See [timeout limits](REFERENCE.md#isolated-tool-timeouts).

### FR-07 — Bound tool output and preserve batch results

Tool return values MUST become text, clipped to the 8,000-character content budget
with a truncation notice when needed. Ordinary known failures MAY return tool errors
for model recovery. Supported budget/cancellation/uncertainty/journal stops MUST retain
completed results and pair unstarted requests with explicit error results.

The harness MUST NOT automatically retry tools. After a recoverable error, the model
may propose another call; this is distinct from automatic execution retry.

### FR-08 — Journal actions durably when enabled

With a `journal=` implementation, the parent MUST commit
`started` before invocation and commit an outcome afterward, durably. No start commit means no
dispatch. Denied, invalid and truncated requests MUST create no action record.
Records MUST omit arguments, outputs and exception details.

Unresolved uncertain/started actions or actions in an unclosed run MUST block a new
task before model activity. Failed outcome commits MUST preserve the obtained result
but stop later work. Human resolution MUST record a known outcome with confirmation,
never execute/replay a tool, repair a session or infer external success.

### FR-09 — Persist resumable conversation safely

Optional sessions MUST save JSON via temporary-file replacement at complete protocol
boundaries, not between a tool request and its result batch. With journaling, saves
also flush/fsync before run closure, including the directory on POSIX. A session is
`{"version": 2, "messages": [...], "compaction": null | {"start": n, "summary": "..."}}`;
a legacy JSON list of messages still loads, without compaction state, and is
rewritten in version 2 on the next save. Invalid JSON, an unknown version, a
non-list `messages` or invalid compaction state MUST raise `SessionLoadError` from
`Agent(...)` and leave the file unchanged. Session history alone MUST NOT be treated
as proof of action completion.

A `session_store=` implementation MUST give the same guarantees: all-or-nothing
saves, durable saves when a journal is enabled, and `SessionLoadError` (via
`validate_session`) for invalid stored data. History can change before the latest
message (steering), so stores MUST NOT assume it is append-only. Stores that detect a
second writer raise `SessionConflictError`; `SessionFile` assumes a single writer.
Save failures propagate out of `run_result`, as file errors do.

### FR-10 — Compact context without splitting tool exchanges

History is never shortened. The model receives a view built from it: when a
compaction exists, one user message with the latest summary, then the history from
`start`; otherwise the full history. Above the configured JSON-character threshold
(measured on that view), the agent MAY summarize the older part of the view and
move `start` forward to an assistant boundary, so tool calls stay with their results;
a previous summary is folded into the new one. When `compact_at_tokens` is set, an
estimated token threshold replaces the
character threshold: the latest main call's reported `input_tokens` plus characters
added since, divided by `CHARS_PER_TOKEN` (4), or characters alone when usage is
unknown or the history was just compacted. It is an estimate, not a tokenizer count.
It MUST avoid repeated compaction without sufficient history growth.
Empty summaries or expected summarizer failures MUST leave the view unchanged; summary
calls still consume the shared budget. Summaries are lossy context for the model, not
durable action records; reviewers read the full history in the session. Sessions
therefore grow without bound even though each model request stays bounded.

### FR-11 — Provide safe, ordered observations

Opt-in callbacks MUST receive versioned immutable events, increasing sequence numbers
and a new run ID per task. Default payloads MUST omit content; explicit content opt-in
MAY include sensitive text/arguments/results. Ordinary callback exceptions MUST disable
that run's observer and set `event_callback_failed`, not authorize or retry tools.
`BaseException` propagates; Ctrl+C uses cancellation cleanup. See [event details](HARNESS_EVENTS.md).

JSONL traces MUST independently default to metadata-only: task/model/tool content,
summaries and exception messages require `trace_include_content=True`. Reported
usage in metadata-only traces MUST allowlist nonnegative integer input/output tokens.
Caller `trace_context` remains trusted, unredacted metadata. Full-content trace opt-in
retains the existing content fields; sessions/results are not redacted by these flags.

### FR-12 — Normalize providers and constrain model transport

Adapters MUST support Anthropic Messages, OpenAI Responses and OpenAI-compatible Chat
Completions using standard-library HTTP/SSE. Model HTTP redirects MUST be rejected,
including same-origin redirects. HTTP error bodies MUST close without being read;
provider diagnostic/stream errors MUST not be interpolated into normal error text.
Selected connection/HTTP failures MAY retry before streaming; interrupted streams MUST
fail rather than retry and duplicate output. These protections apply to model transport.
The separate teaching `http_get` tool also rejects redirects, so approval covers the
only URL it fetches; it does not otherwise restrict hosts or addresses.
Every adapter MUST raise `ModelError(message, category)` with one fixed category:
`rate_limited` (429), `auth` (401/403), `request_rejected` (other 4xx and 3xx),
`server_error` (5xx), `network`, `malformed_response`, `incomplete_response` or
`provider_error` (an error event in the stream); `unknown` is the default for
application-raised errors and unlisted values are rejected. The category, never the
message, is traced and emitted as metadata. All adapters MUST pass the same
conformance scenarios (`ProviderConformanceTests`). Because error bodies are never
read, a context-window overflow is not distinguished from other rejected requests:
it is `request_rejected`, is not retried and stops the task as `model_error`.
Applications avoid it with a `compact_at_tokens` threshold below the context window.

### FR-13 — Retrieve bounded, untrusted local evidence

`search_documents` MUST use a separate knowledge directory outside the writable
workspace, bounded top-level UTF-8 documents and deterministic keyword-ranked
passages with source IDs. Evidence MUST remain tool-result data, not execution
authority. This is lexical retrieval, not vector search or verified factual reasoning.
File tools MUST reject resolved paths outside the workspace; this is not race-free
filesystem confinement. See [retrieval limits](REFERENCE.md#corpus-and-limits).

### FR-14 — Keep evaluation separate and comparable

HotpotQA evaluation MUST default to offline smoke; only explicit `--live` opts into
paid model calls. Per-question workers MUST receive label-free question/candidate
inputs; gold scoring remains on the host. Sequential researcher/reviewer agents MUST
share total evaluation budgets, not receive fresh full budgets per role.

Artifacts MUST use fresh output directories, record input/source/model-setting
fingerprints and reject incompatible references before workers run. Offline quality
scores MUST remain null. Matching recorded settings MUST NOT be presented as proof
of unchanged provider behavior or a general multi-agent accuracy advantage.

### FR-15 — Expose bounded memory as untrusted data

Applications MAY bind a namespace to memory tools; the model MUST NOT select the
namespace. `recall` returns bounded JSON marked untrusted; `remember` and `forget`
use the ordinary risky-action approval, policy and journal gates. Model-created
memories carry their actual run ID. Default traces and events omit memory content.
Database stores apply namespace quotas transactionally and never evict on a full
quota. Namespaces are labels, not authorization. Memory cannot guarantee a model
will ignore injected instructions. See [storage spec](STORAGE_SPEC.md#33-memorystore-and-memory-tools-in-harnessmemorypy).

## 5. Terminal states and invariants

| Run status | Meaning / representative stop reasons |
| --- | --- |
| `completed` | Final model response obtained, normally `end_turn`; not a correctness claim |
| `max_tokens` / `refusal` | Final response truncated or declined by provider |
| `model_error` | Expected main model/API/response failure |
| `interrupted` | User cancellation; already performed effects are not rolled back |
| `stopped` | Step/budget cap, `usage_unavailable`, `tool_timeout`, `tool_outcome_unknown`, `tool_policy_error`, `journal_error` or `journal_recovery_required` |

Tool/journal outcomes are `completed`, `failed`, `timed_out`, `outcome_unknown` and
`not_run`; journal records additionally begin at `started`. Failure does not always
mean no effect: `outcome_unknown` is an independent flag. Unexpected harness exceptions
can propagate; observer terminal metadata may report `error`/`harness_error`, which
is not a returned `RunResult.status` value.

1. A model request never directly executes an arbitrary function or shell command.
2. Validation → policy → approval → optional durable start → invocation is ordered.
3. Completed tool results survive supported batch stops; remaining calls are paired.
4. Uncertain outcomes stop further model/tool activity; there is no automatic replay.
5. Sessions, traces, callbacks and journals have distinct jobs; none guarantees exactly-once effects.

## 6. Defaults and configuration ownership

| Setting | Default / scope |
| --- | --- |
| `max_steps` | 12 main-loop steps per task |
| `compact_at` | 40,000 JSON characters; teaching CLI configuration can override it |
| `compact_at_tokens` | Unset; a positive integer switches compaction to an estimated token threshold |
| Tool output | 8,000 content characters, plus truncation notice; core accepts `max_tool_output` |
| `limits`, tool timeout, journal, policy, callbacks | Unset/off unless explicitly configured |
| `quiet`, event content, trace content | `False` |
| Trace / library session | `trace.jsonl` / no session unless supplied |
| CLI interactive session | `session.json`, configurable with `HARNESS_SESSION` |
| Workspace / knowledge | Teaching configuration only: `./workspace` / `./knowledge`; explicit paths or CLI environment settings |
| Core `writable_roots` | Empty tuple; applications declare all writable roots for journal path exclusion |

CLI startup loads simple `.env` assignments without shell expansion; existing shell
values win. Importing `harness` performs no application configuration or filesystem
initialization. Application imports also perform no setup; the teaching factory
creates the workspace only when explicitly invoked. Library callers explicitly
pass runtime controls. `HARNESS_READ_ONLY`,
`HARNESS_MAX_*`, tool-timeout and journal CLI settings are not inherited as controls
by HotpotQA workers.
Full variable/provider reference: [REFERENCE](REFERENCE.md#configuration).

## 7. Acceptance scenarios and test traceability

Paths below are relative to `tests/`; scenarios summarize existing regression coverage.

| ID / requirements | Given / when | Required observable result | Test modules |
| --- | --- | --- | --- |
| AC-01 / FR-01,02,07 | A valid registered call then a final answer | One execution, paired result, final text; unsupported arguments return errors | `test_mini_harness.py` |
| AC-02 / FR-03,04 | Read-only policy and a write request | No approval, journal action or worker; `not_run` denial | `test_tool_policy.py` |
| AC-03 / FR-03,07 | Predicate raises after an earlier completed call | Preserve prior result; stop as `tool_policy_error`; no later dispatch | `test_tool_policy.py` |
| AC-04 / FR-05,10 | Compaction and main requests share a call cap | Summary consumes capacity; subsequent work stops at the same cap | `test_harness_runtime.py` |
| AC-05 / FR-05,07 | Multi-call batch exhausts attempt budget | Completed result retained; remaining calls explicitly not run | `test_harness_runtime.py` |
| AC-06 / FR-05 | Token-limited model omits usage | Unknown totals remain null; stop with `usage_unavailable` | `test_harness_runtime.py` |
| AC-07 / FR-06,07 | Isolated tool hangs or loses its result | Stop without retry; uncertain outcome and paired remaining results | `test_tool_execution.py` |
| AC-08 / FR-08,09 | Process dies around an action/journal commit | Restart blocks before model calls; manual resolution never replays | `test_execution_journal.py`, `journal_crash_worker.py` |
| AC-09 / FR-11 | Observer raises or attempts payload mutation | Ordinary failure disables observation; immutable payloads protect state | `test_harness_events.py` |
| AC-10 / FR-12 | Model endpoint redirects or emits diagnostic errors | Redirect blocked; diagnostic content absent from ordinary results/traces | `test_model_clients.py`, `test_mini_harness.py` |
| AC-13 / FR-12 | The same text, tool-call, stream-failure and HTTP-failure scenarios reach each adapter | Identical normalized responses; identical error categories; no retry after streaming began | `test_mini_harness.py` (`ProviderConformanceTests`) |
| AC-14 / FR-08,09 | The same session and journal scenarios reach each storage backend | Exact round trips, rewrites before the latest message, invalid data rejected, recovery required after an unfinished action, second writer refused where supported | `test_storage_conformance.py` (`storage_conformance.py` suites) |
| AC-15 / FR-15 | Scoped memory, denied writes and poisoned recall | Namespace isolation and quota; denied writes create no journal action; recalled text is marked untrusted and omitted from default observations | `test_memory.py`, `test_storage_conformance.py`, `test_support_evaluation.py` |
| AC-11 / FR-13 | Malicious text is retrieved from an approved document | Evidence remains data, with corpus/output limits and source metadata | `test_mini_harness.py` |
| AC-12 / FR-14 | Offline paired evaluation or mismatched reference | Null quality; shared role budgets; mismatch rejected before inference | `test_evaluate_hotpotqa.py`, `test_hotpotqa_multi_agent.py` |

Separation acceptance examples (SEP-01–12) are covered by `test_harness_package.py`
and `test_harness_separation.py` (copied package, retired-module absence and configured tools),
`test_hotpotqa_separation.py` (public composition, package imports/commands and
independent corpora), `test_evaluation_separation.py` (host ownership, module
commands, label-free parser and copied runtime/inventory application), plus shared
fingerprint preflight in `test_hotpotqa_multi_agent.py`.
Public admission ordering/failure and detached trace metadata are covered in
`test_harness_package.py`.
Strict registration and trace privacy are covered in `test_tool_contracts.py` and
`test_trace_privacy.py`. Application-owned answer validation and local runtime
distribution are covered in `test_support_agent.py` and `test_runtime_distribution.py`.

Run the offline suite with `python3 -m unittest -q`. Installed full-dataset checks
require prepared data and `HARNESS_TEST_HOTPOTQA_DEV=1`. Unit/smoke success establishes
mechanics, not real-model prompt-injection resistance or answer accuracy.

## 8. Operational limitations and change discipline

Protect the database and connection credentials. Durability depends on storage.
This is a single-owner recovery aid, not a concurrent scheduler. Keep the same
journal across restarts; switching/removing it bypasses recovery protection.

Read-only tools can disclose data. Workers inherit credentials, filesystem and
network access; `worker_env` can withhold environment credentials only. Predicates, callbacks, approval and inline tools can block indefinitely;
task deadlines cannot preempt them. Worker cleanup cannot undo remote work or guarantee
termination of detached descendants. Content-enabled traces, sessions (which keep the
full history), results and ordinary inline tool errors may contain sensitive data. Metadata-only traces/events
minimize content, but IDs, names and caller annotations can still disclose information.
There is no automatic retention, encryption or general redaction layer.

For a behavioral change: identify affected FR/AC IDs, amend the contract, add focused
tests, implement conservatively, then run the relevant offline checks. Shared-source
fingerprint changes require fresh paired evaluation references; historical artifacts
MUST NOT be overwritten or silently reinterpreted. Repeated-call detection, parallel
tools, tenant isolation and automatic recovery are not implemented features.