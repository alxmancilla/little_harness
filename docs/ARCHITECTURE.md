# Little Harness architecture

This is the implementation companion to the [behavioral specification](HARNESS_SPEC.md).
It describes the current standard-library design, not a proposed framework migration.
The [separation specification](HARNESS_SEPARATION_SPEC.md) and
[migration plan](HARNESS_MIGRATION_PLAN.md) record the completed boundaries: runtime,
teaching/HotpotQA applications, host evaluation and an unrelated inventory example.

## 1. Explain it in one minute

The model proposes actions; `Agent` decides whether and how they run. The parent
process owns conversation state, budgets, policy checks, approval and journal commits.
Provider adapters translate a common message format. Tools run sequentially, normally
inline; an optional disposable process makes a tool timeout enforceable. Sessions
preserve conversation, traces aid diagnosis, events support embedding, and the
optional SQLite journal blocks unsafe continuation after uncertain outcomes.

None of these mechanisms is an OS sandbox or exactly-once external transaction.

## 2. Component view

Solid arrows indicate calls/data flow; dotted arrows show optional controls or
observations. The full diagram includes process and storage boundaries:
[editable Mermaid source](architecture.mmd).

````mermaid
flowchart LR
    App["CLI adapter / application"] --> Agent["Agent - harness/agent.py"]
    Agent <--> Model["Model adapters / scripted model"]
    Agent --> Gates["Validate, policy, approval"] --> Exec["Inline tool / isolated worker"]
    Agent --> Budget["Shared ExecutionBudget"]
    Agent --> Session[("Session JSON")]; Agent --> Trace[("Trace JSONL")]
    Gates --> Journal[("Optional SQLite journal")]; Agent --> Events["Optional RunEvent callback"]
````

This overview compresses execution ordering; use the detailed source and sequence
view below for the journal commit-before-invocation requirement.

### Source ownership

| File / symbol | Responsibility | Important boundary |
| --- | --- | --- |
| `harness/agent.py`: `Agent` | Run lifecycle, conversation, compaction, dispatch and saving | Requires explicit tools/paths; imports no application |
| `harness/tools.py`: `ToolRegistry`, `validate_tool_input` | Instance-owned registry; primitive schemas with optional choices and descriptions | Tool semantic checks still belong inside functions |
| `harness/model_clients.py`: `HTTPModel` and adapters | HTTP/SSE, pre-stream retry, normalized responses, provider selection | Depends on no agent state; rejects model HTTP redirects |
| `harness/runtime.py` | Immutable limits/usage/results and mutable per-task budget | Shared across main and summary calls; no provider dependency |
| `harness/policy.py`: `ToolPolicy` | Explicit read-only declarations and argument predicates | Trusted, synchronous parent code; not authorization isolation |
| `harness/tool_execution.py` | Callable descriptor, fresh worker, JSON protocol, timeout and cleanup | Module functions or JSON-bound partials; no closures/bound methods or parent-memory propagation |
| `harness/session.py`: `SessionFile`, `SessionLoadError` | Load and atomically save version 2 session files (full history plus compaction point); reads legacy lists | No agent state; durable fsync only when the agent has a journal |
| `harness/journal.py`: `ExecutionJournal` | Durable action metadata, recovery guard, inspect/resolve CLI | No arguments/outputs; no replay or rollback |
| `harness/events.py`: `RunEvent` | Versioned, deeply immutable observation snapshots | Delivery/failure handling lives in `Agent._emit` |
| `agents/teaching/` | Explicit config, tool/agent factories, arithmetic/files/retrieval tools, demos and CLI | Imports the public runtime; no configuration loading or workspace creation on import |
| `agents/hotpotqa/` | Explicit config, restricted tools, per-agent accounting and sequential researcher/reviewer workflow | Composes plain runtime instances with public hooks; no legacy globals/private runtime overrides |
| `evaluation/hotpotqa/` | Host preparation/loading, per-question runners, scoring and comparisons | Applications do not import evaluation; only the host reads labels |
| `examples/inventory_agent.py` | Unrelated read-only inventory application | Imports only public `harness` exports and stdlib; copyable without teaching/HotpotQA/evaluation files |

Optional controls are ordinary Python objects/modules, **not separate services**.
New applications import `harness` and supply prompts, registries and writable roots.
Package import has no workspace/configuration side effects. Only package APIs and
module commands are supported. The runtime directory can be copied without application or
root files; see the [separation diagram](harness-separation.mmd) for dependencies.
Read-only declarations and timeout settings are registry metadata, not fields sent
to the model in tool schemas. The built-in document search has no embeddings,
external vector database or background indexing service.

## 3. Run and dispatch sequence

Detailed interaction view: [editable Mermaid sequence diagram](dispatch-sequence.mmd).
The diagram illustrates the main path; terminal stops short-circuit its loops rather
than continuing to later calls or steps.

1. `run_result()` rejects overlap, creates a fresh `ExecutionBudget` and run ID, emits
   run start, then checks journal recovery before task/model work.
2. `_run()` appends/saves the task and iterates main steps. `_compact()` may summarize
   older context through `_call_model()`, using the same budget as normal calls. It
   moves the view's start point; it never shortens the saved history.
3. `_model_context()` builds the model's input from history: the latest summary, then
   messages from the compaction point. `_call_model()` admits/counts the logical call, delivers model observations and
   records usage. HTTP/SSE or a scripted callable returns normalized content.
4. For each non-truncated tool request, the agent admits an attempt. `_run_tool()`
   performs allowlist lookup, then calls one gate method per check in order:
   `_gate_admission` (optional application admission), `_gate_known_tool`,
   `_gate_validation` (primitive validation), `_gate_policy` (optional policy),
   `_gate_timeout` (timeout-config validation) and `_gate_approval` (risky-action
   approval). The first gate that rejects returns a not-run `ToolCallOutcome`.
5. After budget rechecks, `_dispatch()` lets an enabled journal commit `started`, emits
   a tool-start observation, rechecks available budget, then `_invoke_tool()` executes
   inline or starts the isolated worker. Policies/approval never run in that worker.
6. A trustworthy tool outcome is committed to the journal when enabled. The loop
   retains its result before finish notifications/printing/tracing. Completed and
   remaining not-run results are preserved together on supported terminal stops.
7. A final model answer or stop is persisted. When safe, the journal run closes after
   saving; terminal events follow finalization. The caller receives `RunResult`.

Denied calls can return errors to the model; policy errors, exhausted budgets,
timeouts, uncertain outcomes and journal failures stop further work. There is no
automatic tool retry. Callbacks may fail or cancel, so notification delivery is not
proof that execution did or did not happen.

### Process boundary

The default inline function runs in the parent's interpreter and can mutate its
memory. An isolated tool starts a fresh Python interpreter, imports the registered
module-level function, receives JSON over stdin and returns a bounded JSON outcome.
JSON-only bound values in `functools.partial` also travel over stdin. The teaching
factory binds its own configuration this way; it needs no private runtime override.
Imports must be safe and script startup guarded. Worker stdout/stderr is discarded;
the private result protocol is separate from tool printing.

The timeout covers startup/import/execution/transfer after launch, with bounded
best-effort cleanup overhead. On POSIX, cleanup targets the worker process group;
other platforms stop only the worker. Both modes inherit application access to files,
credentials and network. Detached descendants/remote operations can outlive cleanup.

## 4. State, persistence and recovery

| State/artifact | Owner and lifetime | What it does not establish |
| --- | --- | --- |
| `Agent.messages` | In-memory conversation across successive tasks | Durable proof of tool effects |
| `ExecutionBudget` | Fresh counters/deadline for each task | Hard preemption or exact billing |
| Session JSON | Optional atomic conversation snapshot; interactive CLI default | Automatic output replay or action reconciliation |
| Trace JSONL | Append-only metadata by default; content explicitly opt-in; default `trace.jsonl` | Replay engine, secret-redacted log or authoritative action ledger |
| SQLite journal | Optional durable run/action metadata across restarts | External exactly-once execution or cancellation |
| Event callback | Synchronous ephemeral observations during a task | Durable queue, permission or complete delivery |

A journal action begins at `started`, then records `completed`, `failed`, `timed_out`,
`outcome_unknown` or `not_run`. Unknown effects and actions from an unclosed run block
subsequent tasks. A completed action can still require recovery if a crash left its
run unclosed and its result may be absent from the session.

Recovery is intentionally manual: stop affected workers, inspect external effects,
inspect journal metadata, then confirm a known outcome. `resolve` preserves the
original record and adds the human assertion; it does not run tools or fix history.
See [recovery procedure](REFERENCE.md#recover-without-replaying).

## 5. Separate evaluation architecture

The evaluation tools are clients of the harness, not hidden stages of every task.
The host validates/prepares the dataset, keeps gold labels for scoring and launches
a fresh worker process per question. The single-agent worker creates one restricted
agent. The multi-agent worker runs researcher then reviewer sequentially in the
**same question worker**, with separate histories and a bounded untrusted handoff.

````mermaid
flowchart LR
    Data["Prepared HotpotQA"] --> Host["evaluation/hotpotqa/evaluate.py"]
    Host -->|"label-free payload"| Worker["Fresh question worker"]
    Worker --> Single["Single restricted Agent"]
    Worker --> Researcher["Researcher Agent"] -->|"bounded handoff"| Reviewer["Reviewer Agent"]
    Single --> Result["Prediction and usage"]; Reviewer --> Result
    Result --> Score["Host scoring / comparison"]; Data -->|"gold labels"| Score
````

| Module | Role |
| --- | --- |
| `evaluation/hotpotqa/prepare.py`, `dev.py` | Preparation, provenance/integrity and isolated question loading |
| `evaluation/hotpotqa/run_question.py` | Standalone one-question lesson, offline unless `--live` |
| `evaluation/hotpotqa/evaluate.py` | Cohort selection, per-question process orchestration and artifacts |
| `agents/hotpotqa/common.py` | Shared restricted tools, config, public-runtime factory, admission budget, metering and metadata |
| `agents/hotpotqa/single.py` | Single-agent question application and module command |
| `agents/hotpotqa/multi.py` | Sequential role workflow and shared total evaluation budgets |
| `agents/hotpotqa/prediction.py` | Label-free strict prediction contract; no evaluation dependency |
| `evaluation/hotpotqa/scoring.py` | Host-side answer/evidence scores; re-exports the same parser |
| `evaluation/hotpotqa/comparison.py` | Reference preflight and recorded-setting/resource comparisons |

Research receives at most half the total model/tool budget, rounded down; review
receives what remains. Compaction is disabled for these roles. Candidate tools use
per-question in-memory data. General CLI policy/journal/timeout settings do not
silently configure evaluation controls. Only `--live` opts into paid model calls.
`run_agent` and `run_agents` accept explicit config/runtime/tool factories. The
default factory composes plain `harness.Agent`: `tool_admission` enforces the
application budget, metadata-only `tool_finished` events count errors, and
`trace_context` adds role labels. Search/arithmetic code is reused from teaching
tools through bound configuration, not a global registry. Workers do not create
workspaces; they expose no writing tools. Prediction parsing is application-owned;
workers import no evaluation modules and can run without host files. The host keeps
scoring and gold labels outside that import/input boundary. Package module commands
retain dataset defaults, including when run from another directory with the repository
on `PYTHONPATH`. Source/input fingerprints reject incompatible references and cover
package implementations; host-only source hashes are recorded separately. Offline smoke
has no accuracy score. See [evaluation contract](../datasets/HOTPOTQA_BASELINE.md).

## 6. Design rationale and implementation limits

- **One synchronous owner:** makes dispatch and result ordering understandable;
  sacrifices concurrency and cannot preempt blocking callbacks/inline functions.
- **Provider adapters:** keep transport changes outside the loop; internal messages
  are normalized, but provider-specific reasoning state is not portable.
- **Fresh process for timed tools:** allows enforced waiting bounds without killing
  threads inside the parent; requires importable functions and explicit JSON inputs.
- **SQLite before effects:** reduces silent loss after crashes without inventing
  exactly-once guarantees or unsafe automatic retries.
- **Separate observations and authority:** callbacks serve applications; policy and
  approval control dispatch. Content minimization is opt-in-path-specific, not global.
- **Separate experiment host:** keeps labels out of worker inputs and records evidence
  for comparisons; matching metadata cannot verify provider-side model stability.

For behavior changes, start with the requirement and acceptance IDs in the
[specification](HARNESS_SPEC.md), then change the responsible module and focused tests.