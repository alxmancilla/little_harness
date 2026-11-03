# Little Harness separation specification

Status: package-only implementation; all four separation phases complete. Validation is recorded in
[the migration plan](HARNESS_MIGRATION_PLAN.md). This specification describes the
current ownership contracts, not a new framework. Root compatibility layers are removed.

## 1. Goal and constraints

Reuse the execution engine for unrelated agents without importing the teaching
application, HotpotQA prompts, tools, datasets or evaluation code. Preserve Python
standard-library-only operation, explicit package APIs, security controls and
historical artifacts. Do not change `.env`, model settings, prompts or scoring.

Non-goals: a new agent framework, distributed scheduling, parallel tools, database
memory, a provider rewrite, an OS sandbox or exactly-once external actions.

## 2. Ownership and dependency direction

The harness owns the model/tool loop, conversation and compaction, limits/results,
approval, policy, events, durable action records and isolated execution. Applications
own prompts, tool implementations/registries, paths, provider selection and their
configuration. Workflows own delegation and shared budgets across agents. Evaluation
owns labels, scoring, frozen inputs, experiment contracts and comparisons.

Dependencies MUST point from applications/workflows to the runtime, never back.
The delivered layout has `harness/`, `agents/`, `examples/` and `evaluation/`.
Names describe ownership, not services or processes. Root aliases/scripts are not supported.

All runtime implementations live in `harness/`. Copying the complete directory is
supported without any application/root files. Packaging/install metadata is not yet provided.
`agents/teaching/` owns explicit configuration, tool factories, demos and CLI.
HotpotQA applications/workflows live under `agents/hotpotqa/`. Host preparation,
loading, scoring, comparison and runners live under `evaluation/hotpotqa/`. The
unrelated inventory example in `examples/` imports only the public runtime and
standard library. No runtime/application import depends on evaluation.

## 3. Required contracts

| ID | Requirement |
| --- | --- |
| SEP-01 | Importing `harness` MUST NOT import applications or evaluation modules; load `.env`; read application configuration; or create files/directories. |
| SEP-02 | `harness.Agent` MUST require an explicit tool mapping. `{}` means no tools. There MUST be no package-global default registry. |
| SEP-03 | `ToolRegistry.tool(...)` MUST derive flat schemas/metadata from the function signature and register only into that registry. |
| SEP-04 | The model callable MUST retain `(system, messages, tools, on_text=None) -> response`; block shapes, usage accounting and stop classifications MUST remain compatible. |
| SEP-05 | Runtime paths/settings MUST be per-instance values. Application workspace protection MUST be supplied as `writable_roots`; the core MUST NOT know `HARNESS_WORKSPACE` or `HARNESS_KNOWLEDGE`. |
| SEP-06 | Allowlist -> validation -> policy -> approval -> journal -> execution ordering MUST remain unchanged, including deadline rechecks and no automatic replay/retry. |
| SEP-07 | Journal files and sidecars MUST remain separate from trace/session files and outside every declared writable root. Callers MUST declare all application-writable roots; omission is not filesystem isolation. |
| SEP-08 | Event immutability/order, observer-failure handling, uncertain outcomes, atomic session saving and subprocess cancellation MUST retain their current guarantees. |
| SEP-09 | Only package APIs and module commands are supported. There MUST be no root aliases, compatibility agent subclasses or package-global tool registry. Applications MUST pass explicit configuration. |
| SEP-10 | Evaluation MUST fingerprint every executable runtime file, including package initialization. Missing/changed shared hashes MUST fail reference preflight before workers/output creation. |
| SEP-11 | Historical run files/frozen fingerprints MUST NOT be rewritten. Structural changes require fresh paired references, not patched old metadata. |
| SEP-12 | Every phase MUST pass focused regressions, the offline full suite and fresh single-/multi-agent smoke evaluations before its checkpoint. |

`harness.Agent` retains `run()` display text and `run_result()` structured results.
`tools` is keyword-only and required; tool functions remain trusted Python code.
`max_tool_output` is a positive character cap (default 8000), `compact_at` defaults
to 40000, and `writable_roots` defaults to an empty tuple. Construction does not
create those roots. Trace/session/journal I/O retains its existing run-time behavior.

The explicit tool mapping snapshots membership, not callable state or arbitrary
trusted registry-entry mutations. Use separate registries and separate agent/model
instances for independent runs. An Agent is not re-entrant or thread-safe.

Two public extension points replace application overrides of private methods:
`tool_admission(name, available)` may reject an attempt before validation, but
cannot bypass the allowlist, policy, approval or journal. `trace_context` snapshots
JSON annotations such as a role; runtime fields win and reserved envelope keys are
rejected. Neither hook carries HotpotQA logic into the runtime. See the
[interface contract](HARNESS_SPEC.md#31-agent-and-extension-interfaces) for details.

## 4. Package-only application interfaces

Execution logic has one implementation in `harness/agent.py`. Applications import
`harness` directly. There are no legacy built-in wrappers or mutable global settings.
Teaching tools, demos, `.env` loading and CLI live in `agents/teaching/`;
the CLI runs as `python3 -m agents.teaching.cli`.
Teaching factories accept frozen `TeachingConfig` values, not process-global paths;
only explicit `create_agent` setup creates the workspace. `create_tools` has no I/O.
Configuration is bound positionally with `functools.partial`, never advertised in
model schemas. Timed execution supports JSON-bound partials of importable module
functions, without pickle or inline fallback; closures remain unsupported.

HotpotQA workers now compose plain `harness.Agent` instances through public APIs.
`run_agent` / `run_agents` accept explicit `HotpotQAConfig`, `runtime_factory` and
`tool_factory` dependencies. Config is immutable and reads environment paths only
when explicitly requested; only CLI entrypoints load `.env`. Tool factories reuse
teaching search/arithmetic implementations through explicit bindings, never globals.
`make_tools` also accepts `search_fn` for a supplied retrieval implementation.

The application owns `ToolBudget`: known attempts consume its budget before
validation, unknown/exhausted calls are recoverable rejections, and truncated
calls do not consume it. Role traces use `trace_context`; errors are counted from
metadata-only finish events. Prompts, schemas, shared role allocations, handoff,
prediction parsing and result fields stay unchanged. There are no compatibility
agent subclasses; callers pass config to the composed runtime factories.

Host evaluation/preparation commands run under `python3 -m evaluation.hotpotqa.<module>`;
repository-relative dataset defaults stay fixed when invoked elsewhere.
The strict label-free parser lives in `agents/hotpotqa/prediction.py`; the host
scorer uses that same parser. Applications run without evaluation files,
and no gold labels/scoring code enter the worker import graph. Shared source
fingerprints include the parser, canonical scorer and package initialization;
host implementation hashes are recorded separately. No historical manifest
is rewritten to accept the new layout.

The supported embedding API is the public `harness` exports and documented options
and hooks in [the interface contract](HARNESS_SPEC.md#31-agent-and-extension-interfaces).
The inventory example validates that surface with its own prompt, model callable
and read-only inventory registry, including JSON-bound isolated execution.

## 5. Acceptance examples

- In a fresh directory containing a sentinel `.env`, import and run a tool-free
  custom agent. No application import, `.env` load or workspace creation occurs.
- Two application registries with the same tool name produce different results;
  neither advertises or dispatches the other's private tools.
- An undeclared model-requested tool is rejected; an explicit empty mapping cannot
  fall back to teaching tools. A missing mapping is a construction error.
- Tests construct explicit per-test tools/configuration; risky calls, quiet denial,
  timeout cleanup and recovery retain their behavior. Retired root modules are absent.
- Journal paths beneath any declared writable root are rejected, as are collisions
  involving trace/session files and journal sidecars.
- A new custom module-level tool runs in a timed worker without importing teaching
  tools or creating a workspace; unsupported callable forms still fail closed.
- A copied `harness/` directory runs a custom timed tool with policies, events and
  journal records without root imports. Two teaching factories retain independent
  workspaces/corpora both inline and in timed workers, with configuration hidden from schemas.
- A missing or altered fingerprint for each new runtime file rejects paired
  evaluation before inference. Fresh offline pairs retain recorded-settings parity.
- HotpotQA package imports have no `.env`/workspace side effects. Module worker
  commands honor explicit environment paths; independent corpora and supplied
  factories do not leak state. Applications override no private runtime methods.
- Package evaluation/preparation/lesson commands retain options, defaults and
  artifact formats. Evaluation imports create no workspace or load `.env`.
  HotpotQA applications run from a copy without evaluation or root modules.
- Copying only `harness/` and `examples/` runs an inventory lookup with an isolated
  tool, read-only policy, event callback and journal. Independent inventory snapshots
  cannot be replaced by model arguments. Paired results/traces retain Phase 3 parity.

## 6. Architectural view

Editable diagram: [harness-separation.mmd](harness-separation.mmd). The existing
[behavior specification](HARNESS_SPEC.md) remains the detailed dispatch contract;
this document governs ownership and migration-specific differences.