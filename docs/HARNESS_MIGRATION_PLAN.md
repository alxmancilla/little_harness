# Little Harness migration plan and checkpoints

Companion: [separation specification](HARNESS_SEPARATION_SPEC.md).
Each phase is a bounded checkpoint, not permission to silently change behavior.

**Current state: package-only.** The checkpoints below are chronological history,
not support promises. All root compatibility modules, legacy agent adapters and
global-registry fixtures have since been removed. Use `harness` imports,
`python3 -m agents.teaching.cli`, and `python3 -m evaluation.hotpotqa.<module>`.
Historical run artifacts and their frozen fingerprints are not migrated; create
fresh paired references with the current package sources.

## Phase 1 — extract the reusable execution engine

1. Create `harness.Agent` and instance-owned `ToolRegistry`.
2. Keep neutral root support modules canonical; export their public runtime types
   through `harness` without importing any application.
3. Replace the legacy loop with a compatibility subclass; retain built-ins/CLI.
4. Add new runtime files to shared evaluation fingerprints.
5. Add import isolation, custom-agent, registry, journal-root and worker tests.

Gate: all existing tests, new separation regressions, full installed-data offline
suite, fresh ten-question single-/multi-agent smoke pair, clean whitespace check,
and unchanged existing run artifacts plus `.env`. No paid/model-network calls.

Delivered scope: execution is reusable now; teaching/HotpotQA applications are not
yet relocated. The package still needs the neutral root support modules alongside
it. Phase 1 does not claim a standalone distributable package.

### Phase 1 checkpoint — complete

- Reusable loop/registry extracted; legacy execution methods share the same
  implementation. No prompt, provider, scoring or artifact-format changes.
- `HARNESS_TEST_HOTPOTQA_DEV=1 python3 -m unittest -q`: **283 tests pass**.
  New regressions cover package import/run isolation, mandatory allowlists,
  separate registries, metadata/validation, custom timed workers, writable roots,
  journal sidecars, core budgets/events/policies and legacy class/loop compatibility.
- Fresh ten-question single-agent and multi-agent offline evaluations: **20/20
  completed, zero tool errors**, matching recorded settings and verified package
  source fingerprints. Scores remain null; these are protocol checks, not accuracy
  evidence. Smoke outputs are outside the repository in a fresh temporary directory.
- All **539 existing run files** and `.env` are checksum-unchanged.
- Local documentation links and `git diff --check` pass; the separation diagram
  renders. No third-party dependencies or paid/model-network calls were added.
- Remaining work begins at Phase 2. Neutral root support modules, teaching tools,
  HotpotQA applications and evaluation code have NOT yet been relocated.

## Phase 2 — consolidate runtime modules and application configuration

Move neutral leaf modules into `harness/`, retaining root compatibility imports and
recovery/worker commands. Keep class identity stable across old/new imports. Add
explicit application-owned configuration/tool factories; move teaching tools/demo
out of the legacy entrypoint. Preserve worker importability and per-instance paths.

Gate: import the copied runtime package without repository application files; run
two independently configured workspaces in one process; verify old/new class
identity, CLI environment precedence, subprocess tools and complete fingerprints.

### Phase 2 checkpoint — complete

- All six neutral support implementations now live under `harness/`; root aliases
  preserve module/class identity and monkeypatch behavior in either import order.
  Root recovery/worker commands remain supported. The complete runtime directory
  runs without any root/application files; install metadata is not yet provided.
- `agents/teaching/` owns configuration, tools, prompts, demos and CLI. Frozen
  `TeachingConfig`, `create_tools` and `create_agent` allow independent workspaces
  and corpora using only the public runtime API. Imports/tool factories create no
  files; explicit agent setup creates its workspace. The legacy entrypoint retains
  thin wrappers, its live default registry/globals and historical import behavior.
- Timed workers additionally accept `functools.partial` bindings of importable
  module functions with JSON-only values. Teaching configuration travels over
  stdin, not model schemas/argv or process-global overrides. Closures/arbitrary
  callable objects still fail closed; no pickle or inline fallback was added.
- `HARNESS_TEST_HOTPOTQA_DEV=1 python3 -m unittest -q`: **293 tests pass**.
  Ten new regressions cover copied-package execution with a custom timed tool,
  policy/events/journal, both import orders, no-I/O teaching imports/factories,
  schema parity, immutable config, independent inline/timed workspaces/corpora,
  model attempts to replace hidden config, factory policy/root protection,
  unsupported partials, recovery commands and CLI environment precedence.
- Fresh ten-question single-/multi-agent offline pair: **20/20 completed, zero
  tool errors**, matching recorded settings and all current shared source hashes.
  Every result field matches Phase 1 except runtime; frozen inputs are identical.
  Scores remain null: these checks do not measure model accuracy.
- All **539 historical run files** and `.env` are checksum-unchanged. Smoke outputs
  are outside the repository in `harness-phase2-3xm6f2au` under the OS temporary
  directory. No dependencies or paid/model-network calls were added.
- Specification, source ownership, README and diagrams reflect Phase 2. Source
  fingerprints include canonical runtime files, compatibility aliases and teaching
  package dependencies; missing/changed shared hashes still fail closed.
- Local documentation links, IDE diagnostics and `git diff --check` pass; the
  updated separation diagram renders.
- Remaining work begins at Phase 3: HotpotQA application/workflow relocation.
  Evaluation relocation and an unrelated example remain Phase 4.

## Phase 3 — separate agent applications and workflows

Move single-agent and multi-agent HotpotQA logic under `agents/hotpotqa/`. Make
factories accept the runtime and tool/config dependencies explicitly. Keep role
prompts, label-free input construction, evidence handling, workflow limits and
prediction formats unchanged. Root worker commands remain wrappers.

Gate: both architectures run through the new imports with identical deterministic
offline predictions, tool errors and budget accounting. Assert the runtime has no
imports of applications/workflows. Use fresh paired source fingerprints.

### Phase 3 checkpoint — complete

- Shared HotpotQA logic and single-/multi-agent workers now live under
  `agents/hotpotqa/`. Root imports are identity-preserving aliases and old worker
  commands remain supported. Both workers also run as package module commands.
- Frozen `HotpotQAConfig` and explicit runtime/tool factories replace legacy path
  globals. Imports and tool factories perform no application setup or `.env` loading.
  Only worker CLI entrypoints load `.env`; general harness CLI controls are ignored.
  Search/arithmetic implementations are reused through explicit teaching-tool bindings.
- Workflows compose plain `harness.Agent` instances. Public `tool_admission` and
  `trace_context` replace private dispatch/trace overrides; metadata-only finish
  events count tool errors. Admission cannot authorize unknown tools or bypass
  validation, policy, approval, journal or worker checks. Thin legacy classes remain.
- Existing evaluation budget rules, role allocations, bounded untrusted handoff,
  prompts, schemas, prediction formats and result fields are preserved. Shared
  fingerprints cover canonical common/single modules and package initialization;
  multi-agent manifests also fingerprint the canonical workflow.
- `HARNESS_TEST_HOTPOTQA_DEV=1 python3 -m unittest -q`: **306 tests pass**.
  Thirteen new tests cover import identity/isolation, independent corpora/observations,
  supplied runtime/tool/search dependencies, mixed request budget semantics, legacy
  adapters, root/module command parity, ignored general CLI controls, admission
  ordering/failures and detached/validated trace context. Fingerprint regressions
  reject every missing/changed shared source and verify the multi-only module.
- Fresh ten-question single-/multi-agent offline pair: **20/20 completed, zero
  tool errors**, matching recorded settings and every current source hash. Every
  result field and every question trace matches Phase 2 except timings/timestamps;
  frozen inputs are identical. Scores remain null, not model-accuracy evidence.
- All **539 historical run files** and `.env` are checksum-unchanged. Final smoke
  outputs are `single-final/` and `multi-final/` in OS temporary directory
  `harness-phase3-l_9cqnyo`. No dependencies or paid/model-network calls were added.
- Specification, architecture, README and evaluation guide reflect Phase 3. Local
  documentation links, IDE diagnostics and `git diff --check` pass; the separation
  diagram renders. Remaining work is Phase 4: evaluation relocation and an unrelated
  application example.

## Phase 4 — separate evaluation and demonstrate another application

Move preparation/scoring/runner/comparison code under `evaluation/hotpotqa/`,
preserving root commands and historical artifacts. Add a small unrelated custom
agent example using only the public runtime API. Update ownership diagrams and
freeze the supported API based on validated use cases, not speculative abstractions.

Gate: CLI compatibility, evaluation label separation, deterministic offline parity,
source-boundary coverage, and standalone runtime/example execution. No dependency
addition, paid evaluation or deployment is implied by this plan.

### Phase 4 checkpoint — complete

- Host preparation, dataset loading, scoring, comparison and runners now live in
  `evaluation/hotpotqa/`. Identity-preserving root aliases retain patch seams;
  preparation/evaluation/lesson root commands and package module commands expose
  the same options. Repository-relative dataset defaults remain stable from other
  working directories. No prompt, scoring-rule or artifact-format change was made.
- The unchanged strict prediction parser lives in `agents/hotpotqa/prediction.py`
  and is re-exported by the host scorer. Applications import no evaluation modules
  and run from a copied distribution without host/root files. Canonical parser,
  scorer and initialization hashes are shared fingerprints; canonical host modules
  and compatibility wrappers are recorded as host source provenance.
- Added `examples/inventory_agent.py`: its own prompt, deterministic offline model
  and read-only stock tool, using only stdlib and public runtime exports. Copying
  `harness/` plus `examples/` supports isolated tool execution with policies, events
  and a journal. Inventory snapshots stay independent and cannot be replaced by
  model arguments. No `.env` loading or teaching/HotpotQA dependency is involved.
- `HARNESS_TEST_HOTPOTQA_DEV=1 python3 -m unittest -q`: **316 tests pass**.
  Ten new regressions cover root/module import identity and no import-time setup,
  stable paths, copied application execution without evaluation, byte-identical
  preparation/refusal to overwrite, CLI help parity, real-data commands from another
  directory, label-free parsing, and copied/independent inline/isolated inventory
  behavior. Existing fingerprint regressions now cover canonical host/parser files.
- Fresh ten-question single-/multi-agent offline pair: **20/20 completed, zero
  tool errors**, matching recorded settings and every current source fingerprint.
  All result fields and all 20 question traces match Phase 3 except timings and
  timestamps; frozen inputs are identical. Quality remains null, not accuracy evidence.
- All **539 historical run files** and `.env` are checksum-unchanged. Smoke outputs
  are `single/` and `multi/` in OS temporary directory `harness-phase4-w86zxwlt`.
  No dependencies or paid/model-network calls were added.
- Supported public API, source ownership, README, evaluation guide and diagrams
  reflect the completed separation. Local documentation links, IDE diagnostics and
  `git diff --check` pass; the separation diagram renders.
- All four planned phases are complete. The runtime is independently copyable;
  install/distribution metadata, new frameworks and deployment remain out of scope.

## Post-separation cleanup checkpoint — complete

- Shared worker argument parsing, model configuration, prompt hashing and exclusive
  output writing now live in `agents/hotpotqa/worker.py`. Completed-answer parsing
  also shares one helper, preserving failure/cancellation priority. Role filtering,
  orchestration, budgets, prompts and artifact formats stay unchanged.
- Root and package worker commands remain supported. The shared helper is included
  in source fingerprints; missing/changed hashes still fail reference preflight.
  Historical manifests are not rewritten; new comparisons require fresh paired runs.
- Reusable scripted models, protocol/dataset fixtures, compaction histories and
  temporary-directory/subprocess test bases live in `tests/helpers.py`. Tests no
  longer import fixtures from other test modules or borrow their setup methods.
  Importing helpers does not load applications, read `.env` or create workspaces;
  legacy application initialization is explicit and temporary directories are cleaned.
- `HARNESS_TEST_HOTPOTQA_DEV=1 python3 -m unittest -q`: **319 tests pass**.
  Three new regressions cover helper import isolation, offline defaults/exclusive
  output, and both mocked live entrypoints' limits, metadata and prompt hashes.
- Fresh ten-question single-/multi-agent offline pair: **20/20 completed, zero tool
  errors**, matching recorded settings and verified current source fingerprints.
  Every result field and all 20 traces match Phase 4 except timings/timestamps;
  frozen inputs are identical. Quality remains null, not model-accuracy evidence.
- All **539 historical run files** and `.env` are checksum-unchanged. Smoke outputs
  are `single/` and `multi/` in OS temporary directory `harness-cleanup-8avmfrd8`.
  No dependencies or paid/model-network calls were added. Dataset deletion,
  `.gitignore` changes and shared-tool relocation were not part of this cleanup.

## Package-only checkpoint — complete

- Removed all 16 root compatibility modules, legacy agent subclasses, global
  registry adapters and worker override hooks. File removals are recoverable;
  historical run files and frozen fingerprints are not migrated or rewritten.
- Applications and tests import package APIs and pass explicit tools/configuration.
  Supported commands use `python3 -m agents.teaching.cli`,
  `python3 -m evaluation.hotpotqa.<module>` and `python3 -m harness` for journal
  inspection/human recovery. The recovery command avoids double-import warnings.
- Current guides, README, diagrams and `.env.example` use package-only commands.
  Earlier checkpoints above are historical, not compatibility promises.
- `HARNESS_TEST_HOTPOTQA_DEV=1 python3 -m unittest -q`: **318 tests pass**.
  Local documentation links/anchors, source parsing and `git diff --check` pass;
  no application/runtime/test imports depend on a retired root module.
- Fresh ten-question single-/multi-agent offline pair: **20/20 completed, zero
  tool errors**, matching recorded settings and every current source fingerprint.
  All result fields match the previous cleanup pair except task/role runtimes;
  frozen inputs are identical and quality scores remain null.
- All **539 historical run files**, dataset data and `.env` are checksum-unchanged.
  Smoke outputs are `single-smoke/` and `multi-smoke/` in OS temporary directory
  `harness-package-only-zblyu939`. No dependencies or paid/model-network calls added.

## Execution and rollback rules

- Validate and record delivered versus remaining work at each checkpoint.
- Do not mix prompt/model/scoring changes with file-boundary migrations.
- Roll back a failed phase's source changes, not earlier work or historical outputs.
- Do not automatically replay journal actions or repair experiment metadata.
- Never modify `.env`; write smoke artifacts into a fresh temporary directory.