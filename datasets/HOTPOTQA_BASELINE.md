# HotpotQA single-agent and multi-agent comparison

**Purpose:** establish a measured single-agent baseline, then run exactly the same
inputs through a multi-agent version. Dataset preparation and offline retrieval are
prerequisites—not evidence that either architecture answers correctly.

This baseline uses the **original ten teaching question IDs**, five comparison and
five bridge, in their pinned order. Each question receives **only its own original
candidate passages**, not the demo's shared 100-file corpus. The full 7,405-question
development split remains the larger pool; this first baseline intentionally does
not launch a full-dataset experiment. All code uses Python's standard library.

## Run it

First prepare the [full development split](HOTPOTQA_DEV.md), if not already present:
`python3 -m evaluation.hotpotqa.prepare --full-dev`.

Check all ten examples without network/model calls:
`python3 -m evaluation.hotpotqa.evaluate --output runs/single-agent-hardened-smoke`.
This performs real retrieval and sentence reading, then deliberately abstains.
It is marked **offline-smoke-not-an-accuracy-baseline**; quality scores are `null`,
not misleading zeros or synthetic model-accuracy results.

Start with **one paid-model example** using your configured provider:
`python3 -m evaluation.hotpotqa.evaluate --live --limit 1 --output runs/single-agent-hardened-live-one`.

Then explicitly run the ten-question live baseline:
`python3 -m evaluation.hotpotqa.evaluate --live --output runs/single-agent-hardened-v1`.
`--live` opts into API costs. The current directory's `.env` and existing environment
select the provider/model; neither is edited. Use a **new output directory** each
time. Existing results, including symlink destinations, are never overwritten.
`--limit 1` chooses the first pinned question, not a random or balanced sample.

### Revised guidance and reruns

After the first live baseline exposed formatting and reader-argument failures,
the single-agent instructions and tool descriptions were tightened:

- The final response explicitly forbids code fences, preambles and prose outside
  the JSON object. Strict parsing remains unchanged: no output repair or hidden retry.
- Baseline-only retrieval notices use the same JSON citation contract, instead of
  the general harness's bracket-style citations. General harness tools are unchanged.
- Search results now expose `document` (exact filename), `title` and `sentence_count`.
  The reader schema lists valid filenames and numeric bounds. Article titles and
  `#char-` source IDs are not valid reader filenames; errors explain how to recover.
- The prompt asks the agent to read each cited sentence first. This is guidance,
  not enforced evidence verification; inspect `read_evidence_fraction` and traces.

The questions, candidate text, scoring rules and budgets are unchanged. Extra search
metadata stays within the same output budget; lowest-ranked matches may be omitted
if needed to keep the JSON intact. Previous runs are **not rewritten or rescored**.
Prompt and code hashes distinguish the new configuration; input fingerprints should
still match the original run for the same selected questions.

The earlier revised live run was saved as `runs/single-agent-baseline-v2`.
Offline tests do not establish improved live-model behavior. Since these ten examples
informed the changes, treat them as development examples, not an untouched test set.
Use the current configuration for both architectures in the comparison below;
do not attribute differences from the old configuration solely to agent count.

### Hardened experiment version

Current runs record `version: 2` and `experiment_version: hardened-v1` in `run.json`.
The security/correctness review changed shared harness and worker code:

- Model API redirects are rejected, including same-origin redirects; configure the
  final endpoint directly. HTTP error bodies are closed without reading, and stream
  error details are replaced with generic messages before reaching logs or results.
- Tool arguments must be objects with the required fields and correct primitive
  types; unknown fields are rejected for schemas that disallow them. Validation
  happens before approval/execution. Invalid requests return tool errors and consume
  a tool attempt in these evaluations. Tool functions still enforce semantic bounds.
- Interruption and terminal failures take priority over an earlier truncated model
  response in both workers. An interrupted question stops the remaining evaluation.
- Results record versioned model metadata: provider/model, a SHA-256 fingerprint of
  the full configured base URL, output-token cap and provider-default sampling policy.
  The URL itself is not saved in this metadata. Fingerprints may differ if URL
  credentials or query parameters change, even for the same service.

**Create a fresh pair of runs.** Existing V2 and multi-agent V1 results remain
historical artifacts; their shared code hashes no longer pass reference preflight.
They are not rewritten or silently upgraded. Use the hardened commands above and
below, with new directory names if those destinations already exist. Matching
recorded settings cannot establish unchanged provider-side behavior.

### Structural refactor and reference compatibility

Provider code lives in `harness/model_clients.py`;
shared prompts, tools, metering and status helpers live in `agents/hotpotqa/common.py`
and workers run as `python3 -m agents.hotpotqa.single` or `python3 -m agents.hotpotqa.multi`.
Root aliases, commands and compatibility classes are removed. Prompts, tool schemas, budgets, scoring and
artifact formats; `hardened-v1` remains the behavior version.

Code fingerprints now include both extracted modules. Missing or changed shared
hashes fail reference preflight before inference. Even pre-refactor `hardened-v1`
runs need a fresh paired reference; existing results are never rewritten. Use new
output names if the examples below already exist. Host-side preparation, dataset
loading, runner and comparison modules are also fingerprinted in `run.json`.

The shared `harness/runtime.py` module is fingerprinted too. Core harness limits
and structured results preserve the workers' existing explicit evaluation budgets
and artifact format; the optional `HARNESS_MAX_*` settings apply only to the general
harness CLI. References from before this runtime addition require a fresh pair.

`harness/tool_execution.py` is also fingerprinted. Isolated tool timeouts are opt-in;
HotpotQA workers do not inherit `HARNESS_TOOL_TIMEOUT_SECONDS` and keep their
in-memory tools inline. References predating this module need a fresh paired run.

`harness/journal.py` is fingerprinted too. The optional journal backend settings
applies only to the general harness CLI, not evaluation workers. Journal metadata
does not change evaluation artifact formats; older references require a fresh pair.

`harness/events.py` is also fingerprinted. Event callbacks and quiet embedding are
opt-in library APIs; evaluation workers retain their existing output and artifact
formats. References from before this addition require a fresh paired run.

`harness/policy.py` is fingerprinted too. Policies are opt-in; evaluation workers do
not inherit `HARNESS_READ_ONLY` and retain their existing tool allowlists. Artifact
formats and evaluation behavior are unchanged; older references need a fresh pair.

Separation Phase 1 also fingerprints `harness/__init__.py`, `harness/agent.py` and
`harness/tools.py`. The loop/schema implementation lives there. Missing or changed
package hashes fail reference preflight. Create a fresh pair; never rewrite old
run metadata to make pre-extraction results appear comparable.

The shared boundary fingerprints every runtime file under `harness/`
and every teaching package file under `agents/teaching/`
(including package initialization). HotpotQA reuses teaching search/arithmetic
implementations through explicit configuration, so these remain dependencies.
The whole runtime directory can now be copied independently. References made
before this move need a fresh pair; neither prompts/scoring nor artifact formats
changed.

Phase 3 fingerprints `agents/hotpotqa/__init__.py`, `common.py` and `single.py`
as shared dependencies; multi-agent runs also record `agents/hotpotqa/multi.py`.
Only package modules are fingerprinted and supported. Both applications
compose plain `harness.Agent` instances using public admission/trace hooks, retaining
the previous tool-attempt budget rules, prompts, schemas and outputs. Imports and
tool factories do not load `.env` or create workspaces. CLI entrypoints explicitly
load `.env` and resolve candidate/workspace paths; library callers supply
`HotpotQAConfig` rather than global settings. General CLI compaction/limits,
policy/journal/timeouts do not configure these workers. Earlier fingerprints need
a fresh paired reference.

Phase 4 places preparation/loading/scoring/comparison/host runners under
`evaluation/hotpotqa/`. The supported preparation/evaluation/lesson commands are
`python3 -m evaluation.hotpotqa.prepare`, `python3 -m evaluation.hotpotqa.evaluate`
and `python3 -m evaluation.hotpotqa.run_question` with the same options. Default
dataset locations stay repository-relative, even from another working directory.
Scoring remains host-only. The unchanged strict parser now lives in
`agents/hotpotqa/prediction.py` and is re-exported by the scorer; workers have no
evaluation imports and run without host files. Shared fingerprints add the parser,
canonical scorer and package initialization; canonical host modules are recorded
as source provenance. Again, create a fresh paired reference after removal of the root adapters;
no old manifests, scores, prompts or artifact formats are rewritten.

## What the single agent does

One instance of the existing `Agent` loop handles research and answering, with fresh
history per question. Its enforced tool allowlist is:

- `search_documents`: existing keyword search, at most three 800-character windows,
  plus filename/title/sentence-count metadata for the reader.
- `read_document`: read 1–3 numbered sentences from a selected candidate; each
  sentence is capped at 1,200 characters, with an explicit truncation flag.
- `calculator`: the existing safe arithmetic interpreter.

No arbitrary filesystem, write, clock or HTTP tools are advertised **or executable**.
The host passes only one label-free question and its candidate texts to an isolated
worker process. Original answers, types, difficulty, supporting-fact labels, other
questions and earlier results never enter agent input. The evaluator alone reads
gold data after the prediction. Retrieved text remains untrusted evidence.

The agent returns only a JSON object with `answer` (a short string),
`supporting_facts` (exact article-title/zero-based-sentence pairs) and `abstain`
(boolean). Abstention uses an empty answer and evidence list. Markdown, prose,
malformed JSON, refusals and truncated final responses are not silently repaired
or sent through extra model calls. Invalid evidence is retained as a false positive,
not filtered out to improve scores.

### Default per-question budgets

- At most **12 logical model calls** (`--max-steps`).
- At most **12 tool executions**, including tool attempts that raise errors
  (`--max-tool-calls`). Over-budget/forbidden requests return tool errors, not actions.
- At most **2,048 output tokens per model call** (`--max-output-tokens`).
- **180 seconds wall time** for the worker (`--timeout`). A timeout kills the worker.

Compaction is disabled so hidden summarization calls do not change the model-call
budget. Provider HTTP retries are not separate logical calls and may affect actual
billing. Calls requesting truncated tool arguments are recorded but not executed.
Model sampling settings use provider defaults; identical inputs do not guarantee
identical model outputs. Keep effective model settings and **total** budgets
comparable when introducing multiple agents, rather than giving each agent a fresh
full budget. Safe calls do not require human approval; risky tools are unavailable.

## Saved artifacts

Each new output directory contains:

- `run.json`: selected IDs, dataset checksum, input fingerprint, code hashes,
  allowed tools, budgets, mode and experiment version.
- `inputs.json`: frozen **label-free** question/candidate inputs for the future
  multi-agent run. Its SHA-256 is the `input_fingerprint` in the run and summary.
- `q01/` … `q10/`: per-question `trace.jsonl`, `agent.log` and worker `result.json`
  when produced. Each result records the resolved provider/model, endpoint fingerprint,
  generation settings and prompt hash. Setup failures/timeouts may not produce every
  worker artifact.
- `results.jsonl`: incremental per-question predictions, statuses, metrics and scores.
- `predictions.json`: HotpotQA-compatible `answer` and `sp` dictionaries keyed by ID.
- `summary.json`: aggregate scores, score denominators, status counts and resource use.

`runs/` is ignored by Git. Temporary candidate/workspace directories are cleaned up;
traces and results are retained outside the agent workspace. No session is loaded
or saved. Provider error bodies and worker stderr are not persisted, since they can
contain credentials; `worker_error` indicates a startup/configuration/worker failure.
Ctrl+C saves a partial report and marks remaining examples `not_run` without calling
more models. A crash of the host itself may leave only the incremental artifacts.

## Metrics and interpretation

Live quality metrics follow the [HotpotQA evaluator](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py):

- `em`: normalized answer exact match; `f1`, `prec`, `recall`: answer-token overlap.
  Normalization lowercases, removes ASCII punctuation/articles and fixes whitespace.
  Incorrect `yes`, `no` or `noanswer` answers do not get partial token credit.
- `sp_*`: exact match and overlap of **sets of title/sentence pairs**.
- `joint_*`: combined answer and supporting-fact scores using HotpotQA's formula.

Scores are **fractions from 0 to 1**, macro-averaged across the selected examples.
Failures and abstentions stay in the denominator; invalid predictions receive zero
quality scores. Invalid original gold sentence labels exclude only that example's
supporting/joint metrics, with separate denominators; answer scoring remains valid.
The known development label error is not in the initial ten-example cohort.

Resource metrics include logical model calls (including failed calls), requested,
executed and rejected tool calls, tool errors, agent runtime and reported input/output
tokens. Missing/incomplete usage is `null`, not a fabricated zero. Timeout/worker
failure counts are also unknown. Runtime excludes worker setup; `wall_seconds`
includes dataset verification, setup and orchestration. **`cost_usd` is `null`**:
no pricing or billing data is assumed, and cached-token/provider pricing can differ.

`citation_validity` measures whether cited pairs exist in the supplied candidates.
`read_evidence_fraction` measures whether cited pairs appeared through `read_document`
(possibly as truncated excerpts). Neither proves the sentence entails the answer.
Inspect traces and gold evidence as well as scores. Ten deliberately balanced public
questions cannot establish general quality, and may have appeared in model training.

## Sequential multi-agent comparison

The multi-agent version uses **two separate conversations with the same model**:

1. A **researcher** investigates and produces a draft using the current tools and schema.
2. A **reviewer** receives the question, structured draft and bounded tool evidence,
   checks the entities/relationship and writes the only scored final prediction.

Both roles reuse the unchanged single-agent tools, label-free candidate inputs,
strict parser and host-only scorer. No new dependencies, external retrieval or gold
labels enter either role. This is sequential orchestration, not parallel agents.

### Shared budgets and handoff

The defaults remain **12 total model calls and 12 total tool executions per question**,
not 12 per role. The researcher can use at most half of each budget, rounded down;
the reviewer receives all unused calls/tools. At least two model calls are required.
The output-token limit applies per call; the same worker timeout covers both roles.
Compaction stays disabled. Rejected tool requests do not execute; failed executions
consume the budget. Handoff construction needs no extra model call, and its elapsed
time is included in the question's runtime. Equal caps do not imply equal actual
token usage, runtime or cost; inspect these separately.

The handoff is at most 16,000 characters. Whole JSON sentence-read results are
preferred over discovery snippets; overflowing evidence or drafts are omitted,
never cut into broken JSON. Invalid drafts are not forwarded as prose. The reviewer
is instructed to treat the entire handoff as untrusted data and independently read
cited sentences when possible. This is guidance, not proof of injection resistance
or mandatory verification. A malformed draft or exhausted researcher budget can
still be reviewed. Provider errors, interruption, refusal or truncated researcher
responses stop the worker. Reviewer failures never fall back to the draft answer.

### Run and compare

First run an **offline plumbing comparison**, with new output directories:

- `python3 -m evaluation.hotpotqa.evaluate --output runs/single-agent-hardened-smoke`
- `python3 -m evaluation.hotpotqa.evaluate --architecture multi-agent --reference runs/single-agent-hardened-smoke --output runs/multi-agent-hardened-smoke`

Both smoke roles deliberately abstain. Quality scores and quality deltas remain
`null`; successful smoke runs do not measure model accuracy.

First create the **fresh hardened live single-agent baseline** using the command
above. Keep the same provider/model/endpoint settings and explicitly opt into the
ten-question paid multi-agent run:

`python3 -m evaluation.hotpotqa.evaluate --live --architecture multi-agent --reference runs/single-agent-hardened-v1 --output runs/multi-agent-hardened-v1`

The reference must cover the same cohort: `--limit 1` requires a one-question
single-agent reference, not the saved ten-question baseline. Non-default budgets
must match too. Before starting any workers, `--reference` checks frozen input
bytes, per-question fingerprints/order, dataset snapshot, mode, budgets, allowlist,
compaction and hashes of the harness, provider clients, common tools/utilities,
single-agent worker and scorer. Any missing or different shared hash is rejected.
The host runner hash is recorded
but may differ because it now supports both architectures. Reference artifacts
are read-only; a setup mismatch fails without creating the new output directory.

Afterward, `comparison.json` reports answer/evidence/joint score deltas, paired
wins/ties/losses, status counts, score denominators and resource totals/deltas.
Recorded provider/model, endpoint fingerprint and generation settings are compared
**after** the run; all questions in both runs must have the same complete metadata.
Different, legacy or missing settings set `recorded_settings_match` and `comparable`
to `false`. A matching result has `comparison_scope: recorded-settings-only` and
explicit limitations: it does not verify gateway routing, provider-side model
revisions, sampling defaults or model-training contamination. Keep those conditions
stable yourself. Failures remain in quality denominators; unknown usage remains
`null`, and actual billing cost is still unknown.

Multi-agent results also retain `roles` with draft/final predictions, role status,
allocated budgets and measured calls/tools/tokens; traces identify `researcher`,
`reviewer` and the handoff. `read_evidence_fraction` covers sentence reads by either
role; `reviewer_read_evidence_fraction` covers only the reviewer's independent reads.
Neither metric proves entailment. `summary.json` adds the total handoff count.

This ten-question development comparison is descriptive, not a causal finding or
general accuracy estimate. Role prompts and the handoff policy are part of this
architecture variant. Avoid repeatedly tuning on these ten questions and claiming
an untouched test result; use a fresh larger cohort and repeated runs before
drawing broad conclusions.

Run offline regression tests with
`python3 -m unittest -v tests.test_evaluate_hotpotqa tests.test_hotpotqa_multi_agent`.