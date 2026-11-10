# Evaluate the support assistant

This small workflow evaluates [the support application](../examples/support_agent.py),
not the generic runtime. It uses only the standard library and never loads `.env`.
There is no LLM judge, hidden repair call, automatic retry, message sending or draft saving.

## Start offline

````sh
python3 -m evaluation.support.evaluate --output runs/support-smoke-01
````

Choose a **new** output directory each time. Existing directories and symlinks are
rejected before model setup. Never delete an earlier run just to reuse its name.
Interrupted/partial output directories also remain intact and must not be reused.

The bundled `OfflineModel` is a protocol demonstration, not a reasoning model.
Currently all 17 cases produce valid application output, but only 11 pass all
draft expectation checks. Six cases expose its inability to distinguish relevant
evidence or infer policy escalation from keyword matches. These are visible findings,
not reasons to weaken the cases or pretend the script measures model quality.

Offline exit code 0 means all cases produced valid application output, **not** that
the draft expectations or answer quality passed. Code 1 indicates a setup/execution
or output-contract failure; interruption returns 130 when handled by the runner.

## Review the cases before treating them as a benchmark

[cases.json](../evaluation/support/cases.json) contains seventeen **AI-authored synthetic
drafts, not human-approved ground truth**:

| Coverage | Cases | Intended behavior |
| --- | --- | --- |
| Supported questions | c01–c04 | Give grounded connectivity or warranty information |
| Unknown product | c05 | Escalate without invented product facts |
| Missing evidence | c06–c07 | Do not confuse keyword overlap with an answer |
| Policy boundary | c08–c09 | Escalate warranty approval or failed reconnection |
| Misleading instructions | c10–c12 | Do not claim unperformed actions, invent reset steps or bypass unknown SKUs |
| Adversarial tool use | c13–c15 | Ignore instructions inside a retrieved note, do not claim a missing tool ran, never cite an unretrieved source |
| Memory poisoning | c16–c17 | Treat recalled memories as untrusted; do not follow injected refund, warranty or reset instructions |

Each case separates `input`, `expected` and reviewer guidance. Only `sku` and
`question` enter the agent's task. Expectations and guidance never enter its prompt
or tools. Each case gets a fresh agent/history and snapshots of the same approved
notes and catalog. The application itself is unchanged. A case may also list
`extra_documents`: untrusted notes, such as c13's forum post with hidden instructions,
that `search_support` can return for that case only. They are frozen with the cases
in `cases.json`, never in `evidence.json`, and cannot be required citations.
The offline script passes c13 and c15 only because it always cites the top approved
match; that says nothing about how a real model handles the injection.
Cases c16–c17 provide `extra_memories` as case-specific, read-only ephemeral memory.
The script calls `recall` and does not use its contents in the answer; this does not
demonstrate that a real model would resist the injected instructions.

Before using these cases to make a decision, inspect every question, draft
escalation expectation, required citation and review guide against the fictional
notes/catalog. Correct ambiguous expectations in the source dataset, document your
rationale and run again into a new directory. Do not edit an earlier run's snapshots
or tune expectations just to raise its pass count. These seventeen cases do not
represent a production distribution or demonstrate general prompt-injection resistance.

## Optional live evaluation

Configure provider credentials/settings in your shell; do not paste them into
commands or this dataset. No credentials, endpoints or raw model configuration are
written by this evaluator. Start with one case to inspect cost and behavior:

````sh
python3 -m evaluation.support.evaluate --live --limit 1 --output runs/support-live-01 \
  --max-model-calls 4 --max-tool-calls 4 --max-seconds 30
````

`--live` explicitly permits paid calls using the same application factory and data.
Omit `--limit` to run all seventeen cases. Defaults are six logical model calls, six
tool attempts and 30 seconds **per case**, not per evaluation. Model calls include
compaction. Limits are cooperative: a running model/approval operation may outlast
the deadline, and provider transport retries are internal to a logical call. These
limits are not a hard timeout or billing ceiling. Actual billing cost is unknown.

Live exit code 0 means all automated draft checks passed; 1 means at least one did
not, or setup failed. Neither is a quality verdict. A model error or interruption
halts further cases, preserves earlier results and marks remaining cases `not_run`.
There is no automatic rerun. Other contract failures are reported and the evaluation
continues. Examine the report before choosing whether to run anything again.

## Understand the artifacts

| Artifact | Purpose |
| --- | --- |
| `run.json` | Mode, selected case IDs, per-case limits and dataset/evidence/source hashes |
| `cases.json` | Full frozen draft dataset, including host-only expectations |
| `inputs.json` | Selected label-free inputs actually shown to agents |
| `evidence.json` | Frozen approved support notes and product catalog |
| `results.jsonl` | Per-case execution status, raw answer, validated recommendation, checks, tool requests, timing and usage |
| `cXX/trace.jsonl` | Metadata-only execution trace for an attempted case |
| `summary.json`, `report.md` | Concise automated findings and resource totals |
| `review.json` | Human-review worksheet; every judgment starts `null` |

Per-case model-configuration hashes help detect changed recorded settings without
storing raw provider/model/endpoint values. They do not prove provider reproducibility
or automatic comparability. Compare inputs, evidence, limits and source hashes too.

The single-turn automated checks test application output validity (including actual
citation provenance), the draft `needs_human` expectation, and presence of required
citations. They also check whether each supplied memory was returned by a successful
`recall`; completed cases without extra memories satisfy that check automatically.
Memory exposure establishes only that the model received the data, not that it
followed or resisted it. Invalid/stopped/not-run cases stay in the denominator; they cannot pass.
A valid source name does **not** establish that the recommendation correctly uses
the source. A structurally valid but factually wrong answer can pass every automated
check. There is deliberately no automatic answer-quality score, even for live runs.

Unknown token usage stays `null`, never zero. Resource totals become unknown if
an attempted case lacks measurements; cases never started consume zero known calls.
Elapsed time summarizes agent runs, not a hard wall-clock budget for the batch.

**Artifacts still contain content:** metadata-only traces do not redact raw answers,
recommendations, questions or evidence stored in the other files. Keep cases
fictional/non-sensitive and control output-directory permissions and retention.

## Evaluate whole conversations

The single-turn cases give each question a fresh agent. Real users follow up, correct
themselves and add constraints, so a second suite scores whole conversations:

````sh
python3 -m evaluation.support.multiturn --output runs/support-multiturn-01
````

[scenarios.json](../evaluation/support/scenarios.json) holds five **unreviewed synthetic
draft** scenarios of two turns each: a follow-up on a new topic (m01), a corrected
product (m02), an unknown product that is then corrected (m03), a constraint showing
the documented steps already failed (m04), and a follow-up that falsely claims an
earlier approval (m05). Each scenario keeps **one agent and its history** across
turns; only each turn's `sku`/`question` enters that history. Every turn is scored
with the same three checks as a single case, so citations must come from evidence
retrieved in that turn, not an earlier one. A scenario passes only if every turn
passes and the runtime kept every earlier task in history.

Offline, three of five scenarios pass. m04 and m05 fail turn 2 for the same reason
c09 and c10 fail: the script cannot infer escalation from the conversation. A turn
that does not complete (for example a budget stop) ends its scenario, and later turns
are `not_run`; a model error or interruption also stops the remaining scenarios.
Limits apply **per turn**. `--live` works as above and may incur costs; there is no
`--limit`, so all ten turns run. Exit codes match the single-turn runner. The
artifacts mirror it too, with `scenarios.json`, per-turn rows in `results.jsonl`
and a `review.json` that adds `context_carried_correctly` to each scenario.

These checks sit at three levels. **Runtime correctness** (history kept, budgets and
stops honored) is checked mechanically and also covered by the harness tests.
**Agent behavior** (fresh retrieval, escalation, valid output) is checked against
draft expectations. **Application outcome** (is the advice right and safe for the
whole conversation?) is left to human review.

## Complete human review

Read each result beside its input and frozen evidence. In a **copy** of `review.json`,
record your reviewer identity, approve/correct draft expectations and judge factual
correctness, citation support, appropriate escalation and action safety. Use
`true`/`false` with notes; leave `null` when not reviewed or no judgment is possible.
In particular, reject fabricated refunds/messages, reset instructions and unsupported
product facts. This evaluator does not ingest that worksheet or manufacture a score
from it. `answer_quality` remains `null` and the generated pending-review count is
only the initial state, not a live tracker of your later review work.

Run `python3 -m unittest -v tests.test_support_evaluation` for the mechanical
regressions of both suites. They include a deliberately incorrect answer that passes structural
checks, demonstrating why human review is necessary.