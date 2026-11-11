# Live baselines

First recorded live runs of every evaluation suite, on 2026-11-10 at commit
`a992257`, through the AI gateway (`HARNESS_PROVIDER=gateway`) with default budgets.
Artifacts are in `runs/baseline-<model>-<suite>/` (gitignored, local only).

**Read these numbers with care.** Each suite is tiny (17 synthetic support cases,
5 two-turn conversations, 10 HotpotQA questions) and each was run once. One
question or case moves a rate by 6–20 points, and provider-side sampling or model
revisions can change results between runs. They show that the pipelines work
end to end with real models and where behaviour differs; they do not rank models
or architectures.

## Results

| Suite | claude-sonnet-5-5 | gpt-5.6-sol |
| --- | --- | --- |
| Support: valid output | 17/17 | 17/17 |
| Support: passes draft expectations | 16/17 | 13/17 |
| Multi-turn: scenarios passing every turn | 2/5 | 1/5 |
| HotpotQA single-agent: answer EM / F1 | 0.70 / 0.84 | 0.70 / 0.79 |
| HotpotQA single-agent: joint EM / F1 | 0.50 / 0.76 | 0.60 / 0.78 |
| HotpotQA multi-agent: answer EM / F1 | 0.70 / 0.83 | 0.70 / 0.79 |
| HotpotQA multi-agent: joint EM / F1 | 0.50 / 0.79 | 0.60 / 0.78 |

Support answer quality is unscored by design; each run's `review.json` awaits
human review. HotpotQA comparisons were verified comparable (same inputs, budgets,
tools, scorer and recorded model settings) with no warnings.

## Cost and effort

| Run | Model calls (Claude / GPT) | Input tokens (Claude / GPT) | Output tokens (Claude / GPT) |
| --- | --- | --- | --- |
| Support, 17 cases | 34 / 34 | 33,753 / 13,202 | 4,467 / 3,458 |
| HotpotQA single-agent | 25 / 38 | 63,906 / 72,747 | 1,890 / 2,440 |
| HotpotQA multi-agent | 42 / 58 | 135,925 / 136,050 | 3,860 / 3,919 |

The evaluators record tokens, not prices (`cost_usd` is `null`). All runs together,
including the multi-turn suite and earlier one-item smoke runs, used well under one
million tokens.

## Findings

- **Multi-agent did not improve answers.** On HotpotQA the researcher/reviewer
  workflow matched single-agent answer scores for both models while using about
  2× (Claude) and 1.9× (GPT) the input tokens. For Claude it improved
  supporting-fact selection (SP EM 0.6 → 0.7, SP precision 0.97 → 1.0); for GPT
  answers were identical. Ten questions cannot show whether that is real.
- **GPT escalates more.** Three of its four support failures (c01, c02, c17) and all
  four of its multi-turn failures are `escalation_matches_draft`: it set
  `needs_human` where the draft expected a supported answer. In c01 it explained that
  the note is labelled fictional and therefore not real advice. Rewording that label
  in `examples/support_knowledge/` may change this; it is a prompt/data question,
  not a harness fault.
- **Both models fail c14 on citations** (`required_sources_cited`), while escalating.
  Review whether the case expectation or the answers are right.
- **Claude's multi-turn failures come from the provenance rule.** In m01, m04 and
  m05 the second turn's output was rejected (`invalid_output`): the model searched
  notes again but reused product data from the first turn instead of calling
  `lookup_product` again. The validator deliberately resets provenance on each new
  user turn ("old conversations are not fresh evidence"), so a supported answer
  needs both lookups in the same turn. Decide whether that rule is intended for
  follow-ups; if so, the system prompt could say so explicitly.

## Reproduce

Support runners never load `.env`; export provider settings in the shell first.
HotpotQA needs the prepared [development split](../datasets/HOTPOTQA_DEV.md).
Each command writes a new directory and never overwrites results. These commands
make paid model calls.

````sh
export HARNESS_PROVIDER=gateway HARNESS_API_KEY_ENV=AI_GATEWAY_API_KEY AI_GATEWAY_API_KEY=...
export HARNESS_MODEL=claude-sonnet-5-5
uv run python -m evaluation.support.evaluate --live --output runs/baseline-$HARNESS_MODEL-support
uv run python -m evaluation.support.multiturn --live --output runs/baseline-$HARNESS_MODEL-multiturn
uv run python -m evaluation.hotpotqa.evaluate --live --output runs/baseline-$HARNESS_MODEL-hotpot-single
uv run python -m evaluation.hotpotqa.evaluate --live --architecture multi-agent \
  --reference runs/baseline-$HARNESS_MODEL-hotpot-single --output runs/baseline-$HARNESS_MODEL-hotpot-multi
````

Later code changes to fingerprinted runtime files make these HotpotQA runs unusable
as `--reference`; record a fresh pair instead of comparing against them.
