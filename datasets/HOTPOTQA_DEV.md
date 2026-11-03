# HotpotQA: full development split

The harness supports all **7,405 distractor / validation questions**, in their
original order: **5,918 bridge** and **1,487 comparison** questions. No training,
Wikipedia dump, embedding service or Python package installation is needed.

## Prepare and run

Download and validate the full split once:
`python3 -m evaluation.hotpotqa.prepare --full-dev`.
This creates `datasets/hotpotqa-dev/`, which is ignored by Git. Preparation never
calls an LLM and refuses to overwrite an existing directory.

Run question 1 through offline keyword retrieval:
`python3 -m evaluation.hotpotqa.run_question --question 1`.
Try the last question with `python3 -m evaluation.hotpotqa.run_question --question 7405`.
Use `--id` instead of `--question` to select an original ID from `questions.json`.

To use your configured LLM for **one** question, explicitly opt in:
`python3 -m evaluation.hotpotqa.run_question --question 1 --live`.
Provider configuration still comes from the environment and your current directory's
`.env`; the runner does not change that file. Live runs incur normal API costs and
may request the existing human approval for risky tools. No bulk evaluation runs
are started automatically.

Each run creates a fresh temporary knowledge directory containing **only that
question's candidates**, plus a fresh writable workspace and conversation history.
Environment overrides prevent `.env` from redirecting those directories to gold
labels or previous workspaces. The child process receives only the question and
candidate paragraphs—not answers, labels or other questions. Temporary directories,
including any approved workspace writes, are removed when the run ends. The harness
still appends its trace to `trace.jsonl` in your current directory; it saves no session.

## Files and size

- `source.json`: the unmodified downloaded JSON, including original gold labels.
- `questions.json`: label-free IDs, question text and candidate filenames.
- `answers.json`: answers, types, difficulty and supporting-fact labels; evaluator only.
- `manifest.json`: source URL/checksum, license, counts and original label issues.

The pinned JSON mirror is **61,065,698 bytes (58.2 MiB)**, larger than the approximate
44 MB listed for the original download. The prepared directory is about **64.2 MiB**.
The runner rereads the JSON per invocation, then renders at most ten small files;
it does not create 73,700 files or search the complete split for every question.
There is deliberately no shared `knowledge/` directory for the development split.
Do not point `HARNESS_KNOWLEDGE` or `HARNESS_WORKSPACE` at this dataset directory.

Keep labels and source snapshots outside any agent-accessible workspace, including
workspaces exposed by custom tools. Separation is a tool-access boundary in this
educational harness, not protection against arbitrary filesystem tools you add.

## Original data quirks—not silently repaired

- **7,345** questions have ten candidates; **60** have between two and nine.
  All **73,700 candidate occurrences** are preserved, with no padding or dropped questions.
- Question `5ae61bfd5542992663a4f261` includes supporting-fact sentence **902** for
  `Jimmy Butler (basketball)`, whose supplied paragraph has only five sentences.
  That original label is retained exactly and flagged as `sentence_out_of_range`
  in the manifest. Review it before interpreting supporting-fact scores.
- Every record is checked before preparation writes any output: unique ID,
  nonempty question, supported type, candidate structure, valid paragraph titles,
  bounded document sizes and supporting-fact references. Out-of-range nonnegative
  sentence labels are reported, not treated as valid sentence references.

## Provenance and reproducibility

Default downloads use a public JSON mirror pinned to Hugging Face commit
`7e54db4656209750ff487f6fdf8e39a66dba136b`:
[namlh2004/hotpotqa](https://huggingface.co/datasets/namlh2004/hotpotqa/tree/7e54db4656209750ff487f6fdf8e39a66dba136b).
The download must match SHA-256
`e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8` before use.
This pins the mirror artifact; it is not a claim of independent publisher authentication.

For an offline rebuild into a new directory:
`python3 -m evaluation.hotpotqa.prepare --full-dev --input datasets/hotpotqa-dev/source.json --output datasets/hotpotqa-dev-rebuilt`.
Then use `python3 -m evaluation.hotpotqa.run_question --dataset datasets/hotpotqa-dev-rebuilt --question 1`.
Local JSON inputs must contain all 7,405 records. Their own checksum and whether
they match the pinned mirror are recorded honestly; local files are not silently
attributed to the default download URL. Each run checks the saved snapshot checksum.
Custom output directories are not automatically ignored by Git.

Offline tests use small synthetic fixtures. To additionally validate the installed
snapshot and run first/last/short-candidate/flagged-label examples through the CLI:
`HARNESS_TEST_HOTPOTQA_DEV=1 python3 -m unittest -v tests.test_hotpotqa_dev`.
This performs no download or live-model calls.

## Educational limitations

The bundled [ten-question demo](hotpotqa/README.md) is unchanged. Unlike its shared
100-document corpus, full-development runs retain each question's original candidate
set. Keyword retrieval is still limited to three 800-character windows per call;
answering may require multiple searches. The offline demo quotes passages and does
**not** produce or score answers. A live answer is not an official benchmark score:
use the separate [single-agent baseline](HOTPOTQA_BASELINE.md) for measured runs of
the original ten IDs, with answer/supporting-fact/joint scoring and frozen inputs
for the paired multi-agent comparison. The `evaluation.hotpotqa.run_question` demo
does not score answers; orchestration is provided by the evaluation runner. Public
benchmarks may have been in model training; inspect evidence, not just correct answers.

## Attribution and license

**HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering**,
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen,
Ruslan Salakhutdinov and Christopher D. Manning, EMNLP 2018.
[Homepage](https://hotpotqa.github.io/) · [Paper](https://arxiv.org/abs/1809.09600) ·
[Original repository and license](https://github.com/hotpotqa/hotpot).

Data and adapted paragraphs are **[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)**.
Wikipedia contributors are identified by article titles in the source and gold files.
Changes: separated questions/labels, reported an existing label issue, and rendered
selected candidates with titles and zero-based sentence numbers. Retain this
attribution/license when redistributing data or adaptations. The harness code's
license is unchanged.