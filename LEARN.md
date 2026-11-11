# Learn agent development with Little Harness

**Goal:** learn agent development from first principles, then apply explicit
execution controls. Build, inspect and test a loop—not merely run one.
Start here before reading the runtime specifications.

## Before you start

Know Python functions, imports, loops, lists, dictionaries and basic JSON.
No machine-learning background is required.

1. Download the repository ZIP from its hosting page and extract it, or clone it
   using Git. Open the extracted/cloned folder in your editor.
2. Check that this folder contains `README.md`, `LEARN.md`, `harness/` and `lessons/`.
   This is the **repository root**; do not open only the `lessons/` subfolder.
3. Open a terminal in that folder. In VS Code, use **Terminal → New Terminal**.
4. Run the version check below. Use Python **3.11+**. On Windows, replace `python3`
   with `python` or `py -3` if that is your Python launcher.

````sh
python3 --version
````

There are no third-party dependencies to install, API keys to configure or datasets
to download. You do **not** need to copy `.env.example` or run `pip install`.

The labs never load `.env`, contact a provider or write a session/workspace.
Lab 0 uses plain Python without importing the harness. In Labs 1–4, runtime traces
go into a fresh temporary directory and are deleted afterward.
`--messages` prints only the synthetic conversation; `--events` in Labs 1–4 prints
ordered metadata event names. Neither is a view into a model's private reasoning.

## Your first success: watch a completed run

Before writing code, see the whole interaction working:

````sh
python3 -m lessons.preview
````

Expect **status `completed`, answer `3`, two model calls and one tool execution**,
followed by `Preview complete.` You have a working setup. The preview uses an
existing demonstration; it does not load your exercises or reference solutions.
Add `--messages --events` if you want to inspect the conversation and event names.

Next, build that loop yourself in Lab 0. You will use the harness API only after
understanding what the loop needs to do. If the preview does not work, start with
[troubleshooting](#troubleshooting) rather than configuring a provider.

## What do we mean by an agent?

- A **function** maps inputs to outputs: count the words in a string.
- A **fixed workflow** chooses the steps in application code: count, then format.
- A **model-driven agent loop** lets a model propose the next tool call or a final
  answer. Trusted application code decides what may actually execute.
- The **harness** runs that loop, maintains conversation history and enforces the
  application's controls. A prompt alone is not an authorization mechanism.

These labs use **scripted models**: a simpler one-block interface in Lab 0, then
the same interface used by real providers in Labs 1–4. Each script selects responses
using Python conditionals, not learned reasoning. This makes
protocol behavior reproducible and free; it does **not** establish model accuracy,
planning ability, prompt effectiveness or prompt-injection resistance.

For this tiny counting task, calling a function directly is simpler than using an
agent. We use it to make the execution mechanics visible—not to justify an LLM.

## The first-principles learning path

Allow roughly 30–45 minutes for Lab 0, 20–30 minutes for Labs 1–3 and 45–60 minutes
for the application capstone.

| Lab | Learning outcome | Read first |
| --- | --- | --- |
| 0. From functions to agents | Build a loop yourself; discover why validation and limits exist before using `Agent` | [Walkthrough](lessons/00-from-functions-to-agents.md) |
| 1. Follow the loop | Explain who requests, validates and executes a tool; inspect the conversation | [Walkthrough](lessons/01-follow-the-loop.md) |
| 2. Build a tool | Define a schema-backed function and distinguish a valid call from a correct answer | [Walkthrough](lessons/02-build-a-tool.md) |
| 3. Diagnose a failure | Repair invalid arguments, verify execution never happened and recognize a budget stop | [Walkthrough](lessons/03-diagnose-a-failure.md) |
| 4. Build an application | Own two read-only tools, data snapshots, agent wiring, budgets and a new test | [Capstone](lessons/04-build-an-application.md) |

After the preview, begin with [Lab 0](lessons/00-from-functions-to-agents.md). Run
its starter to see the direct function and fixed workflow:

````sh
python3 -m lessons.lab00_basics
````

The CLI labels this **exercise mode**. An untouched loop reports `not_implemented`:
that means the exercise is ready to edit, not that your setup is broken.
The walkthrough divides the loop into five cumulative checkpoints:

| Checkpoint | What you add |
| --- | --- |
| 1 | Finish with a text response |
| 2 | Execute one valid tool request |
| 3 | Return the tool result to the model |
| 4 | Reject invalid requests and allow a repair |
| 5 | Enforce call limits, including the final-answer call |

Start with `python3 -m lessons.lab00_basics --check --stage 1`. Until you implement
that checkpoint, its failed check is expected. Advance using the next command
printed after each success; `--check` without `--stage` assesses the whole loop.
Assessment stops at the first unfinished checkpoint so you can focus on one thing.

After building the loop, compare it with the supported runtime in Lab 1:

````sh
python3 -m lessons.lab01_loop --messages --events
````

Use the [annotated execution trace](lessons/annotated-trace.md) to connect its output
to the code. Then work through the labs in order.

## How exercises and feedback work

1. Read the lab and predict its output before running it.
2. Run the starter, then edit only the relevant function in
   [lessons/exercises.py](lessons/exercises.py) for Labs 0–3. The capstone has its
   own application file, [lessons/capstone_exercises.py](lessons/capstone_exercises.py).
3. Run that lab with `--check`. Read the descriptions of any failing checks.
   For Lab 0, start with `--check --stage 1`; later stages also check earlier work.
4. Use the hints before consulting [reference solutions](lessons/solutions.py)
   or the separate [capstone reference](lessons/capstone_solution.py).
5. Explain the result in your own words; passing checks alone is not the goal.

````sh
python3 -m lessons.lab00_basics --check
python3 -m lessons.lab01_loop --check
python3 -m lessons.lab02_tools --check
python3 -m lessons.lab03_failures --check
python3 -m lessons.lab04_capstone --check
````

**Untouched exercises intentionally fail these checks** (exit status 1). Lab 2
initially produces a valid but incorrect answer; Lab 3 initially stops at a call
limit. Lab 0 initially reports `not_implemented` until you build the loop. The
capstone initially advertises no tools and cannot produce the requested quote.
Ordinary demo commands exit 0 when they run, even for those expected states;
use `--check` for exercise assessment. Invalid command options exit nonzero.
The deliberate caught error in Lab 0 is shown only with `--scenario invalid`,
not during the starter or completed preview. With `--check --messages`, Lab 0
prints the checkpoint conversations, which use their own synthetic test inputs.

`--solution` selects reference functions without modifying your work:

````sh
python3 -m lessons.lab00_basics --solution --check
python3 -m lessons.lab01_loop --solution --check
python3 -m lessons.lab02_tools --solution --check
python3 -m lessons.lab03_failures --solution --check
python3 -m lessons.lab04_capstone --solution --check
````

All five should pass. These checks assess the exercises, not a live LLM.
Finish the capstone by adding a test you own and explaining the observed controls;
passing its supplied checks is not the whole assignment.

## Troubleshooting

| What you see | What to do |
| --- | --- |
| `python3` not found | Install Python 3.11+ or use an existing `python` / Windows `py -3` launcher. Check its version before continuing. |
| `needs Python 3.11 or newer` | Your `python3` is older (macOS's built-in one is 3.9). Run the same command with `python3.11`, `python3.12` etc., or with `uv run python` if you use uv. |
| `No module named lessons` | Open a terminal in the folder containing `README.md` and `lessons/`, not inside `lessons/`. Run `python3 -m lessons.preview`, not a `.py` file directly. |
| `SyntaxError` or `IndentationError` after editing | Check the edited function's syntax and indentation. Save the file before rerunning; a syntax error must be fixed before the checker can load it. |
| Preview works, but a checkpoint fails | Your setup is working. Read the named checkpoint and hint; edit only the relevant function. Do not add credentials or change `.env`. |
| Lab 0 says `not_implemented` | This is the starter, not a runtime failure. Begin with `--check --stage 1`. |
| Later Lab 0 checkpoints are not shown | Assessment stops at the first unfinished checkpoint. Fix it before moving on. |
| `--stage requires --check` | Use `--check --stage 1` together. Stages select assessment, not a demo scenario. |
| Capstone says an application contract is unfinished | Edit `lessons/capstone_exercises.py`, not the earlier exercises or the runtime. Read the printed hint; assessment stops at the first unfinished contract. |
| You want to inspect a scenario or a different limit | Run `--scenario` / `--max-model-calls` separately from `--check`; assessment uses fixed test inputs. |
| Your loop appears stuck | Press Ctrl-C. Check that it either returns or requests another model response within a finite call loop. The checker is not a sandbox. |
| You want to undo an experiment | Save a separate copy of your work first, then use editor Undo/local history. `--solution` runs the reference without overwriting your work; avoid resetting the whole repository. |

## Vocabulary to keep straight

| Term | Meaning here |
| --- | --- |
| Model call | One request for the next response, including a final-answer request |
| Tool schema | Tool name, description and supported argument shape exposed to the model |
| Tool attempt | A requested call admitted by the budget, even if validation rejects it |
| Tool execution | The registered function is actually invoked |
| Tool result | Output or error paired with the original tool-call ID |
| Completed run | The loop ended normally—not proof the answer is correct |
| Conversation | Recorded messages; not durable knowledge or proof an external action completed |
| Trace / event | Diagnostic record / observation callback; not permission to execute a tool |

## After this milestone

These next topics already have runtime support or examples, but are **not yet
guided labs**:

1. Adapt your [capstone application](lessons/04-build-an-application.md) to another
   domain; the [inventory example](examples/inventory_agent.py) is another reference.
2. Go beyond the capstone's budgets and read-only policy with
   [approval and execution controls](docs/REFERENCE.md#execution-controls) and
   [argument policies](docs/TOOL_POLICIES.md). A worker timeout is not a security sandbox.
3. Explore [document retrieval](docs/REFERENCE.md#teaching-tools-and-retrieval): require evidence,
   distinguish document data from instructions, and admit insufficient information.
4. Try a [real provider](README.md#run-with-a-real-model) only with explicit cost
   consent and bounded calls. Never share keys, private prompts or sensitive traces.
5. Learn [evaluation](datasets/HOTPOTQA_BASELINE.md): keep expected answers outside
   agent input, reserve untouched cases and compare quality against calls/tokens/time.
   The [support evaluation](docs/SUPPORT_EVALUATION.md) also scores whole offline
   conversations, where a follow-up turn corrects or adds to an earlier one.
6. Compare single and multiple agents only after establishing a baseline; more
   agents do not automatically mean better results.
7. Handle [long conversations](docs/REFERENCE.md#traces-sessions-and-compaction):
   `compact_at_tokens` keeps each model request small while the saved history stays
   complete, and `get_steering` lets a person add guidance between tool batches.
   A summary can lose details, and steering text is guidance, not permission.

## For instructors and study groups

- Start offline; no student should need an API subscription to complete the labs.
- Run the preview first, then ask learners to show each Lab 0 checkpoint before
  moving on. Record where they need help; passing tests is not a usability study.
- Ask learners to predict the next message before showing it. Do not require hidden
  chain-of-thought; explain observable requests, results and authorization instead.
- Have learners demonstrate one failing case, their fix and one additional test.
- Assess whether they can distinguish attempted/executed calls and completion/correctness.
- Keep personal data and credentials out of submissions; use the supplied synthetic text.
- Obtain permission before any optional paid-model exercise. Scripted success is
  not evidence that the same behavior will occur with a live model.

## Tell us how it went

Finished a lab, or got stuck? Share it with the
[lab feedback form](https://github.com/alxmancilla/little_harness/issues/new?template=lab-feedback.yml).
It takes about two minutes, and knowing where people stop is what improves the labs.
