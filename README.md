# Little Harness

**A small Python runtime for agents, with explicit controls.**

Includes teaching applications and reproducible evaluation tools, built entirely
with the **Python 3 standard library**. No third-party packages, agent framework,
database service or API key are needed for the offline examples.

The core loop is simple: ask a model, validate its tool requests, execute permitted
tools, return their results and repeat. Applications supply the prompt and tools;
the runtime handles execution, conversation state and optional controls.

## Learn agent development

Start with [the learning path](LEARN.md): five guided offline labs begin with
functions and a loop you build yourself, then introduce the harness, building a
tool and diagnosing a failed call. Finish with a two-tool application capstone
where you own the registry, data, policies and budgets. Each lab includes an
exercise, checks, hints and a separate reference solution.

Use **Python 3.9+**. Download or clone the repository, open its folder in your
editor and open a terminal there. Follow the [setup checklist](LEARN.md#before-you-start)
if you are new to running Python projects. Nothing needs to be installed with `pip`.

**First, watch a completed preview—no editing or reference solution required:**

````sh
python3 -m lessons.preview
````

Expect answer `3`, two model calls and one tool execution. Then open
[Lab 0](lessons/00-from-functions-to-agents.md) and build the loop in five small
checkpoints. Edit only `run_loop` in [lessons/exercises.py](lessons/exercises.py).
**The first check intentionally fails until you implement that checkpoint:**

````sh
python3 -m lessons.lab00_basics --check --stage 1
````

Feedback stops at the first unfinished checkpoint and gives a hint and next command.
Use the hints before selecting `--solution`; see [troubleshooting](LEARN.md#troubleshooting)
if you get stuck.

The labs ignore `.env` and provider settings; no configuration, API key, paid calls
or dataset downloads are needed. Lab 0 uses plain Python without importing the
harness; its loop is teaching code, not a second supported runtime. Labs 1–4 use
`Agent` with temporary traces that are deleted afterward. An
[annotated execution trace](lessons/annotated-trace.md) connects the messages to the code.

After Lab 3, build the [independent application capstone](lessons/04-build-an-application.md)
in `lessons/capstone_exercises.py`. Check it with `python3 -m lessons.lab04_capstone --check`,
then add your own test. Earlier exercise files are separate and stay untouched.

## Quick start

Use **Python 3.9+** and run commands from the repository root. A virtual environment
is optional; there is nothing to install with `pip`.

````sh
python3 -m agents.teaching.cli mock
python3 -m agents.teaching.cli rag-mock
python3 -m examples.inventory_agent --sku WIDGET
````

- **`mock`** calculates a value, timestamps it and writes a report in `workspace/`.
  Risky tools are explicitly auto-approved in this scripted demo.
- **`rag-mock`** retrieves and quotes passages from the fictional `knowledge/`
  documents, with source citations. It needs no dataset download.
- **Inventory lookup** demonstrates a separate application with its own read-only
  tool and deterministic model. It prints a structured result.

These commands make no model/network calls. Teaching demos append to `trace.jsonl`;
inventory uses `inventory-trace.jsonl`. Scripted success demonstrates mechanics,
**not model accuracy**. Teaching configuration can be overridden by your existing
environment or `.env`.

## Project structure

| Location | Responsibility |
| --- | --- |
| [harness/](harness/) | Reusable loop, tool registry, provider clients, budgets, events, policies, workers and SQLite journal |
| [agents/teaching/](agents/teaching/) | Teaching tools, explicit configuration, factories, scripted demos and CLI |
| [agents/hotpotqa/](agents/hotpotqa/) | Restricted question-answering tools, single-agent and researcher/reviewer workflows, shared worker helpers |
| [evaluation/hotpotqa/](evaluation/hotpotqa/) | Dataset preparation, per-question execution, host-only scoring and comparisons |
| [examples/inventory_agent.py](examples/inventory_agent.py) | Independent application using the public runtime API |
| [lessons/](lessons/) | Guided offline labs, learner exercises, checks and reference solutions; start at [LEARN.md](LEARN.md) |
| [tests/](tests/) | Offline regressions; reusable fixtures in `tests/helpers.py` |
| [datasets/](datasets/) | Dataset guides and the attributed ten-question HotpotQA demo |
| [docs/](docs/) | Runtime contracts, architecture diagrams and migration history |

Little Harness uses the Python package name `harness`. Only the package API and
module commands are supported. Import from `harness`;
there are no root compatibility scripts, aliases or global tool registry.
The complete `harness/` directory is independently copyable. An optional
[standard-library wheel builder](docs/BUILD_AN_AGENT.md#reuse-without-a-framework)
packages just the runtime for local installation. Importing it does not load `.env`,
create a workspace or register application tools.

## Run with a real model

Configure a [provider](docs/REFERENCE.md#providers) and its credential locally, then run a task or
start an interactive conversation. **These commands can incur API costs.**

````sh
python3 -m agents.teaching.cli "What is 17 * 23 + 4?"
python3 -m agents.teaching.cli
````

Interactive mode resumes `session.json`. Type `exit` to leave; Ctrl+C during a task
cancels that task, not its past effects. One-shot and scripted runs do not save or
resume a session.

Select a provider with `HARNESS_PROVIDER` (`anthropic` by default, also `openai`,
`openai-chat` and gateway routes) and set its credential locally. Use
[.env.example](.env.example) as a template and keep credentials out of source control.
See the [provider list and configuration reference](docs/REFERENCE.md#providers)
for every route and setting.

## Build on the runtime

Supply a model callable, system prompt and explicit tool mapping. This complete
example runs offline without application imports:

````python
from harness import Agent
def model(system, messages, tools, on_text=None):
    return {"stop_reason": "end_turn", "content": [{"type": "text", "text": "Hello."}]}
agent = Agent(model, "Be helpful.", tools={}, quiet=True)
result = agent.run_result("Say hello.")
print(result.to_dict())
````

`tools={}` disables tools; omitting `tools` is an error for `harness.Agent`.
`run()` returns display text. `run_result()` returns an immutable result containing
status, stop reason, answer, usage and elapsed time; `to_dict()` is JSON-serializable.
Inspect status as well as answer text: a retained answer does not imply completion.

### Register application tools

Schemas come from parameter annotations and defaults; descriptions come from
docstrings. Pass the resulting registry as `Agent(..., tools=tools)`.

````python
from harness import ToolRegistry
tools = ToolRegistry()
@tools.tool(read_only=True)
def word_count(text: str) -> str:
    """Count words in supplied text."""
    return str(len(text.split()))
````

Parameter types are `str`, `int`, `float` and `bool`. Use `Literal["asc", "desc"]`
(all strings or all integers) for fixed choices, and `Annotated[str, "description"]`
to describe a parameter to the model. The runtime checks object shape, required
fields, unknown arguments, primitive types and choices before execution;
tools still enforce their own path, range and authorization rules.
Registration rejects unsupported/missing annotations, incompatible defaults and
duplicate names rather than silently guessing a schema. Keyword-only parameters
and builtin postponed annotations work; variadic/positional-only parameters,
coroutines and generators are unsupported. Return annotations are not output schemas.

Use `risky=True` to require approval. `read_only=True` is a separate trusted
declaration, not something inferred from a tool's name or lack of approval.
Use separate registries, agents and trace/session/journal paths for independent
applications. One agent supports sequential tasks, not overlapping runs.

For a complete tool-using example, read [inventory_agent.py](examples/inventory_agent.py).
Try its isolated-worker path with `python3 -m examples.inventory_agent --sku GADGET --tool-timeout 3`.
To reuse the teaching tools with independent paths, use `TeachingConfig`,
`create_tools` and `create_agent` from `agents.teaching`; its factories do not load `.env`.

### Build a useful application

Follow [the application recipe](docs/BUILD_AN_AGENT.md) for an annotated loop-reading
path, application-owned output validation and a realistic support assistant:

````sh
python3 -m examples.support_agent
python3 -m examples.support_agent "warranty review" --sku GADGET
python3 -m examples.support_agent "penguin migration"
````

It combines approved local notes and product data, validates structured output and
actual citation provenance, and escalates when evidence is missing. Offline runs
make no network calls; `--live` explicitly uses provider environment settings and
can incur costs. This example never loads `.env`. Optional `--save-draft PATH`
asks for human approval after validation, never overwrites and never sends anything.
Data and policies are fictional; scripted success is not an accuracy evaluation.

Next, [evaluate the support assistant](docs/SUPPORT_EVALUATION.md) with twelve
synthetic draft cases: `python3 -m evaluation.support.evaluate --output runs/support-smoke-01`.
To score whole conversations (follow-ups, corrections, added constraints), run the five
multi-turn scenarios: `python3 -m evaluation.support.multiturn --output runs/support-multiturn-01`.
Both runners are offline by default, preserve prior runs and separate mechanical
checks from unscored human review. Paid model calls require explicit `--live`.

### Execution controls

Optional controls are passed explicitly to `Agent`: per-task budgets (`limits=`),
tool policies (`tool_policy=`), human approval (`approve=`), event callbacks (`on_event=`),
tool timeouts (`tool_timeout_seconds=`), the timed-tool worker environment (`worker_env=`),
crash-recovery journaling (`journal_path=`),
session persistence (`session_path=`) and between-step steering (`get_steering=`).
See the [execution controls reference](docs/REFERENCE.md#execution-controls),
including [isolated tool timeouts](docs/REFERENCE.md#isolated-tool-timeouts) and
[recovery without replaying](docs/REFERENCE.md#recover-without-replaying).

## Teaching tools, retrieval and evaluation

The teaching application provides calculator, time, workspace file, HTTP and
keyword document-search tools; the core runtime has none built in. Risky tools
require approval by default. Document retrieval (RAG) is a normal tool call with
cited passages, not a separate framework.

A HotpotQA evaluator compares a single agent with a researcher/reviewer workflow
on ten development questions, offline by default:
`python3 -m evaluation.hotpotqa.evaluate --output runs/single-smoke`.

See the reference for [teaching tools and corpus limits](docs/REFERENCE.md#teaching-tools-and-retrieval)
and the [HotpotQA evaluation workflow](docs/REFERENCE.md#hotpotqa-evaluation).

## Safety and operational limits

- **Not a security sandbox:** path checks and read-only policies constrain supplied
  tools, not arbitrary Python code or inherited process privileges.
- **Data remains sensitive:** prompts, document passages, tool results, traces and
  sessions may contain private content. Traces/events default to metadata only;
  full-content opt-ins and sessions, which keep the full history, still
  disclose content. Metadata and caller
  annotations are not secret-redacted. Curate documents and control provider/file access.
- **No automatic recovery of effects:** cancellation preserves known results but
  does not roll back actions. Inspect unknown outcomes before retrying.
- **No hard cost guarantee:** model calls and inline operations can overshoot
  cooperative time/token caps. Tool workers have no CPU/memory sandbox.
- **HTTP hardening is limited:** model clients reject redirects and sanitize error
  diagnostics; normal model/tool content can still reach results, sessions, console
  output and explicitly enabled content traces/events.

See [traces, sessions and compaction](docs/REFERENCE.md#traces-sessions-and-compaction)
for trace content opt-ins and history-compaction behavior.

## Tests

Tests use the standard library and need no provider credentials or network calls.
Run from the repository root:

````sh
python3 -m unittest -q
````

Focused suites and installed-data checks are listed in the [test reference](docs/REFERENCE.md#tests).
GitHub Actions runs the same command on Python 3.9 through 3.14 for every push to
`main` and every pull request ([workflow](.github/workflows/tests.yml)).

## Further reading

- [Reference: providers, configuration, controls, retrieval and evaluation](docs/REFERENCE.md)
- [Build a useful application and validate its output](docs/BUILD_AN_AGENT.md)
- [Evaluate the support assistant and review its answers](docs/SUPPORT_EVALUATION.md)
- [Runtime specification and supported API](docs/HARNESS_SPEC.md)
- [Architecture, process boundaries and editable diagrams](docs/ARCHITECTURE.md)
- [Event callbacks and quiet embedding](docs/HARNESS_EVENTS.md)
- [Tool policies and approval ordering](docs/TOOL_POLICIES.md)
- [HotpotQA comparison workflow](datasets/HOTPOTQA_BASELINE.md)
- [Separation contract](docs/HARNESS_SEPARATION_SPEC.md) and [migration checkpoints](docs/HARNESS_MIGRATION_PLAN.md)
