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

<augment_code_snippet mode="EXCERPT">
````sh
python3 -m lessons.preview
````
</augment_code_snippet>

Expect answer `3`, two model calls and one tool execution. Then open
[Lab 0](lessons/00-from-functions-to-agents.md) and build the loop in five small
checkpoints. Edit only `run_loop` in [lessons/exercises.py](lessons/exercises.py).
**The first check intentionally fails until you implement that checkpoint:**

<augment_code_snippet mode="EXCERPT">
````sh
python3 -m lessons.lab00_basics --check --stage 1
````
</augment_code_snippet>

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

<augment_code_snippet mode="EXCERPT">
````sh
python3 -m agents.teaching.cli mock
python3 -m agents.teaching.cli rag-mock
python3 -m examples.inventory_agent --sku WIDGET
````
</augment_code_snippet>

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

Configure a [provider](#providers) and its credential locally, then run a task or
start an interactive conversation. **These commands can incur API costs.**

<augment_code_snippet mode="EXCERPT">
````sh
python3 -m agents.teaching.cli "What is 17 * 23 + 4?"
python3 -m agents.teaching.cli
````
</augment_code_snippet>

The application-owned CLI also supports `python3 -m agents.teaching.cli` with the
same task, `mock` and `rag-mock` forms. Interactive mode resumes `session.json`.
Type `exit` to leave; Ctrl+C during a task cancels that task, not its past effects.
One-shot and scripted runs do not save or resume a session.

### Providers

`harness.make_model()` selects a streaming client using the environment. Adapters
normalize provider responses into one model/tool message format.

| `HARNESS_PROVIDER` | API | Credential variable |
| --- | --- | --- |
| `anthropic` (default) | Anthropic Messages | `ANTHROPIC_API_KEY` |
| `openai` | OpenAI Responses | `OPENAI_API_KEY` |
| `openai-chat` | OpenAI-compatible Chat Completions | `OPENAI_API_KEY` |
| `gateway-openai` | Grove AI Gateway, OpenAI route | `AI_GATEWAY_API_KEY` |
| `gateway-anthropic` | Grove AI Gateway, Anthropic route | `AI_GATEWAY_API_KEY` |
| `gateway` | Chooses Anthropic for model IDs starting with `claude`, otherwise OpenAI | `AI_GATEWAY_API_KEY` |

Set `HARNESS_MODEL` explicitly for `openai` and `openai-chat`; other routes have
defaults in [model_clients.py](harness/model_clients.py). `HARNESS_BASE_URL` can
target a compatible proxy or local server, where credentials may be unnecessary.
Chat Completions uses `max_tokens`; use the Responses adapter for OpenAI models
that require a different generation interface. Provider reasoning state is not
preserved across adapters.

### Configuration

Use [.env.example](.env.example) as a template. If `.env` already exists, edit it
deliberately rather than overwriting it. Keep credentials out of source control,
commands shared with others and traces.
This configuration is for the application CLIs, not the offline labs in [LEARN.md](LEARN.md).

The teaching CLI and HotpotQA worker CLIs load `.env` from their current working
directory. Existing environment variables win, including empty values. The loader
supports assignments, quotes, comments and optional `export`; it does not execute
shell code or expand variables. Library callers must load configuration explicitly.
Live startup prints the selected provider, model and a credential-filtered endpoint.

| Setting | Default | Purpose |
| --- | --- | --- |
| `HARNESS_PROVIDER` | `anthropic` | Provider route from the table above |
| `HARNESS_MODEL` | Route-dependent | Model ID |
| `HARNESS_BASE_URL` | Route-dependent | Final API base URL; redirects are rejected |
| `HARNESS_API_KEY_ENV` | Route-dependent | Name of the environment variable holding the credential, not its value |
| `ANTHROPIC_WORKSPACE_ID` | Unset | Anthropic workspace selection when needed |
| `HARNESS_WORKSPACE` | `workspace` | Teaching file-tool directory |
| `HARNESS_KNOWLEDGE` | `knowledge` | Approved retrieval directory, outside the workspace |
| `HARNESS_SESSION` | `session.json` | Interactive conversation file |
| `HARNESS_COMPACT_CHARS` | `40000` | History-size threshold for model-written summaries, in JSON characters |
| `HARNESS_MAX_MODEL_CALLS` | Unset | Per-task logical calls, including summaries |
| `HARNESS_MAX_TOOL_CALLS` | Unset | Per-task tool attempts, including invalid/denied requests |
| `HARNESS_MAX_TOKENS` | Unset | Per-task reported input + output tokens |
| `HARNESS_MAX_SECONDS` | Unset | Cooperative task deadline in seconds |
| `HARNESS_TOOL_TIMEOUT_SECONDS` | Unset | Positive timeout for isolated tool execution |
| `HARNESS_READ_ONLY` | `0` | `1` permits only explicitly read-only tools |
| `HARNESS_JOURNAL` | Unset | SQLite journal path outside application-writable directories |
| `HARNESS_TRACE_CONTENT` | `0` | Teaching CLI only: `1` explicitly includes potentially sensitive content in traces |

Runtime library callers pass controls explicitly to `Agent`. HotpotQA evaluation
uses its own command-line budgets and does not inherit the general teaching CLI
budget, timeout, policy or journal settings.

## Build on the runtime

Supply a model callable, system prompt and explicit tool mapping. This complete
example runs offline without application imports:

<augment_code_snippet mode="EXCERPT">
````python
from harness import Agent
def model(system, messages, tools, on_text=None):
    return {"stop_reason": "end_turn", "content": [{"type": "text", "text": "Hello."}]}
agent = Agent(model, "Be helpful.", tools={}, quiet=True)
result = agent.run_result("Say hello.")
print(result.to_dict())
````
</augment_code_snippet>

`tools={}` disables tools; omitting `tools` is an error for `harness.Agent`.
`run()` returns display text. `run_result()` returns an immutable result containing
status, stop reason, answer, usage and elapsed time; `to_dict()` is JSON-serializable.
Inspect status as well as answer text: a retained answer does not imply completion.

### Register application tools

Schemas come from parameter annotations and defaults; descriptions come from
docstrings. Pass the resulting registry as `Agent(..., tools=tools)`.

<augment_code_snippet mode="EXCERPT">
````python
from harness import ToolRegistry
tools = ToolRegistry()
@tools.tool(read_only=True)
def word_count(text: str) -> str:
    """Count words in supplied text."""
    return str(len(text.split()))
````
</augment_code_snippet>

Parameter types are `str`, `int`, `float` and `bool`. The runtime checks object
shape, required fields, unknown arguments and primitive types before execution;
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

<augment_code_snippet mode="EXCERPT">
````sh
python3 -m examples.support_agent
python3 -m examples.support_agent "warranty review" --sku GADGET
python3 -m examples.support_agent "penguin migration"
````
</augment_code_snippet>

It combines approved local notes and product data, validates structured output and
actual citation provenance, and escalates when evidence is missing. Offline runs
make no network calls; `--live` explicitly uses provider environment settings and
can incur costs. This example never loads `.env`. Optional `--save-draft PATH`
asks for human approval after validation, never overwrites and never sends anything.
Data and policies are fictional; scripted success is not an accuracy evaluation.

Next, [evaluate the support assistant](docs/SUPPORT_EVALUATION.md) with twelve
synthetic draft cases: `python3 -m evaluation.support.evaluate --output runs/support-smoke-01`.
The runner is offline by default, preserves prior runs and separates mechanical
checks from unscored human review. Paid model calls require explicit `--live`.

### Execution controls

| Capability | Library interface | Important limitation |
| --- | --- | --- |
| Shared per-task budget | `RunLimits(...)`, passed as `limits=` | Token/time caps are cooperative, not hard billing or execution limits |
| Tool dispatch policy | `ToolPolicy(read_only=True, argument_checks=...)` | Application checks, not filesystem/network confinement |
| Human approval | `approve(name, args)` | Only trusted application code decides; `quiet=True` denies the default prompt |
| Observation | `on_event=callback` | Synchronous immutable events; metadata-only unless content is explicitly enabled |
| Trace privacy | `trace_include_content=False` | JSONL metadata only by default; content opt-in is independent of event settings |
| Tool timeouts | `tool_timeout_seconds=` or tool decorator | Worker processes do not provide security isolation |
| Crash recovery | `journal_path=` | Human-reviewed outcomes, not automatic replay or exactly-once effects |
| Conversation persistence | `session_path=` | Saved history alone cannot establish whether an action completed |

For example, `RunLimits(max_model_calls=12, max_tool_calls=12, max_seconds=60)`
caps further work within each task, including compaction. Unknown token usage stays
`None`; enabling a token cap stops with `usage_unavailable` when usage is missing.
Use call/tool/time limits, rather than token limits, for scripted demos without usage.

See the [public API and execution contract](docs/HARNESS_SPEC.md),
[event guide](docs/HARNESS_EVENTS.md) and [policy guide](docs/TOOL_POLICIES.md).

### Isolated tool timeouts

Tools run inline unless a positive timeout is configured. Timed tools run in fresh
Python workers after validation, policy and approval. Functions must be importable
at module level; JSON-bound `functools.partial` functions are also supported.
Closures, lambdas and bound methods are rejected, never silently run inline.

Workers use the smaller of the tool timeout and remaining task deadline; bounded
cleanup adds time. Parent-memory changes do not propagate. Workers inherit process
access, including environment, filesystem and network: **this is not an OS sandbox**.
Cleanup is best-effort, including POSIX process-group cleanup where available.

A timeout or uncertain outcome stops further work without automatic retry. A write
or remote action may already have happened. Inspect effects before continuing;
stopping a worker does not undo them.

### Recover without replaying

Opt into journaling with `HARNESS_JOURNAL=execution.sqlite3` or `journal_path=`.
Keep the journal on a trusted local filesystem, outside writable roots and separate
from traces/sessions; its parent must exist. Custom applications declare their
writable directories through `writable_roots=` and enforce their own tool paths.

The journal commits action metadata before dispatch and records the outcome after.
It stores no tool arguments or outputs. Uncertain actions and unfinished runs with
actions block new tasks before model calls. Keep the same journal across restarts.

Stop affected processes, inspect pending records and verify external effects first:

<augment_code_snippet mode="EXCERPT">
````sh
python3 -m harness inspect execution.sqlite3 --pending --limit 100
````
</augment_code_snippet>

Only after human verification, replace `ACTION_ID` with the inspected ID and record
the actual outcome (`completed`, `failed` or `not_run`):

<augment_code_snippet mode="EXCERPT">
````sh
python3 -m harness resolve execution.sqlite3 ACTION_ID --outcome completed --confirm
````
</augment_code_snippet>

Resolution never executes tools, restores outputs or repairs the conversation.
Review saved history before continuing. Journaling is a local, single-owner recovery
aid, not a distributed scheduler or an exactly-once guarantee.

## Teaching tools and retrieval

The teaching application provides the following tools; the core runtime has none
built in. Live `write_file` and `http_get` requests require approval by default.

| Tool | Behavior |
| --- | --- |
| `calculator` | Arithmetic through an AST interpreter, not `eval()` |
| `current_time` | UTC timestamp |
| `list_files`, `read_file` | Workspace-relative file access with escape checks |
| `write_file` | Create or overwrite a workspace file; risky |
| `http_get` | Fetch an HTTP(S) response as text; risky |
| `search_documents` | Bounded keyword retrieval with passage source IDs |

**Retrieval-augmented generation (RAG)** supplies external evidence before an answer.
Here it is a normal tool call, not a separate framework: search approved documents,
read the returned evidence and cite it. Ranking uses keyword overlap, not embeddings,
semantic similarity or confidence scores. Conversation summaries are memory, not
external evidence.

Try a no-match case with `python3 -m agents.teaching.cli rag-mock "penguin migration"`.
The [bundled HotpotQA demo](datasets/hotpotqa/README.md) adds ten real questions
and 100 candidate paragraphs without a download. Point retrieval only at its
`knowledge/` subdirectory, never the parent containing answers and source snapshots.

### Corpus and limits

- Use approved, non-sensitive UTF-8 `.md`/`.txt` files directly in `knowledge/` or
  `HARNESS_KNOWLEDGE`, outside the writable workspace. No recursive search or index.
- The directory may contain at most 100 entries, including skipped files. Files
  over 64 KiB, invalid UTF-8, symlinks, hidden files and disallowed names are skipped.
- Filenames start with an ASCII letter/number and use letters, numbers, dots,
  underscores or hyphens. Credential-like names are filtered, but this is **not
  secret detection**; ordinary files can still contain sensitive material.
- Queries are at most 500 characters. Each call returns up to three 800-character
  passages with 100-character overlap, within the default 8,000-character output cap.
- Source IDs identify filename/character offsets, not permanent citations after
  edits. No matches do not prove no answer exists. Documents remain untrusted data;
  prompt instructions do not guarantee resistance to malicious document content.

## HotpotQA evaluation

The evaluator compares a single agent with a **sequential researcher/reviewer**
workflow using identical label-free inputs and shared total budgets. Only the host
scorer reads gold answers. Evaluation covers the original ten selected IDs, **not
the entire development split**.

Prepare the 7,405-question development source once, if it is not already installed.
This downloads about 61 MB, makes no model calls and refuses to overwrite an existing
output directory:

<augment_code_snippet mode="EXCERPT">
````sh
python3 -m evaluation.hotpotqa.prepare --full-dev
python3 -m evaluation.hotpotqa.run_question --question 1
````
</augment_code_snippet>

Then run an offline paired smoke check. Use **new output directories** each time:

<augment_code_snippet mode="EXCERPT">
````sh
python3 -m evaluation.hotpotqa.evaluate --output runs/single-smoke
python3 -m evaluation.hotpotqa.evaluate --architecture multi-agent --reference runs/single-smoke --output runs/multi-smoke
````
</augment_code_snippet>

Smoke runs exercise real retrieval and sentence reading, then deliberately abstain.
Quality scores stay `null`: they are not an accuracy baseline. Outputs include
frozen inputs, per-question traces/results, summaries and code/model fingerprints;
the paired run also writes `comparison.json`.

For a first **paid** evaluation, configure a provider and explicitly opt in to one
question with `python3 -m evaluation.hotpotqa.evaluate --live --limit 1 --output runs/single-live-one`.
The [comparison guide](datasets/HOTPOTQA_BASELINE.md) explains full paired live runs,
budgets, metrics and artifacts. [The dataset guide](datasets/HOTPOTQA_DEV.md) covers
provenance, attribution, offline rebuilds and original label quirks.

References must match current source, inputs and recorded settings. Keep historical
artifacts unchanged and generate fresh pairs after code changes. Matching settings
cannot prove unchanged provider behavior; ten development examples do not establish
a general architecture advantage. Package commands such as
`python3 -m evaluation.hotpotqa.evaluate` are the supported entrypoints.

## Safety and operational limits

- **Not a security sandbox:** path checks and read-only policies constrain supplied
  tools, not arbitrary Python code or inherited process privileges.
- **Data remains sensitive:** prompts, document passages, tool results, traces and
  sessions may contain private content. Traces/events default to metadata only;
  full-content opt-ins and sessions still disclose content. Metadata and caller
  annotations are not secret-redacted. Curate documents and control provider/file access.
- **No automatic recovery of effects:** cancellation preserves known results but
  does not roll back actions. Inspect unknown outcomes before retrying.
- **No hard cost guarantee:** model calls and inline operations can overshoot
  cooperative time/token caps. Tool workers have no CPU/memory sandbox.
- **HTTP hardening is limited:** model clients reject redirects and sanitize error
  diagnostics; normal model/tool content can still reach results, sessions, console
  output and explicitly enabled content traces/events.

**Trace migration:** consumers reading task/model/tool content must explicitly set
`trace_include_content=True` (teaching CLI: `HARNESS_TRACE_CONTENT=1`). Event content
is controlled separately. HotpotQA evaluation explicitly retains full benchmark traces;
historical artifacts are not rewritten. Support traces default to metadata only.

History compaction uses model-written summaries measured in characters, not tokens.
It preserves recent tool/result pairs but may lose information. Sessions store
conversation state; traces are append-only observations, not replay instructions.
Use a new session path for an unrelated conversation, and distinct paths for each
application.

## Tests

Tests use the standard library and need no provider credentials or network calls.
Run from the repository root:

<augment_code_snippet mode="EXCERPT">
````sh
python3 -m unittest -q
python3 -m unittest -v tests.test_harness_package
python3 -m unittest -v tests.test_first_principles_lab tests.test_learning_labs
python3 -m unittest -v tests.test_learning_onboarding
python3 -m unittest -v tests.test_learning_capstone
python3 -m unittest -v tests.test_tool_contracts tests.test_trace_privacy tests.test_support_agent tests.test_runtime_distribution
python3 -m unittest -v tests.test_support_evaluation
````
</augment_code_snippet>

After preparing the full development set, include the installed-data checks with
`HARNESS_TEST_HOTPOTQA_DEV=1 python3 -m unittest -q`. This does not download data or
make live-model calls. Shared fixtures live in [tests/helpers.py](tests/helpers.py);
importing them does not initialize applications.

## Further reading

- [Build a useful application and validate its output](docs/BUILD_AN_AGENT.md)
- [Evaluate the support assistant and review its answers](docs/SUPPORT_EVALUATION.md)
- [Runtime specification and supported API](docs/HARNESS_SPEC.md)
- [Architecture, process boundaries and editable diagrams](docs/ARCHITECTURE.md)
- [Event callbacks and quiet embedding](docs/HARNESS_EVENTS.md)
- [Tool policies and approval ordering](docs/TOOL_POLICIES.md)
- [HotpotQA comparison workflow](datasets/HOTPOTQA_BASELINE.md)
- [Separation contract](docs/HARNESS_SEPARATION_SPEC.md) and [migration checkpoints](docs/HARNESS_MIGRATION_PLAN.md)
