# Little Harness reference

Detailed configuration, execution controls, teaching tools, evaluation and testing
reference. Start with the [README](../README.md) for an overview and quick start.

## Providers

`harness.make_model()` selects a streaming client using the environment. Adapters
normalize provider responses into one model/tool message format. An unknown
provider, missing credential or missing model raises `harness.ModelConfigError`
before any request; the bundled command-line tools print its message and exit with code 1.

| `HARNESS_PROVIDER` | API | Credential variable |
| --- | --- | --- |
| `anthropic` (default) | Anthropic Messages | `ANTHROPIC_API_KEY` |
| `openai` | OpenAI Responses | `OPENAI_API_KEY` |
| `openai-chat` | OpenAI-compatible Chat Completions | `OPENAI_API_KEY` |
| `gateway-openai` | Grove AI Gateway, OpenAI route | `AI_GATEWAY_API_KEY` |
| `gateway-anthropic` | Grove AI Gateway, Anthropic route | `AI_GATEWAY_API_KEY` |
| `gateway` | Chooses Anthropic for model IDs starting with `claude`, otherwise OpenAI | `AI_GATEWAY_API_KEY` |

Set `HARNESS_MODEL` explicitly for `openai` and `openai-chat`; other routes have
defaults in [model_clients.py](../harness/model_clients.py). `HARNESS_BASE_URL` can
target a compatible proxy or local server, where credentials may be unnecessary.
Chat Completions uses `max_tokens`; use the Responses adapter for OpenAI models
that require a different generation interface. Provider reasoning state is not
preserved across adapters.

## Configuration

Use [.env.example](../.env.example) as a template. If `.env` already exists, edit it
deliberately rather than overwriting it. Keep credentials out of source control,
commands shared with others and traces.
This configuration is for the application CLIs, not the offline labs in [LEARN.md](../LEARN.md).

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

## Execution controls

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
| Between-step steering | `get_steering=callable` | Called only after a saved tool batch; text guides the model but cannot authorize tools or cancel a running one |

For example, `RunLimits(max_model_calls=12, max_tool_calls=12, max_seconds=60)`
caps further work within each task, including compaction. Unknown token usage stays
`None`; enabling a token cap stops with `usage_unavailable` when usage is missing.
Use call/tool/time limits, rather than token limits, for scripted demos without usage.

See the [public API and execution contract](HARNESS_SPEC.md),
[event guide](HARNESS_EVENTS.md) and [policy guide](TOOL_POLICIES.md).

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

````sh
python3 -m harness inspect execution.sqlite3 --pending --limit 100
````

Only after human verification, replace `ACTION_ID` with the inspected ID and record
the actual outcome (`completed`, `failed` or `not_run`):

````sh
python3 -m harness resolve execution.sqlite3 ACTION_ID --outcome completed --confirm
````

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
| `http_get` | Fetch an HTTP(S) response as text; risky. Redirects are not followed: the model must request, and a person approve, the new URL |
| `search_documents` | Bounded keyword retrieval with passage source IDs |

**Retrieval-augmented generation (RAG)** supplies external evidence before an answer.
Here it is a normal tool call, not a separate framework: search approved documents,
read the returned evidence and cite it. Ranking uses keyword overlap, not embeddings,
semantic similarity or confidence scores. Conversation summaries are memory, not
external evidence.

Try a no-match case with `python3 -m agents.teaching.cli rag-mock "penguin migration"`.
The [bundled HotpotQA demo](../datasets/hotpotqa/README.md) adds ten real questions
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

````sh
python3 -m evaluation.hotpotqa.prepare --full-dev
python3 -m evaluation.hotpotqa.run_question --question 1
````

Then run an offline paired smoke check. Use **new output directories** each time:

````sh
python3 -m evaluation.hotpotqa.evaluate --output runs/single-smoke
python3 -m evaluation.hotpotqa.evaluate --architecture multi-agent --reference runs/single-smoke --output runs/multi-smoke
````

Smoke runs exercise real retrieval and sentence reading, then deliberately abstain.
Quality scores stay `null`: they are not an accuracy baseline. Outputs include
frozen inputs, per-question traces/results, summaries and code/model fingerprints;
the paired run also writes `comparison.json`.

For a first **paid** evaluation, configure a provider and explicitly opt in to one
question with `python3 -m evaluation.hotpotqa.evaluate --live --limit 1 --output runs/single-live-one`.
The [comparison guide](../datasets/HOTPOTQA_BASELINE.md) explains full paired live runs,
budgets, metrics and artifacts. [The dataset guide](../datasets/HOTPOTQA_DEV.md) covers
provenance, attribution, offline rebuilds and original label quirks.

References must match current source, inputs and recorded settings. Keep historical
artifacts unchanged and generate fresh pairs after code changes. Matching settings
cannot prove unchanged provider behavior; ten development examples do not establish
a general architecture advantage. Package commands such as
`python3 -m evaluation.hotpotqa.evaluate` are the supported entrypoints.

## Traces, sessions and compaction

**Trace migration:** consumers reading task/model/tool content must explicitly set
`trace_include_content=True` (teaching CLI: `HARNESS_TRACE_CONTENT=1`). Event content
is controlled separately. HotpotQA evaluation explicitly retains full benchmark traces;
historical artifacts are not rewritten. Support traces default to metadata only.

History compaction uses model-written summaries. By default its threshold counts
JSON characters. Library callers can pass `compact_at_tokens=` to `Agent` to use an
estimated token threshold instead: the last reported input tokens plus about one
token per four characters added since, or characters alone when usage is unknown.
For live models, set `compact_at_tokens` comfortably below the model's context window,
leaving room for the system prompt, tool schemas and the reply. The harness does not
detect or recover from a provider's "context too long" error: it arrives as an HTTP
400 (`ModelError` category `request_rejected`) and the task stops with `model_error`.
Compaction changes only what the model sees: the latest summary followed by recent
messages, keeping tool calls with their results. The summary may lose information,
but the conversation history itself is never shortened. A session file stores that
full history plus the compaction point and summary, so a reviewer can always see
what a summary replaced and a resumed session sends the same view. Sessions therefore
grow over time; older sessions saved as a plain list still load. Sessions store
conversation state; traces are append-only observations, not replay instructions.
Use a new session path for an unrelated conversation, and distinct paths for each
application.

## Tests

Run focused suites from the repository root:

````sh
python3 -m unittest -v tests.test_harness_package
python3 -m unittest -v tests.test_first_principles_lab tests.test_learning_labs
python3 -m unittest -v tests.test_learning_onboarding
python3 -m unittest -v tests.test_learning_capstone
python3 -m unittest -v tests.test_tool_contracts tests.test_trace_privacy tests.test_support_agent tests.test_runtime_distribution
python3 -m unittest -v tests.test_support_evaluation
````

After preparing the full development set, include the installed-data checks with
`HARNESS_TEST_HOTPOTQA_DEV=1 python3 -m unittest -q`. This does not download data or
make live-model calls. Shared fixtures live in [tests/helpers.py](../tests/helpers.py);
importing them does not initialize applications.

