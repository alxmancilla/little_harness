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
| `HARNESS_JOURNAL_BACKEND`, `HARNESS_JOURNAL_ID` | Unset | Teaching CLI journal backend (`postgres` or `mongodb`) and stable journal ID |
| `HARNESS_POSTGRES_DSN`, `HARNESS_MONGODB_URI` | Unset | Connection string for the selected teaching CLI journal backend |
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
| Worker environment | `worker_env={...}` | Replaces the environment of timed-tool workers only; inline tools still see the parent's |
| Crash recovery | `journal=` with an `ActionJournal` | Human-reviewed outcomes, not automatic replay or exactly-once effects |
| Conversation persistence | `session_path=`, or `session_store=` with any `SessionStore` | Saved history alone cannot establish whether an action completed |
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
To keep API keys and other secrets out of timed tools, pass `worker_env=` with only
the variables they need, for example `{"PATH": os.environ["PATH"]}`. The mapping
becomes the worker's whole environment (Windows also keeps `SYSTEMROOT`); `None`, the
default, copies the parent environment. It does not limit files, network or inline tools.
Cleanup is best-effort, including POSIX process-group cleanup where available.

A timeout or uncertain outcome stops further work without automatic retry. A write
or remote action may already have happened. Inspect effects before continuing;
stopping a worker does not undo them.

### Recover without replaying

Opt into journaling by passing a PostgreSQL or MongoDB `ActionJournal` as `journal=`.
For the teaching CLI, set `HARNESS_JOURNAL_BACKEND` and `HARNESS_JOURNAL_ID` and
provide the matching connection variable. Protect the database and its credentials.

The journal commits action metadata before dispatch and records the outcome after.
It stores no tool arguments or outputs. Uncertain actions and unfinished runs with
actions block new tasks before model calls. Keep the same journal across restarts.

Stop affected processes, inspect pending records and verify external effects first:

Use the database-specific `journal inspect` command below.

Only after human verification, replace `ACTION_ID` with the inspected ID and record
the actual outcome (`completed`, `failed` or `not_run`):

Use the corresponding `journal resolve` command with `--confirm`.

Resolution never executes tools, restores outputs or repairs the conversation.
Review saved history before continuing. Journaling is a single-owner recovery
aid, not a distributed scheduler or an exactly-once guarantee.

### PostgreSQL sessions and journals (optional)

`stores.postgres` provides `PostgresSessionStore` and `PostgresJournal` for
PostgreSQL 18, passed as `session_store=` and `journal=`. They keep the file
session and journal contracts and add a check against a second session writer
([storage specification](STORAGE_SPEC.md)). Install the driver, then name the
environment variable that holds the connection string; never pass the string itself:

````sh
uv sync --extra postgres
export APP_POSTGRES_DSN='postgresql://app@db.example/app?sslmode=verify-full'
python -m stores.postgres init --dsn-env APP_POSTGRES_DSN
````

````python
from stores.postgres import PostgresJournal, PostgresSessionStore

agent = Agent(model, system, tools=tools, session_store=PostgresSessionStore("user-7", dsn_env="APP_POSTGRES_DSN"),
              journal=PostgresJournal("support-worker", dsn_env="APP_POSTGRES_DSN"))
````

- **Setup and ownership.** `init` creates the tables in the `little_harness` schema
  (`--schema` to change it) and records their owner and version. Without `init`,
  the first use creates them if the role may. Tables of the same name without that
  record, or with another version, are never adopted.
- **Least privilege.** After `init`, the application's role needs only `USAGE` on
  the schema and `SELECT`, `INSERT`, `UPDATE` and `DELETE` on its tables.
- **Connections.** Use TLS (`sslmode=verify-full`). Connecting is limited to 10
  seconds and each statement to 5 (`statement_timeout=`). Every write commits
  synchronously. Errors carry fixed messages, never the driver's text.
- **One writer per session.** A second agent saving the same session ID gets
  `SessionConflictError`, and the first agent's save is kept.
- **Recovery** uses the shared journal rules:

````sh
python -m stores.postgres journal inspect --dsn-env APP_POSTGRES_DSN --journal-id support-worker --pending
python -m stores.postgres journal resolve --dsn-env APP_POSTGRES_DSN --journal-id support-worker \
  ACTION_ID --outcome completed --confirm
````

For local development, `compose.yaml` starts PostgreSQL 18 on `127.0.0.1:55432`
without a password or TLS; see [Tests](#tests) for running the database suites.

### MongoDB sessions and journals (optional)

`stores.mongodb` provides `MongoSessionStore` and `MongoJournal` for MongoDB 9.0,
with the same guarantees, options and commands as the PostgreSQL adapters. The
server must be a replica set (one member is enough), because saves and journal checks
use transactions.

````sh
uv sync --extra mongodb
export APP_MONGODB_URI='mongodb://app@db.example/?replicaSet=rs0&tls=true'
python -m stores.mongodb init --uri-env APP_MONGODB_URI
````

````python
from stores.mongodb import MongoJournal, MongoSessionStore

agent = Agent(model, system, tools=tools, session_store=MongoSessionStore("user-7", uri_env="APP_MONGODB_URI"),
              journal=MongoJournal("support-worker", uri_env="APP_MONGODB_URI"))
````

- **Storage.** Collections prefixed `lh_` in the `little_harness` database
  (`--database` to change it), each with a schema validator. Every message is its own
  document holding JSON text, so keys chosen by the model never become field names.
- **Least privilege.** After `init`, the application's user needs only `readWrite` on
  that database.
- **Connections.** Use TLS. Connecting is limited to 10 seconds and each operation to
  5 (`timeout=`); writes are acknowledged by a majority and journaled. One client is
  shared per connection string; pass `client=` to use your own `MongoClient`.
- **After a crash mid-save.** The dead process's transaction stays open on the server
  for up to `transactionLifetimeLimitSeconds` (60 by default). Until then, the session
  still loads its last saved state, but saves to it fail with `SessionStoreError`; the
  same applies to journal writes. Wait, or end the session with `killSessions`.
- **Recovery:** `python -m stores.mongodb journal inspect|resolve --uri-env NAME
  --journal-id ID`, with the same output and confirmation as the other journals.

`compose.yaml` also starts a single-node MongoDB 9.0 replica set on `127.0.0.1:57017`.

### Long-term memory (optional)

`PostgresMemoryStore` and `MongoMemoryStore` keep persistent memory separate from session
history and the action journal. Construct it explicitly; nothing is injected into
the model context without a `recall` call. For example, the support factory can
bind an application-selected namespace:

````python
from harness import ToolPolicy
from examples import support_agent
from stores.postgres import PostgresJournal, PostgresMemoryStore
store = PostgresMemoryStore("support", dsn_env="APP_POSTGRES_DSN")
agent = support_agent.create_agent(model, documents, products, memory_store=store,
    memory_namespace="tenant:42/user:7", quiet=False, tool_policy=ToolPolicy(),
    journal=PostgresJournal("support-worker", dsn_env="APP_POSTGRES_DSN"))
````

With this configuration `recall` is read-only and `remember`/`forget` ask for human
approval before the journal records and executes them. The support factory's default
`ToolPolicy(read_only=True)` blocks both writes; `memory_tools(..., allow_write=False)`
also omits them entirely. Never auto-approve based on model or memory text.

- Namespaces are scope labels, **not access control**; the application must enforce
  authorization and protect the database and its credentials.
- Each namespace defaults to 10,000 records; full quotas reject additions without
  eviction. Text is limited to 2,000 characters; `search` uses case-insensitive ASCII
  word matches and ranks by shared words, recency and ID. Both adapters use indexed
  word arrays to find candidates, then rank them in the application.
- `recall` returns JSON marked `untrusted`; complete records that do not fit are
  omitted. An output limit too small even for an empty envelope fails the tool.
  Instructions inside memory have no authority, but a model can still follow them.
- `delete` and `purge` delete records. This does not erase copies in backups,
  previously saved sessions, traces, filesystem snapshots
  or storage hardware. Protect those separately.
- `memory_tools(store, namespace)` binds closures and runs inline only. For isolated
  timeouts, use `stores.memory.memory_worker_tools` as shown below. Default traces and
  events omit memory content; content opt-ins, session history and model output may contain it.
- The support evaluation includes synthetic, unreviewed memory-poisoning cases. Its
  `cases.json` retains the synthetic memory; per-case fixtures are ephemeral. Exposure checks
  only establish that the model received it, not that the answer resisted it.

#### Memory tools with isolated timeouts

Configure the connection variable in the parent environment, then pass its **name**
to the worker registry and its value only through `worker_env`:

````python
import os
from harness import ToolPolicy
from examples import support_agent
from stores.memory import memory_worker_tools
from stores.postgres import PostgresJournal
registry = memory_worker_tools("postgres", "support", "tenant:42/user:7", env="APP_POSTGRES_DSN")
agent = support_agent.create_agent(model, documents, products, memory_registry=registry,
    tool_timeout_seconds=10, worker_env={"APP_POSTGRES_DSN": os.environ["APP_POSTGRES_DSN"]},
    tool_policy=ToolPolicy(), quiet=False,
    journal=PostgresJournal("support-worker", dsn_env="APP_POSTGRES_DSN"))
````

Use `"mongodb"` and the URI variable name for MongoDB; `location=` selects the
schema or database, and `quota=` and `timeout=` configure the backend. The tool
timeout applies to the whole child process, including connection and imports; the
backend timeout applies to its database operations. `memory_allow_write=False` or
the registry's `allow_write=False` omits writes. Policy and approval still run in the
parent before the journal and worker. The worker receives the run ID and output limit
privately, not as model arguments. PostgreSQL backend timeouts must be at least one
millisecond.

Each timed invocation starts a fresh process and connection; MongoDB cannot reuse
the parent's client pool across processes. Monitor database connection counts and
latency before enabling this for high-volume workloads. A timeout or interrupted write
can have committed: inspect and resolve the journal and external state before retrying.
An interrupted MongoDB transaction may hold locks until its server-side lifetime expires.
`worker_env` limits inherited environment variables, not filesystem or network access;
it is not a security sandbox.

## Teaching tools and retrieval

The teaching application provides the following tools; these are not built into the
core runtime. Live `write_file` and `http_get` requests require approval by default.

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
python3 -m unittest -v tests.test_memory tests.test_storage_conformance
````

After preparing the full development set, include the installed-data checks with
`HARNESS_TEST_HOTPOTQA_DEV=1 python3 -m unittest -q`. This does not download data or
make live-model calls. Shared fixtures live in [tests/helpers.py](../tests/helpers.py);
importing them does not initialize applications.

Storage backends share the contract suites in
[tests/storage_conformance.py](../tests/storage_conformance.py). The PostgreSQL and
MongoDB tests skip unless the extra is installed and a test database is named; each
test creates and drops its own schema or database:

````sh
docker compose up -d --wait
uv sync --extra postgres --extra mongodb
export LITTLE_HARNESS_TEST_POSTGRES_DSN=postgresql://postgres@127.0.0.1:55432/postgres
export LITTLE_HARNESS_TEST_MONGODB_URI='mongodb://127.0.0.1:57017/?directConnection=true'
uv run python -m unittest -v tests.test_storage_conformance tests.test_storage_postgres tests.test_storage_mongodb
````

The [storage workflow](../.github/workflows/storage.yml) runs them in CI.

