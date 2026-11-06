# Little Harness storage specification

**Status:** interfaces, PostgreSQL and MongoDB session, journal and memory adapters,
and memory tools are implemented; timed memory workers remain planned. See also the
[harness specification](HARNESS_SPEC.md). **Targets:** Python 3.11+,
PostgreSQL 18, MongoDB 9.0 (replica set), psycopg 3.3, PyMongo 4.19.

**MUST** denotes a required behavior once implemented; **MAY** an optional one.
Requirements are numbered `ST-xx`; acceptance scenarios `SAC-xx`.

## 1. Purpose and scope

Applications can keep three kinds of data outside local files:

| Data | Interfaces and implementations |
| --- | --- |
| Conversation state | `SessionStore`; `SessionFile`, PostgreSQL and MongoDB stores |
| Action journal | `ActionJournal`; PostgreSQL and MongoDB journals |
| Long-term memory | `MemoryStore`, memory tools, PostgreSQL and MongoDB stores |

### Goals

- Keep `harness/` standard-library only: interfaces and defaults live there; database
  adapters live in a separate optional package, `stores/`.
- Retain `session_path=` and its file format; journals and memory use database adapters.
- Give every backend the same contract, proven by shared conformance tests, as
  `ProviderConformanceTests` does for model adapters.
- Expose memory to the model only as ordinary, visible, policy-gated tools.

### Non-goals

- Several processes driving one session or journal at once. Stores detect this and
  fail; they do not coordinate it.
- One transaction spanning session, journal and memory, even in one database.
- Embeddings, vector search or automatic memory extraction (possible later revision).
- Memory added to the prompt without a tool call, or summaries written to memory
  automatically.
- Authorization or tenant isolation. Namespaces are application-chosen labels, not
  access control; use database roles for that.
- Storing traces or events in a database (`on_event` remains the integration point).
- A migration framework. Each schema has one version; upgrades are explicit commands.

## 2. Vocabulary and boundaries

- **Backend:** `file` for sessions (standard library, in `harness/`); `postgres` and
  `mongodb` for sessions, journals and memory (in `stores/`, importing drivers lazily).
- **Session ID / journal ID:** application-chosen strings (1–200 characters,
  `[A-Za-z0-9._:-]`) naming one session or one journal inside a database. They play
  the role the file path plays today.
- **Namespace:** application-chosen memory scope (1–200 characters,
  `[A-Za-z0-9._:/-]`), such as `tenant:42/user:7`. The model never supplies or sees it.
- **Revision:** a per-session counter used to detect a second writer.
- **Connection settings:** a DSN or URI read from an environment variable whose
  *name* the application passes. Values never appear in arguments, traces, events,
  error messages or command lines.

Trust boundaries are unchanged: stored messages, tool results and memory text are
data, never instructions or authority. Memory written by the model is untrusted
input to later sessions.

## 3. Interfaces

### 3.1 `SessionStore` (in `harness/session.py`)

````python
class SessionStore(Protocol):
    def load(self) -> tuple[list, dict | None] | None: ...   # None: no session yet
    def save(self, messages: list, compaction: dict | None, *, durable: bool) -> None: ...
````

- One instance serves one session. `SessionFile(path)` implements it.
- `Agent(session_store=...)` is new and mutually exclusive with `session_path=`.
- Validation moves into one shared function, `validate_session(...)`, used by every
  backend, so all of them raise the same `SessionLoadError` for the same bad data.

### 3.2 `ActionJournal` (in `harness/journal.py`)

The interface has these methods: `prepare`,
`assert_ready`, `begin`, `finish`, `close_run`, `inspect`, `resolve`.

- `Agent(journal=...)` accepts an `ActionJournal` implementation.
- `JournalError` and `JournalRecoveryRequired` stay the only errors the agent sees.

### 3.3 `MemoryStore` and memory tools (in `harness/memory.py`)

````python
class MemoryStore(Protocol):
    def add(self, namespace, text, *, kind, created_by, run_id=None) -> str: ...
    def search(self, namespace, query, *, limit=5) -> list[MemoryRecord]: ...
    def delete(self, namespace, memory_id) -> bool: ...
    def list(self, namespace, *, limit=100) -> list[MemoryRecord]: ...
    def purge(self, namespace) -> int: ...
````

`MemoryRecord` is a frozen dataclass: `memory_id`, `namespace`, `text`, `kind`
(`fact`, `preference` or `note`), `created_by` (`model` or `application`),
`created_at` (UTC ISO 8601) and `run_id`.

`memory_tools(store, namespace, *, allow_write=True)` returns a `ToolRegistry`
whose entries the application merges into its own tools mapping:

| Tool | Model arguments | Classification |
| --- | --- | --- |
| `recall` | `query: str` | Read-only |
| `remember` | `text: str`, `kind: Literal["fact", "preference", "note"]` | Risky: approval and journal apply |
| `forget` | `memory_id: str` | Risky |

The namespace is bound by the application and is not a tool argument.
`allow_write=False` registers only `recall`.

## 4. Requirements

### ST-01 — Keep the core standard-library only

`import harness` MUST NOT import `psycopg`, `pymongo` or `stores`. Adapters MUST
import their driver only when constructed, and fail with a clear `ImportError`
naming the missing optional dependency. Existing defaults, file formats and the
2,000-line scale of `harness/` are preserved; adapters do not count toward it.

### ST-02 — Save sessions atomically, with one writer

- A `save` MUST be all-or-nothing: after a crash, `load` returns either the previous
  or the new history and compaction point, never a mix.
- Database stores MUST keep a revision and save only if the stored revision still
  matches the one they last loaded or saved. A mismatch MUST raise
  `SessionConflictError` (a `SessionStoreError`) and leave stored data unchanged.
- Other save failures MUST raise `SessionStoreError` without connection details.
  It subclasses `OSError`, so handlers written for file errors also catch it.
  Like a file save error today, it propagates out of `run_result`; no further model
  or tool calls happen in that task.
- `durable=True` (used when a journal is enabled) MUST mean the save survives a
  server crash once `save` returns. FR-09's ordering is kept: the session is saved
  durably before the journal run is closed.
- `load` MUST apply `validate_session` and raise `SessionLoadError` for invalid
  stored data, leaving it unchanged.
- Earlier messages can change between saves (steering text is added to the latest
  user turn), so stores MUST NOT assume history is append-only.

### ST-03 — Keep journal guarantees on every backend

Each journal MUST meet FR-08 exactly:

- `begin` commits a `started` record durably before dispatch. If `begin` fails or
  does not return, the tool does not run.
- Checking for blocking records and inserting the new one happen in one serialized
  transaction, so two concurrent `begin` calls cannot both pass the check.
- `finish`, `close_run` and `resolve` are durable, single-record transitions that
  fail when the record is not in the expected state.
- No arguments, outputs, exception details or credentials are stored.
- A journal never adopts storage with an unknown owner or a newer schema version.
- `inspect` is read-only and bounded; `resolve` requires confirmation and never runs
  a tool.

### ST-04 — Bound and scope memory

- Every `MemoryStore` call MUST filter by namespace; no call reads or changes
  another namespace.
- `text` MUST be 1–2,000 characters; `query` 1–500; `limit` 1–20 for `search` and
  1–1,000 for `list`. A namespace holds at most 10,000 records by default
  (configurable). `add` beyond the quota MUST fail without evicting anything.
- `search` MUST be lexical: it returns records sharing at least one word with the
  query, best matches first; ties go to newer records, then to `memory_id`. Ranking
  details may differ between backends; membership and limits may not.
- `delete` and `purge` MUST remove text from the store, not only hide it.

### ST-05 — Present memory to the model as untrusted evidence

- `recall` MUST return JSON `{"untrusted": true, "memories": [...]}` with
  `memory_id`, `kind`, `created_by`, `created_at` and `text`, within
  `max_tool_output`.
- `remember` and `forget` MUST be registered as risky, so policy, approval and the
  journal apply; `ToolPolicy(read_only=True)` blocks them.
- `remember` records `created_by="model"` and the current run ID. Applications
  writing directly through the store record `created_by="application"`.
- Memory text MUST NOT enter traces or events unless content is explicitly enabled.
- Memory tools run inline by default. Timed execution using a JSON-bound configuration
  and worker reconnection is planned for Phase 5; the current closure bindings cannot
  run in isolated workers. With `worker_env`, workers must receive the required variable.

### ST-06 — Own schemas explicitly

- Each backend records `{application: "little-harness", component, schema_version: 1}`
  for every component it creates (session, journal, memory).
- `prepare()` MAY create missing objects only when none of the expected names already
  exist with another owner or version. Otherwise it MUST fail, never adopt or alter.
- Explicit setup commands create schemas for deployments whose runtime role has no
  DDL rights: `python -m stores.postgres init --dsn-env NAME` and
  `python -m stores.mongodb init --uri-env NAME`. At runtime, `prepare()` then only
  checks.

### ST-07 — Handle connections and secrets safely

- Adapters accept either an application-created connection pool or client, or the
  *name* of an environment variable holding the DSN or URI. They never accept a
  connection string as a command-line argument.
- Driver errors MUST be mapped to `SessionStoreError`, `JournalError` or
  `MemoryStoreError` with fixed messages, never the driver's text (which can include
  hosts, user names or query fragments).
- Every operation MUST be time-bounded: connect at most 10 seconds and each statement
  or command at most 5 seconds by default, configurable. Retries happen only where
  section 5 allows them.
- Documentation MUST describe a least-privilege role (data access only) and TLS.

### ST-08 — Prove each backend with the same tests

- One conformance suite per interface runs against every backend. The standard-library
  backends always run.
- PostgreSQL and MongoDB runs happen only when `LITTLE_HARNESS_TEST_POSTGRES_DSN` or
  `LITTLE_HARNESS_TEST_MONGODB_URI` is set. Each test uses unique IDs and cleans up.
- `python3 -m unittest` stays offline and dependency-free. A separate CI job runs
  the database suites against `postgres:18` and a single-node `mongo:9.0` replica set.
- Crash tests kill a worker process between `begin` and `finish`, and during a session
  save, then check that recovery is required or that the previous revision loads.

### ST-09 — Keep human recovery available

`python3 -m harness.journal inspect|resolve PATH` is unchanged. Database journals
get the same subcommands with the same output and confirmation rules:
`python -m stores.postgres journal inspect --dsn-env NAME --journal-id ID`, and the
`stores.mongodb` equivalent.

## 5. Backend mapping

### 5.1 PostgreSQL 18

| Concern | Design |
| --- | --- |
| Layout | One schema (default `little_harness`). Tables `meta`, `sessions`, `session_messages`, `journals`, `journal_runs`, `journal_actions`, `memory_namespaces`, `memories` |
| Messages | One row per message, `(session_id, idx)` key, the message as JSON **text** plus its SHA-256. Text keeps the exact round trip; `jsonb` would reject `\u0000` and reorder keys |
| Session save | One transaction: lock the `sessions` row (`FOR UPDATE`), compare revision, find the first changed message by hash, delete from there, insert the rest, then update revision, count and compaction |
| Durability | `SET LOCAL synchronous_commit = on` in every write transaction, so a server-level `off` cannot weaken it |
| Journal serialization | `begin` and `resolve` lock the `journals` row for their journal ID (`SELECT … FOR UPDATE`), then check and insert |
| Memory search | A GIN-indexed array of lower-case ASCII words selects scoped candidates; application code ranks them by matching words, recency and ID |
| Retries | None for writes. A connection lost during `COMMIT` is reported as a failure; for the journal, that leaves at most a `started` record, which requires recovery |

### 5.2 MongoDB 9.0

| Concern | Design |
| --- | --- |
| Deployment | A replica set is required (a single node is enough), because sessions and the journal use multi-document transactions |
| Layout | One database (configurable); collections `lh_meta`, `lh_sessions`, `lh_session_messages`, `lh_journals`, `lh_journal_runs`, `lh_journal_actions`, `lh_memory_namespaces`, `lh_memories`, each with a `$jsonSchema` validator |
| Messages | One document per message, holding the message as a JSON **string**. Keys the model chooses (such as `$where` or `a.b`) never become BSON field names, and no session approaches the 16 MiB document limit |
| Session save | One transaction: compare-and-set `revision` on the `lh_sessions` document, delete from the first changed index, insert the rest |
| Durability | Write concern `{w: "majority", j: true}` and read concern `majority` for loads; transactions use `snapshot` |
| Journal serialization | `begin` and `resolve` increment a counter on the `lh_journals` document inside the transaction. MongoDB detects only write-write conflicts, so this write is what stops two concurrent `begin` calls from both passing the check |
| Memory search | A compound multikey index over store ID, namespace and lower-case ASCII word array selects candidates; application code ranks them by matching words, recency and ID |
| Retries | Only the driver's labelled cases: `TransientTransactionError` reruns the whole transaction (nothing was committed) and `UnknownTransactionCommitResult` retries the commit, both within the 5-second bound. If the outcome is still unknown, the call fails; a journal `begin` then authorizes nothing |
| Crashed writer | Transactions belong to logical sessions, not connections, so a process that dies mid-transaction leaves it open on the server, holding its uncommitted writes, until `transactionLifetimeLimitSeconds` (60 by default) or an operator's `killSessions`. Loads still return the last committed revision; saves and journal writes to the same session or journal fail closed with `SessionStoreError` or `JournalError` until then |
| Clients | One `MongoClient` (connection pool) per connection string per process, closed at exit; applications may pass their own `client=` instead |

## 6. Acceptance scenarios

Each scenario runs against every backend through the conformance suites (ST-08).

| ID / requirement | Scenario | Expected |
| --- | --- | --- |
| SAC-01 / ST-01 | Import `harness` without database drivers installed | Succeeds; constructing a database adapter names the missing optional dependency |
| SAC-02 / ST-02 | Save, change the latest user turn (steering), save, reload | Exactly the second history and compaction point |
| SAC-03 / ST-02 | Process killed during a session save | The previous or the new revision loads, never a mix |
| SAC-04 / ST-02 | Two agents load one session; both save | The second save raises `SessionConflictError`; stored data is the first save |
| SAC-05 / ST-02 | Stored session with invalid compaction state | `SessionLoadError`; data unchanged |
| SAC-06 / ST-03 | Process killed between `begin` and `finish` | The next task stops with `journal_recovery_required` before any model call |
| SAC-07 / ST-03 | Two concurrent `begin` calls for different runs on one journal | At most one succeeds; the other raises `JournalRecoveryRequired` |
| SAC-08 / ST-03, ST-06 | Journal pointed at storage with another owner or a newer schema | `JournalError`; nothing created or altered |
| SAC-09 / ST-04 | Same text added in two namespaces; search and purge one | Only that namespace's records are returned and removed |
| SAC-10 / ST-04 | Quota reached | `add` fails; no existing record is evicted |
| SAC-11 / ST-05 | `remember` under `ToolPolicy(read_only=True)`, and with approval denied | `not_run`; nothing stored; no journal action |
| SAC-12 / ST-05 | A stored memory contains an injected instruction | `recall` returns it inside `{"untrusted": true, …}`; traces hold no memory text by default |
| SAC-13 / ST-07 | Database unreachable or credentials wrong | Fixed error message without host, user or query text, within the time bound |

## 7. Decisions

Settled in Phase 0 of the [plan](STORAGE_PLAN.md):

1. **Dependency declaration:** `pyproject.toml` with no runtime dependencies and two
   optional extras, `postgres` (`psycopg[binary]>=3.3,<4`) and `mongodb`
   (`pymongo>=4.19,<5`), locked in `uv.lock`. Install with
   `uv sync --extra postgres --extra mongodb`.
2. **Package location:** `stores/` inside this repository, not published separately.
3. **Memory limits:** as in ST-04: 2,000 characters per memory, 10,000 memories per
   namespace by default.
4. **Local and CI databases:** `compose.yaml` (development only) and
   `.github/workflows/storage.yml`.

Still open: MongoDB Atlas Search and Vector Search could serve a later vector
revision; revision 1 uses a standard multikey index, so it runs on a local
replica set.

