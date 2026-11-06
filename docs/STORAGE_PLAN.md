# Little Harness storage plan and checkpoints

Companion: [storage specification](STORAGE_SPEC.md). Each phase is a bounded
checkpoint with its own gate; no phase changes default behavior or file formats.

**Current state: Phase 0 complete; Phase 1 next.** Python 3.11+ is in place.

Every gate includes: `python3 -m unittest -q` offline on Python 3.11 and 3.14 with no
third-party packages installed, CI green on 3.11–3.14, docs updated with the code,
and the HotpotQA fingerprint list updated for any new `harness/*.py` file.

## Phase 0 — decisions and dependency approval

1. Settle the open decisions in spec section 7: how dependencies are declared, the
   package name, memory limits.
2. Get explicit approval to add the two optional drivers, then add them with the
   package manager. Nothing in `harness/` depends on them.
3. Add a `docker compose` file (or documented commands) for `postgres:18` and a
   single-node `mongo:9.0` replica set, for local development only.
4. Add a separate, manually or label-triggered CI workflow `storage.yml` with both
   services; the existing `tests.yml` stays dependency-free.

Gate: decisions recorded in the spec; the new workflow starts both services and
checks them successfully.

### Phase 0 checkpoint — complete

- Decisions recorded in [spec section 7](STORAGE_SPEC.md#7-decisions).
- `pyproject.toml` declares no runtime dependencies and two optional extras, added
  with `uv add --optional`: `postgres` (`psycopg[binary]>=3.3,<4`) and `mongodb`
  (`pymongo>=4.19,<5`). `uv.lock` pins them (psycopg 3.3.8, PyMongo 4.19.0).
  `tools/package_runtime.py` reads its version and Python requirement from it.
- `compose.yaml` starts `postgres:18` and a single-node `mongo:9.0` replica set on
  `127.0.0.1:55432` and `127.0.0.1:57017` (configurable), with no passwords or volumes.
- `tools/check_storage_services.py` prints only server versions; verified locally
  against PostgreSQL 18.6 and MongoDB 9.0.4, including a majority-acknowledged
  MongoDB transaction.
- `.github/workflows/storage.yml` runs the check on Python 3.11 and 3.14 when
  storage files change, or on demand. `tests.yml` is unchanged and dependency-free.
- `python3 -m unittest -q`: 441 tests pass on Python 3.11 and 3.14 (2 skipped).

## Phase 1 — interfaces in the core (standard library only)

1. `harness/session.py`: `SessionStore` protocol, shared `validate_session`,
   `SessionStoreError` and `SessionConflictError`. `SessionFile` implements the protocol.
2. `harness/journal.py`: `ActionJournal` protocol; `ExecutionJournal` implements it.
3. `Agent(session_store=..., journal=...)`, each mutually exclusive with the existing
   path argument. The reserved-path checks apply only to file-based stores.
4. `tests/storage_conformance.py`: `SessionStoreConformanceTests` and
   `JournalConformanceTests`, run against `SessionFile` and `ExecutionJournal`.
5. Update `HARNESS_SPEC.md` (FR-08, FR-09, section 3.1), `REFERENCE.md` and
   `ARCHITECTURE.md`.

Gate: every existing test passes unchanged; conformance suites pass on the
standard-library backends; SAC-01, SAC-02, SAC-05 pass; `harness/` grows by no more
than about 80 lines.

## Phase 2 — PostgreSQL 18 sessions and journal

1. `stores/postgres.py`: `PostgresSessionStore`, `PostgresJournal`, the `init` and
   `journal inspect|resolve` commands, schema ownership checks (ST-06).
2. Error mapping and time bounds (ST-07), including a test that driver error text
   never reaches messages, traces or events.
3. Run both conformance suites against PostgreSQL; add crash tests (SAC-03, SAC-06)
   and the concurrency tests (SAC-04, SAC-07) using two processes.
4. Docs: setup, least-privilege role, TLS, recovery commands.

Gate: SAC-01 to SAC-08 and SAC-13 pass against PostgreSQL 18 in `storage.yml`; the
offline suite is unchanged.

## Phase 3 — MongoDB 9.0 sessions and journal

1. `stores/mongodb.py`: `MongoSessionStore`, `MongoJournal`, collection validators,
   `init` and `journal inspect|resolve` commands.
2. Transactions with majority write concern, the journal-document counter for
   serialization, and only the driver-labelled retries (spec 5.2).
3. Same conformance, crash and concurrency tests as Phase 2, plus a test that model
   tool inputs with `$` or `.` keys round-trip exactly.
4. Docs: replica set requirement, setup, role, TLS.

Gate: SAC-01 to SAC-08 and SAC-13 pass against MongoDB 9.0 in `storage.yml`.

## Phase 4 — memory in the core

1. `harness/memory.py`: `MemoryStore` protocol, `MemoryRecord`, `MemoryStoreError`,
   `memory_tools(...)` and a standard-library SQLite reference store.
2. `MemoryStoreConformanceTests`, run against the SQLite reference.
3. A support-assistant example or lesson showing `recall` and an approved `remember`,
   with `ToolPolicy(read_only=True)` blocking writes.
4. Evaluation: memory-poisoning cases in the support evaluation, where a stored
   memory carries an injected instruction (SAC-12).
5. Docs: `HARNESS_SPEC.md` (new FR), `REFERENCE.md`, `ARCHITECTURE.md`, `LEARN.md`
   if a lesson is added.

Gate: SAC-09 to SAC-12 pass on SQLite; no memory text in default traces or events;
`harness/` stays at roughly 2,200 lines or fewer.

## Phase 5 — PostgreSQL 18 and MongoDB 9.0 memory

1. `PostgresMemoryStore` (stored generated `tsvector` column, GIN index) and
   `MongoMemoryStore` (compound text index with a namespace prefix).
2. Memory conformance suite against both; namespace isolation, quota and purge tests.
3. Timed memory tools via a JSON-bound configuration, with `worker_env` passing only
   the connection variable.

Gate: SAC-09 to SAC-13 pass on both databases in `storage.yml`.

## Phase 6 — review and release notes

1. Re-read `STORAGE_SPEC.md` against the code; change its status to as-built for the
   delivered phases.
2. Update `README.md` (optional controls, further reading) and the blog post if wanted.
3. Record measured overhead per save, `begin` and `search` on both databases, as
   information rather than a guarantee.

## Risks

| Risk | Mitigation |
| --- | --- |
| Concurrent `begin` calls both pass the check | Explicit row lock (PostgreSQL) or counter write (MongoDB); two-process tests in SAC-07 |
| Ambiguous commit outcome | Journal fails closed: no dispatch, and a possible `started` record requires recovery |
| Driver errors leak hosts or credentials | Fixed error messages; tests that search outputs for the DSN and host |
| Model-written memory poisons later sessions | Writes are risky and approved; `recall` marks results untrusted; poisoning evaluation cases |
| History rewrites make saves slow | Message-per-row storage, rewriting only from the first changed message |
| Core grows past its teaching size | Adapters live in `stores/`; line budgets in the Phase 1 and 4 gates |
