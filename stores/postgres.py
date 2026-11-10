"""PostgreSQL 18 session, journal and memory stores (docs/STORAGE_SPEC.md, section 5.1).

Needs the optional extra: uv sync --extra postgres. Connection settings come from an
environment variable named by the application, or from an application-supplied
connect() callable; they never appear in errors, traces or command lines.

  python -m stores.postgres init --dsn-env NAME [--schema NAME]
  python -m stores.postgres journal inspect --dsn-env NAME --journal-id ID [--pending]
  python -m stores.postgres journal resolve --dsn-env NAME --journal-id ID ACTION_ID --outcome O --confirm
"""

import hashlib
import json
import os
import re
from contextlib import contextmanager
from uuid import uuid4

from harness.journal import (JournalError, JournalRecoveryRequired, check_inspect_limit, check_outcome,
                             check_resolution, inspection_record)
from harness.memory import MemoryRecord, MemoryStoreError, _limit, _namespace, _rank, _text, _validate_add, _WORDS
from harness.session import (SESSION_VERSION, SessionConflictError, SessionLoadError, SessionStoreError,
                             validate_session)
from stores._common import APPLICATION, SCHEMA_VERSION, check_id, storage_main

try:
    import psycopg
    from psycopg import sql
    from psycopg.rows import dict_row
except ImportError:
    psycopg = None

CONNECT_TIMEOUT_SECONDS = 10
SCHEMA_NAME = re.compile(r"[a-z_][a-z0-9_]{0,62}")
TABLES = {"session": ("sessions", "session_messages"),
          "journal": ("journals", "journal_runs", "journal_actions"),
          "memory": ("memory_namespaces", "memories")}
DDL = {
    "session": """
        CREATE TABLE {schema}.sessions (
            session_id text PRIMARY KEY, revision bigint NOT NULL, message_count integer NOT NULL,
            compaction text, updated_at timestamptz NOT NULL DEFAULT now());
        CREATE TABLE {schema}.session_messages (
            session_id text NOT NULL REFERENCES {schema}.sessions ON DELETE CASCADE,
            idx integer NOT NULL, message text NOT NULL, sha256 bytea NOT NULL,
            PRIMARY KEY (session_id, idx));""",
    "journal": """
        CREATE TABLE {schema}.journals (journal_id text PRIMARY KEY);
        CREATE TABLE {schema}.journal_runs (
            journal_id text NOT NULL REFERENCES {schema}.journals, run_id text NOT NULL,
            closed boolean NOT NULL DEFAULT false, PRIMARY KEY (journal_id, run_id));
        CREATE TABLE {schema}.journal_actions (
            action_id text PRIMARY KEY, journal_id text NOT NULL, run_id text NOT NULL,
            tool_call_id text NOT NULL, tool_name text NOT NULL, risky boolean NOT NULL,
            started_at timestamptz NOT NULL, finished_at timestamptz, status text NOT NULL,
            outcome_unknown boolean NOT NULL DEFAULT false, resolution text, resolved_at timestamptz,
            FOREIGN KEY (journal_id, run_id) REFERENCES {schema}.journal_runs);
        CREATE INDEX ON {schema}.journal_actions (journal_id) WHERE resolution IS NULL;""",
    "memory": """
        CREATE TABLE {schema}.memory_namespaces (
            store_id text NOT NULL, namespace text NOT NULL, PRIMARY KEY (store_id, namespace));
        CREATE TABLE {schema}.memories (
            memory_id text PRIMARY KEY, store_id text NOT NULL, namespace text NOT NULL,
            text text NOT NULL, kind text NOT NULL, created_by text NOT NULL,
            created_at timestamptz NOT NULL, run_id text, words text[] NOT NULL,
            FOREIGN KEY (store_id, namespace) REFERENCES {schema}.memory_namespaces);
        CREATE INDEX ON {schema}.memories (store_id, namespace, created_at DESC, memory_id);
        CREATE INDEX ON {schema}.memories USING gin (words);"""}
PENDING = "a.resolution IS NULL AND (NOT r.closed OR a.outcome_unknown OR a.status = 'started')"
ACTIONS = """{schema}.journal_actions a JOIN {schema}.journal_runs r USING (journal_id, run_id)"""


class _PostgresStorage:
    """Connections, time bounds, durable commits and schema ownership (ST-06, ST-07)."""

    component = None

    def __init__(self, name, *, dsn_env, connect, schema, statement_timeout):
        if psycopg is None:
            raise ImportError("PostgreSQL storage needs the optional 'postgres' extra: uv sync --extra postgres")
        check_id(name)
        if (dsn_env is None) == (connect is None):
            raise TypeError("pass exactly one of dsn_env (a variable name) or connect (a callable)")
        if connect is not None and not callable(connect):
            raise TypeError("connect must be a callable returning a new psycopg connection")
        if dsn_env is not None and not os.environ.get(dsn_env):
            raise ValueError(f"environment variable {dsn_env} is not set")
        if type(schema) is not str or not SCHEMA_NAME.fullmatch(schema):
            raise ValueError("schema must be a lower-case SQL identifier")
        if type(statement_timeout) not in (int, float) or not 0.001 <= statement_timeout <= 300:
            raise ValueError("statement_timeout must be between 0.001 and 300 seconds")
        self.name, self.schema, self._dsn_env, self._connect = name, schema, dsn_env, connect
        self._timeout_ms = str(int(statement_timeout * 1000))
        self._ready = False

    def _error(self):
        raise NotImplementedError

    def _sql(self, text):
        return sql.SQL(text).format(schema=sql.Identifier(self.schema))

    def _open(self):
        if self._connect is not None:
            return self._connect()
        return psycopg.connect(os.environ[self._dsn_env], connect_timeout=CONNECT_TIMEOUT_SECONDS)

    @contextmanager
    def _transaction(self, *, read_only=False):
        """One transaction per operation; driver errors become fixed messages (ST-07)."""
        try:
            connection = self._open()
        except (psycopg.Error, KeyError, OSError):
            raise self._error() from None
        try:
            with connection:
                connection.row_factory = dict_row
                if read_only:
                    connection.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
                    connection.read_only = True
                # Transaction-local settings: bounded statements, UTC times, and durable commits
                # even if the server or role defaults synchronous_commit to off.
                connection.execute("SELECT set_config('statement_timeout', %s, true), "
                                   "set_config('TimeZone', 'UTC', true), "
                                   "set_config('synchronous_commit', 'on', true)", (self._timeout_ms,))
                yield connection
        except psycopg.Error:
            raise self._error() from None

    def _exists(self, connection, table):
        return connection.execute("SELECT to_regclass(format('%%I.%%I', %s::text, %s::text)) AS found",
                                  (self.schema, table)).fetchone()["found"] is not None

    def _check_schema(self, connection, *, create):
        """Use only tables this component recorded; create them only where none exist."""
        key = int.from_bytes(hashlib.sha256(f"{APPLICATION}:{self.schema}".encode()).digest()[:8], "big", signed=True)
        connection.execute("SELECT pg_advisory_xact_lock(%s)", (key,))
        if self._exists(connection, "meta"):
            row = connection.execute(self._sql("SELECT application, schema_version FROM {schema}.meta "
                                               "WHERE component = %s"), (self.component,)).fetchone()
            if row is not None:
                if (row["application"], row["schema_version"]) != (APPLICATION, SCHEMA_VERSION):
                    raise self._error()
                return
        if not create or any(self._exists(connection, table) for table in TABLES[self.component]):
            raise self._error()
        connection.execute(self._sql("CREATE SCHEMA IF NOT EXISTS {schema}"))
        connection.execute(self._sql("CREATE TABLE IF NOT EXISTS {schema}.meta (component text PRIMARY KEY, "
                                     "application text NOT NULL, schema_version integer NOT NULL)"))
        connection.execute(self._sql(DDL[self.component]))
        connection.execute(self._sql("INSERT INTO {schema}.meta VALUES (%s, %s, %s)"),
                           (self.component, APPLICATION, SCHEMA_VERSION))

    def ensure_schema(self, *, create=True):
        """Check this component's tables, creating them in an empty schema when allowed."""
        with self._transaction() as connection:
            self._check_schema(connection, create=create)

    def prepare(self):
        with self._transaction() as connection:
            self._check_schema(connection, create=True)
            self._register(connection)
        self._ready = True

    def _register(self, connection):
        pass


class PostgresSessionStore(_PostgresStorage):
    """A SessionStore: one row per message, rewritten only from the first changed message.

    Every save is durable (synchronous commit), whatever `durable` asks for, and is
    refused with SessionConflictError if another writer saved since this handle's load.
    """

    component = "session"

    def __init__(self, session_id, *, dsn_env=None, connect=None, schema="little_harness", statement_timeout=5):
        super().__init__(session_id, dsn_env=dsn_env, connect=connect, schema=schema,
                         statement_timeout=statement_timeout)
        self.revision, self._hashes = 0, []

    def __str__(self):
        return f"PostgreSQL session {self.name}"

    def _error(self):
        return SessionStoreError("PostgreSQL session operation failed; inspect the session before continuing")

    def load(self):
        if not self._ready:
            self.prepare()
        with self._transaction(read_only=True) as connection:
            head = connection.execute(self._sql("SELECT revision, message_count, compaction FROM {schema}.sessions "
                                                "WHERE session_id = %s"), (self.name,)).fetchone()
            rows = [] if head is None else connection.execute(self._sql(
                "SELECT idx, message, sha256 FROM {schema}.session_messages WHERE session_id = %s ORDER BY idx"),
                (self.name,)).fetchall()
        if head is None:
            self.revision, self._hashes = 0, []
            return None
        try:
            if [row["idx"] for row in rows] != list(range(head["message_count"])):
                raise ValueError("stored messages are incomplete")
            data = {"version": SESSION_VERSION, "messages": [json.loads(row["message"]) for row in rows],
                    "compaction": None if head["compaction"] is None else json.loads(head["compaction"])}
        except ValueError as e:
            raise SessionLoadError(f"{self} is not a valid session ({e}); fix or delete it to start fresh") from None
        loaded = validate_session(data, self)
        self.revision, self._hashes = head["revision"], [bytes(row["sha256"]) for row in rows]
        return loaded

    def save(self, messages, compaction, *, durable=True):
        # JSON text, not jsonb: exact round trips, and \u0000 stays an escape (ST-02, spec 5.1).
        texts = [json.dumps(message) for message in messages]
        hashes = [hashlib.sha256(text.encode("utf-8")).digest() for text in texts]
        start = next((index for index, (old, new) in enumerate(zip(self._hashes, hashes)) if old != new),
                     min(len(self._hashes), len(hashes)))
        if not self._ready:
            self.prepare()
        with self._transaction() as connection:
            connection.execute(self._sql("INSERT INTO {schema}.sessions (session_id, revision, message_count) "
                                         "VALUES (%s, 0, 0) ON CONFLICT DO NOTHING"), (self.name,))
            stored = connection.execute(self._sql("SELECT revision FROM {schema}.sessions WHERE session_id = %s "
                                                  "FOR UPDATE"), (self.name,)).fetchone()["revision"]
            if stored != self.revision:
                raise SessionConflictError(f"{self} changed since it was loaded; nothing was saved")
            connection.execute(self._sql("DELETE FROM {schema}.session_messages WHERE session_id = %s AND idx >= %s"),
                               (self.name, start))
            if start < len(texts):
                with connection.cursor() as cursor:
                    cursor.executemany(self._sql("INSERT INTO {schema}.session_messages VALUES (%s, %s, %s, %s)"),
                                       [(self.name, index, texts[index], hashes[index])
                                        for index in range(start, len(texts))])
            connection.execute(self._sql("UPDATE {schema}.sessions SET revision = revision + 1, message_count = %s, "
                                         "compaction = %s, updated_at = now() WHERE session_id = %s"),
                               (len(texts), None if compaction is None else json.dumps(compaction), self.name))
        self.revision, self._hashes = stored + 1, hashes


class PostgresMemoryStore(_PostgresStorage):
    """Scoped memory with serialized per-namespace quota checks and indexed word lookup."""

    component = "memory"

    def __init__(self, store_id, *, dsn_env=None, connect=None, schema="little_harness", statement_timeout=5,
                 quota=10000):
        if type(quota) is not int or quota < 1:
            raise ValueError("quota must be a positive integer")
        super().__init__(store_id, dsn_env=dsn_env, connect=connect, schema=schema,
                         statement_timeout=statement_timeout)
        self.quota = quota

    def _error(self):
        return MemoryStoreError()

    def _lock_namespace(self, connection, namespace):
        connection.execute(self._sql("INSERT INTO {schema}.memory_namespaces VALUES (%s, %s) "
                                     "ON CONFLICT DO NOTHING"), (self.name, namespace))
        connection.execute(self._sql("SELECT 1 FROM {schema}.memory_namespaces "
                                     "WHERE store_id = %s AND namespace = %s FOR UPDATE"), (self.name, namespace))

    def _ensure(self):
        if not self._ready:
            self.prepare()

    @staticmethod
    def _record(row):
        return MemoryRecord(row["memory_id"], row["namespace"], row["text"], row["kind"], row["created_by"],
                            row["created_at"].isoformat(), row["run_id"])

    def add(self, namespace, text, *, kind, created_by, run_id=None):
        _validate_add(namespace, text, kind, created_by, run_id)
        self._ensure()
        memory_id = uuid4().hex
        with self._transaction() as connection:
            self._lock_namespace(connection, namespace)
            count = connection.execute(self._sql("SELECT count(*) AS n FROM {schema}.memories "
                                                 "WHERE store_id = %s AND namespace = %s"),
                                       (self.name, namespace)).fetchone()["n"]
            if count >= self.quota:
                raise MemoryStoreError()
            connection.execute(self._sql("INSERT INTO {schema}.memories VALUES "
                                         "(%s, %s, %s, %s, %s, %s, clock_timestamp(), %s, %s)"),
                               (memory_id, self.name, namespace, text, kind, created_by, run_id,
                                sorted(set(_WORDS.findall(text.lower())))))
        return memory_id

    def search(self, namespace, query, *, limit=5):
        _namespace(namespace)
        _text(query, 500, "query")
        _limit(limit, 20)
        words = set(_WORDS.findall(query.lower()))
        self._ensure()
        with self._transaction(read_only=True) as connection:
            rows = connection.execute(self._sql("SELECT * FROM {schema}.memories WHERE store_id = %s "
                                                "AND namespace = %s AND words && %s"),
                                      (self.name, namespace, list(words))).fetchall()
        return _rank([self._record(row) for row in rows], words, limit)

    def list(self, namespace, *, limit=100):
        _namespace(namespace)
        _limit(limit, 1000)
        self._ensure()
        with self._transaction(read_only=True) as connection:
            rows = connection.execute(self._sql("SELECT * FROM {schema}.memories WHERE store_id = %s "
                                                "AND namespace = %s ORDER BY created_at DESC, memory_id LIMIT %s"),
                                      (self.name, namespace, limit)).fetchall()
        return [self._record(row) for row in rows]

    def delete(self, namespace, memory_id):
        _namespace(namespace)
        _text(memory_id, 200, "memory_id")
        self._ensure()
        with self._transaction() as connection:
            self._lock_namespace(connection, namespace)
            return bool(connection.execute(self._sql("DELETE FROM {schema}.memories WHERE store_id = %s "
                                                      "AND namespace = %s AND memory_id = %s"),
                                           (self.name, namespace, memory_id)).rowcount)

    def purge(self, namespace):
        _namespace(namespace)
        self._ensure()
        with self._transaction() as connection:
            self._lock_namespace(connection, namespace)
            return connection.execute(self._sql("DELETE FROM {schema}.memories WHERE store_id = %s AND namespace = %s"),
                                      (self.name, namespace)).rowcount


class PostgresJournal(_PostgresStorage):
    """A durable ActionJournal (FR-08, ST-03).

    begin and resolve lock this journal's row first, so concurrent calls check and
    write one at a time. No retries: a failed or
    ambiguous commit raises JournalError, so no tool is dispatched.
    """

    component = "journal"

    def __init__(self, journal_id, *, dsn_env=None, connect=None, schema="little_harness", statement_timeout=5):
        super().__init__(journal_id, dsn_env=dsn_env, connect=connect, schema=schema,
                         statement_timeout=statement_timeout)

    def __str__(self):
        return f"PostgreSQL journal {self.name}"

    def _error(self):
        return JournalError()

    def _register(self, connection):
        connection.execute(self._sql("INSERT INTO {schema}.journals VALUES (%s) ON CONFLICT DO NOTHING"), (self.name,))

    def _lock(self, connection):
        if connection.execute(self._sql("SELECT 1 FROM {schema}.journals WHERE journal_id = %s FOR UPDATE"),
                              (self.name,)).fetchone() is None:
            raise JournalError()

    def assert_ready(self):
        self.prepare()
        with self._transaction(read_only=True) as connection:
            if connection.execute(self._sql(f"SELECT 1 FROM {ACTIONS} WHERE a.journal_id = %s AND {PENDING} LIMIT 1"),
                                  (self.name,)).fetchone():
                raise JournalRecoveryRequired()

    def begin(self, run_id, tool_call_id, tool_name, risky):
        """Commit before dispatch; an unreturned begin never authorizes execution."""
        action_id = uuid4().hex
        with self._transaction() as connection:
            self._lock(connection)
            if connection.execute(self._sql(f"""SELECT 1 FROM {ACTIONS} WHERE a.journal_id = %s
                    AND a.resolution IS NULL AND (a.status = 'started' OR a.outcome_unknown
                        OR (NOT r.closed AND a.run_id <> %s)) LIMIT 1"""), (self.name, run_id)).fetchone():
                raise JournalRecoveryRequired()
            connection.execute(self._sql("INSERT INTO {schema}.journal_runs (journal_id, run_id) VALUES (%s, %s) "
                                         "ON CONFLICT DO NOTHING"), (self.name, run_id))
            if connection.execute(self._sql("SELECT closed FROM {schema}.journal_runs WHERE journal_id = %s "
                                            "AND run_id = %s"), (self.name, run_id)).fetchone()["closed"]:
                raise JournalError()
            connection.execute(self._sql("""INSERT INTO {schema}.journal_actions
                (action_id, journal_id, run_id, tool_call_id, tool_name, risky, started_at, status)
                VALUES (%s, %s, %s, %s, %s, %s, clock_timestamp(), 'started')"""),
                               (action_id, self.name, run_id, tool_call_id, tool_name, bool(risky)))
        return action_id

    def finish(self, action_id, status, outcome_unknown=False):
        check_outcome(status, outcome_unknown)
        with self._transaction() as connection:
            updated = connection.execute(self._sql("""UPDATE {schema}.journal_actions
                SET status = %s, outcome_unknown = %s, finished_at = clock_timestamp()
                WHERE journal_id = %s AND action_id = %s AND status = 'started' AND resolution IS NULL"""),
                                         (status, bool(outcome_unknown), self.name, action_id)).rowcount
            if updated != 1:
                raise JournalError()

    def close_run(self, run_id):
        """Called only after the harness has saved a coherent terminal conversation."""
        with self._transaction() as connection:
            connection.execute(self._sql("UPDATE {schema}.journal_runs SET closed = true "
                                         "WHERE journal_id = %s AND run_id = %s"), (self.name, run_id))

    def inspect(self, *, pending_only=False, limit=100):
        """Read-only, bounded metadata; never creates tables, journals or records."""
        check_inspect_limit(limit)
        with self._transaction(read_only=True) as connection:
            self._check_schema(connection, create=False)
            if connection.execute(self._sql("SELECT 1 FROM {schema}.journals WHERE journal_id = %s"),
                                  (self.name,)).fetchone() is None:
                raise JournalError()
            rows = connection.execute(self._sql(f"""SELECT a.action_id, a.run_id, a.tool_call_id, a.tool_name,
                    a.risky, a.started_at, a.finished_at, a.status, a.outcome_unknown, a.resolution, a.resolved_at,
                    r.closed AS run_closed, ({PENDING}) AS recovery_required
                FROM {ACTIONS} WHERE a.journal_id = %s {"AND " + PENDING if pending_only else ""}
                ORDER BY a.started_at DESC, a.action_id DESC LIMIT %s"""), (self.name, limit)).fetchall()
        records = []
        for row in rows:
            for name in ("started_at", "finished_at", "resolved_at"):
                row[name] = row[name].isoformat() if row[name] is not None else None
            records.append(inspection_record(row))
        return records

    def resolve(self, action_id, outcome, *, confirm=False):
        """Human assertion, not proof, approval, undo, session repair or a retry."""
        check_resolution(outcome, confirm)
        with self._transaction() as connection:
            self._lock(connection)
            if not connection.execute(self._sql(f"SELECT 1 FROM {ACTIONS} WHERE a.journal_id = %s "
                                                f"AND a.action_id = %s AND {PENDING}"), (self.name, action_id)).fetchone():
                raise ValueError("action does not require recovery")
            connection.execute(self._sql("UPDATE {schema}.journal_actions SET resolution = %s, "
                                         "resolved_at = clock_timestamp() WHERE journal_id = %s AND action_id = %s"),
                               (outcome, self.name, action_id))


def main(argv=None):
    storage_main(argv, description=__doc__, backend="PostgreSQL", env_option="--dsn-env",
                 location_option="--schema", default_location="little_harness",
                 session_store=lambda name, env, schema: PostgresSessionStore(name, dsn_env=env, schema=schema),
                 journal=lambda name, env, schema: PostgresJournal(name, dsn_env=env, schema=schema),
                 memory_store=lambda name, env, schema: PostgresMemoryStore(name, dsn_env=env, schema=schema))


if __name__ == "__main__":
    main()

