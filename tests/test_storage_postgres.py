"""PostgreSQL 18 adapters against the shared storage contracts (docs/STORAGE_SPEC.md).

Database tests run only when LITTLE_HARNESS_TEST_POSTGRES_DSN is set and the
postgres extra is installed; each test uses its own schema and drops it.
"""

import json
import os
import subprocess
import sys
import textwrap
import threading
import time
import unittest
from unittest import mock
from uuid import uuid4

from harness import Agent, JournalError, JournalRecoveryRequired, MemoryStoreError, SessionStoreError
from stores import postgres
from stores.postgres import PostgresJournal, PostgresMemoryStore, PostgresSessionStore
from tests.helpers import REPO, RecordingModel, TemporaryDirectoryTestCase, reply
from tests.storage_conformance import (JournalConformanceTests, MemoryStoreConformanceTests,
                                       SessionStoreConformanceTests, history)

ENV = "LITTLE_HARNESS_TEST_POSTGRES_DSN"
needs_driver = unittest.skipIf(postgres.psycopg is None, "install the postgres extra")
needs_database = unittest.skipUnless(postgres.psycopg is not None and os.environ.get(ENV),
                                     f"set {ENV} and install the postgres extra")


class SchemaPerTest:
    def setUp(self):
        super().setUp()
        self.schema = "lh_test_" + uuid4().hex[:12]
        self.addCleanup(self.drop_schema)

    def sql(self, text, params=()):
        with postgres.psycopg.connect(os.environ[ENV]) as connection:
            cursor = connection.execute(postgres.sql.SQL(text).format(
                schema=postgres.sql.Identifier(self.schema)), params)
            return cursor.fetchall() if cursor.description else None

    def drop_schema(self):
        with postgres.psycopg.connect(os.environ[ENV]) as connection:
            connection.execute(postgres.sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                postgres.sql.Identifier(self.schema)))

    def run_child(self, code):
        """A separate process that may exit abruptly (os._exit) in the middle of an operation."""
        prelude = f"import os, psycopg\nENV, SCHEMA = {ENV!r}, {self.schema!r}\n"
        return subprocess.run([sys.executable, "-c", prelude + textwrap.dedent(code)], cwd=REPO,
                              env={"PATH": os.defpath, "PYTHONPATH": str(REPO), ENV: os.environ[ENV]},
                              capture_output=True, text=True, timeout=30)


@needs_database
class PostgresSessionStoreTests(SchemaPerTest, SessionStoreConformanceTests, TemporaryDirectoryTestCase):
    detects_conflicts = True

    def make_store(self, name):
        return PostgresSessionStore(name, dsn_env=ENV, schema=self.schema)

    def store_raw(self, name, data):
        # save() writes compaction state as given; only load() validates it.
        self.make_store(name).save(data["messages"], data["compaction"], durable=False)

    def test_process_killed_during_a_save_leaves_the_previous_revision(self):
        self.make_store("crash").save(history("one", "two"), None, durable=True)
        child = self.run_child("""
            from stores.postgres import PostgresSessionStore
            store = PostgresSessionStore("crash", dsn_env=ENV, schema=SCHEMA)
            store.load()
            original = psycopg.Cursor.executemany
            def insert_then_die(self, *args, **kwargs):
                original(self, *args, **kwargs)
                os._exit(3)  # Rows were deleted and inserted, but never committed.
            psycopg.Cursor.executemany = insert_then_die
            store.save([{"role": "user", "content": "one"}, {"role": "assistant", "content": "CHANGED"},
                        {"role": "user", "content": "three"}], None, durable=True)
        """)
        self.assertEqual(child.returncode, 3, child.stderr)
        store = self.make_store("crash")
        self.assertEqual(store.load(), (history("one", "two"), None))
        store.save(history("one", "two", "three"), None, durable=True)
        self.assertEqual(self.make_store("crash").load()[0], history("one", "two", "three"))

    def test_messages_before_the_first_change_are_not_rewritten(self):
        store = self.make_store("rewrite")
        store.save(history("one", "two", "three"), None, durable=False)
        before = dict(self.sql("SELECT idx, xmin::text FROM {schema}.session_messages"))
        store.save(history("one", "two", "THREE", "four"), None, durable=False)
        after = dict(self.sql("SELECT idx, xmin::text FROM {schema}.session_messages"))
        self.assertEqual((after[0], after[1]), (before[0], before[1]))
        self.assertNotEqual(after[2], before[2])
        self.assertEqual(sorted(after), [0, 1, 2, 3])


@needs_database
class PostgresMemoryTests(SchemaPerTest, MemoryStoreConformanceTests, TemporaryDirectoryTestCase):
    memory_error = MemoryStoreError

    def make_memory_store(self, name, *, quota=10000):
        return PostgresMemoryStore(name, dsn_env=ENV, schema=self.schema, quota=quota)

    def test_concurrent_adds_share_namespace_quota(self):
        store = self.make_memory_store("quota", quota=1)
        store.prepare()
        barrier = threading.Barrier(2)
        outcomes = []

        def add(value):
            barrier.wait(timeout=5)
            try:
                outcomes.append(self.make_memory_store("quota", quota=1).add(
                    "n", value, kind="note", created_by="application"))
            except MemoryStoreError:
                outcomes.append(None)

        threads = [threading.Thread(target=add, args=(value,)) for value in ("first", "second")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
            self.assertFalse(thread.is_alive())
        self.assertEqual(len(outcomes), 2)
        self.assertEqual(sum(value is not None for value in outcomes), 1)
        self.assertEqual(len(store.list("n")), 1)

    def test_unowned_tables_are_not_adopted(self):
        self.sql("CREATE SCHEMA {schema}; CREATE TABLE {schema}.memories (unrelated text)")
        with self.assertRaises(MemoryStoreError):
            self.make_memory_store("unowned").list("n")
        self.assertEqual(self.sql("SELECT count(*) FROM {schema}.memories")[0][0], 0)


@needs_database
class PostgresJournalTests(SchemaPerTest, JournalConformanceTests, TemporaryDirectoryTestCase):
    def make_journal(self, name):
        return PostgresJournal(name, dsn_env=ENV, schema=self.schema)

    def make_foreign(self, name):
        self.make_journal(name).prepare()
        self.sql("UPDATE {schema}.meta SET schema_version = 99 WHERE component = 'journal'")
        return self.make_journal(name)

    def test_tables_without_an_ownership_record_are_never_adopted(self):
        self.sql("CREATE SCHEMA {schema}; CREATE TABLE {schema}.journal_actions (unrelated text)")
        with self.assertRaises(JournalError):
            self.make_journal("unowned").prepare()
        self.assertEqual(self.sql("SELECT count(*) FROM {schema}.journal_actions"), [(0,)])

    def test_begin_waits_for_a_concurrent_begin_and_then_sees_its_open_run(self):
        self.make_journal("serial").prepare()
        holder = postgres.psycopg.connect(os.environ[ENV])
        self.addCleanup(holder.close)
        query = lambda text: postgres.sql.SQL(text).format(schema=postgres.sql.Identifier(self.schema))
        holder.execute(query("SELECT 1 FROM {schema}.journals WHERE journal_id = 'serial' FOR UPDATE"))
        outcome = {}

        def second_begin():
            try:
                self.make_journal("serial").begin("run-b", "call-1", "probe", False)
                outcome["result"] = "began"
            except JournalRecoveryRequired:
                outcome["result"] = "blocked"

        thread = threading.Thread(target=second_begin)
        thread.start()
        time.sleep(0.5)
        self.assertTrue(thread.is_alive(), "begin must wait for the journal row lock")
        holder.execute(query("INSERT INTO {schema}.journal_runs (journal_id, run_id) VALUES ('serial', 'run-a')"))
        holder.execute(query("""INSERT INTO {schema}.journal_actions (action_id, journal_id, run_id, tool_call_id,
            tool_name, risky, started_at, status) VALUES ('a1', 'serial', 'run-a', 'c', 'probe', false, now(),
            'completed')"""))
        holder.commit()
        thread.join(10)
        self.assertEqual(outcome, {"result": "blocked"})

    def test_process_killed_after_begin_blocks_the_next_task_before_any_model_call(self):
        child = self.run_child("""
            from stores.postgres import PostgresJournal
            journal = PostgresJournal("crash", dsn_env=ENV, schema=SCHEMA)
            journal.assert_ready()
            journal.begin("run-1", "call-1", "probe", True)
            os._exit(0)  # The process dies after the start commit, before any finish.
        """)
        self.assertEqual(child.returncode, 0, child.stderr)
        model = RecordingModel(reply("must not be called"))
        result = Agent(model, "sys", tools={}, journal=self.make_journal("crash"), quiet=True,
                       trace_path=self.root / "trace.jsonl").run_result("go")
        self.assertEqual((result.status, result.stop_reason), ("stopped", "journal_recovery_required"))
        self.assertEqual(model.calls, [])

    def test_command_line_initializes_inspects_and_resolves_without_running_anything(self):
        def command(*args):
            return subprocess.run([sys.executable, "-m", "stores.postgres", *args], cwd=REPO, capture_output=True,
                                  text=True, timeout=30, env={"PATH": os.defpath, "PYTHONPATH": str(REPO),
                                                              ENV: os.environ[ENV]})

        target = ("--dsn-env", ENV, "--schema", self.schema)
        initialized = command("init", *target)
        self.assertEqual(initialized.returncode, 0, initialized.stderr)
        self.assertIn("is ready", initialized.stdout)
        PostgresMemoryStore("cli", dsn_env=ENV, schema=self.schema).ensure_schema(create=False)
        journal = self.make_journal("cli")
        journal.assert_ready()
        action = journal.begin("run-1", "call-1", "probe", True)
        pending = command("journal", "inspect", *target, "--journal-id", "cli", "--pending")
        self.assertEqual([r["action_id"] for r in json.loads(pending.stdout)], [action])
        resolved = command("journal", "resolve", *target, "--journal-id", "cli", action, "--outcome", "not_run",
                           "--confirm")
        self.assertIn("no action was executed", resolved.stdout)
        self.assertEqual(json.loads(command("journal", "inspect", *target, "--journal-id", "cli",
                                            "--pending").stdout), [])
        missing = command("journal", "inspect", "--dsn-env", "LITTLE_HARNESS_UNSET_VARIABLE", "--journal-id", "cli")
        self.assertEqual(missing.returncode, 1)
        self.assertIn("PostgreSQL storage operation failed", missing.stderr)


@needs_driver
class PostgresFailureTests(TemporaryDirectoryTestCase):
    """No database needed: nothing reachable on port 1. Errors must not leak connection details."""

    SECRET_DSN = "postgresql://leak-user:leak-password@127.0.0.1:1/leak-database"

    def setUp(self):
        super().setUp()
        patcher = mock.patch.dict(os.environ, {"LH_UNREACHABLE_DSN": self.SECRET_DSN})
        patcher.start()
        self.addCleanup(patcher.stop)

    def assertNoLeak(self, text):
        for fragment in ("leak", "127.0.0.1", "password"):
            self.assertNotIn(fragment, text)

    def test_unreachable_database_raises_fixed_errors_quickly(self):
        started = time.monotonic()
        for operation, error in ((lambda: PostgresSessionStore("s", dsn_env="LH_UNREACHABLE_DSN").load(),
                                  SessionStoreError),
                                 (lambda: PostgresJournal("j", dsn_env="LH_UNREACHABLE_DSN").assert_ready(),
                                  JournalError)):
            with self.subTest(error=error.__name__), self.assertRaises(error) as caught:
                operation()
            self.assertNoLeak(str(caught.exception))
            self.assertTrue(caught.exception.__suppress_context__)
        self.assertLess(time.monotonic() - started, 15)

    def test_agent_stops_on_an_unreachable_journal_without_recording_details(self):
        events = []
        model = RecordingModel(reply("must not be called"))
        agent = Agent(model, "sys", tools={}, journal=PostgresJournal("j", dsn_env="LH_UNREACHABLE_DSN"), quiet=True,
                      trace_path=self.root / "trace.jsonl", on_event=events.append, event_include_content=True)
        result = agent.run_result("go")
        self.assertEqual((result.status, result.stop_reason, model.calls), ("stopped", "journal_error", []))
        self.assertNoLeak(json.dumps(result.to_dict()) + (self.root / "trace.jsonl").read_text()
                          + json.dumps([event.data for event in events], default=str))

    def test_configuration_errors_name_the_variable_never_a_value(self):
        with self.assertRaisesRegex(ValueError, "LITTLE_HARNESS_UNSET_VARIABLE is not set"):
            PostgresJournal("j", dsn_env="LITTLE_HARNESS_UNSET_VARIABLE")
        for options in ({}, {"dsn_env": "LH_UNREACHABLE_DSN", "connect": lambda: None}):
            with self.subTest(options=sorted(options)), self.assertRaises(TypeError):
                PostgresJournal("j", **options)
        for name, options in (("bad id!", {}), ("j", {"schema": "Bad-Schema"}), ("j", {"statement_timeout": 0})):
            with self.subTest(name=name, options=options), self.assertRaises(ValueError):
                PostgresJournal(name, dsn_env="LH_UNREACHABLE_DSN", **options)

    def test_statement_timeout_cannot_round_down_to_disabled(self):
        for adapter in (PostgresSessionStore, PostgresJournal, PostgresMemoryStore):
            with self.subTest(adapter=adapter.__name__), self.assertRaises(ValueError):
                adapter("x", dsn_env="LH_UNREACHABLE_DSN", statement_timeout=0.0009)
            store = adapter("x", dsn_env="LH_UNREACHABLE_DSN", statement_timeout=0.001)
            self.assertEqual(store._timeout_ms, "1")


@unittest.skipIf(postgres.psycopg is not None, "the postgres extra is installed")
class MissingDriverTests(unittest.TestCase):
    def test_adapters_name_the_missing_optional_extra(self):
        for adapter in (PostgresSessionStore, PostgresJournal):
            with self.subTest(adapter=adapter.__name__), self.assertRaisesRegex(ImportError, "postgres' extra"):
                adapter("x", dsn_env="ANY")

