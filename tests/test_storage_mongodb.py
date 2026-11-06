"""MongoDB 9.0 adapters against the shared storage contracts (docs/STORAGE_SPEC.md).

Database tests run only when LITTLE_HARNESS_TEST_MONGODB_URI names a replica set and
the mongodb extra is installed; each test uses its own database and drops it.
"""

import json
import os
import subprocess
import sys
import textwrap
import threading
import time
import unittest
from datetime import datetime, timezone
from unittest import mock
from uuid import uuid4

from harness import Agent, JournalError, JournalRecoveryRequired, SessionStoreError
from stores import mongodb
from stores.mongodb import MongoJournal, MongoSessionStore
from tests.helpers import REPO, RecordingModel, TemporaryDirectoryTestCase, reply
from tests.storage_conformance import JournalConformanceTests, SessionStoreConformanceTests, history

ENV = "LITTLE_HARNESS_TEST_MONGODB_URI"
needs_driver = unittest.skipIf(mongodb.pymongo is None, "install the mongodb extra")
needs_database = unittest.skipUnless(mongodb.pymongo is not None and os.environ.get(ENV),
                                     f"set {ENV} and install the mongodb extra")


class DatabasePerTest:
    def setUp(self):
        super().setUp()
        self.database = "lh_test_" + uuid4().hex[:12]
        self.client = mongodb.MongoClient(os.environ[ENV], serverSelectionTimeoutMS=10000, tz_aware=True)
        self.addCleanup(self.client.close)
        self.addCleanup(self.client.drop_database, self.database)
        self.db = self.client[self.database]

    def run_child(self, code):
        """A separate process that may exit abruptly (os._exit) in the middle of an operation."""
        prelude = f"import os, pymongo\nENV, DATABASE = {ENV!r}, {self.database!r}\n"
        return subprocess.run([sys.executable, "-c", prelude + textwrap.dedent(code)], cwd=REPO,
                              env={"PATH": os.defpath, "PYTHONPATH": str(REPO), ENV: os.environ[ENV]},
                              capture_output=True, text=True, timeout=30)


@needs_database
class MongoSessionStoreTests(DatabasePerTest, SessionStoreConformanceTests, TemporaryDirectoryTestCase):
    detects_conflicts = True

    def make_store(self, name):
        return MongoSessionStore(name, client=self.client, database=self.database)

    def store_raw(self, name, data):
        # save() writes compaction state as given; only load() validates it.
        self.make_store(name).save(data["messages"], data["compaction"], durable=False)

    def test_process_killed_during_a_save_leaves_the_previous_revision_and_fails_closed(self):
        self.make_store("crash").save(history("one", "two"), None, durable=True)
        child = self.run_child("""
            from stores.mongodb import MongoSessionStore
            store = MongoSessionStore("crash", uri_env=ENV, database=DATABASE)
            store.load()
            original = pymongo.collection.Collection.insert_many
            def insert_then_die(self, *args, **kwargs):
                original(self, *args, **kwargs)
                os._exit(3)  # Documents were deleted and inserted, but the transaction never committed.
            pymongo.collection.Collection.insert_many = insert_then_die
            store.save([{"role": "user", "content": "one"}, {"role": "assistant", "content": "CHANGED"},
                        {"role": "user", "content": "three"}], None, durable=True)
        """)
        self.assertEqual(child.returncode, 3, child.stderr)
        store = MongoSessionStore("crash", client=self.client, database=self.database, timeout=1)
        self.assertEqual(store.load(), (history("one", "two"), None))
        # Unlike a PostgreSQL connection, the dead writer's transaction stays open on the server
        # (up to transactionLifetimeLimitSeconds, 60 by default) and keeps its writes: saves are refused.
        with self.assertRaises(SessionStoreError):
            store.save(history("one", "two", "three"), None, durable=True)
        self.client.admin.command("killAllSessions", [])  # The operator's way to end it early.
        store.save(history("one", "two", "three"), None, durable=True)
        self.assertEqual(self.make_store("crash").load()[0], history("one", "two", "three"))

    def test_messages_before_the_first_change_are_not_rewritten_and_validators_apply(self):
        store = self.make_store("rewrite")
        store.save(history("one", "two", "three"), None, durable=False)
        before = {doc["idx"]: doc["_id"] for doc in self.db.lh_session_messages.find()}
        store.save(history("one", "two", "THREE", "four"), None, durable=False)
        after = {doc["idx"]: doc["_id"] for doc in self.db.lh_session_messages.find()}
        self.assertEqual((after[0], after[1]), (before[0], before[1]))
        self.assertNotEqual(after[2], before[2])
        self.assertEqual(sorted(after), [0, 1, 2, 3])
        with self.assertRaises(mongodb.PyMongoError):
            self.db.lh_sessions.insert_one({"_id": "malformed", "message_count": "many"})


@needs_database
class MongoJournalTests(DatabasePerTest, JournalConformanceTests, TemporaryDirectoryTestCase):
    def make_journal(self, name):
        return MongoJournal(name, client=self.client, database=self.database)

    def make_foreign(self, name):
        self.make_journal(name).prepare()
        self.db.lh_meta.update_one({"_id": "journal"}, {"$set": {"schema_version": 99}})
        return self.make_journal(name)

    def test_collections_without_an_ownership_record_are_never_adopted(self):
        self.db.lh_journal_actions.insert_one({"unrelated": True})
        with self.assertRaises(JournalError):
            self.make_journal("unowned").prepare()
        self.assertEqual(self.db.lh_journal_actions.count_documents({}), 1)
        self.assertIsNone(self.db.lh_meta.find_one({"_id": "journal"}))

    def test_begin_waits_for_a_concurrent_begin_and_then_sees_its_open_run(self):
        self.make_journal("serial").prepare()
        holder = self.client.start_session()
        self.addCleanup(holder.end_session)
        holder.start_transaction(write_concern=mongodb.WriteConcern("majority"))
        self.db.lh_journals.update_one({"_id": "serial"}, {"$inc": {"seq": 1}}, session=holder)
        self.db.lh_journal_runs.insert_one({"journal_id": "serial", "run_id": "run-a", "closed": False},
                                           session=holder)
        self.db.lh_journal_actions.insert_one({
            "_id": "a1", "journal_id": "serial", "run_id": "run-a", "tool_call_id": "c", "tool_name": "probe",
            "risky": False, "started_at": datetime.now(timezone.utc), "finished_at": None, "status": "completed",
            "outcome_unknown": False, "resolution": None, "resolved_at": None}, session=holder)
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
        self.assertTrue(thread.is_alive(), "begin must conflict with the uncommitted counter write and retry")
        holder.commit_transaction()
        thread.join(10)
        self.assertEqual(outcome, {"result": "blocked"})

    def test_process_killed_after_begin_blocks_the_next_task_before_any_model_call(self):
        child = self.run_child("""
            from stores.mongodb import MongoJournal
            journal = MongoJournal("crash", uri_env=ENV, database=DATABASE)
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
            return subprocess.run([sys.executable, "-m", "stores.mongodb", *args], cwd=REPO, capture_output=True,
                                  text=True, timeout=30, env={"PATH": os.defpath, "PYTHONPATH": str(REPO),
                                                              ENV: os.environ[ENV]})

        target = ("--uri-env", ENV, "--database", self.database)
        self.assertIn("is ready", command("init", *target).stdout)
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
        missing = command("journal", "inspect", "--uri-env", "LITTLE_HARNESS_UNSET_VARIABLE", "--journal-id", "cli")
        self.assertEqual(missing.returncode, 1)
        self.assertIn("MongoDB storage operation failed", missing.stderr)


@needs_driver
class MongoFailureTests(TemporaryDirectoryTestCase):
    """No database needed: nothing reachable on port 1. Errors must not leak connection details."""

    SECRET_URI = "mongodb://leak-user:leak-password@127.0.0.1:1/?directConnection=true"

    def setUp(self):
        super().setUp()
        patcher = mock.patch.dict(os.environ, {"LH_UNREACHABLE_URI": self.SECRET_URI})
        patcher.start()
        self.addCleanup(patcher.stop)

    def assertNoLeak(self, text):
        for fragment in ("leak", "127.0.0.1", "password"):
            self.assertNotIn(fragment, text)

    def test_unreachable_database_raises_fixed_errors_within_the_time_bound(self):
        started = time.monotonic()
        for operation, error in ((lambda: MongoSessionStore("s", uri_env="LH_UNREACHABLE_URI", timeout=1).load(),
                                  SessionStoreError),
                                 (lambda: MongoJournal("j", uri_env="LH_UNREACHABLE_URI", timeout=1).assert_ready(),
                                  JournalError)):
            with self.subTest(error=error.__name__), self.assertRaises(error) as caught:
                operation()
            self.assertNoLeak(str(caught.exception))
            self.assertTrue(caught.exception.__suppress_context__)
        self.assertLess(time.monotonic() - started, 10)

    def test_agent_stops_on_an_unreachable_journal_without_recording_details(self):
        events = []
        model = RecordingModel(reply("must not be called"))
        agent = Agent(model, "sys", tools={}, journal=MongoJournal("j", uri_env="LH_UNREACHABLE_URI", timeout=1),
                      quiet=True, trace_path=self.root / "trace.jsonl", on_event=events.append,
                      event_include_content=True)
        result = agent.run_result("go")
        self.assertEqual((result.status, result.stop_reason, model.calls), ("stopped", "journal_error", []))
        self.assertNoLeak(json.dumps(result.to_dict()) + (self.root / "trace.jsonl").read_text()
                          + json.dumps([event.data for event in events], default=str))

    def test_configuration_errors_name_the_variable_never_a_value(self):
        with self.assertRaisesRegex(ValueError, "LITTLE_HARNESS_UNSET_VARIABLE is not set"):
            MongoJournal("j", uri_env="LITTLE_HARNESS_UNSET_VARIABLE")
        for options in ({}, {"uri_env": "LH_UNREACHABLE_URI", "client": object()}, {"client": object()}):
            with self.subTest(options=sorted(options)), self.assertRaises(TypeError):
                MongoJournal("j", **options)
        for name, options in (("bad id!", {}), ("j", {"database": "bad/name"}), ("j", {"timeout": 0})):
            with self.subTest(name=name, options=options), self.assertRaises(ValueError):
                MongoJournal(name, uri_env="LH_UNREACHABLE_URI", **options)


@unittest.skipIf(mongodb.pymongo is not None, "the mongodb extra is installed")
class MissingDriverTests(unittest.TestCase):
    def test_adapters_name_the_missing_optional_extra(self):
        for adapter in (MongoSessionStore, MongoJournal):
            with self.subTest(adapter=adapter.__name__), self.assertRaisesRegex(ImportError, "mongodb' extra"):
                adapter("x", uri_env="ANY")
