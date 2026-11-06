"""Standard-library storage backends against the shared storage contracts."""

import json
import sqlite3

from harness import (Agent, ExecutionJournal, SessionConflictError, SessionFile, SessionStoreError,
                     validate_session)
from tests.helpers import TemporaryDirectoryTestCase
from tests.storage_conformance import JournalConformanceTests, SessionStoreConformanceTests


class FileSessionStoreTests(SessionStoreConformanceTests, TemporaryDirectoryTestCase):
    def make_store(self, name):
        return SessionFile(self.root / f"{name}.json")

    def store_raw(self, name, data):
        (self.root / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")


class RevisionCheckedSessionStore:
    """Test double for database stores: one shared mapping, compare-and-set revisions."""

    def __init__(self, database, name):
        self.database, self.name, self.revision = database, name, 0

    def load(self):
        if self.name not in self.database:
            return None
        self.revision, text = self.database[self.name]
        return validate_session(json.loads(text), f"session {self.name}")

    def save(self, messages, compaction, *, durable):
        if self.database.get(self.name, (0, None))[0] != self.revision:
            raise SessionConflictError("session changed since it was loaded; nothing was saved")
        self.revision += 1
        self.database[self.name] = (self.revision, json.dumps({"version": 2, "messages": messages,
                                                                "compaction": compaction}))


class RevisionCheckedSessionStoreTests(SessionStoreConformanceTests, TemporaryDirectoryTestCase):
    detects_conflicts = True

    def setUp(self):
        super().setUp()
        self.database = {}

    def make_store(self, name):
        return RevisionCheckedSessionStore(self.database, name)

    def store_raw(self, name, data):
        self.database[name] = (1, json.dumps(data))


class SQLiteJournalTests(JournalConformanceTests, TemporaryDirectoryTestCase):
    def make_journal(self, name):
        return ExecutionJournal(self.root / f"{name}.sqlite3")

    def make_foreign(self, name):
        path = self.root / f"{name}.sqlite3"
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE unrelated (value TEXT)")
        connection.close()
        return ExecutionJournal(path)


class AgentStorageOptionTests(TemporaryDirectoryTestCase):
    def agent(self, **options):
        return Agent(lambda *_: None, "sys", tools={}, trace_path=self.root / "trace.jsonl", quiet=True, **options)

    def test_store_and_path_options_are_exclusive_and_type_checked(self):
        store, journal = SessionFile(self.root / "s.json"), ExecutionJournal(self.root / "j.sqlite3")
        for options in ({"session_store": store, "session_path": self.root / "other.json"},
                        {"session_store": object()},
                        {"journal": journal, "journal_path": self.root / "other.sqlite3"},
                        {"journal": object()}):
            with self.subTest(options=sorted(options)), self.assertRaises(TypeError):
                self.agent(**options)

    def test_session_store_errors_are_os_errors_for_existing_handlers(self):
        self.assertTrue(issubclass(SessionConflictError, SessionStoreError))
        self.assertTrue(issubclass(SessionStoreError, OSError))

    def test_file_journal_still_cannot_share_the_session_file(self):
        session = self.root / "shared.sqlite3"
        with self.assertRaises(ValueError):
            self.agent(session_store=SessionFile(session), journal=ExecutionJournal(session))

    def test_importing_harness_loads_no_database_driver(self):
        child = self.child("""
            import sys, harness
            print(sorted(name for name in ("psycopg", "pymongo", "bson", "stores") if name in sys.modules))
        """)
        self.assertEqual((child.returncode, child.stdout.strip()), (0, "[]"), child.stderr)
