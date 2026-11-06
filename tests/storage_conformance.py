"""Shared storage contracts (docs/STORAGE_SPEC.md, ST-08).

Mix a suite into a TestCase that provides self.root (a temporary directory) and the
backend hooks below. Each hook call with the same name returns a new handle to the
same storage; a new name is empty.
"""

import json
from datetime import datetime, timezone

from harness import (Agent, JournalError, JournalRecoveryRequired, SessionConflictError, SessionLoadError,
                     SessionStore, ToolRegistry)
from harness.journal import ActionJournal
from tests.helpers import RecordingModel, reply, tool_calls

# Model-chosen keys and text that are awkward for databases must round-trip exactly.
AWKWARD = {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "probe", "input": {
    "$where": "1", "a.b": "dotted", "nul": "x\u0000y", "snow": "\u2603", "nested": {"$gt": [1, 2.5, None]}}}]}
COMPACTION = {"start": 1, "summary": "earlier turns"}


def history(*texts):
    return [{"role": "user" if index % 2 == 0 else "assistant", "content": text} for index, text in enumerate(texts)]


class SessionStoreConformanceTests:
    """Hooks: make_store(name). Optional: store_raw(name, data) writes decoded data unvalidated;
    detects_conflicts = True for stores that refuse a second writer (ST-02)."""

    detects_conflicts = False

    def make_store(self, name):
        raise NotImplementedError

    def store_raw(self, name, data):
        self.skipTest("backend cannot write unvalidated session data")

    def test_store_implements_the_session_store_interface(self):
        self.assertIsInstance(self.make_store("interface"), SessionStore)

    def test_missing_session_loads_as_none(self):
        self.assertIsNone(self.make_store("missing").load())

    def test_history_and_compaction_round_trip_exactly(self):
        messages = [*history("hello", "hi"), AWKWARD]
        for durable in (False, True):
            with self.subTest(durable=durable):
                name = f"round-trip-{durable}"
                self.make_store(name).save(messages, COMPACTION, durable=durable)
                self.assertEqual(self.make_store(name).load(), (messages, COMPACTION))
        self.make_store("no-compaction").save(messages, None, durable=False)
        self.assertEqual(self.make_store("no-compaction").load(), (messages, None))

    def test_saved_messages_may_change_anywhere_not_only_at_the_end(self):
        store = self.make_store("rewrite")
        store.save(history("one", "two", "three"), None, durable=False)
        store.save(history("one", "TWO", "three", "four"), COMPACTION, durable=False)
        store.save(history("one", "TWO", "three", "four, steered"), COMPACTION, durable=False)
        self.assertEqual(self.make_store("rewrite").load(), (history("one", "TWO", "three", "four, steered"),
                                                             COMPACTION))

    def test_invalid_stored_compaction_raises_and_is_left_unchanged(self):
        raw = {"version": 2, "messages": history("one"), "compaction": {"start": 5, "summary": "too far"}}
        self.store_raw("invalid", raw)
        for _ in range(2):
            with self.assertRaisesRegex(SessionLoadError, "invalid compaction state"):
                self.make_store("invalid").load()

    def test_second_writer_is_refused_and_first_save_is_kept(self):
        if not self.detects_conflicts:
            self.skipTest("this store assumes a single writer")
        self.make_store("conflict").save(history("start"), None, durable=False)
        first, second = self.make_store("conflict"), self.make_store("conflict")
        first.load(), second.load()
        first.save(history("start", "first"), None, durable=True)
        with self.assertRaises(SessionConflictError):
            second.save(history("start", "second"), None, durable=True)
        self.assertEqual(self.make_store("conflict").load(), (history("start", "first"), None))

    def test_agent_saves_through_the_store_and_a_new_agent_resumes(self):
        options = dict(tools={}, trace_path=self.root / "session-trace.jsonl", quiet=True)
        Agent(RecordingModel(reply("first answer")), "sys", session_store=self.make_store("agent"),
              **options).run_result("first task")
        model = RecordingModel(reply("second answer"))
        resumed = Agent(model, "sys", session_store=self.make_store("agent"), **options)
        self.assertEqual(resumed.run_result("second task").answer, "second answer")
        seen = model.calls[0]
        self.assertEqual((len(seen), seen[0]["content"], seen[-1]["content"]), (3, "first task", "second task"))
        self.assertEqual(len(self.make_store("agent").load()[0]), 4)


def probe():
    return "probed"


class JournalConformanceTests:
    """Hooks: make_journal(name). Optional: make_foreign(name) returns a journal whose
    storage already holds data with another owner or schema version (ST-06)."""

    def make_journal(self, name):
        raise NotImplementedError

    def make_foreign(self, name):
        self.skipTest("backend has no foreign-storage fixture")

    def prepared(self, name):
        """The agent always prepares (via assert_ready) before its first begin."""
        journal = self.make_journal(name)
        journal.prepare()
        return journal

    def test_journal_implements_the_action_journal_interface(self):
        self.assertIsInstance(self.make_journal("interface"), ActionJournal)

    def test_completed_closed_run_needs_no_recovery_and_stores_metadata_only(self):
        journal = self.make_journal("completed")
        journal.assert_ready()
        action = journal.begin("run-1", "call-1", "probe", True)
        journal.finish(action, "completed")
        journal.close_run("run-1")
        self.make_journal("completed").assert_ready()
        [record] = self.make_journal("completed").inspect()
        self.assertEqual((record["action_id"], record["status"], record["risky"], record["recovery_required"]),
                         (action, "completed", True, False))
        self.assertFalse({"arguments", "input", "output", "error"} & record.keys())

    def test_unfinished_action_requires_recovery_until_a_confirmed_resolution(self):
        journal = self.prepared("crashed")
        action = journal.begin("run-1", "call-1", "probe", True)  # The process "dies" here.
        with self.assertRaises(JournalRecoveryRequired):
            self.make_journal("crashed").assert_ready()
        [record] = self.make_journal("crashed").inspect(pending_only=True)
        self.assertEqual((record["status"], record["outcome_unknown"]), ("outcome_unknown", True))
        with self.assertRaises(ValueError):
            journal.resolve(action, "completed")  # Not confirmed.
        journal.resolve(action, "completed", confirm=True)
        self.make_journal("crashed").assert_ready()
        with self.assertRaises(ValueError):
            journal.resolve(action, "completed", confirm=True)  # No longer pending.

    def test_an_unclosed_run_blocks_a_second_run_from_starting_actions(self):
        journal = self.prepared("two-runs")
        journal.finish(journal.begin("run-a", "call-1", "probe", False), "completed")
        with self.assertRaises(JournalRecoveryRequired):
            self.make_journal("two-runs").begin("run-b", "call-1", "probe", False)
        self.assertEqual({r["run_id"] for r in journal.inspect()}, {"run-a"})
        journal.close_run("run-a")
        self.make_journal("two-runs").begin("run-b", "call-1", "probe", False)

    def test_invalid_or_repeated_transitions_fail_without_changing_records(self):
        journal = self.prepared("transitions")
        action = journal.begin("run-1", "call-1", "probe", True)
        for status, unknown in (("bogus", False), ("timed_out", False), ("completed", True)):
            with self.subTest(status=status, unknown=unknown), self.assertRaises(ValueError):
                journal.finish(action, status, unknown)
        journal.finish(action, "failed")
        with self.assertRaises(JournalError):
            journal.finish(action, "completed")
        self.assertEqual(journal.inspect()[0]["status"], "failed")
        for limit in (0, 1001, True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                journal.inspect(limit=limit)

    def test_foreign_or_newer_storage_is_never_adopted(self):
        with self.assertRaises(JournalError):
            self.make_foreign("foreign").prepare()

    def test_agent_records_risky_actions_through_the_journal(self):
        tools = ToolRegistry()
        tools.tool(risky=True)(probe)
        model = RecordingModel(tool_calls(("call-1", "probe", {})), reply("done"))
        agent = Agent(model, "sys", tools=tools, journal=self.make_journal("agent"), approve=lambda *_: True,
                      trace_path=self.root / "journal-trace.jsonl", quiet=True)
        self.assertEqual(agent.run_result("go").status, "completed")
        [record] = self.make_journal("agent").inspect()
        self.assertEqual((record["tool_name"], record["status"], record["run_closed"]), ("probe", "completed", True))
        self.assertNotIn("probed", json.dumps(record))


class MemoryStoreConformanceTests:
    """Hooks: make_memory_store(name, *, quota=10000), memory_error exception type.

    The same name opens the same backing store; a new name starts empty. The
    fixture supplies the shared MemoryStoreError exception type.
    """

    def make_memory_store(self, name, *, quota=10000):
        raise NotImplementedError

    def test_memory_round_trips_with_provenance_across_handles(self):
        store = self.make_memory_store("round-trip")
        memory_id = store.add("tenant:1/user:7", "Snowman ☃; $where and a.b", kind="note",
                              created_by="model", run_id="run-1")
        [record] = self.make_memory_store("round-trip").list("tenant:1/user:7")
        self.assertIsInstance(memory_id, str)
        self.assertTrue(memory_id)
        self.assertEqual((record.memory_id, record.namespace, record.text, record.kind,
                          record.created_by, record.run_id),
                         (memory_id, "tenant:1/user:7", "Snowman ☃; $where and a.b", "note", "model", "run-1"))
        self.assertEqual(datetime.fromisoformat(record.created_at.replace("Z", "+00:00")).utcoffset(),
                         timezone.utc.utcoffset(None))

    def test_search_delete_and_purge_are_isolated_by_namespace(self):
        store = self.make_memory_store("scope")
        first = store.add("tenant:1", "shared telescope", kind="fact", created_by="application")
        second = store.add("tenant:2", "shared telescope", kind="fact", created_by="application")
        self.assertNotEqual(first, second)
        self.assertEqual([record.memory_id for record in store.search("tenant:1", "telescope")], [first])
        self.assertFalse(store.delete("tenant:1", second))
        self.assertEqual(store.purge("tenant:1"), 1)
        self.assertEqual(store.list("tenant:1"), [])
        self.assertEqual([record.memory_id for record in store.list("tenant:2")], [second])
        self.assertEqual(store.purge("tenant:1"), 0)

    def test_search_matches_whole_words_case_insensitively_and_obeys_limit(self):
        store = self.make_memory_store("search")
        first = store.add("n", "Blue telescope", kind="fact", created_by="application")
        second = store.add("n", "Red telescope", kind="note", created_by="application")
        store.add("n", "telescopes are different", kind="preference", created_by="application")
        store.add("n", "Unrelated astronomy", kind="note", created_by="application")
        self.assertEqual({record.memory_id for record in store.search("n", "TELESCOPE")}, {first, second})
        self.assertEqual(len(store.search("n", "telescope", limit=1)), 1)
        self.assertEqual(store.search("n", "missing"), [])

    def test_quota_is_per_namespace_and_rejection_does_not_evict(self):
        store = self.make_memory_store("quota", quota=2)
        first = store.add("a", "first", kind="fact", created_by="application")
        second = store.add("a", "second", kind="note", created_by="application")
        with self.assertRaises(self.memory_error):
            store.add("a", "third", kind="fact", created_by="application")
        other = store.add("b", "other", kind="fact", created_by="application")
        self.assertEqual({record.memory_id for record in store.list("a")}, {first, second})
        self.assertEqual([record.memory_id for record in store.list("b")], [other])
        self.assertTrue(store.delete("a", first))
        self.assertFalse(store.delete("a", first))
        store.add("a", "replacement", kind="fact", created_by="application")
        self.assertEqual(len(store.list("a")), 2)

    def test_search_ranks_exact_two_word_match_above_one_word_match(self):
        store = self.make_memory_store("ranking")
        store.add("n", "telescope", kind="fact", created_by="application")
        better = store.add("n", "blue telescope", kind="fact", created_by="application")
        self.assertEqual(store.search("n", "blue telescope")[0].memory_id, better)

    def test_length_and_limit_boundaries(self):
        store = self.make_memory_store("bounds")
        for text in ("", "x" * 2001):
            with self.subTest(text_length=len(text)), self.assertRaises(ValueError):
                store.add("n", text, kind="fact", created_by="application")
        store.add("n", "x" * 2000, kind="fact", created_by="application")
        for query in ("", "x" * 501):
            with self.subTest(query_length=len(query)), self.assertRaises(ValueError):
                store.search("n", query)
        store.search("n", "x" * 500)
        for limit in (0, 21, True):
            with self.subTest(search_limit=limit), self.assertRaises(ValueError):
                store.search("n", "x", limit=limit)
        for limit in (0, 1001, True):
            with self.subTest(list_limit=limit), self.assertRaises(ValueError):
                store.list("n", limit=limit)
