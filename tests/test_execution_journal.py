"""SQLite durability boundaries, fail-closed dispatch, and offline human recovery."""

import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from harness import ExecutionJournal, JournalError, JournalRecoveryRequired
from tests import helpers as f
from tests import tool_fixtures as tools

from harness import agent as h
REPO = f.REPO


class JournalTests(f.HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.path = self.dir / "execution.sqlite3"
        self.journal = ExecutionJournal(self.path)
        self.journal.assert_ready()

    def test_committed_records_have_stable_ids_and_unclosed_runs_require_review(self):
        first = self.journal.begin("run-a", "provider-id", "tool", True)
        record = ExecutionJournal(self.path).inspect()[0]
        self.assertEqual(record["action_id"], first)
        self.assertEqual(record["recorded_status"], "started")
        self.assertEqual(record["status"], "outcome_unknown")
        self.assertTrue(record["recovery_required"])
        self.journal.finish(first, "completed")
        with self.assertRaises(JournalRecoveryRequired):
            self.journal.assert_ready()
        second = self.journal.begin("run-a", "provider-id", "tool", False)
        self.assertNotEqual(first, second)
        self.journal.finish(second, "failed")
        self.journal.close_run("run-a")
        self.journal.assert_ready()
        self.assertTrue(all(not r["recovery_required"] for r in self.journal.inspect()))
        self.assertEqual(self.journal.inspect(pending_only=True), [])
        if os.name == "posix":
            self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_read_only_inspection_does_not_create_or_rewrite_records(self):
        action_id = self.journal.begin("run", "call", "tool", True)
        before = self.path.read_bytes()
        self.assertEqual(self.journal.inspect()[0]["action_id"], action_id)
        self.assertEqual(self.path.read_bytes(), before)
        missing = self.dir / "missing.sqlite3"
        with self.assertRaises(JournalError):
            ExecutionJournal(missing).inspect()
        self.assertFalse(missing.exists())
        for limit in (0, 1001, True):
            with self.assertRaises(ValueError):
                self.journal.inspect(limit=limit)

    def test_resolution_is_explicit_preserves_history_and_cannot_rewrite_closed_success(self):
        first = self.journal.begin("run", "a", "tool", True)
        self.journal.finish(first, "completed")
        second = self.journal.begin("run", "b", "tool", True)
        for outcome, confirm in (("completed", False), ("uncertain", True)):
            with self.assertRaises(ValueError):
                self.journal.resolve(first, outcome, confirm=confirm)
        self.journal.resolve(first, "completed", confirm=True)
        with self.assertRaises(JournalRecoveryRequired):
            self.journal.assert_ready()
        self.journal.resolve(second, "not_run", confirm=True)
        self.journal.assert_ready()
        rows = {r["action_id"]: r for r in self.journal.inspect()}
        self.assertEqual(rows[second]["recorded_status"], "started")
        self.assertEqual(rows[second]["resolution"], "not_run")
        self.assertIsNotNone(rows[second]["resolved_at"])
        self.assertTrue(rows[second]["outcome_unknown"])
        with self.assertRaises(ValueError):
            self.journal.resolve(second, "completed", confirm=True)
        third = self.journal.begin("new-run", "c", "tool", False)
        self.journal.finish(third, "completed")
        self.journal.close_run("new-run")
        with self.assertRaises(ValueError):
            self.journal.resolve(third, "failed", confirm=True)

    def test_concurrent_begins_cannot_both_authorize_actions(self):
        barrier = threading.Barrier(2)

        def begin(run):
            barrier.wait(timeout=5)
            try:
                return ExecutionJournal(self.path).begin(run, "call", "tool", True)
            except JournalRecoveryRequired:
                return None

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(begin, ("one", "two")))
        self.assertEqual(sum(r is not None for r in results), 1)
        self.assertEqual(len(self.journal.inspect()), 1)

    def test_unrelated_or_newer_databases_are_not_adopted_or_modified(self):
        unrelated = self.dir / "unrelated.sqlite3"
        with sqlite3.connect(unrelated) as connection:
            connection.execute("CREATE TABLE other (value TEXT)")
        future = self.dir / "future.sqlite3"
        self.path.replace(future)
        with sqlite3.connect(future) as connection:
            connection.execute("PRAGMA user_version = 999")
        for path in (unrelated, future):
            before = path.read_bytes()
            with self.assertRaises(JournalError):
                ExecutionJournal(path).assert_ready()
            self.assertEqual(path.read_bytes(), before)

    def test_database_lock_fails_with_a_bounded_sanitized_error(self):
        connection = sqlite3.connect(self.path)
        self.addCleanup(connection.close)
        connection.execute("BEGIN IMMEDIATE")
        started = time.monotonic()
        with self.assertRaises(JournalError) as raised:
            self.journal.begin("run", "call", "tool", True)
        self.assertLess(time.monotonic() - started, 4)
        self.assertNotIn(str(self.path), str(raised.exception))
        self.assertNotIn("locked", str(raised.exception))

    def test_finished_outcomes_cannot_be_overwritten(self):
        action_id = self.journal.begin("run", "call", "tool", True)
        self.journal.finish(action_id, "completed")
        with self.assertRaises(JournalError):
            self.journal.finish(action_id, "failed", True)
        self.assertEqual(self.journal.inspect()[0]["status"], "completed")


class JournalIntegrationTests(f.HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.path = self.dir / "execution.sqlite3"

    def agent(self, model, **kwargs):
        kwargs.setdefault("journal_path", self.path)
        return super().agent(model, **kwargs)

    def test_start_is_committed_before_inline_execution_and_no_payload_is_stored(self):
        observed = []

        def sample(value: str):
            observed.extend(ExecutionJournal(self.path).inspect())
            return "synthetic-output-not-for-journal"

        self.register_tool(sample)
        batch = f.tool_calls(("same-id", "sample", {"value": "synthetic-argument-not-for-journal"}))
        agent = self.agent([batch, f.reply("done")])
        result = agent.run_result("go")
        self.assertEqual(result.status, "completed")
        self.assertEqual(observed[0]["recorded_status"], "started")
        row = agent.journal.inspect()[0]
        self.assertEqual(row["status"], "completed")
        self.assertTrue(row["run_closed"])
        self.assertFalse(row["recovery_required"])
        self.assertNotIn(b"synthetic-argument", self.path.read_bytes())
        self.assertNotIn(b"synthetic-output", self.path.read_bytes())
        events = [r for r in self.trace_records() if r["kind"] in ("tool_started", "tool")]
        self.assertEqual([r["action_id"] for r in events], [row["action_id"]] * 2)
        self.assertEqual([r["run_id"] for r in events], [row["run_id"]] * 2)
        self.assertIsNone(agent._journal_run_id)

    def test_rejected_denied_unknown_and_truncated_requests_do_not_create_action_records(self):
        batch = f.tool_calls(("bad", "calculator", {}), ("missing", "missing", {}),
                             ("denied", "write_file", {"path": "x", "content": "x"}),
                             ("cut", "calculator", {"expression": "2+2"}))
        batch["truncated_tool_ids"] = ["cut"]
        agent = self.agent([batch, f.reply("done")], approve=lambda *_: False)
        result = agent.run_result("go")
        self.assertEqual(result.usage.tool_errors, 4)
        self.assertEqual(agent.journal.inspect(), [])

    def test_risky_inline_failure_stops_and_blocks_later_tasks_without_replaying(self):
        self.register_tool(tools.record_action, risky=True)
        effect = self.dir / "effects.txt"
        batch = f.tool_calls(("a", "record_action", {"path": str(effect), "mode": "done"}),
                             ("b", "record_action", {"path": str(effect), "mode": "fail"}),
                             ("c", "record_action", {"path": str(effect), "mode": "later"}))
        model = f.RecordingModel(batch, f.reply("must not call"))
        session = self.dir / "session.json"
        agent = self.agent(model, session_path=session)
        result = agent.run_result("go")
        self.assertEqual(result.stop_reason, "tool_outcome_unknown")
        self.assertEqual(effect.read_text().splitlines(), ["done", "fail"])
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(result.usage.tool_calls_attempted, 2)
        saved = session.read_bytes()
        later = f.RecordingModel(f.reply("must not call"))
        stopped = self.agent(later, session_path=session).run_result("continue")
        self.assertEqual(stopped.stop_reason, "journal_recovery_required")
        self.assertEqual(stopped.usage.model_calls, 0)
        self.assertEqual(later.calls, [])
        self.assertEqual(session.read_bytes(), saved)
        self.assertEqual(agent.messages[-1]["content"][0]["content"], "recorded done")
        self.assertIn("not run", agent.messages[-1]["content"][2]["content"])
        pending = agent.journal.inspect(pending_only=True)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["status"], "failed")
        self.assertTrue(pending[0]["outcome_unknown"])

    def test_isolated_timeout_and_crash_are_durable_and_block_recovery(self):
        self.register_tool(tools.record_action, risky=True)
        for mode in ("hang", "crash"):
            with self.subTest(mode=mode):
                path = self.dir / (mode + ".sqlite3")
                effect = self.dir / (mode + ".txt")
                batch = f.tool_calls(("a", "record_action", {"path": str(effect), "mode": mode}))
                agent = self.agent([batch], journal_path=path, tool_timeout_seconds=1)
                result = agent.run_result("go")
                row = agent.journal.inspect()[0]
                self.assertEqual(row["status"], "timed_out" if mode == "hang" else "outcome_unknown")
                self.assertEqual(effect.read_text().splitlines(), [mode])
                self.assertTrue(row["recovery_required"])
                self.assertEqual(result.status, "stopped")
                self.assertEqual(self.agent([], journal_path=path).run_result("go").stop_reason,
                                 "journal_recovery_required")

    def test_begin_failure_pairs_not_run_results_without_running_any_action(self):
        execute = mock.Mock()
        batch = f.tool_calls(("a", "current_time", {}), ("b", "current_time", {}))
        agent = self.agent([batch])
        with mock.patch.object(agent.journal, "begin", side_effect=JournalError()), \
                mock.patch.dict(self.tools["current_time"], {"fn": execute}):
            result = agent.run_result("go")
        self.assertEqual(result.stop_reason, "journal_error")
        execute.assert_not_called()
        self.assertTrue(all("not run" in r["content"] for r in agent.messages[-1]["content"]))
        self.assertEqual(agent.journal.inspect(), [])

    def test_unavailable_or_corrupt_journal_stops_before_any_model_call(self):
        corrupt = self.dir / "corrupt.sqlite3"
        corrupt.write_bytes(b"not a database")
        for path in (corrupt, self.dir / "missing-parent/execution.sqlite3"):
            model = f.RecordingModel(f.reply("must not call"))
            result = self.agent(model, journal_path=path).run_result("go")
            self.assertEqual((result.status, result.stop_reason), ("stopped", "journal_error"))
            self.assertEqual(result.usage.model_calls, 0)
            self.assertEqual(model.calls, [])
        self.assertEqual(corrupt.read_bytes(), b"not a database")

    def test_deadline_during_start_commit_records_known_not_run_and_does_not_dispatch(self):
        now = [0.0]
        execute = mock.Mock()
        agent = self.agent([f.tool_calls(("a", "current_time", {}))], limits=h.RunLimits(max_seconds=1))
        begin = agent.journal.begin

        def slow_begin(*args):
            action_id = begin(*args)
            now[0] = 2.0
            return action_id

        with mock.patch("harness.runtime.time.monotonic", side_effect=lambda: now[0]), \
                mock.patch.object(agent.journal, "begin", side_effect=slow_begin), \
                mock.patch.dict(self.tools["current_time"], {"fn": execute}):
            result = agent.run_result("go")
        execute.assert_not_called()
        self.assertEqual(result.stop_reason, "max_seconds")
        row = agent.journal.inspect()[0]
        self.assertEqual(row["status"], "not_run")
        self.assertFalse(row["recovery_required"])

    def test_safe_inline_errors_remain_recoverable(self):
        batch = f.tool_calls(("bad", "calculator", {"expression": "1/0"}),
                             ("good", "calculator", {"expression": "2+2"}))
        agent = self.agent([batch, f.reply("four")])
        result = agent.run_result("go")
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.usage.tool_errors, 1)
        self.assertEqual(agent.journal.inspect(pending_only=True), [])

    def test_finish_failure_preserves_completed_result_but_stops_and_requires_review(self):
        execute = mock.Mock(return_value="action completed")
        model = f.RecordingModel(f.tool_calls(("a", "current_time", {}), ("b", "current_time", {})))
        agent = self.agent(model)
        with mock.patch.object(agent.journal, "finish", side_effect=JournalError()), \
                mock.patch.dict(self.tools["current_time"], {"fn": execute}):
            result = agent.run_result("go")
        self.assertEqual(result.stop_reason, "journal_error")
        execute.assert_called_once()
        self.assertEqual(agent.messages[-1]["content"][0]["content"], "action completed")
        self.assertFalse(agent.messages[-1]["content"][0]["is_error"])
        self.assertIn("not run", agent.messages[-1]["content"][1]["content"])
        self.assertEqual(agent.journal.inspect()[0]["recorded_status"], "started")
        self.assertEqual(self.agent([]).run_result("continue").stop_reason, "journal_recovery_required")

    def test_close_failure_retains_final_answer_but_leaves_known_actions_for_review(self):
        agent = self.agent([f.tool_calls(("a", "calculator", {"expression": "2+2"})), f.reply("four")])
        with mock.patch.object(agent.journal, "close_run", side_effect=JournalError()):
            result = agent.run_result("go")
        self.assertEqual((result.status, result.stop_reason, result.answer), ("stopped", "journal_error", "four"))
        row = agent.journal.inspect()[0]
        self.assertEqual(row["status"], "completed")
        self.assertTrue(row["recovery_required"])
        self.assertEqual(self.agent([]).run_result("continue").stop_reason, "journal_recovery_required")

    def test_cancellation_records_uncertainty_and_preserves_completed_results(self):
        def interrupt():
            raise KeyboardInterrupt

        self.register_tool(interrupt, risky=True)
        batch = f.tool_calls(("a", "calculator", {"expression": "2+2"}), ("b", "interrupt", {}),
                             ("c", "current_time", {}))
        agent = self.agent([batch])
        result = agent.run_result("go")
        self.assertEqual(result.status, "interrupted")
        saved = agent.messages[-1]["content"]
        self.assertEqual(saved[0]["content"], "4")
        self.assertIn("outcome unknown", saved[1]["content"])
        self.assertIn("not run", saved[2]["content"])
        self.assertEqual(len(agent.journal.inspect(pending_only=True)), 2)
        self.assertEqual(self.agent([]).run_result("continue").stop_reason, "journal_recovery_required")

    def test_journal_must_be_outside_workspace_and_not_overlap_session_or_trace(self):
        for path, kwargs in ((self.config.workspace / "execution.sqlite3", {}),
                             (self.path, {"trace_path": self.path}),
                             (self.path, {"trace_path": Path(str(self.path) + "-journal")}),
                             (self.path, {"session_path": self.path})):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                kwargs.setdefault("trace_path", self.trace)
                self.agent([], journal_path=path, **kwargs)

    def test_reused_provider_ids_get_distinct_action_and_run_ids(self):
        call = f.tool_calls(("same-id", "calculator", {"expression": "2+2"}))
        agent = self.agent([call, f.reply("done"), call, f.reply("done again")])
        self.assertEqual(agent.run_result("one").status, "completed")
        self.assertEqual(agent.run_result("two").status, "completed")
        rows = agent.journal.inspect()
        self.assertEqual(len({r["action_id"] for r in rows}), 2)
        self.assertEqual(len({r["run_id"] for r in rows}), 2)


class CrashRecoveryTests(f.HarnessTestCase):
    def test_abrupt_parent_exit_before_tool_after_effect_after_commit_and_after_session_save(self):
        phases = (("before_tool", 21, "started"), ("after_effect", 22, "started"),
                  ("after_finish", 23, "completed"), ("after_save", 24, "completed"))
        for phase, code, status in phases:
            with self.subTest(phase=phase):
                root = self.dir / phase
                root.mkdir()
                env = {"PATH": os.defpath, "PYTHONPATH": str(REPO), "HARNESS_WORKSPACE": str(root / "workspace")}
                child = subprocess.run([sys.executable, "-m", "tests.journal_crash_worker", phase, str(root)],
                                       cwd=root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
                self.assertEqual(child.returncode, code)
                journal = ExecutionJournal(root / "execution.sqlite3")
                row = journal.inspect()[0]
                self.assertEqual(row["recorded_status"], status)
                self.assertTrue(row["recovery_required"])
                effects = root / "effects.txt"
                expected = "" if phase == "before_tool" else "effect\n"
                self.assertEqual(effects.read_text() if effects.exists() else "", expected)
                model = f.RecordingModel(f.reply("reviewed; no repeat"))
                session = root / "session.json"
                before = session.read_bytes()
                agent = self.agent(model, journal_path=journal.path, session_path=session)
                result = agent.run_result("continue")
                self.assertEqual(result.stop_reason, "journal_recovery_required")
                self.assertEqual(model.calls, [])
                self.assertEqual(session.read_bytes(), before)
                journal.resolve(row["action_id"], "not_run" if phase == "before_tool" else "completed", confirm=True)
                self.assertEqual(agent.run_result("reviewed; do not repeat the previous action").status, "completed")
                self.assertEqual(len(model.calls), 1)
                self.assertEqual(effects.read_text() if effects.exists() else "", expected)
                self.assertEqual(journal.inspect()[0]["action_id"], row["action_id"])


class JournalCLITests(f.HarnessTestCase):
    def command(self, *arguments):
        return subprocess.run([sys.executable, "-m", "harness", *map(str, arguments)],
                              cwd=self.dir, env={"PATH": os.defpath, "PYTHONPATH": str(REPO)},
                              capture_output=True, text=True, timeout=10)

    def test_inspect_is_offline_read_only_and_resolution_requires_confirmation(self):
        missing = self.dir / "missing.sqlite3"
        child = self.command("inspect", missing)
        self.assertEqual(child.returncode, 1)
        self.assertFalse(missing.exists())
        self.assertFalse((self.dir / "workspace").exists())
        path = self.dir / "execution.sqlite3"
        journal = ExecutionJournal(path)
        journal.assert_ready()
        action_id = journal.begin("run", "call", "tool", True)
        before = path.read_bytes()
        child = self.command("inspect", path, "--pending", "--limit", "1")
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(json.loads(child.stdout)[0]["action_id"], action_id)
        self.assertEqual(path.read_bytes(), before)
        child = self.command("resolve", path, action_id, "--outcome", "not_run")
        self.assertNotEqual(child.returncode, 0)
        self.assertEqual(path.read_bytes(), before)
        child = self.command("resolve", path, action_id, "--outcome", "not_run", "--confirm")
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertIn("no action was executed", child.stdout)
        journal.assert_ready()

    def test_mock_cli_supports_journaling_with_and_without_isolated_workers(self):
        for timeout in (None, "3"):
            root = self.dir / ("inline" if timeout is None else "isolated")
            root.mkdir()
            path = root / "execution.sqlite3"
            env = {"PATH": os.defpath, "PYTHONPATH": str(REPO),
                   "HARNESS_WORKSPACE": str(root / "workspace"), "HARNESS_JOURNAL": str(path)}
            if timeout is not None:
                env["HARNESS_TOOL_TIMEOUT_SECONDS"] = timeout
            child = subprocess.run([sys.executable, "-m", "agents.teaching.cli", "mock"],
                                   cwd=root, env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(child.returncode, 0, child.stderr)
            self.assertNotIn("[stopped:", child.stdout)
            journal = ExecutionJournal(path)
            self.assertGreater(len(journal.inspect()), 0)
            self.assertEqual(journal.inspect(pending_only=True), [])
            self.assertTrue((root / "workspace/report.txt").exists())