"""Fail-closed dispatch using a non-durable journal test double."""

import json
import os
import subprocess
import sys
from unittest import mock

from harness import JournalError
from tests import helpers as f
from tests import tool_fixtures as tools
from tests.storage_fakes import FakeJournal
from tests.test_storage_postgres import ENV, SchemaPerTest, needs_database
from stores.postgres import PostgresJournal

from harness import agent as h
class JournalIntegrationTests(f.HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.journal = FakeJournal()

    def agent(self, model, **kwargs):
        kwargs.setdefault("journal", self.journal)
        return super().agent(model, **kwargs)

    def test_start_is_committed_before_inline_execution_and_no_payload_is_stored(self):
        observed = []

        def sample(value: str):
            observed.extend(self.journal.inspect())
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
        self.assertNotIn("synthetic-argument", json.dumps(agent.journal.inspect()))
        self.assertNotIn("synthetic-output", json.dumps(agent.journal.inspect()))
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
                journal = FakeJournal()
                effect = self.dir / (mode + ".txt")
                batch = f.tool_calls(("a", "record_action", {"path": str(effect), "mode": mode}))
                agent = self.agent([batch], journal=journal, tool_timeout_seconds=1)
                result = agent.run_result("go")
                row = agent.journal.inspect()[0]
                self.assertEqual(row["status"], "timed_out" if mode == "hang" else "outcome_unknown")
                self.assertEqual(effect.read_text().splitlines(), [mode])
                self.assertTrue(row["recovery_required"])
                self.assertEqual(result.status, "stopped")
                self.assertEqual(self.agent([], journal=journal).run_result("go").stop_reason,
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

    def test_unavailable_journal_stops_before_any_model_call(self):
        model = f.RecordingModel(f.reply("must not call"))
        agent = self.agent(model)
        with mock.patch.object(agent.journal, "assert_ready", side_effect=JournalError()):
            result = agent.run_result("go")
        self.assertEqual((result.status, result.stop_reason), ("stopped", "journal_error"))
        self.assertEqual(result.usage.model_calls, 0)
        self.assertEqual(model.calls, [])

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

    def test_reused_provider_ids_get_distinct_action_and_run_ids(self):
        call = f.tool_calls(("same-id", "calculator", {"expression": "2+2"}))
        agent = self.agent([call, f.reply("done"), call, f.reply("done again")])
        self.assertEqual(agent.run_result("one").status, "completed")
        self.assertEqual(agent.run_result("two").status, "completed")
        rows = agent.journal.inspect()
        self.assertEqual(len({r["action_id"] for r in rows}), 2)
        self.assertEqual(len({r["run_id"] for r in rows}), 2)


@needs_database
class CrashRecoveryTests(SchemaPerTest, f.HarnessTestCase):
    def test_abrupt_parent_exit_before_tool_after_effect_after_commit_and_after_session_save(self):
        phases = (("before_tool", 21, "started"), ("after_effect", 22, "started"),
                  ("after_finish", 23, "completed"), ("after_save", 24, "completed"))
        for phase, code, status in phases:
            with self.subTest(phase=phase):
                root = self.dir / phase
                root.mkdir()
                env = {"PATH": os.defpath, "PYTHONPATH": str(f.REPO), ENV: os.environ[ENV]}
                child = subprocess.run([sys.executable, "-m", "tests.journal_crash_worker", phase, str(root), self.schema],
                                       cwd=root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
                self.assertEqual(child.returncode, code)
                journal = PostgresJournal(phase, dsn_env=ENV, schema=self.schema)
                row = journal.inspect()[0]
                self.assertEqual(row["recorded_status"], status)
                self.assertTrue(row["recovery_required"])
                effects = root / "effects.txt"
                expected = "" if phase == "before_tool" else "effect\n"
                self.assertEqual(effects.read_text() if effects.exists() else "", expected)
                model = f.RecordingModel(f.reply("reviewed; no repeat"))
                session = root / "session.json"
                before = session.read_bytes()
                agent = self.agent(model, journal=journal, session_path=session)
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
    def test_teaching_cli_selects_and_passes_each_database_journal(self):
        from agents.teaching import cli

        for backend, target, connection_option in (
                ("postgres", "stores.postgres.PostgresJournal", {"dsn_env": "HARNESS_POSTGRES_DSN"}),
                ("mongodb", "stores.mongodb.MongoJournal", {"uri_env": "HARNESS_MONGODB_URI"})):
            with self.subTest(backend=backend), \
                    mock.patch.dict(os.environ, {"HARNESS_JOURNAL_BACKEND": backend,
                                                 "HARNESS_JOURNAL_ID": "teaching"}, clear=True), \
                    mock.patch.object(sys, "argv", ["cli", "mock"]), \
                    mock.patch.object(cli, "load_env"), mock.patch(target) as journal_factory, \
                    mock.patch.object(cli, "create_agent") as create_agent, mock.patch("builtins.print"):
                create_agent.return_value.run.return_value = "done"
                cli.main()
                journal_factory.assert_called_once_with("teaching", **connection_option)
                self.assertIs(create_agent.call_args.kwargs["journal"], journal_factory.return_value)

    def test_legacy_journal_setting_is_rejected_without_creating_or_modifying_file(self):
        path = self.dir / "old-journal"
        path.write_bytes(b"original journal")
        env = {"PATH": os.defpath, "PYTHONPATH": str(f.REPO), "HARNESS_JOURNAL": str(path)}
        child = subprocess.run([sys.executable, "-m", "agents.teaching.cli", "mock"],
                               cwd=self.dir, env=env, capture_output=True, text=True, timeout=20)
        self.assertNotEqual(child.returncode, 0)
        self.assertIn("no longer supported", child.stderr)
        self.assertEqual(path.read_bytes(), b"original journal")
        self.assertFalse((self.dir / "workspace").exists())