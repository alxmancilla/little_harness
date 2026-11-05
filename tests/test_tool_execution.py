"""Real disposable workers and their integration with budgets and session safety."""

import json
import math
import os
import signal
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

from harness import tool_execution as execution
from tests import helpers as f
from tests import tool_fixtures as tools

from harness import agent as h


class WorkerTests(f.HarnessTestCase):
    def execute(self, fn, args=None, **kwargs):
        return execution.run_isolated_tool(fn, args or {}, kwargs.pop("timeout", 3),
                                           kwargs.pop("max_output", 8000), **kwargs)

    def test_success_is_bounded_and_worker_noise_cannot_corrupt_the_protocol(self):
        value = "hello \u2603"
        result = self.execute(tools.echo, {"value": value})
        self.assertEqual(result, execution.ToolOutcome("completed", value, False))
        large = self.execute(tools.large_output, max_output=100)
        self.assertEqual(large.status, "completed")
        self.assertTrue(large.output.startswith("x" * 100))
        self.assertIn("truncated, 20000 characters total", large.output)
        self.assertNotIn("noise-must-not-leak", self.stdout.getvalue())

    def test_worker_exceptions_are_sanitized_and_risky_failures_are_uncertain(self):
        for risky in (False, True):
            with self.subTest(risky=risky):
                result = self.execute(tools.fail, risky=risky)
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.outcome_unknown, risky)
                self.assertTrue(result.is_error)
                self.assertNotIn("exception-detail", result.output)

    def test_closures_and_callable_objects_are_not_silently_executed_inline(self):
        local = mock.Mock()
        for fn in (local, lambda: local()):
            result = self.execute(fn)
            self.assertEqual(result.status, "failed")
            self.assertFalse(result.outcome_unknown)
            self.assertIn("importable", result.output)
        local.assert_not_called()

    def test_parent_memory_is_not_changed_by_worker(self):
        self.assertEqual(self.execute(tools.change_memory).output, "1")
        self.assertEqual(self.execute(tools.change_memory).output, "1")
        self.assertEqual(tools.LOCAL_STATE, [])

    def test_timeout_is_bounded_and_reaps_a_term_resistant_worker(self):
        if os.name != "posix":
            self.skipTest("POSIX signal handling")
        marker = self.dir / "worker.pid"
        started = time.monotonic()
        result = self.execute(tools.stubborn_worker, {"path": str(marker)}, timeout=1)
        self.assertLess(time.monotonic() - started, 4)
        self.assertEqual(result.status, "timed_out")
        self.assertTrue(result.outcome_unknown)
        self.assertTrue(marker.exists(), "worker must have entered the tool")
        with self.assertRaises(ProcessLookupError):
            os.kill(int(marker.read_text()), 0)

    def test_start_failure_is_known_not_run_and_connection_failure_is_uncertain(self):
        with mock.patch.object(execution.subprocess, "Popen", side_effect=OSError("private detail")):
            result = self.execute(tools.echo, {"value": "hello"})
        self.assertFalse(result.outcome_unknown)
        self.assertIn("not run", result.output)
        self.assertNotIn("private detail", result.output)
        process = mock.Mock()
        process.communicate.side_effect = OSError("private detail")
        with mock.patch.object(execution.subprocess, "Popen", return_value=process), \
                mock.patch.object(execution, "_stop_worker") as cleanup:
            result = self.execute(tools.echo, {"value": "hello"})
        self.assertTrue(result.outcome_unknown)
        cleanup.assert_called_once_with(process)

    def test_invalid_worker_responses_are_uncertain_not_successful(self):
        for output in (b"not json", b"{}", b"[]", b'{"status":"completed","output":null}'):
            process = mock.Mock(returncode=0)
            process.communicate.return_value = (output, None)
            with self.subTest(output=output), \
                    mock.patch.object(execution.subprocess, "Popen", return_value=process), \
                    mock.patch.object(execution, "_stop_worker"):
                result = self.execute(tools.echo, {"value": "hello"})
            self.assertEqual(result.status, "outcome_unknown")

    def test_cleanup_failure_does_not_replace_an_uncertain_outcome(self):
        process = mock.Mock(pid=123)
        process.communicate.side_effect = subprocess.TimeoutExpired("worker", 1)
        process.wait.side_effect = subprocess.TimeoutExpired("worker", 1)
        with mock.patch.object(execution.subprocess, "Popen", return_value=process), \
                mock.patch.object(execution.os, "killpg", create=True):
            result = self.execute(tools.echo, {"value": "hello"})
        self.assertEqual(result.status, "timed_out")
        self.assertEqual(process.wait.call_count, 2)

    @unittest.skipUnless(os.name == "posix", "POSIX process group cleanup")
    def test_descendants_are_stopped_after_success_crash_and_timeout(self):
        for mode in ("done", "crash", "hang"):
            marker = self.dir / (mode + ".pid")
            result = self.execute(tools.spawn_child, {"path": str(marker), "mode": mode}, timeout=1)
            self.assertEqual(result.status, {"done": "completed", "crash": "outcome_unknown",
                                             "hang": "timed_out"}[mode])
            self.assertTrue(marker.exists())
            pid = int(marker.read_text())
            # Orphaned children may remain zombies until init reaps them. A zombie is stopped.
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True,
                                       text=True, timeout=2).stdout.strip()
                if not state or state.startswith("Z"):
                    break
                time.sleep(0.02)
            else:
                os.kill(pid, signal.SIGKILL)
                self.fail("worker descendant was still running after cleanup")


class TimeoutIntegrationTests(f.HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.register_tool(tools.record_action, risky=True)

    def test_timeout_crash_and_risky_failure_stop_without_retries_and_save_the_whole_batch(self):
        for mode in ("hang", "crash", "fail"):
            with self.subTest(mode=mode):
                effects = self.dir / (mode + ".txt")
                batch = f.tool_calls(*[(cid, "record_action", {"path": str(effects), "mode": action})
                                       for cid, action in (("a", "done"), ("b", mode), ("c", "later"))])
                model = f.RecordingModel(batch, f.reply("must not call"))
                session = self.dir / (mode + "-session.json")
                agent = self.agent(model, tool_timeout_seconds=1, session_path=session)
                result = agent.run_result("go")
                reason = "tool_timeout" if mode == "hang" else "tool_outcome_unknown"
                self.assertEqual((result.status, result.stop_reason), ("stopped", reason))
                self.assertEqual(effects.read_text().splitlines(), ["done", mode])
                self.assertEqual(len(model.calls), 1)
                self.assertEqual(result.usage.tool_calls_requested, 3)
                self.assertEqual(result.usage.tool_calls_attempted, 2)
                self.assertEqual(result.usage.tool_errors, 2)
                saved = f.saved_messages(session)[-1]["content"]
                self.assertEqual([r["tool_use_id"] for r in saved], ["a", "b", "c"])
                self.assertEqual(saved[0]["content"], "recorded done")
                self.assertFalse(saved[0]["is_error"])
                self.assertIn("outcome unknown", saved[1]["content"])
                self.assertIn("not run", saved[2]["content"])
                self.assertEqual(agent.messages[-1]["content"], saved)
                last_tool = [r for r in self.trace_records() if r["kind"] == "tool"][-1]
                self.assertTrue(last_tool["outcome_unknown"])
                self.assertEqual(last_tool["execution_status"],
                                 {"hang": "timed_out", "crash": "outcome_unknown", "fail": "failed"}[mode])
                self.assertNotIn("exception-detail", self.trace.read_text())
                resumed = self.agent([f.reply("reviewed")], session_path=session)
                self.assertEqual(resumed.messages, agent.messages)

    def test_validation_denial_and_bad_configuration_do_not_launch_workers(self):
        approve = mock.Mock(return_value=False)
        calls = [("a", "record_action", {"path": 3}),
                 ("b", "record_action", {"path": str(self.dir / "no.txt")}),
                 ("c", "missing", {})]
        with mock.patch.object(h, "run_isolated_tool") as execute:
            result = self.agent([f.tool_calls(*calls), f.reply("done")], approve=approve,
                                tool_timeout_seconds=1).run_result("go")
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.usage.tool_errors, 3)
        approve.assert_called_once()
        execute.assert_not_called()
        with mock.patch.dict(self.tools["record_action"], {"timeout_seconds": -1}), \
                mock.patch.object(h, "run_isolated_tool") as execute:
            agent = self.agent([], approve=approve, tool_timeout_seconds=1)
            outcome = agent._run_tool(f.tool_calls(calls[1])["content"][0])
        self.assertTrue(outcome.is_error)
        self.assertFalse(outcome.dispatched)
        self.assertIn("configuration", outcome.output)
        execute.assert_not_called()
        approve.assert_called_once()

    def test_per_tool_override_does_not_change_the_advertised_schema(self):
        self.tools.tool(timeout_seconds=0.25)(tools.echo)
        self.assertEqual(self.tools["echo"]["timeout_seconds"], 0.25)
        self.assertNotIn("timeout_seconds", self.tools["echo"]["schema"])
        agent = self.agent([], tool_timeout_seconds=3)
        with mock.patch.object(h, "run_isolated_tool", return_value=execution.ToolOutcome("completed", "ok", False)) as run:
            outcome = agent._run_tool(f.tool_calls(("a", "echo", {"value": "ok"}))["content"][0])
        self.assertEqual(outcome, h.ToolCallOutcome("ok", False, execution.ToolOutcome("completed", "ok", False),
                                                     dispatched=True))
        self.assertEqual(run.call_args.args[2], 0.25)

    def test_remaining_task_deadline_shortens_an_isolated_tool_timeout(self):
        effects = self.dir / "deadline.txt"
        model = f.RecordingModel(f.tool_calls(("a", "record_action", {"path": str(effects), "mode": "hang"})))
        result = self.agent(model, limits=h.RunLimits(max_seconds=1), tool_timeout_seconds=10).run_result("go")
        self.assertEqual((result.status, result.stop_reason), ("stopped", "max_seconds"))
        self.assertLess(result.elapsed_seconds, 4)
        self.assertEqual(effects.read_text().splitlines(), ["hang"])
        self.assertEqual(len(model.calls), 1)

    def test_deadline_during_approval_prevents_worker_start(self):
        now = [0.0]

        def approve(*args):
            now[0] = 2.0
            return True

        batch = f.tool_calls(("a", "record_action", {"path": str(self.dir / "no.txt")}))
        agent = self.agent([batch], approve=approve, limits=h.RunLimits(max_seconds=1), tool_timeout_seconds=3)
        with mock.patch("harness.runtime.time.monotonic", side_effect=lambda: now[0]), \
                mock.patch.object(h, "run_isolated_tool") as execute:
            result = agent.run_result("go")
        self.assertEqual(result.stop_reason, "max_seconds")
        execute.assert_not_called()

    def test_builtin_file_tools_use_parent_selected_workspace_in_the_worker(self):
        workspace = self.config.workspace
        workspace.mkdir()
        batch = f.tool_calls(("a", "write_file", {"path": "result.txt", "content": "saved"}),
                             ("b", "read_file", {"path": "result.txt"}))
        agent = self.agent([batch, f.reply("done")], tool_timeout_seconds=3)
        result = agent.run_result("go")
        self.assertEqual(result.status, "completed")
        self.assertEqual((workspace / "result.txt").read_text(), "saved")
        self.assertEqual(agent.messages[-2]["content"][1]["content"], "saved")

    def test_cancellation_reaps_a_real_worker_and_preserves_completed_and_unstarted_calls(self):
        effects = self.dir / "interrupted.txt"
        batch = f.tool_calls(("a", "calculator", {"expression": "2+2"}),
                             ("b", "record_action", {"path": str(effects), "mode": "hang"}),
                             ("c", "record_action", {"path": str(effects), "mode": "later"}))
        session = self.dir / "interrupted-session.json"
        agent = self.agent([batch], tool_timeout_seconds=5, session_path=session)
        original_popen = execution.subprocess.Popen
        children = []

        def popen(*args, **kwargs):
            child = original_popen(*args, **kwargs)
            children.append(child)
            if len(children) == 2:
                def interrupt(data, timeout):
                    # Deliver stdin, then interrupt while the tool is known to be in flight.
                    child.stdin.write(data)
                    child.stdin.close()
                    deadline = time.monotonic() + 3
                    while not effects.exists() and time.monotonic() < deadline:
                        time.sleep(0.01)
                    self.assertTrue(effects.exists())
                    raise KeyboardInterrupt

                child.communicate = interrupt
            return child

        with mock.patch.object(execution.subprocess, "Popen", side_effect=popen):
            result = agent.run_result("go")
        self.assertEqual(result.status, "interrupted")
        self.assertEqual(len(children), 2)
        self.assertTrue(all(child.poll() is not None for child in children))
        self.assertEqual(effects.read_text().splitlines(), ["hang"])
        saved = f.saved_messages(session)[-1]["content"]
        self.assertEqual(saved[0]["content"], "4")
        self.assertIn("outcome unknown", saved[1]["content"])
        self.assertIn("not run", saved[2]["content"])
        self.assertEqual(result.usage.tool_calls_attempted, 2)

    def test_timeout_values_are_positive_finite_and_validated_without_echoing_them(self):
        for value in (0, -1, True, "private-invalid-setting", math.inf, math.nan):
            with self.subTest(value=value):
                for operation in (lambda: self.tools.tool(timeout_seconds=value),
                                  lambda: self.agent([], tool_timeout_seconds=value)):
                    with self.assertRaises(ValueError) as raised:
                        operation()
                    self.assertNotIn("private-invalid-setting", str(raised.exception))

    def test_cli_mock_can_execute_application_tools_in_workers(self):
        env = {"PATH": os.defpath, "PYTHONPATH": str(f.REPO), "HARNESS_WORKSPACE": str(self.dir / "workspace"),
               "HARNESS_TOOL_TIMEOUT_SECONDS": "3"}
        child = subprocess.run([sys.executable, "-m", "agents.teaching.cli", "mock"], cwd=self.dir,
                               env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertIn("=== FINAL ===", child.stdout)
        self.assertNotIn("[stopped:", child.stdout)
        self.assertTrue((self.dir / "workspace/report.txt").exists())
        for value in ("0", "-1", "nan", "inf", "private-invalid-setting"):
            env["HARNESS_TOOL_TIMEOUT_SECONDS"] = value
            child = subprocess.run([sys.executable, "-m", "agents.teaching.cli", "mock"], cwd=self.dir,
                                   env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(child.returncode, 1)
            self.assertIn("Invalid execution limits", child.stderr)
            self.assertNotIn("private-invalid-setting", child.stderr)