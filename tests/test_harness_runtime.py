"""Core budgets/results, without real clocks, network calls or provider credentials."""

import json
import math
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest import mock

from tests import helpers as f
from tests.helpers import used

from harness import agent as h


class RuntimeTests(f.HarnessTestCase):
    def test_result_has_explicit_status_clean_answer_and_serializable_usage(self):
        result = self.agent([used(f.reply("[stopped: just quoted text]"))]).run_result("go")
        self.assertEqual((result.status, result.stop_reason), ("completed", "end_turn"))
        self.assertEqual(result.answer, result.text)
        self.assertEqual(result.usage.model_calls, 1)
        self.assertEqual(result.usage.total_tokens, 5)
        self.assertGreaterEqual(result.elapsed_seconds, 0)
        self.assertEqual(json.loads(json.dumps(result.to_dict()))["usage"]["input_tokens"], 3)
        with self.assertRaises(FrozenInstanceError):
            result.status = "changed"
        with self.assertRaises(FrozenInstanceError):
            result.usage.model_calls = 9

    def test_terminal_statuses_do_not_need_trace_parsing(self):
        for response, status, reason in ((f.reply("cut", "max_tokens"), "max_tokens", "max_tokens"),
                                         (f.reply("no", "refusal"), "refusal", "refusal"),
                                         (h.ModelError("offline failure"), "model_error", "model_error"),
                                         (KeyboardInterrupt(), "interrupted", "interrupted")):
            with self.subTest(status=status):
                result = self.agent(f.RecordingModel(response)).run_result("go")
                self.assertEqual((result.status, result.stop_reason), (status, reason))
                self.assertEqual(result.usage.model_calls, 1)
                if isinstance(response, BaseException):
                    self.assertEqual(result.answer, "")
                    self.assertIsNone(result.usage.total_tokens)
                else:
                    self.assertNotIn("warning", result.answer)
                    self.assertIn("warning", result.text)

    def test_step_exhaustion_has_structured_and_display_results(self):
        result = self.agent([f.tool_calls(("a", "current_time", {}))], max_steps=1).run_result("go")
        self.assertEqual((result.status, result.stop_reason), ("stopped", "max_steps"))
        self.assertEqual(result.text, "[stopped: hit the 1 step budget]")

    def test_counts_invalid_denied_unknown_and_truncated_requests_honestly(self):
        batch = f.tool_calls(("ok", "calculator", {"expression": "2+2"}),
                             ("bad", "calculator", {}),
                             ("denied", "write_file", {"path": "no.txt", "content": "no"}),
                             ("unknown", "missing", {}), ("cut", "write_file", {}))
        batch["truncated_tool_ids"] = ["cut"]
        approve = mock.Mock(return_value=False)
        result = self.agent([used(batch), used(f.reply("done"))], approve=approve).run_result("go")
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.usage.tool_calls_requested, 5)
        self.assertEqual(result.usage.tool_calls_attempted, 4)
        self.assertEqual(result.usage.tool_errors, 4)
        self.assertEqual(result.usage.total_tokens, 10)
        approve.assert_called_once()

    def test_batch_budget_stop_preserves_completed_and_unstarted_tool_results(self):
        execute = mock.Mock(return_value="done")
        model = f.RecordingModel(f.tool_calls(("a", "current_time", {}), ("b", "current_time", {})),
                                 f.reply("must not call"))
        session = self.dir / "session.json"
        agent = self.agent(model, limits=h.RunLimits(max_tool_calls=1), session_path=session)
        with mock.patch.dict(self.tools["current_time"], {"fn": execute}):
            result = agent.run_result("go")
        self.assertEqual((result.status, result.stop_reason), ("stopped", "max_tool_calls"))
        self.assertEqual(result.usage.tool_calls_requested, 2)
        self.assertEqual(result.usage.tool_calls_attempted, 1)
        self.assertEqual(result.usage.tool_errors, 1)
        execute.assert_called_once()
        self.assertEqual(len(model.calls), 1)
        saved = json.loads(session.read_text())[-1]["content"]
        self.assertEqual([r["tool_use_id"] for r in saved], ["a", "b"])
        self.assertEqual(saved[0]["content"], "done")
        self.assertIn("not run", saved[1]["content"])

    def test_model_cap_includes_successful_and_failed_compaction(self):
        for summary in (used(f.reply("SUMMARY")), h.ModelError("summary unavailable")):
            with self.subTest(failed=isinstance(summary, BaseException)):
                model = f.RecordingModel(summary, used(f.reply("must not call")))
                agent = self.agent(model, compact_at=1500, limits=h.RunLimits(max_model_calls=1))
                agent.messages = f.compaction_history()
                result = agent.run_result("continue")
                self.assertEqual(result.stop_reason, "max_model_calls")
                self.assertEqual(result.usage.model_calls, 1)
                self.assertEqual(len(model.calls), 1)

    def test_model_cap_stops_a_loop_before_the_next_call(self):
        model = f.RecordingModel(used(f.tool_calls(("a", "current_time", {}))), f.reply("must not call"))
        result = self.agent(model, limits=h.RunLimits(max_model_calls=1)).run_result("go")
        self.assertEqual(result.stop_reason, "max_model_calls")
        self.assertEqual(result.usage.model_calls, 1)
        self.assertEqual(result.usage.tool_calls_attempted, 1)
        self.assertEqual(len(model.calls), 1)

    def test_invalid_arguments_consume_attempt_budget_before_later_valid_actions(self):
        execute = mock.Mock()
        batch = f.tool_calls(("bad", "current_time", {"extra": True}), ("good", "current_time", {}))
        with mock.patch.dict(self.tools["current_time"], {"fn": execute}):
            result = self.agent([batch], limits=h.RunLimits(max_tool_calls=1)).run_result("go")
        self.assertEqual(result.stop_reason, "max_tool_calls")
        self.assertEqual(result.usage.tool_calls_attempted, 1)
        self.assertEqual(result.usage.tool_errors, 2)
        execute.assert_not_called()

    def test_compaction_and_main_call_share_token_usage(self):
        agent = self.agent([used(f.reply("SUMMARY")), used(f.reply("answer"), 7, 4)], compact_at=1500,
                           limits=h.RunLimits(max_model_calls=2, max_tokens=16))
        agent.messages = f.compaction_history()
        result = agent.run_result("continue")
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.usage.model_calls, 2)
        self.assertEqual(result.usage.input_tokens, 10)
        self.assertEqual(result.usage.output_tokens, 6)

    def test_compaction_token_exhaustion_prevents_main_call(self):
        model = f.RecordingModel(used(f.reply("SUMMARY")), used(f.reply("must not call")))
        agent = self.agent(model, compact_at=1500, limits=h.RunLimits(max_tokens=5))
        agent.messages = f.compaction_history()
        result = agent.run_result("continue")
        self.assertEqual(result.stop_reason, "max_total_tokens")
        self.assertEqual(len(model.calls), 1)

    def test_missing_or_invalid_token_usage_stops_work_when_a_cap_is_requested(self):
        for value in (None, -1, True, 2.5, "3"):
            with self.subTest(value=value):
                model = f.RecordingModel(used(f.tool_calls(("a", "current_time", {})), value),
                                         used(f.reply("must not call")))
                execute = mock.Mock()
                with mock.patch.dict(self.tools["current_time"], {"fn": execute}):
                    result = self.agent(model, limits=h.RunLimits(max_tokens=100)).run_result("go")
                self.assertEqual(result.stop_reason, "usage_unavailable")
                self.assertIsNone(result.usage.input_tokens)
                self.assertEqual(result.usage.output_tokens, 2)
                self.assertEqual(result.usage.tool_calls_attempted, 0)
                execute.assert_not_called()
                self.assertEqual(len(model.calls), 1)

    def test_unknown_usage_stays_unknown_without_disabling_normal_runs(self):
        result = self.agent([f.tool_calls(("a", "current_time", {})), used(f.reply("ok"))]).run_result("go")
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.usage.model_calls, 2)
        self.assertIsNone(result.usage.input_tokens)
        self.assertIsNone(result.usage.output_tokens)

    def test_tokens_at_cap_block_tools_and_overshoot_retains_final_answer(self):
        for response, limit in ((used(f.tool_calls(("a", "current_time", {}))), 5),
                                (used(f.reply("answer")), 4)):
            with self.subTest(limit=limit):
                agent = self.agent([response], limits=h.RunLimits(max_tokens=limit))
                result = agent.run_result("go")
                self.assertEqual(result.stop_reason, "max_total_tokens")
                self.assertEqual(result.usage.total_tokens, 5)
                self.assertEqual(result.usage.tool_calls_attempted, 0)
                if limit == 4:
                    self.assertEqual(result.answer, "answer")

    def test_deadline_after_model_response_prevents_tools_and_pairs_results(self):
        now = [0.0]

        def slow_model(*args, **kwargs):
            now[0] = 2.0
            return used(f.tool_calls(("a", "current_time", {})))

        agent = self.agent(slow_model, limits=h.RunLimits(max_seconds=1))
        with mock.patch("harness.runtime.time.monotonic", side_effect=lambda: now[0]):
            result = agent.run_result("go")
        self.assertEqual(result.stop_reason, "max_seconds")
        self.assertEqual(result.elapsed_seconds, 2)
        self.assertEqual(result.usage.tool_calls_attempted, 0)
        self.assertEqual(agent.messages[-1]["content"][0]["tool_use_id"], "a")

    def test_deadline_during_approval_does_not_execute_the_approved_action(self):
        now = [0.0]

        def approve(*args):
            now[0] = 2.0
            return True

        execute = mock.Mock()
        agent = self.agent([used(f.tool_calls(("a", "write_file", {"path": "x", "content": "x"})))],
                           approve=approve, limits=h.RunLimits(max_seconds=1))
        with mock.patch("harness.runtime.time.monotonic", side_effect=lambda: now[0]), \
                mock.patch.dict(self.tools["write_file"], {"fn": execute}):
            result = agent.run_result("go")
        self.assertEqual(result.stop_reason, "max_seconds")
        self.assertEqual(result.usage.tool_calls_attempted, 1)
        execute.assert_not_called()

    def test_deadline_after_tool_preserves_its_result_without_automatic_retry(self):
        now = [0.0]

        def slow_tool():
            now[0] = 2.0
            return "action completed"

        model = f.RecordingModel(used(f.tool_calls(("a", "current_time", {}))), f.reply("must not call"))
        agent = self.agent(model, limits=h.RunLimits(max_seconds=1))
        with mock.patch("harness.runtime.time.monotonic", side_effect=lambda: now[0]), \
                mock.patch.dict(self.tools["current_time"], {"fn": slow_tool}):
            result = agent.run_result("go")
        self.assertEqual(result.stop_reason, "max_seconds")
        self.assertEqual(agent.messages[-1]["content"][0]["content"], "action completed")
        self.assertFalse(agent.messages[-1]["content"][0]["is_error"])
        self.assertEqual(len(model.calls), 1)

    def test_deadline_after_final_response_is_not_reported_as_success(self):
        now = [0.0]

        def slow_model(*args, **kwargs):
            now[0] = 2.0
            return used(f.reply("late answer"))

        with mock.patch("harness.runtime.time.monotonic", side_effect=lambda: now[0]):
            result = self.agent(slow_model, limits=h.RunLimits(max_seconds=1)).run_result("go")
        self.assertEqual((result.status, result.stop_reason), ("stopped", "max_seconds"))
        self.assertEqual(result.answer, "late answer")

    def test_interrupted_tool_batch_counts_attempts_without_replaying_completed_work(self):
        execute = mock.Mock(side_effect=["completed", KeyboardInterrupt()])
        batch = used(f.tool_calls(("a", "current_time", {}), ("b", "current_time", {}),
                                  ("c", "current_time", {})))
        with mock.patch.dict(self.tools["current_time"], {"fn": execute}):
            result = self.agent([batch]).run_result("go")
        self.assertEqual(result.status, "interrupted")
        self.assertEqual(result.usage.tool_calls_requested, 3)
        self.assertEqual(result.usage.tool_calls_attempted, 2)
        self.assertEqual(result.usage.tool_errors, 2)
        self.assertEqual(result.usage.total_tokens, 5)
        self.assertEqual(execute.call_count, 2)

    def test_limits_reset_per_task_and_old_results_do_not_change(self):
        agent = self.agent([used(f.reply("first")), used(f.reply("second"), 8, 9)],
                           limits=h.RunLimits(max_model_calls=1))
        first = agent.run_result("one")
        second = agent.run_result("two")
        self.assertEqual((first.answer, second.answer), ("first", "second"))
        self.assertEqual(first.usage.model_calls, second.usage.model_calls)
        self.assertEqual(first.usage.total_tokens, 5)
        self.assertEqual(second.usage.total_tokens, 17)

    def test_zero_limits_prevent_the_corresponding_operation(self):
        for name, reason in (("max_model_calls", "max_model_calls"), ("max_tokens", "max_total_tokens"),
                             ("max_seconds", "max_seconds")):
            with self.subTest(name=name):
                model = f.RecordingModel(f.reply("must not call"))
                result = self.agent(model, limits=h.RunLimits(**{name: 0})).run_result("go")
                self.assertEqual(result.stop_reason, reason)
                self.assertEqual(result.usage.model_calls, 0)
                self.assertEqual(result.usage.total_tokens, 0)
                self.assertEqual(model.calls, [])

    def test_programming_errors_still_propagate_and_clear_active_accounting(self):
        agent = self.agent(f.RecordingModel(RuntimeError("programming bug")))
        with self.assertRaisesRegex(RuntimeError, "programming bug"):
            agent.run_result("go")
        self.assertIsNone(agent._budget)

    def test_limit_configuration_is_validated_without_echoing_values(self):
        for name in ("max_model_calls", "max_tool_calls", "max_tokens", "max_seconds"):
            for value in (-1, True, "1"):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    h.RunLimits(**{name: value})
        for value in (math.inf, math.nan):
            with self.assertRaises(ValueError):
                h.RunLimits(max_seconds=value)
        self.assertEqual(h.RunLimits.from_env({"HARNESS_MAX_MODEL_CALLS": "2", "HARNESS_MAX_SECONDS": "0.5"}),
                         h.RunLimits(max_model_calls=2, max_seconds=0.5))
        with self.assertRaises(ValueError) as raised:
            h.RunLimits.from_env({"HARNESS_MAX_TOKENS": "private-invalid-setting"})
        self.assertNotIn("private-invalid-setting", str(raised.exception))

    def test_cli_applies_limits_before_paid_calls_or_actions_without_echoing_bad_values(self):
        command = [sys.executable, "-m", "agents.teaching.cli"]
        cases = (("HARNESS_MAX_MODEL_CALLS", "0", "max_model_calls"),
                 ("HARNESS_MAX_TOOL_CALLS", "0", "max_tool_calls"),
                 ("HARNESS_MAX_SECONDS", "0", "max_seconds"),
                 ("HARNESS_MAX_TOKENS", "100", "usage_unavailable"),
                 ("HARNESS_MAX_TOKENS", "private-invalid-setting", "Invalid execution limits"))
        for number, (name, value, expected) in enumerate(cases):
            root = self.dir / str(number)
            root.mkdir()
            env = {"PATH": os.defpath, "PYTHONPATH": str(f.REPO),
                   "HARNESS_WORKSPACE": str(root / "workspace"), name: value}
            with self.subTest(setting=name, value=value):
                child = subprocess.run([*command, "mock"], cwd=root, env=env,
                                       capture_output=True, text=True, timeout=10)
                self.assertIn(expected, child.stdout + child.stderr)
                self.assertNotIn("private-invalid-setting", child.stdout + child.stderr)
                self.assertFalse((root / "workspace/report.txt").exists())
                self.assertEqual(child.returncode, 1 if value == "private-invalid-setting" else 0)