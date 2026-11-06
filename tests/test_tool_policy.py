"""Opt-in dispatch policy safety, without model/network calls or real writes."""

import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest import mock

from harness import ToolPolicy
from tests import helpers as f
from tests import tool_fixtures
from tests.storage_fakes import FakeJournal as fake_journal

from harness import agent as h


class PolicyTests(f.HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.events = []

    def agent(self, model, **kwargs):
        kwargs.setdefault("on_event", self.events.append)
        kwargs.setdefault("quiet", True)
        return super().agent(model, **kwargs)

    def select(self, kind):
        return [event.data for event in self.events if event.kind == kind]

    def test_builtin_declarations_are_explicit_and_not_part_of_model_schemas(self):
        names = {name for name, entry in self.tools.items() if entry.get("read_only") is True}
        self.assertEqual(names, {"calculator", "current_time", "list_files", "read_file", "search_documents"})
        self.assertFalse(self.tools["write_file"]["read_only"])
        self.assertFalse(self.tools["http_get"]["read_only"])
        self.assertTrue(all("read_only" not in entry["schema"] for entry in self.tools.values()))

    def test_read_only_hides_blocked_schemas_but_still_enforces_hallucinated_requests(self):
        advertised = []
        execute = mock.Mock()
        approve = mock.Mock(return_value=True)
        batch = f.tool_calls(("write", "write_file", {"path": "report", "content": "data"}),
                             ("http", "http_get", {"url": "https://example.invalid"}),
                             ("unclassified", "custom", {}), ("safe", "calculator", {"expression": "2+2"}))
        responses = iter([batch, f.reply("done")])

        def model(system, messages, tools, on_text=None):
            advertised.append({tool["name"] for tool in tools})
            return next(responses)

        custom = dict(self.tools["current_time"], fn=execute)
        custom.pop("read_only")
        tools = dict(self.tools, custom=custom)
        tools["write_file"] = dict(tools["write_file"], fn=execute)
        tools["http_get"] = dict(tools["http_get"], fn=execute)
        agent = self.agent(model, tools=tools, tool_policy=ToolPolicy(read_only=True), approve=approve,
                           journal=fake_journal(), tool_timeout_seconds=3)
        with mock.patch.object(h, "run_isolated_tool", return_value=h.ToolOutcome("completed", "4", False)) as worker:
            result = agent.run_result("go")
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.usage.tool_calls_attempted, 4)
        self.assertEqual(result.usage.tool_errors, 3)
        self.assertTrue(all("write_file" not in names and "http_get" not in names and "custom" not in names
                            for names in advertised))
        approve.assert_not_called()
        execute.assert_not_called()
        worker.assert_called_once()
        self.assertEqual([row["tool_name"] for row in agent.journal.inspect()], ["calculator"])
        self.assertEqual([e["status"] for e in self.select("tool_finished")],
                         ["not_run", "not_run", "not_run", "completed"])

    def test_read_only_uses_exact_true_and_does_not_equate_nonrisky_with_readonly(self):
        for declaration in (None, False, 1, "yes", True):
            with self.subTest(declaration=declaration):
                entry = dict(self.tools["current_time"], fn=mock.Mock(return_value="ok"))
                entry["read_only"] = declaration
                agent = self.agent([f.tool_calls(("a", "current_time", {})), f.reply("done")],
                                   tools={"current_time": entry}, tool_policy=ToolPolicy(read_only=True))
                agent.run("go")
                self.assertEqual(entry["fn"].call_count, int(declaration is True))

    def test_read_only_does_not_bypass_approval_for_a_risky_declared_reader(self):
        entry = dict(self.tools["current_time"], risky=True, fn=mock.Mock(return_value="ok"))
        approve = mock.Mock(return_value=False)
        agent = self.agent([f.tool_calls(("a", "current_time", {})), f.reply("done")],
                           tools={"current_time": entry}, approve=approve, tool_policy=ToolPolicy(read_only=True))
        agent.run("go")
        approve.assert_called_once()
        entry["fn"].assert_not_called()
        self.assertEqual(self.select("tool_policy")[0]["reason"], "allowed")

    def test_argument_checks_require_literal_true_and_denial_is_recoverable(self):
        for decision in (False, None, 1, "yes", {}, True):
            with self.subTest(decision=decision):
                execute = mock.Mock(return_value="ok")
                model = f.RecordingModel(f.tool_calls(("a", "current_time", {})), f.reply("done"))
                policy = ToolPolicy(argument_checks={"current_time": lambda args: decision})
                with mock.patch.dict(self.tools["current_time"], {"fn": execute}):
                    result = self.agent(model, tool_policy=policy).run_result("go")
                self.assertEqual(result.status, "completed")
                self.assertEqual(execute.call_count, int(decision is True))
                self.assertEqual(result.usage.tool_errors, int(decision is not True))
                self.assertEqual(len(model.calls), 2)

    def test_validation_precedes_policy_and_policy_precedes_approval_journal_and_execution(self):
        order = []

        def check(args):
            order.append("policy")
            return True

        def approve(*args):
            order.append("approval")
            return True

        def execute(**args):
            order.append("execution")
            return "done"

        policy = ToolPolicy(argument_checks={"write_file": check})
        agent = self.agent([f.tool_calls(("invalid", "write_file", {}),
                                        ("valid", "write_file", {"path": "note", "content": "data"})),
                            f.reply("done")], tool_policy=policy, approve=approve,
                           journal=fake_journal())
        begin = agent.journal.begin

        def record(*args):
            order.append("journal")
            return begin(*args)

        with mock.patch.dict(self.tools["write_file"], {"fn": execute}), \
                mock.patch.object(agent.journal, "begin", side_effect=record):
            result = agent.run_result("go")
        self.assertEqual(order, ["policy", "approval", "journal", "execution"])
        self.assertEqual(result.usage.tool_errors, 1)
        self.assertEqual(len(self.select("tool_policy")), 1)

    def test_read_only_gate_precedes_argument_checks(self):
        check = mock.Mock(return_value=True)
        policy = ToolPolicy(read_only=True, argument_checks={"write_file": check})
        self.agent([f.tool_calls(("a", "write_file", {"path": "note", "content": "data"})), f.reply("done")],
                   tool_policy=policy).run("go")
        check.assert_not_called()

    def test_argument_check_exception_stops_without_approval_action_or_error_detail_leak(self):
        marker = "synthetic-private-policy-detail"
        check = mock.Mock(side_effect=ValueError(marker))
        approve = mock.Mock()
        execute = mock.Mock()
        model = f.RecordingModel(f.tool_calls(("a", "write_file", {"path": "note", "content": "data"}),
                                              ("b", "current_time", {})), f.reply("must not call"))
        agent = self.agent(model, tool_policy=ToolPolicy(argument_checks={"write_file": check}),
                           approve=approve, journal=fake_journal(),
                           session_path=self.dir / "session.json", event_include_content=True)
        with mock.patch.dict(self.tools["write_file"], {"fn": execute}), \
                mock.patch.object(h, "run_isolated_tool") as worker:
            result = agent.run_result("go")
        self.assertEqual((result.status, result.stop_reason), ("stopped", "tool_policy_error"))
        self.assertNotIn("outcome unknown", result.text)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(result.usage.tool_calls_attempted, 1)
        self.assertEqual(result.usage.tool_errors, 2)
        self.assertEqual(agent.journal.inspect(), [])
        approve.assert_not_called()
        execute.assert_not_called()
        worker.assert_not_called()
        self.assertEqual([e["status"] for e in self.select("tool_finished")], ["not_run", "not_run"])
        saved = f.saved_messages(agent.session_path)[-1]["content"]
        self.assertEqual([r["tool_use_id"] for r in saved], ["a", "b"])
        combined = self.trace.read_text() + json.dumps([e.to_dict() for e in self.events]) + result.text
        self.assertNotIn(marker, combined)

    def test_predicate_and_approval_receive_detached_snapshots_not_executed_arguments(self):
        original = {"path": "allowed.txt", "content": "original"}
        execute = mock.Mock(return_value="done")

        def check(args):
            self.assertEqual(args["path"], "allowed.txt")
            args["path"] = "check-mutated.txt"
            return True

        def approve(name, args):
            self.assertEqual(args, original)
            args["content"] = "approval-mutated"
            return True

        model = f.RecordingModel(f.tool_calls(("a", "write_file", original)), f.reply("done"))
        with mock.patch.dict(self.tools["write_file"], {"fn": execute}):
            result = self.agent(model, tool_policy=ToolPolicy(argument_checks={"write_file": check}),
                                approve=approve).run_result("go")
        self.assertEqual(result.status, "completed")
        execute.assert_called_once_with(path="allowed.txt", content="original")
        self.assertEqual(original, {"path": "allowed.txt", "content": "original"})

    def test_policy_snapshot_preserves_nested_arguments_and_rejects_non_json_values(self):
        arguments = {"nested": [{"value": "original"}]}

        def mutate(snapshot):
            snapshot["nested"][0]["value"] = "changed"
            return True

        policy = ToolPolicy(argument_checks={"custom": mutate})
        self.assertTrue(policy.check("custom", arguments).allowed)
        self.assertEqual(arguments["nested"][0]["value"], "original")
        for value in (float("nan"), float("inf"), object()):
            with self.subTest(value_type=type(value).__name__):
                self.assertEqual(policy.check("custom", {"nested": value}).reason, "policy_error")

    def test_optional_arguments_remain_omitted_and_predicate_uses_explicit_default(self):
        seen = []

        def check(args):
            seen.append(args)
            return args.get("subdir", ".") == "."

        execute = mock.Mock(return_value="listed")
        with mock.patch.dict(self.tools["list_files"], {"fn": execute}):
            self.agent([f.tool_calls(("a", "list_files", {})), f.reply("done")],
                       tool_policy=ToolPolicy(argument_checks={"list_files": check})).run("go")
        self.assertEqual(seen, [{}])
        execute.assert_called_once_with()

    def test_policy_errors_preserve_previous_completed_results_and_leave_no_pending_actions(self):
        check = mock.Mock(side_effect=RuntimeError("private-detail"))
        model = f.RecordingModel(f.tool_calls(("done", "calculator", {"expression": "2+2"}),
                                              ("bad", "current_time", {}), ("later", "current_time", {})))
        agent = self.agent(model, tool_policy=ToolPolicy(argument_checks={"current_time": check}),
                           journal=fake_journal(), session_path=self.dir / "session.json")
        result = agent.run_result("go")
        self.assertEqual(result.stop_reason, "tool_policy_error")
        saved = f.saved_messages(agent.session_path)[-1]["content"]
        self.assertEqual(saved[0]["content"], "4")
        self.assertTrue(all(r["is_error"] for r in saved[1:]))
        self.assertEqual(agent.journal.inspect(pending_only=True), [])
        self.assertEqual([r["tool_name"] for r in agent.journal.inspect()], ["calculator"])
        check.assert_called_once()

    def test_predicate_time_exhausts_deadline_before_approval_or_dispatch(self):
        now = [0.0]

        def check(args):
            now[0] = 2.0
            return True

        approve = mock.Mock(return_value=True)
        execute = mock.Mock()
        agent = self.agent([f.tool_calls(("a", "write_file", {"path": "note", "content": "data"}))],
                           tool_policy=ToolPolicy(argument_checks={"write_file": check}), approve=approve,
                           limits=h.RunLimits(max_seconds=1), journal=fake_journal())
        with mock.patch("harness.runtime.time.monotonic", side_effect=lambda: now[0]), \
                mock.patch.dict(self.tools["write_file"], {"fn": execute}):
            result = agent.run_result("go")
        self.assertEqual(result.stop_reason, "max_seconds")
        approve.assert_not_called()
        execute.assert_not_called()
        self.assertEqual(agent.journal.inspect(), [])

    def test_denials_consume_attempt_budget_before_later_valid_requests(self):
        agent = self.agent([f.tool_calls(("denied", "current_time", {}),
                                        ("valid", "calculator", {"expression": "2+2"}))],
                           tool_policy=ToolPolicy(argument_checks={"current_time": lambda args: False}),
                           limits=h.RunLimits(max_tool_calls=1))
        result = agent.run_result("go")
        self.assertEqual(result.stop_reason, "max_tool_calls")
        self.assertEqual(result.usage.tool_calls_attempted, 1)
        self.assertEqual(result.usage.tool_errors, 2)
        self.assertEqual(self.select("tool_started"), [])

    def test_events_and_traces_explain_policy_denial_without_leaking_arguments(self):
        marker = "synthetic-private-input"
        agent = self.agent([f.tool_calls(("a", "write_file", {"path": "note", "content": marker})),
                            f.reply("done")], tool_policy=ToolPolicy(read_only=True), event_include_content=True)
        agent.run("go")
        decisions = [e.to_dict() for e in self.events if e.kind == "tool_policy"]
        self.assertEqual(decisions[0]["data"]["reason"], "read_only_required")
        self.assertFalse(decisions[0]["data"]["allowed"])
        self.assertNotIn(marker, json.dumps(decisions))
        traced = [r for r in self.trace_records() if r["kind"] == "tool_policy"]
        self.assertNotIn(marker, json.dumps(traced))
        self.assertEqual(traced[0]["reason"], "read_only_required")
        self.assertEqual(self.select("tool_finished")[0]["status"], "not_run")

    def test_observer_failure_cannot_bypass_a_policy_denial(self):
        def observe(event):
            if event.kind == "tool_policy":
                raise RuntimeError("private-observer-error")

        execute = mock.Mock()
        agent = self.agent([f.tool_calls(("a", "current_time", {})), f.reply("done")], on_event=observe,
                           tool_policy=ToolPolicy(argument_checks={"current_time": lambda args: False}))
        with mock.patch.dict(self.tools["current_time"], {"fn": execute}):
            self.assertEqual(agent.run_result("go").status, "completed")
        execute.assert_not_called()
        self.assertTrue(agent.event_callback_failed)

    def test_policy_checks_apply_before_real_isolated_workers(self):
        entry = dict(self.tools["current_time"], fn=tool_fixtures.change_memory)
        for allowed in (False, True):
            with self.subTest(allowed=allowed):
                agent = self.agent([f.tool_calls(("a", "current_time", {})), f.reply("done")],
                                   tools={"current_time": entry}, tool_timeout_seconds=3,
                                   tool_policy=ToolPolicy(read_only=True,
                                                          argument_checks={"current_time": lambda args: allowed}))
                run_worker = h.run_isolated_tool
                with mock.patch.object(h, "run_isolated_tool", wraps=run_worker) as worker:
                    result = agent.run_result("go")
                self.assertEqual(result.status, "completed")
                self.assertEqual(worker.call_count, int(allowed))
                self.assertEqual(tool_fixtures.LOCAL_STATE, [])

    def test_policy_interrupt_preserves_batch_without_approval_worker_or_action(self):
        for interrupt_in_observer in (False, True):
            with self.subTest(interrupt_in_observer=interrupt_in_observer):
                self.events.clear()

                def observe(event):
                    self.events.append(event)
                    if interrupt_in_observer and event.kind == "tool_policy":
                        raise KeyboardInterrupt()

                check = mock.Mock(return_value=True,
                                  side_effect=None if interrupt_in_observer else KeyboardInterrupt())
                approve = mock.Mock(return_value=True)
                model = f.RecordingModel(f.tool_calls(("a", "write_file", {"path": "note", "content": "data"}),
                                                      ("b", "current_time", {})))
                agent = self.agent(model, on_event=observe, approve=approve,
                                   tool_policy=ToolPolicy(argument_checks={"write_file": check}),
                                   journal=fake_journal(),
                                   session_path=self.dir / "session.json")
                with mock.patch.object(h, "run_isolated_tool") as worker, \
                        mock.patch.dict(self.tools["write_file"], {"fn": mock.Mock()}) as entry:
                    result = agent.run_result("go")
                    entry["fn"].assert_not_called()
                self.assertEqual(result.status, "interrupted")
                approve.assert_not_called()
                worker.assert_not_called()
                self.assertEqual(agent.journal.inspect(), [])
                self.assertEqual(len(model.calls), 1)
                saved = f.saved_messages(agent.session_path)[-1]["content"]
                self.assertEqual([r["tool_use_id"] for r in saved], ["a", "b"])
                self.assertIn("outcome unknown", saved[0]["content"])
                self.assertIn("not run", saved[1]["content"])

    def test_journal_recovery_guard_precedes_model_and_policy(self):
        check = mock.Mock(return_value=True)
        model = f.RecordingModel(f.tool_calls(("a", "current_time", {})))
        agent = self.agent(model, tool_policy=ToolPolicy(argument_checks={"current_time": check}),
                           journal=fake_journal())
        agent.journal.prepare()
        agent.journal.begin("unclosed", "old", "current_time", False)
        result = agent.run_result("go")
        self.assertEqual(result.stop_reason, "journal_recovery_required")
        self.assertEqual(model.calls, [])
        check.assert_not_called()
        self.assertEqual(self.select("tool_policy"), [])

    def test_truncated_and_unknown_calls_do_not_invoke_policy(self):
        check = mock.Mock(return_value=True)
        batch = f.tool_calls(("truncated", "current_time", {}), ("unknown", "missing_tool", {}))
        batch["truncated_tool_ids"] = ["truncated"]
        agent = self.agent([batch, f.reply("done")],
                           tool_policy=ToolPolicy(argument_checks={"current_time": check}))
        result = agent.run_result("go")
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.usage.tool_calls_attempted, 1)
        self.assertEqual(result.usage.tool_errors, 2)
        check.assert_not_called()
        self.assertEqual(self.select("tool_policy"), [])

    def test_explicit_next_task_rechecks_policy_after_error_and_resets_stop_state(self):
        check = mock.Mock(side_effect=[ValueError("private-policy-detail"), True])
        batch = f.tool_calls(("a", "current_time", {}))
        agent = self.agent([batch, batch, f.reply("done")],
                           tool_policy=ToolPolicy(argument_checks={"current_time": check}),
                           journal=fake_journal())
        execute = mock.Mock(return_value="ok")
        with mock.patch.dict(self.tools["current_time"], {"fn": execute}):
            first = agent.run_result("go")
            second = agent.run_result("explicitly continue after policy fix")
        self.assertEqual(first.stop_reason, "tool_policy_error")
        self.assertEqual(second.status, "completed")
        self.assertEqual(second.usage.tool_errors, 0)
        self.assertEqual(check.call_count, 2)
        execute.assert_called_once()
        self.assertEqual(agent.journal.inspect(pending_only=True), [])

    def test_policy_configuration_is_frozen_copied_and_checked_against_allowlist(self):
        checks = {"current_time": lambda args: False}
        policy = ToolPolicy(argument_checks=checks)
        checks["current_time"] = lambda args: True
        self.assertFalse(policy.check("current_time", {}).allowed)
        with self.assertRaises(TypeError):
            policy.argument_checks["current_time"] = lambda args: True
        with self.assertRaises(FrozenInstanceError):
            policy.read_only = True
        for kwargs in ({"tool_policy": {}}, {"tool_policy": ToolPolicy(argument_checks={"misspelled": bool})},
                       {"tool_policy": policy, "tools": {}}):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                self.agent([], **kwargs)
        for kwargs in ({"read_only": 1}, {"argument_checks": None}, {"argument_checks": {"": bool}},
                       {"argument_checks": {"current_time": None}}):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                ToolPolicy(**kwargs)

    def test_custom_tool_declaration_is_opt_in_and_default_behavior_stays_unrestricted(self):
        def custom():
            return "custom result"

        self.tools.tool()(custom)
        self.assertFalse(self.tools["custom"]["read_only"])
        self.assertEqual(self.agent([f.tool_calls(("a", "custom", {})), f.reply("done")]).run("go"), "done")
        self.assertNotIn("tool_policy", [e.kind for e in self.events])
        del self.tools["custom"]  # Deliberate replacement, not accidental duplicate registration.
        self.tools.tool(read_only=True)(custom)
        self.events.clear()
        self.agent([f.tool_calls(("a", "custom", {})), f.reply("done")],
                   tool_policy=ToolPolicy(read_only=True)).run("go")
        self.assertEqual(self.select("tool_finished")[0]["status"], "completed")
        with self.assertRaises(TypeError):
            self.tools.tool(read_only="yes")

    def test_cli_read_only_blocks_demo_write_and_invalid_setting_is_not_echoed(self):
        command = [sys.executable, "-m", "agents.teaching.cli", "mock"]
        env = {"PATH": os.defpath, "PYTHONPATH": str(f.REPO)}
        env.update(HARNESS_WORKSPACE=str(self.dir / "workspace"), HARNESS_READ_ONLY="1")
        result = subprocess.run(command, cwd=self.dir, env=env,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0)
        self.assertIn("read_only_required", result.stdout)
        self.assertNotIn("auto-approved", result.stdout)
        self.assertEqual(list((self.dir / "workspace").rglob("*.txt")), [])
        env["HARNESS_READ_ONLY"] = "synthetic-private-bad-setting"
        result = subprocess.run(command, cwd=self.dir, env=env,
                                capture_output=True, text=True, timeout=20)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Invalid HARNESS_READ_ONLY", result.stderr)
        self.assertNotIn("synthetic-private-bad-setting", result.stdout + result.stderr)