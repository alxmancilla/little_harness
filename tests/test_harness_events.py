"""Ordered observations, safe payload defaults and quiet stdlib embedding."""

import json
from dataclasses import FrozenInstanceError
from unittest import mock

from harness import RunEvent
from tests import helpers as f
from tests import tool_fixtures
from tests.helpers import used
from tests.storage_fakes import FakeJournal as fake_journal

from harness import agent as h
from harness import dispatch


class EventTests(f.HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.events = []

    def agent(self, model, **kwargs):
        kwargs.setdefault("on_event", self.events.append)
        kwargs.setdefault("quiet", True)
        return super().agent(model, **kwargs)

    def kinds(self):
        return [event.kind for event in self.events]

    def select(self, kind):
        return [event for event in self.events if event.kind == kind]

    def test_lifecycle_order_result_and_default_printing(self):
        script = [used(f.tool_calls(("a", "calculator", {"expression": "2+2"}))), used(f.reply("four"))]
        agent = self.agent(script)
        result = agent.run_result("calculate")
        self.assertEqual(result.answer, "four")
        self.assertEqual(self.kinds(), ["run_started", "model_started", "model_finished", "tool_started",
                                        "tool_finished", "model_started", "model_finished", "model_text",
                                        "run_finished"])
        final = self.events[-1].data
        self.assertEqual(final["status"], result.status)
        self.assertEqual(dict(final["usage"]), result.to_dict()["usage"])
        self.assertEqual(self.stdout.getvalue(), "")
        self.assertEqual(self.select("tool_finished")[0].data["status"], "completed")
        self.assertEqual(self.select("model_started")[1].data["model_call"], 2)
        visible = self.agent([f.reply("visible")], quiet=False, on_event=None)
        self.assertEqual(visible.run("go"), "visible")
        self.assertIn("[step 1]", self.stdout.getvalue())

    def test_event_ids_order_and_counters_reset_per_task(self):
        agent = self.agent([f.reply("one"), f.reply("two")])
        agent.run("first")
        first = list(self.events)
        self.events.clear()
        agent.run("second")
        self.assertNotEqual(first[0].run_id, self.events[0].run_id)
        for events in (first, self.events):
            self.assertEqual([e.sequence for e in events], list(range(1, len(events) + 1)))
            self.assertEqual(len({e.run_id for e in events}), 1)
            self.assertTrue(all(e.version == 1 and "+00:00" in e.ts for e in events))
        self.assertEqual(self.select("model_started")[0].data["model_call"], 1)

    def test_metadata_default_excludes_task_text_arguments_outputs_and_errors(self):
        marker = "synthetic-private-content"
        execute = mock.Mock(return_value=marker)
        script = [f.tool_calls(("a", "write_file", {"path": "not-written", "content": marker})),
                  f.reply(marker)]
        with mock.patch.dict(self.tools["write_file"], {"fn": execute}):
            self.agent(script).run(marker)
        serialized = json.dumps([event.to_dict() for event in self.events])
        self.assertNotIn(marker, serialized)
        self.assertNotIn("not-written", serialized)
        self.events.clear()
        self.agent(f.RecordingModel(h.ModelError(marker))).run(marker)
        self.assertNotIn(marker, json.dumps([event.to_dict() for event in self.events]))

    def test_content_opt_in_and_deeply_immutable_detached_snapshots(self):
        agent = self.agent([f.tool_calls(("a", "calculator", {"expression": "2+2"})), f.reply("four")],
                           event_include_content=True)
        agent.run("calculate")
        self.assertEqual(self.events[0].data["task"], "calculate")
        start = self.select("tool_started")[0]
        with self.assertRaises(TypeError):
            start.data["input"]["expression"] = "9+9"
        with self.assertRaises(FrozenInstanceError):
            start.kind = "changed"
        copy = start.to_dict()
        copy["data"]["input"]["expression"] = "9+9"
        agent.messages[1]["content"][0]["input"]["expression"] = "8+8"
        self.assertEqual(start.data["input"]["expression"], "2+2")
        self.assertEqual(self.select("tool_finished")[0].data["output"], "4")
        self.assertEqual(self.events[-1].data["answer"], "four")
        original = {"nested": [{"value": 1}]}
        event = RunEvent("run", 1, "example", "now", original)
        original["nested"][0]["value"] = 2
        self.assertEqual(event.data["nested"][0]["value"], 1)
        self.assertEqual(json.loads(json.dumps(event.to_dict()))["data"]["nested"], [{"value": 1}])

    def test_streaming_and_nonstreaming_text_are_not_duplicated(self):
        def model(system, messages, tools, on_text=None):
            on_text("hel")
            on_text("lo")
            return f.reply("hello")

        self.agent(model, event_include_content=True).run("go")
        text = self.select("model_text")
        self.assertEqual([event.data["phase"] for event in text], ["delta", "delta"])
        self.assertEqual("".join(event.data["text"] for event in text), "hello")
        self.assertLess(self.kinds().index("model_text"), self.kinds().index("model_finished"))
        self.assertEqual(self.stdout.getvalue(), "")

    def test_partial_stream_failure_is_explicit_without_exception_details(self):
        def model(system, messages, tools, on_text=None):
            on_text("partial")
            raise h.ModelError("synthetic-private-error")

        result = self.agent(model).run_result("go")
        self.assertEqual(result.status, "model_error")
        self.assertEqual(self.select("model_finished")[0].data["status"], "failed")
        self.assertEqual(self.kinds()[-2:], ["run_stopped", "run_finished"])
        self.assertNotIn("synthetic-private-error", json.dumps([e.to_dict() for e in self.events]))

    def test_approval_events_precede_dispatch_and_do_not_decide_approval(self):
        execute = mock.Mock(return_value="written")
        for allowed in (True, False):
            with self.subTest(allowed=allowed):
                self.events.clear()
                execute.reset_mock()

                def approve(*args):
                    self.assertEqual(self.events[-1].kind, "approval_requested")
                    self.assertFalse(execute.called)
                    return allowed

                with mock.patch.dict(self.tools["write_file"], {"fn": execute}):
                    self.agent([f.tool_calls(("a", "write_file", {"path": "note", "content": "data"})),
                                f.reply("done")], approve=approve).run("go")
                self.assertEqual(execute.call_count, int(allowed))
                self.assertEqual(self.select("approval_decided")[0].data["decision"],
                                 "allowed" if allowed else "denied")
                self.assertEqual(len(self.select("tool_started")), int(allowed))
                self.assertEqual(self.select("tool_finished")[0].data["status"],
                                 "completed" if allowed else "not_run")

    def test_quiet_default_approval_denies_without_prompt_and_session_resume_is_silent(self):
        execute = mock.Mock()
        session = self.dir / "session.json"
        session.write_text(json.dumps([{"role": "user", "content": "earlier"}]))
        with mock.patch("builtins.input", side_effect=AssertionError("must not prompt")), \
                mock.patch.dict(self.tools["write_file"], {"fn": execute}):
            self.agent([f.tool_calls(("a", "write_file", {"path": "note", "content": "data"})),
                        f.reply("done")], approve=h.ask_human, session_path=session).run("go")
        execute.assert_not_called()
        self.assertEqual(self.stdout.getvalue(), "")

    def test_invalid_unknown_truncated_and_failed_tools_have_honest_outcomes(self):
        batch = f.tool_calls(("invalid", "calculator", {}), ("unknown", "absent", {}),
                             ("cut", "calculator", {}), ("failed", "calculator", {"expression": "1/0"}))
        batch["truncated_tool_ids"] = ["cut"]
        self.agent([batch, f.reply("done")]).run("go")
        self.assertEqual([e.data["tool_call_id"] for e in self.select("tool_finished")],
                         ["invalid", "unknown", "cut", "failed"])
        self.assertEqual([e.data["status"] for e in self.select("tool_finished")],
                         ["not_run", "not_run", "not_run", "failed"])
        self.assertEqual(len(self.select("tool_started")), 1)

    def test_budget_stop_pairs_finished_and_unstarted_requests(self):
        agent = self.agent([f.tool_calls(("a", "current_time", {}), ("b", "current_time", {}))],
                           limits=h.RunLimits(max_tool_calls=1))
        result = agent.run_result("go")
        self.assertEqual([e.data["status"] for e in self.select("tool_finished")], ["completed", "not_run"])
        self.assertEqual(result.stop_reason, "max_tool_calls")
        self.assertEqual(self.events[-1].data["stop_reason"], result.stop_reason)
        self.assertEqual(self.events[-1].data["usage"]["tool_errors"], 1)

    def test_callback_failures_at_every_phase_do_not_retry_or_change_execution(self):
        phases = ("run_started", "model_started", "model_finished", "model_text", "approval_requested",
                  "approval_decided", "tool_started", "tool_finished", "run_finished")
        for phase in phases:
            with self.subTest(phase=phase):
                received = []

                def observe(event):
                    received.append(event)
                    if event.kind == phase:
                        raise ValueError("synthetic-private-listener-error")

                execute = mock.Mock(return_value="written")
                model = f.RecordingModel(f.tool_calls(("a", "write_file", {"path": "note", "content": "data"})),
                                         f.reply("done"))
                agent = self.agent(model, on_event=observe, session_path=self.dir / (phase + ".json"),
                                   journal=fake_journal())
                with mock.patch.dict(self.tools["write_file"], {"fn": execute}):
                    result = agent.run_result("go")
                self.assertEqual(result.status, "completed")
                execute.assert_called_once()
                self.assertEqual(len(model.calls), 2)
                self.assertTrue(agent.event_callback_failed)
                self.assertEqual(received[-1].kind, phase)
                self.assertEqual(agent.journal.inspect(pending_only=True), [])
                self.assertNotIn("synthetic-private-listener-error", self.trace.read_text())
                disabled = [r for r in self.trace_records() if r["kind"] == "event_callback_failed"]
                self.assertEqual((disabled[-1]["event_kind"], disabled[-1].get("error")),
                                 (phase, "ValueError"))

    def test_callback_is_reenabled_next_run_and_reentrant_runs_are_rejected(self):
        calls = []

        def observe(event):
            calls.append(event)
            if len(calls) == 1:
                agent.run("nested")

        agent = self.agent([f.reply("first"), f.reply("second")], on_event=observe)
        self.assertEqual(agent.run("go"), "first")
        self.assertTrue(agent.event_callback_failed)
        self.assertEqual(len(calls), 1)
        self.assertEqual(agent.run("again"), "second")
        self.assertFalse(agent.event_callback_failed)
        self.assertEqual(calls[-1].kind, "run_finished")

    def test_callback_interrupt_after_completed_tool_preserves_batch_and_does_not_reinvoke(self):
        received = []

        def observe(event):
            received.append(event)
            if event.kind == "tool_finished":
                raise KeyboardInterrupt()

        execute = mock.Mock(return_value="completed")
        session = self.dir / "session.json"
        agent = self.agent([f.tool_calls(("a", "current_time", {}), ("b", "current_time", {}))],
                           on_event=observe, session_path=session, journal=fake_journal())
        with mock.patch.dict(self.tools["current_time"], {"fn": execute}):
            result = agent.run_result("go")
        self.assertEqual(result.status, "interrupted")
        execute.assert_called_once()
        saved = f.saved_messages(session)[-1]["content"]
        self.assertEqual(saved[0]["content"], "completed")
        self.assertIn("not run", saved[1]["content"])
        self.assertEqual(received[-1].kind, "tool_finished")
        self.assertEqual(agent.journal.inspect()[0]["recorded_status"], "completed")
        self.assertTrue(agent.journal.inspect()[0]["recovery_required"])

    def test_tool_interrupt_and_timeout_preserve_unknown_status_and_stop_later_work(self):
        for timeout in (False, True):
            with self.subTest(timeout=timeout):
                self.events.clear()
                model = f.RecordingModel(f.tool_calls(("a", "current_time", {}), ("b", "current_time", {})))
                agent = self.agent(model, tool_timeout_seconds=1 if timeout else None)
                outcome = dispatch.ToolOutcome("timed_out", "timeout", True, True)
                patch = (mock.patch.object(dispatch, "run_isolated_tool", return_value=outcome) if timeout else
                         mock.patch.dict(self.tools["current_time"], {"fn": mock.Mock(side_effect=KeyboardInterrupt())}))
                with patch:
                    result = agent.run_result("go")
                self.assertEqual(result.stop_reason, "tool_timeout" if timeout else "interrupted")
                finished = self.select("tool_finished")
                self.assertTrue(finished[0].data["outcome_unknown"])
                self.assertEqual(finished[0].data["status"], "timed_out" if timeout else "outcome_unknown")
                self.assertEqual(finished[1].data["status"], "not_run")
                self.assertEqual(len(model.calls), 1)

    def test_interrupt_during_not_run_notification_keeps_previously_completed_results(self):
        def observe(event):
            if event.kind == "tool_finished" and event.data["status"] == "not_run":
                raise KeyboardInterrupt()

        execute = mock.Mock(return_value="completed")
        session = self.dir / "session.json"
        agent = self.agent([f.tool_calls(("a", "current_time", {}), ("b", "current_time", {}),
                                        ("c", "current_time", {}))], on_event=observe, session_path=session,
                           limits=h.RunLimits(max_tool_calls=1))
        with mock.patch.dict(self.tools["current_time"], {"fn": execute}):
            result = agent.run_result("go")
        execute.assert_called_once()
        self.assertEqual(result.status, "interrupted")
        saved = f.saved_messages(session)[-1]["content"]
        self.assertEqual([r["tool_use_id"] for r in saved], ["a", "b", "c"])
        self.assertEqual(saved[0]["content"], "completed")
        self.assertTrue(all("not run" in r["content"] for r in saved[1:]))
        self.assertEqual(result.usage.tool_errors, 2)

    def test_journal_ids_match_events_and_recovery_blocks_before_model_activity(self):
        journal = fake_journal()
        agent = self.agent([f.tool_calls(("a", "current_time", {})), f.reply("done")], journal=journal)
        agent.run("go")
        row = agent.journal.inspect()[0]
        self.assertEqual(self.events[0].run_id, row["run_id"])
        self.assertEqual(self.select("tool_started")[0].data["action_id"], row["action_id"])
        self.assertEqual(self.select("tool_finished")[0].data["action_id"], row["action_id"])
        agent.journal.begin("unclosed", "other", "current_time", False)
        self.events.clear()
        result = self.agent([], journal=journal).run_result("blocked")
        self.assertEqual(result.stop_reason, "journal_recovery_required")
        self.assertEqual(self.kinds(), ["run_started", "run_stopped", "run_finished"])
        self.assertEqual(result.usage.model_calls, 0)

    def test_journal_outcome_commit_failure_keeps_actual_tool_outcome(self):
        agent = self.agent([f.tool_calls(("a", "current_time", {}), ("b", "current_time", {}))],
                           journal=fake_journal())
        with mock.patch.object(agent.journal, "finish", side_effect=h.JournalError()):
            result = agent.run_result("go")
        self.assertEqual(result.stop_reason, "journal_error")
        self.assertEqual([e.data["status"] for e in self.select("tool_finished")], ["completed", "not_run"])
        self.assertEqual(self.events[-1].data["stop_reason"], "journal_error")

    def test_run_finished_follows_session_save_and_journal_close_including_close_failure(self):
        session = self.dir / "session.json"

        def observe(event):
            self.events.append(event)
            if event.kind == "run_finished":
                self.assertEqual(f.saved_messages(session)[-1]["content"][0]["text"], "done")
                self.assertTrue(agent.journal.inspect()[0]["run_closed"])

        agent = self.agent([f.tool_calls(("a", "current_time", {})), f.reply("done")],
                           on_event=observe, session_path=session, journal=fake_journal())
        self.assertEqual(agent.run_result("go").status, "completed")
        self.assertFalse(agent.event_callback_failed, "observer assertions must not have failed")
        self.events.clear()
        agent = self.agent([f.reply("done")], journal=fake_journal())
        with mock.patch.object(agent.journal, "close_run", side_effect=h.JournalError()):
            result = agent.run_result("go")
        self.assertEqual(result.stop_reason, "journal_error")
        self.assertEqual(self.events[-1].data["stop_reason"], "journal_error")

    def test_real_worker_result_emits_events_in_parent_without_worker_console_noise(self):
        entry = dict(self.tools["current_time"], fn=tool_fixtures.change_memory)
        agent = self.agent([f.tool_calls(("a", "current_time", {})), f.reply("done")],
                           tools={"current_time": entry}, tool_timeout_seconds=3, event_include_content=True)
        self.assertEqual(agent.run_result("go").status, "completed")
        self.assertEqual(self.select("tool_finished")[0].data["status"], "completed")
        self.assertEqual(self.select("tool_finished")[0].data["output"], "1")
        self.assertEqual(tool_fixtures.LOCAL_STATE, [])
        self.assertEqual(self.stdout.getvalue(), "")

    def test_compaction_lifecycle_and_shared_model_call_numbers(self):
        for summary, status in ((f.reply("summary"), "completed"), (h.ModelError("private-error"), "failed"),
                                (f.reply(""), "empty")):
            with self.subTest(status=status):
                self.events.clear()
                agent = self.agent(f.RecordingModel(summary, f.reply("answer")), compact_at=1500)
                agent.messages = f.compaction_history()
                self.assertEqual(agent.run_result("continue").status, "completed")
                self.assertEqual(self.select("compaction_finished")[0].data["status"], status)
                models = self.select("model_started")
                self.assertEqual([e.data["purpose"] for e in models], ["compaction", "main"])
                self.assertEqual([e.data["model_call"] for e in models], [1, 2])
                self.assertEqual(self.stdout.getvalue(), "")

    def test_observer_time_can_exhaust_deadline_before_dispatch(self):
        for phase in ("model_started", "tool_started"):
            with self.subTest(phase=phase):
                self.events.clear()
                now = [0.0]

                def observe(event):
                    self.events.append(event)
                    if event.kind == phase:
                        now[0] = 2.0

                execute = mock.Mock(return_value="must not run")
                model = f.RecordingModel(f.tool_calls(("a", "current_time", {})))
                agent = self.agent(model, on_event=observe, limits=h.RunLimits(max_seconds=1),
                                   journal=fake_journal())
                with mock.patch("harness.runtime.time.monotonic", side_effect=lambda: now[0]), \
                        mock.patch.dict(self.tools["current_time"], {"fn": execute}):
                    result = agent.run_result("go")
                self.assertEqual(result.stop_reason, "max_seconds")
                execute.assert_not_called()
                self.assertEqual(len(model.calls), int(phase == "tool_started"))
                if phase == "tool_started":
                    self.assertEqual(agent.journal.inspect()[0]["status"], "not_run")

    def test_unexpected_error_is_reported_generically_and_still_propagates(self):
        agent = self.agent(f.RecordingModel(ValueError("synthetic-private-error")))
        with self.assertRaises(ValueError):
            agent.run("go")
        self.assertEqual(self.events[-1].data["status"], "error")
        self.assertEqual(self.events[-1].data["stop_reason"], "harness_error")
        self.assertNotIn("synthetic-private-error", json.dumps([e.to_dict() for e in self.events]))
        self.assertIsNone(agent._budget)

    def test_event_configuration_rejects_invalid_values(self):
        for kwargs in ({"on_event": 1}, {"quiet": "yes"}, {"event_include_content": "yes"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(TypeError):
                self.agent([], **kwargs)