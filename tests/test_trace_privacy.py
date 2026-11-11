"""Trace content is independent from event content and never affects execution."""

import json
import os
import subprocess
import sys

from harness import Agent, ModelError, ToolRegistry
from tests.helpers import REPO, Model, TemporaryDirectoryTestCase, compaction_history, reply, tool_calls


class TracePrivacyTests(TemporaryDirectoryTestCase):
    def test_trace_and_session_directories_are_created_and_tracing_can_be_disabled(self):
        runs = self.root / "runs" / "nested"
        agent = Agent(Model(reply("first")), "sys", tools={}, quiet=True,
                      trace_path=runs / "trace.jsonl", session_path=runs / "session.json")
        self.assertEqual(agent.run_result("go").status, "completed")
        self.assertTrue((runs / "trace.jsonl").is_file())
        self.assertTrue((runs / "session.json").is_file())

        cwd = os.getcwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, cwd)
        untraced = Agent(Model(reply("second")), "sys", tools={}, quiet=True, trace_path=None)
        self.assertIsNone(untraced.trace_path)
        self.assertEqual(untraced.run_result("go").answer, "second")
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["runs"])

    def test_defaults_omit_tasks_arguments_outputs_and_provider_usage_extensions(self):
        tools = ToolRegistry()

        @tools.tool(read_only=True)
        def echo(value: str):
            return "private-output:" + value

        events = []
        response = reply("private-answer")
        response["usage"] = {"input_tokens": 3, "output_tokens": 2, "debug": "private-provider-data"}
        agent = Agent(Model(tool_calls(("a", "echo", {"value": "private-argument"})), response), "sys",
                      tools=tools, quiet=True, trace_path=self.root / "trace.jsonl", on_event=events.append)
        result = agent.run_result("private-task")
        text = agent.trace_path.read_text()
        self.assertNotIn("private-", text)
        self.assertNotIn("private-", json.dumps([event.to_dict() for event in events]))
        records = [json.loads(line) for line in text.splitlines()]
        self.assertEqual([r["kind"] for r in records], ["task", "model", "tool", "model"])
        self.assertEqual(records[-1]["usage"], {"input_tokens": 3, "output_tokens": 2})
        self.assertEqual((result.status, result.answer), ("completed", "private-answer"))
        self.assertEqual(agent.messages[2]["content"][0]["content"], "private-output:private-argument")

    def test_event_and_trace_content_opt_ins_are_independent(self):
        for trace_content, event_content in ((False, True), (True, False), (True, True)):
            events = []
            agent = Agent(Model(reply("private-answer")), "sys", tools={}, quiet=True,
                          trace_path=self.root / f"{trace_content}-{event_content}.jsonl",
                          on_event=events.append, trace_include_content=trace_content,
                          event_include_content=event_content)
            agent.run("private-task")
            self.assertEqual("private-answer" in agent.trace_path.read_text(), trace_content)
            self.assertEqual("private-answer" in json.dumps([e.to_dict() for e in events]), event_content)

    def test_compaction_and_custom_model_errors_do_not_leak_into_metadata_traces(self):
        agent = Agent(Model(reply("private-summary")), "sys", tools={}, quiet=True,
                      compact_at=500, trace_path=self.root / "compact.jsonl")
        agent.messages = compaction_history()
        agent._compact(1)
        self.assertNotIn("private-summary", agent.trace_path.read_text())
        self.assertIn("private-summary", agent._model_context()[0]["content"])

        def failing_model(*args, **kwargs):
            raise ModelError("private-error")

        agent = Agent(failing_model, "sys", tools={}, quiet=True, trace_path=self.root / "errors.jsonl")
        self.assertEqual(agent.run_result("private-task").status, "model_error")
        self.assertNotIn("private-", agent.trace_path.read_text())
        with self.assertRaises(TypeError):
            Agent(Model(), "sys", tools={}, trace_include_content=1)

    def test_teaching_cli_defaults_to_metadata_and_requires_explicit_content_opt_in(self):
        env = {"PATH": os.defpath, "PYTHONPATH": str(REPO)}
        for setting in ("0", "1", "invalid-private-setting"):
            directory = self.root / setting
            directory.mkdir()
            child = subprocess.run([sys.executable, "-B", "-m", "agents.teaching.cli", "mock"],
                                   cwd=directory, env={**env, "HARNESS_TRACE_CONTENT": setting},
                                   capture_output=True, text=True, timeout=10)
            if setting.startswith("invalid"):
                self.assertNotEqual(child.returncode, 0)
                self.assertNotIn(setting, child.stderr)
                self.assertFalse((directory / "trace.jsonl").exists())
            else:
                self.assertEqual(child.returncode, 0, child.stderr)
                trace = (directory / "trace.jsonl").read_text()
                self.assertEqual('"content":' in trace, setting == "1")
