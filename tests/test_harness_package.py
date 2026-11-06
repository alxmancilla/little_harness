"""Reusable runtime contracts without relying on teaching or HotpotQA fixtures."""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

from harness import Agent, RunLimits, ToolPolicy, ToolRegistry
from tests.helpers import Model, TemporaryDirectoryTestCase, call, usage_reply as reply
from tests.storage_fakes import FakeJournal


REPO = Path(__file__).resolve().parents[1]


class PackageTests(TemporaryDirectoryTestCase):
    def agent(self, model, tools, **kwargs):
        kwargs.setdefault("quiet", True)
        kwargs.setdefault("trace_path", self.root / "trace.jsonl")
        return Agent(model, "Application-owned instructions", tools=tools, **kwargs)

    def isolated(self, code):
        # Explicit allowlist: no credentials or application configuration inherited.
        return subprocess.run([sys.executable, "-B", "-c", code], cwd=self.root, timeout=15,
                              env={"PATH": os.defpath, "PYTHONPATH": str(REPO),
                                   "HARNESS_COMPACT_CHARS": "not-an-integer",
                                   "HARNESS_WORKSPACE": str(self.root / "missing" / "workspace")},
                              capture_output=True, text=True)

    def test_import_and_tool_free_run_have_no_application_side_effects(self):
        (self.root / ".env").write_text("PACKAGE_SENTINEL=must-not-load\n")
        child = self.isolated(
            "import sys, os; from pathlib import Path; "
            "before = set(Path('.').iterdir()); import harness; "
            "assert set(Path('.').iterdir()) == before; "
            "assert 'PACKAGE_SENTINEL' not in os.environ; "
            "assert not any(n == 'mini_harness' or n.startswith('hotpotqa') "
            "or n.startswith('evaluate_hotpotqa') for n in sys.modules); "
            "model = lambda *a, **k: {'content': [{'type': 'text', 'text': 'hello'}]}; "
            "agent = harness.Agent(model, 'other app', tools={}, quiet=True); "
            "assert agent.compact_at == 40000; assert agent.run('go') == 'hello'; "
            "assert set(Path('.').iterdir()) == before | {Path('trace.jsonl')}; "
            "assert not Path('missing').exists(); assert not Path('workspace').exists()"
        )
        self.assertEqual(child.returncode, 0, child.stderr)

    def test_shared_helpers_import_without_applications_configuration_or_test_modules(self):
        (self.root / ".env").write_text("HELPER_SENTINEL=must-not-load\n")
        child = self.isolated(
            "import os, sys; from pathlib import Path; "
            "before_files = set(Path('.').iterdir()); before_env = dict(os.environ); "
            "import tests.helpers; "
            "assert set(Path('.').iterdir()) == before_files; assert dict(os.environ) == before_env; "
            "assert not any(n == 'mini_harness' or n.startswith(('agents', 'evaluation', "
            "'hotpotqa', 'evaluate_hotpotqa', 'tests.test_')) for n in sys.modules)"
        )
        self.assertEqual(child.returncode, 0, child.stderr)

    def test_tools_are_required_and_empty_mapping_does_not_fall_back(self):
        with self.assertRaises(TypeError):
            Agent(Model(), "sys")
        for invalid in (None, [], [("tool", {})]):
            with self.subTest(invalid=invalid), self.assertRaises(TypeError):
                self.agent(Model(), invalid)
        model = Model(call("calculator", expression="2+2"), reply("no tools"))
        agent = self.agent(model, {})
        result = agent.run_result("go")
        self.assertEqual(result.usage.tool_errors, 1)
        self.assertEqual(model.schemas, [[], []])
        self.assertIn("unknown tool", agent.messages[2]["content"][0]["content"])

    def test_separate_registries_and_membership_snapshots(self):
        agents = []
        for number, text in enumerate(("alpha", "beta")):
            registry = ToolRegistry()

            @registry.tool(read_only=True)
            def identity(value: str = text):
                """Return the application's own identity."""
                return value

            model = Model(call("identity"), call("private"), reply(text))
            agent = self.agent(model, registry, trace_path=self.root / f"{number}.jsonl")

            @registry.tool()
            def private():
                raise AssertionError("registration after construction must not widen the allowlist")

            agents.append((agent, model))
        for (agent, model), expected in zip(agents, ("alpha", "beta")):
            self.assertEqual(agent.run("identity"), expected)
            self.assertEqual(agent.messages[2]["content"][0]["content"], expected)
            self.assertTrue(agent.messages[4]["content"][0]["is_error"])
            self.assertEqual([schema[0]["name"] for schema in model.schemas], ["identity"] * 3)
        self.assertIsNot(agents[0][0].tools["identity"], agents[1][0].tools["identity"])

    def test_registry_schema_metadata_and_validation_precede_approval(self):
        registry, executed = ToolRegistry(), mock.Mock()

        @registry.tool(risky=True, timeout_seconds=1, read_only=True)
        def lookup(count: int, label: str = "x", ratio: float = 1.0, flag: bool = False):
            """Look up application data."""
            executed()

        schema = registry["lookup"]["schema"]
        self.assertEqual(schema["input_schema"]["required"], ["count"])
        self.assertEqual({k: v["type"] for k, v in schema["input_schema"]["properties"].items()},
                         {"count": "integer", "label": "string", "ratio": "number", "flag": "boolean"})
        self.assertNotIn("timeout_seconds", schema)
        approve = mock.Mock(return_value=True)
        agent = self.agent(Model(call("lookup", count=True), reply("invalid")), registry, approve=approve)
        self.assertEqual(agent.run_result("go").usage.tool_errors, 1)
        approve.assert_not_called()
        executed.assert_not_called()

    def test_custom_timed_worker_needs_no_legacy_application(self):
        child = self.isolated(
            "from pathlib import Path; from harness import Agent, ToolRegistry; "
            "from tests.helpers import Model, call, usage_reply as reply, package_echo; "
            "tools = ToolRegistry(); tools.tool(read_only=True)(package_echo); "
            "agent = Agent(Model(call('package_echo', value='custom'), reply('done')), "
            "'other', tools=tools, quiet=True, tool_timeout_seconds=3); "
            "result = agent.run_result('go'); assert result.status == 'completed'; "
            "assert result.usage.tool_errors == 0; "
            "assert agent.messages[2]['content'][0]['content'] == 'custom'; "
            "assert not Path('missing').exists(); assert not Path('workspace').exists()"
        )
        self.assertEqual(child.returncode, 0, child.stderr)

    def test_multiple_writable_roots_with_explicit_journal(self):
        roots = (self.root / "first", self.root / "second")
        journal = FakeJournal()
        agent = self.agent(Model(reply("safe")), {}, journal=journal, writable_roots=roots)
        self.assertEqual(agent.run_result("go").status, "completed")
        self.assertEqual(journal.inspect(), [])
        self.assertFalse(any(root.exists() for root in roots))

    def test_core_policy_budget_events_and_output_cap(self):
        registry = ToolRegistry()

        @registry.tool(read_only=True)
        def read():
            return "application-value"

        events = []
        agent = self.agent(Model(call("read"), reply("done")), registry,
                           limits=RunLimits(max_model_calls=1), on_event=events.append,
                           tool_policy=ToolPolicy(read_only=True), max_tool_output=5)
        result = agent.run_result("go")
        self.assertEqual((result.status, result.stop_reason), ("stopped", "max_model_calls"))
        self.assertEqual((result.usage.tool_calls_attempted, result.usage.tool_errors), (1, 0))
        self.assertTrue(agent.messages[2]["content"][0]["content"].startswith("appli\n[truncated"))
        self.assertEqual(events[-1].kind, "run_finished")
        self.assertEqual([event.sequence for event in events], list(range(1, len(events) + 1)))
        self.assertNotIn("output", next(event.data for event in events if event.kind == "tool_finished"))

    def test_admission_rejects_before_validation_policy_approval_journal_or_worker(self):
        registry, executed = ToolRegistry(), mock.Mock()

        @registry.tool(risky=True, read_only=True)
        def action(value: str):
            executed()

        admission = mock.Mock(return_value="application budget exhausted")
        approve, predicate = mock.Mock(return_value=True), mock.Mock(return_value=True)
        agent = self.agent(Model(call("action"), call("absent"), reply("done")), registry,
                           tool_admission=admission, approve=approve, tool_timeout_seconds=1,
                           journal=FakeJournal(),
                           tool_policy=ToolPolicy(argument_checks={"action": predicate}))
        with mock.patch("harness.tool_execution.subprocess.Popen") as spawn:
            result = agent.run_result("go")
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.usage.tool_errors, 2)
        self.assertEqual(admission.call_args_list, [mock.call("action", True), mock.call("absent", False)])
        for observer in (approve, predicate, executed, spawn):
            observer.assert_not_called()
        self.assertEqual(agent.journal.inspect(), [])
        self.assertEqual(agent.messages[2]["content"][0]["content"], "application budget exhausted")

    def test_admission_cannot_bypass_validation_or_policy(self):
        registry, executed = ToolRegistry(), mock.Mock()

        @registry.tool(risky=True, read_only=False)
        def action(value: str):
            executed()

        admission, approve = mock.Mock(return_value=None), mock.Mock(return_value=True)
        agent = self.agent(Model(call("action"), call("action", value="valid"), reply("done")), registry,
                           tool_admission=admission, approve=approve,
                           tool_policy=ToolPolicy(read_only=True), journal=FakeJournal())
        result = agent.run_result("go")
        self.assertEqual(result.usage.tool_errors, 2)
        self.assertIn("missing", agent.messages[2]["content"][0]["content"])
        self.assertIn("policy", agent.messages[4]["content"][0]["content"])
        self.assertEqual(admission.call_count, 2)
        approve.assert_not_called()
        executed.assert_not_called()
        self.assertEqual(agent.journal.inspect(), [])

    def test_invalid_or_failing_admission_never_dispatches_or_persists_error_details(self):
        registry, executed = ToolRegistry(), mock.Mock()

        @registry.tool(risky=True)
        def action():
            executed()

        for index, decision in enumerate((False, True, "", 7, RuntimeError("private-hook-detail"))):
            admission = mock.Mock(**({"side_effect": decision} if isinstance(decision, Exception)
                                     else {"return_value": decision}))
            approve = mock.Mock(return_value=True)
            trace = self.root / f"hook-{index}.jsonl"
            events = []
            agent = self.agent(Model(call("action")), registry, tool_admission=admission,
                               approve=approve, trace_path=trace, on_event=events.append)
            with self.subTest(decision=index), self.assertRaises((ValueError, RuntimeError)):
                agent.run("go")
            self.assertEqual(events[-1].data["stop_reason"], "harness_error")
            self.assertNotIn("private-hook-detail", trace.read_text())
            approve.assert_not_called()
        executed.assert_not_called()

    def test_trace_context_is_detached_json_and_cannot_replace_runtime_fields(self):
        context = {"role": "reviewer", "labels": ["original"], "step": 999}
        agent = self.agent(Model(reply("done")), {}, trace_context=context)
        context["labels"].append("mutated")
        agent.run("go")
        records = [json.loads(line) for line in agent.trace_path.read_text().splitlines()]
        self.assertTrue(all(record["role"] == "reviewer" and record["labels"] == ["original"] for record in records))
        self.assertEqual(records[-1]["kind"], "model")
        self.assertEqual(records[-1]["step"], 1)
        for invalid in ([], {1: "value"}, {"ts": "spoof"}, {"kind": "spoof"}, {"run_id": "spoof"},
                        {"value": float("nan")}, {"value": object()}):
            with self.subTest(invalid=type(invalid).__name__), self.assertRaises((ValueError, TypeError)):
                self.agent(Model(), {}, trace_context=invalid)
        with self.assertRaises(TypeError):
            self.agent(Model(), {}, tool_admission="not callable")

    def test_runtime_files_are_fingerprinted_and_have_no_application_imports(self):
        from evaluation.hotpotqa.comparison import SHARED_CODE_FILES
        package_files = {str(path.relative_to(REPO)) for path in (REPO / "harness").rglob("*.py")}
        self.assertTrue(package_files.issubset(SHARED_CODE_FILES))
        forbidden = ("mini_harness", "hotpotqa", "evaluate_hotpotqa", "run_hotpotqa", "prepare_hotpotqa",
                     "agents", "evaluation")
        for name in package_files:
            tree = ast.parse((REPO / name).read_text())
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.append(node.module or "")
            self.assertFalse(any(module.startswith(forbidden) for module in imports), name)