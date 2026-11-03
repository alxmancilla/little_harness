"""Application isolation, public composition and package workers."""

import ast
import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from agents.hotpotqa import HotpotQAConfig, make_tools
from agents.hotpotqa import common, multi, single
from harness import Agent, ToolRegistry
from tests.helpers import Model, TemporaryDirectoryTestCase, call, usage_reply as reply


REPO = Path(__file__).resolve().parents[1]
FINAL = json.dumps({"answer": "", "supporting_facts": [], "abstain": True})


def stable(value):
    if isinstance(value, dict):
        return {key: stable(item) for key, item in value.items() if key != "runtime_seconds"}
    return [stable(item) for item in value] if isinstance(value, list) else value


class HotpotQASeparationTests(TemporaryDirectoryTestCase):
    def test_package_imports_do_not_load_configuration_or_initialize_applications(self):
        (self.root / ".env").write_text("HOTPOTQA_SENTINEL=must-not-load\n")
        child = self.child("""
            import os, sys
            from pathlib import Path
            before = set(Path('.').iterdir())
            os.environ['HARNESS_COMPACT_CHARS'] = 'not-an-integer'
            from agents.hotpotqa import common, single, multi, make_tools
            assert 'HOTPOTQA_SENTINEL' not in os.environ
            assert not any(name.startswith('evaluation') for name in sys.modules)
            assert set(make_tools({}, set())) == {'search_documents', 'read_document', 'calculator'}
            assert set(Path('.').iterdir()) == before
            """)
        self.assertEqual(child.returncode, 0, child.stderr)

    def test_application_has_no_private_runtime_overrides_or_legacy_imports(self):
        for path in (REPO / "agents/hotpotqa").glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self.assertFalse(node.name.startswith("_") and not node.name.startswith("__"), path.name)
                if isinstance(node, ast.Import):
                    self.assertNotIn("mini_harness", [alias.name for alias in node.names])
                if isinstance(node, ast.ImportFrom):
                    self.assertNotEqual(node.module, "mini_harness")

    def test_two_explicit_corpora_tools_and_observations_stay_independent(self):
        instances = []
        for label in ("alpha", "beta"):
            config = HotpotQAConfig(self.root / label / "workspace", self.root / label / "knowledge")
            config.knowledge.mkdir(parents=True)
            (config.knowledge / "article.txt").write_text("# " + label + "\n\n[0] nebula " + label)
            documents = {"article.txt": {"title": label, "sentences": ["nebula " + label]}}
            observed = set()
            instances.append((config, observed, make_tools(documents, observed, config=config), label))
        for config, observed, tools, label in reversed(instances):
            found = json.loads(tools["search_documents"]["fn"]("nebula"))
            self.assertEqual(found["results"][0]["title"], label)
            self.assertIn(label, found["results"][0]["text"])
            self.assertEqual(observed, set())
            read = json.loads(tools["read_document"]["fn"]("article.txt"))
            self.assertEqual(read["title"], label)
            self.assertEqual(observed, {(label, 0)})
            self.assertFalse(config.workspace.exists())
        self.assertIsNot(instances[0][2]["calculator"], instances[1][2]["calculator"])

    def test_search_dependency_is_explicit_and_enriched_without_global_patches(self):
        search = mock.Mock(return_value=json.dumps({"status": "ok", "results": [
            {"source": "article.txt#char-0", "text": "nebula"}]}))
        tools = make_tools({"article.txt": {"title": "Nebula", "sentences": ["one", "two"]}},
                           set(), search_fn=search)
        found = json.loads(tools["search_documents"]["fn"]("nebula", 1))
        search.assert_called_once_with("nebula", 1)
        self.assertEqual(found["results"][0]["sentence_count"], 2)
        self.assertEqual(found["notice"], common.BASELINE_EVIDENCE_NOTICE)

    def test_worker_configuration_ignores_general_harness_cli_controls(self):
        config = HotpotQAConfig.from_env({"HARNESS_WORKSPACE": str(self.root / "explicit"),
            "HARNESS_KNOWLEDGE": str(self.root / "candidates"), "HARNESS_COMPACT_CHARS": "not-an-integer",
            "HARNESS_MAX_MODEL_CALLS": "not-an-integer", "HARNESS_TOOL_TIMEOUT_SECONDS": "not-a-number",
            "HARNESS_READ_ONLY": "invalid", "HARNESS_JOURNAL": "must-not-create.sqlite"})
        agent, _ = common.create_agent(Model(reply(FINAL)), "sys", {}, set(), self.root / "trace.jsonl",
                                       config=config)
        self.assertEqual(config.knowledge, self.root / "candidates")
        self.assertIsNone(agent.tool_timeout_seconds)
        self.assertIsNone(agent.tool_policy)
        self.assertIsNone(agent.journal)
        self.assertEqual(agent.compact_at, sys.maxsize)
        self.assertFalse(config.workspace.exists())

    def test_both_architectures_accept_runtime_and_tool_factories(self):
        payload = {"id": "custom", "question": "question", "documents": {}}
        config = HotpotQAConfig(self.root / "workspace", self.root / "knowledge")
        for runner, roles in ((single.run_agent, 1), (multi.run_agents, 2)):
            agents, configs = [], []

            def tool_factory(documents, observed, *, config):
                configs.append(config)
                tools = ToolRegistry()

                @tools.tool(read_only=True)
                def custom(value: str):
                    return "injected:" + value

                return tools

            def runtime_factory(*args, **kwargs):
                agent = Agent(*args, quiet=True, **kwargs)
                agents.append(agent)
                return agent

            responses = [response for _ in range(roles) for response in (call("custom", value="ok"), reply(FINAL))]
            result = runner(payload, Model(*responses), self.root / f"{roles}.jsonl", config=config,
                            runtime_factory=runtime_factory, tool_factory=tool_factory)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["tool_calls_executed"], roles)
            self.assertEqual(result["tool_errors"], 0)
            self.assertEqual(configs, [config] * roles)
            self.assertTrue(all(type(agent) is Agent for agent in agents))
            for agent in agents:
                self.assertEqual(agent.messages[2]["content"][0]["content"], "injected:ok")
            if roles == 2:
                self.assertIsNot(agents[0].messages, agents[1].messages)

    def test_truncated_unknown_malformed_and_exhausted_calls_preserve_accounting(self):
        response = {"content": [
            {"type": "tool_use", "id": name, "name": tool, "input": args}
            for name, tool, args in (("bad", "calculator", []), ("cut", "calculator", {}),
                                     ("unknown", "host_only", {}),
                                     ("ok", "calculator", {"expression": "2+2"}),
                                     ("exhausted", "calculator", {"expression": "3+3"}))],
            "truncated_tool_ids": ["cut"], "stop_reason": "tool_use"}
        with redirect_stdout(io.StringIO()):
            result = single.run_agent({"id": "x", "question": "go", "documents": {}},
                                      Model(response, reply(FINAL)), self.root / "trace.jsonl", max_tool_calls=2)
        self.assertEqual(result["status"], "completed")
        self.assertEqual([result[key] for key in ("tool_calls_requested", "tool_calls_executed",
                                                 "tool_calls_rejected", "tool_errors")], [5, 2, 2, 1])

    def test_composed_budget_uses_public_hooks_and_keeps_counters(self):
        tools = ToolRegistry()

        @tools.tool()
        def lookup():
            return "ok"

        for role in (None, "reviewer"):
            agent, budget = common.create_agent(Model(call("lookup"), call("lookup"), reply("done")),
                "sys", {}, set(), self.root / (str(role) + ".jsonl"), max_tool_calls=1, role=role,
                tool_factory=lambda *args, **kwargs: tools,
                runtime_factory=lambda *args, **kwargs: Agent(*args, quiet=True, **kwargs))
            self.assertEqual(agent.run("go"), "done")
            self.assertIs(type(agent), Agent)
            self.assertEqual((budget.executed_tools, budget.rejected_tools, budget.tool_errors), (1, 1, 0))
            self.assertEqual(budget.max_tool_calls, 1)

    def test_module_workers_honor_explicit_environment(self):
        knowledge = self.root / "knowledge"
        knowledge.mkdir()
        (knowledge / "article.txt").write_text("# Nebula\n\n[0] A nebula shines.")
        payload = {"id": "one", "question": "nebula", "documents": {
            "article.txt": {"title": "Nebula", "sentences": ["A nebula shines."]}}}
        input_path = self.root / "input.json"
        input_path.write_text(json.dumps(payload))
        (self.root / ".env").write_text("HARNESS_KNOWLEDGE=missing\nHARNESS_WORKSPACE=from-file\n")
        for module in ("single", "multi"):
            results = []
            for index in range(2):
                entry = ["-m", "agents.hotpotqa." + module]
                output = self.root / f"{module}{index}.json"
                trace = self.root / f"{module}{index}.jsonl"
                child = subprocess.run([sys.executable, "-B", *entry, "--input", str(input_path),
                                        "--output", str(output), "--trace", str(trace)], cwd=self.root,
                                       env={"PATH": os.defpath, "PYTHONPATH": str(REPO),
                                            "HARNESS_KNOWLEDGE": str(knowledge), "HARNESS_WORKSPACE": "from-shell"},
                                       capture_output=True, text=True, timeout=15)
                self.assertEqual(child.returncode, 0, child.stderr)
                self.assertNotIn("RuntimeWarning", child.stderr)
                results.append(json.loads(output.read_text()))
                self.assertEqual(results[-1]["status"], "completed")
                self.assertEqual(results[-1]["tool_calls_executed"], 2 if module == "single" else 4)
                self.assertEqual(results[-1]["tool_errors"], 0)
            self.assertEqual(stable(results[0]), stable(results[1]))
        self.assertFalse((self.root / "from-file").exists())
        self.assertFalse((self.root / "workspace").exists())