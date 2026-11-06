"""Standalone distribution, explicit configuration and workers."""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import FrozenInstanceError
from functools import partial
from pathlib import Path

from agents.teaching import TeachingConfig, create_agent, create_tools
from harness import ToolPolicy
from harness.tool_execution import run_isolated_tool
from tests.helpers import Model, TemporaryDirectoryTestCase, call, usage_reply as reply
from tests.storage_fakes import FakeJournal


REPO = Path(__file__).resolve().parents[1]
RETIRED_MODULES = ("mini_harness", "harness_runtime", "harness_events", "tool_policy",
                   "tool_execution", "execution_journal", "model_clients", "hotpotqa_common",
                   "hotpotqa_agent", "hotpotqa_multi_agent", "prepare_hotpotqa", "hotpotqa_dev",
                   "run_hotpotqa", "evaluate_hotpotqa", "hotpotqa_scoring", "hotpotqa_comparison")


class SeparationTests(TemporaryDirectoryTestCase):
    def test_retired_root_modules_are_not_available(self):
        child = self.child("""
            import importlib.util, json, sys
            import harness
            from agents import teaching, hotpotqa
            from evaluation import hotpotqa as evaluation
            for name in json.loads(sys.argv[1]):
                assert importlib.util.find_spec(name) is None, name
            """, REPO, json.dumps(RETIRED_MODULES))
        self.assertEqual(child.returncode, 0, child.stderr)

    def test_teaching_import_and_tool_factory_have_no_application_io(self):
        (self.root / ".env").write_text("TEACHING_SENTINEL=must-not-load\n")
        child = self.child("""
            import os, sys
            from pathlib import Path
            before = set(Path('.').iterdir())
            from agents.teaching import TeachingConfig, create_tools
            config = TeachingConfig.from_env({'HARNESS_WORKSPACE': 'missing/deep'})
            tools = create_tools(config)
            assert set(Path('.').iterdir()) == before
            assert 'mini_harness' not in sys.modules
            assert not any(name.startswith('hotpotqa') for name in sys.modules)
            assert 'TEACHING_SENTINEL' not in os.environ
            """)
        self.assertEqual(child.returncode, 0, child.stderr)

    def test_copied_package_runs_tools_policies_events_and_worker_alone(self):
        isolated = self.root / "distribution"
        shutil.copytree(REPO / "harness", isolated / "harness", ignore=shutil.ignore_patterns("__pycache__"))
        (isolated / "custom_tools.py").write_text(
            "import sys\ndef echo(prefix, value: str):\n"
            "    assert 'mini_harness' not in sys.modules\n"
            "    assert not any(n.startswith('agents') for n in sys.modules)\n"
            "    return prefix + value\n")
        child = self.child("""
            import importlib.util, sys
            from functools import partial
            from pathlib import Path
            before = set(Path('.').iterdir())
            import harness
            assert set(Path('.').iterdir()) == before
            assert importlib.util.find_spec('mini_harness') is None
            assert importlib.util.find_spec('model_clients') is None
            from custom_tools import echo
            bound = partial(echo, 'standalone:')
            bound.__name__ = 'echo'
            tools = harness.ToolRegistry()
            tools.tool(read_only=True)(bound)
            assert list(tools['echo']['schema']['input_schema']['properties']) == ['value']
            responses = iter([
                {'content': [{'type': 'tool_use', 'id': 'one', 'name': 'echo', 'input': {'value': 'ok'}}]},
                {'content': [{'type': 'text', 'text': 'done'}]}])
            events = []
            agent = harness.Agent(lambda *a, **k: next(responses), 'custom', tools=tools,
                quiet=True, tool_timeout_seconds=3, on_event=events.append,
                tool_policy=harness.ToolPolicy(read_only=True))
            result = agent.run_result('go')
            assert result.status == 'completed' and result.usage.tool_errors == 0
            assert agent.messages[2]['content'][0]['content'] == 'standalone:ok'
            assert events[-1].kind == 'run_finished'
            assert agent.journal is None
            assert 'agents' not in sys.modules and 'mini_harness' not in sys.modules
            """, isolated)
        self.assertEqual(child.returncode, 0, child.stderr)

    def test_teaching_schemas_do_not_expose_bound_configuration(self):
        tools = create_tools(TeachingConfig(self.root / "workspace", self.root / "knowledge"))
        for name, entry in tools.items():
            self.assertEqual(entry["schema"]["name"], name)
            self.assertNotIn("context", entry["schema"]["input_schema"]["properties"])
            self.assertNotIn(str(self.root), json.dumps(entry["schema"]))
            self.assertEqual(entry["read_only"], name not in ("write_file", "http_get"))

    def test_explicit_configuration_is_frozen_and_does_not_load_environment(self):
        config = TeachingConfig.from_env({"HARNESS_WORKSPACE": str(self.root / "explicit"),
                                          "HARNESS_COMPACT_CHARS": "1234"})
        self.assertEqual(config.compact_at, 1234)
        self.assertEqual(config.workspace, self.root / "explicit")
        self.assertFalse(config.workspace.exists())
        with self.assertRaises(FrozenInstanceError):
            config.workspace = self.root / "other"
        for settings in ({"compact_at": -1}, {"max_tool_output": 0}, {"compact_at": True}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                TeachingConfig(**settings)

    def test_two_workspaces_and_corpora_remain_independent_inline_and_timed(self):
        agents = []
        for timeout in (None, 3):
            for label in ("alpha", "beta"):
                base = self.root / (label + str(timeout))
                config = TeachingConfig(base / "workspace", base / "knowledge")
                config.knowledge.mkdir(parents=True)
                (config.knowledge / "guide.txt").write_text("nebula " + label)
                model = Model(call("write_file", path="note.txt", content=label),
                              call("read_file", path="note.txt"), call("search_documents", query="nebula"),
                              call("read_file", path="../outside.txt"),
                              call("read_file", context={}, path="note.txt"), reply("done"))
                agent = create_agent(model, config, quiet=True, approve=lambda *_: True,
                                     trace_path=base / "trace.jsonl", tool_timeout_seconds=timeout,
                                     journal=FakeJournal())
                agents.append((agent, config, label))
        # Construct all applications first, then interleave their work; no last-config-wins state.
        for agent, config, label in reversed(agents):
            result = agent.run_result("go")
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.usage.tool_errors, 2)
            self.assertEqual((config.workspace / "note.txt").read_text(), label)
            self.assertEqual(agent.messages[4]["content"][0]["content"], label)
            evidence = json.loads(agent.messages[6]["content"][0]["content"])
            self.assertEqual(evidence["results"][0]["text"], "nebula " + label)
            expected = "escapes workspace" if agent.tool_timeout_seconds is None else "tool execution failed"
            self.assertIn(expected, agent.messages[8]["content"][0]["content"])
            self.assertIn("unexpected argument", agent.messages[10]["content"][0]["content"])
            self.assertEqual(len(agent.journal.inspect()), 4)

    def test_factory_declares_writable_root_and_enforces_policy(self):
        config = TeachingConfig(self.root / "workspace", self.root / "knowledge")
        agent = create_agent(Model(call("write_file", path="blocked.txt", content="no"), reply("done")),
                             config, quiet=True, trace_path=self.root / "trace.jsonl",
                             tool_policy=ToolPolicy(read_only=True))
        self.assertEqual(agent.run_result("go").usage.tool_errors, 1)
        self.assertEqual(agent.writable_roots, (config.workspace,))
        self.assertFalse((config.workspace / "blocked.txt").exists())

    def test_partial_worker_rejects_non_json_configuration_and_non_importable_functions(self):
        from agents.teaching.tools import read_file
        from unittest.mock import Mock, patch
        local = Mock()
        for fn in (partial(read_file, object()), partial(lambda: local())):
            with self.subTest(fn=type(fn).__name__), patch("harness.tool_execution.subprocess.Popen") as spawn:
                outcome = run_isolated_tool(fn, {}, 1, 100)
                self.assertEqual(outcome.status, "failed")
                self.assertFalse(outcome.outcome_unknown)
                spawn.assert_not_called()
        local.assert_not_called()

    def test_package_recovery_command_directs_users_to_database_backend(self):
        child = subprocess.run([sys.executable, "-m", "harness"],
                               cwd=self.root, env={"PATH": os.defpath, "PYTHONPATH": str(REPO)},
                               capture_output=True, text=True, timeout=10)
        self.assertNotEqual(child.returncode, 0)
        self.assertIn("stores.postgres", child.stderr)
        self.assertIn("stores.mongodb", child.stderr)
        self.assertNotIn("RuntimeWarning", child.stderr)

    def test_new_cli_respects_dotenv_and_shell_precedence_with_timed_tools(self):
        (self.root / ".env").write_text("HARNESS_WORKSPACE=from-file\nHARNESS_TOOL_TIMEOUT_SECONDS=3\n")
        child = subprocess.run([sys.executable, "-m", "agents.teaching.cli", "mock"], cwd=self.root,
                               env={"PATH": os.defpath, "PYTHONPATH": str(REPO),
                                    "HARNESS_WORKSPACE": "from-shell"},
                               capture_output=True, text=True, timeout=20)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual((self.root / "from-shell/report.txt").read_text(), "17 * 23 + 4 = 395\n")
        self.assertFalse((self.root / "from-file").exists())
        self.assertFalse((self.root / "workspace").exists())
        self.assertNotIn("RuntimeWarning", child.stderr)