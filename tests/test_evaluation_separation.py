"""Host ownership, package commands, parser boundary and independent example."""

import ast
import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

from evaluation.hotpotqa import dev, evaluate, prepare, scoring
from agents.hotpotqa.prediction import parse_prediction
from examples.inventory_agent import create_agent
from harness import ToolPolicy
from tests.helpers import Model, TemporaryDirectoryTestCase, call, records, usage_reply as reply


REPO = Path(__file__).resolve().parents[1]


class EvaluationSeparationTests(TemporaryDirectoryTestCase):
    def command(self, entry, *args):
        return subprocess.run([sys.executable, "-B", *entry, *map(str, args)], cwd=self.root,
                              env={"PATH": os.defpath, "PYTHONPATH": str(REPO)},
                              capture_output=True, text=True, timeout=30)

    def test_package_imports_have_stable_paths_and_no_configuration_io(self):
        (self.root / ".env").write_text("EVALUATION_SENTINEL=must-not-load\n")
        child = self.child("""
            import os, sys
            from pathlib import Path
            before = set(Path('.').iterdir())
            from evaluation.hotpotqa import prepare, dev, evaluate, scoring, comparison, run_question
            root = Path(sys.argv[1])
            assert prepare.DEFAULT_OUTPUT == root / 'datasets/hotpotqa'
            assert dev.DEFAULT_DEV_OUTPUT == root / 'datasets/hotpotqa-dev'
            assert evaluate.REPO == root and evaluate.TEACHING_SUBSET == prepare.DEFAULT_OUTPUT
            assert 'EVALUATION_SENTINEL' not in os.environ
            assert set(Path('.').iterdir()) == before
            """, REPO, str(REPO))
        self.assertEqual(child.returncode, 0, child.stderr)

    def test_applications_run_without_evaluation_or_root_modules_present(self):
        isolated = self.root / "distribution"
        for name in ("harness", "agents"):
            shutil.copytree(REPO / name, isolated / name, ignore=shutil.ignore_patterns("__pycache__"))
        child = self.child("""
            import importlib.util, json, sys
            from pathlib import Path
            from agents.hotpotqa import HotpotQAConfig, single, multi
            from agents.hotpotqa.prediction import parse_prediction
            assert importlib.util.find_spec('evaluation') is None
            assert importlib.util.find_spec('hotpotqa_scoring') is None
            knowledge = Path('knowledge')
            knowledge.mkdir()
            (knowledge / 'article.txt').write_text('# Nebula\\n\\n[0] A nebula shines.')
            config = HotpotQAConfig(knowledge=knowledge)
            payload = {'id': 'one', 'question': 'nebula', 'documents': {
                'article.txt': {'title': 'Nebula', 'sentences': ['A nebula shines.']}}}
            for module, runner, count in ((single, single.run_agent, 2), (multi, multi.run_agents, 4)):
                result = runner(payload, module.smoke_model, Path(str(count) + '.jsonl'), config=config)
                assert result['status'] == 'completed' and result['tool_calls_executed'] == count
                assert result['tool_errors'] == 0
                assert parse_prediction(result['raw_answer'])['abstain']
            assert not any(n.startswith('evaluation') for n in sys.modules)
            assert 'mini_harness' not in sys.modules
            assert not Path('workspace').exists()
            """, isolated)
        self.assertEqual(child.returncode, 0, child.stderr)

    def test_preparation_commands_produce_identical_bytes_and_refuse_overwrite(self):
        source = self.root / "source.json"
        source.write_bytes(prepare.json_bytes(records()))
        for index in range(2):
            entry = ["-m", "evaluation.hotpotqa.prepare"]
            output = self.root / str(index)
            child = self.command(entry, "--input", source, "--output", output)
            self.assertEqual(child.returncode, 0, child.stderr)
            self.assertNotIn("RuntimeWarning", child.stderr)
            expected = prepare.build_files(prepare.select_questions(records()))
            self.assertEqual({str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()}, expected)
            refused = self.command(entry, "--input", source, "--output", output)
            self.assertEqual(refused.returncode, 1)
            self.assertEqual((output / "source.json").read_bytes(), expected["source.json"])

    def test_module_commands_expose_cli_options(self):
        for module, option in (("prepare", "--input"), ("run_question", "--question"),
                               ("evaluate", "--architecture")):
            child = self.command(["-m", "evaluation.hotpotqa." + module], "--help")
            self.assertEqual(child.returncode, 0, child.stderr)
            self.assertNotIn("RuntimeWarning", child.stderr)
            self.assertIn(option, child.stdout)

    def test_prediction_parser_is_shared_without_worker_dependencies_on_evaluation(self):
        self.assertIs(scoring.parse_prediction, parse_prediction)
        self.assertIs(evaluate.parse_prediction, parse_prediction)
        forbidden = ("evaluation", "hotpotqa_scoring", "prepare_hotpotqa", "hotpotqa_dev", "evaluate_hotpotqa")
        for path in (REPO / "agents").rglob("*.py"):
            imports = []
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.append(node.module or "")
            self.assertFalse(any(name.startswith(forbidden) for name in imports), str(path))

    @unittest.skipUnless(os.environ.get("HARNESS_TEST_HOTPOTQA_DEV") == "1", "installed dataset opt-in")
    def test_real_dataset_evaluation_and_lesson_commands_work_from_another_directory(self):
        for module in ("evaluate", "run_question"):
            results = []
            for index in range(2):
                entry = ["-m", "evaluation.hotpotqa." + module]
                args = ["--dataset", dev.DEFAULT_DEV_OUTPUT]
                output = self.root / (module + str(index))
                args += ["--limit", "1", "--output", output] if module == "evaluate" else ["--question", "1"]
                child = self.command(entry, *args)
                self.assertEqual(child.returncode, 0, child.stderr)
                self.assertNotIn("RuntimeWarning", child.stderr)
                if module == "evaluate":
                    summary = json.loads((output / "summary.json").read_text())
                    self.assertEqual(summary["statuses"], {"completed": 1})
                    self.assertEqual(summary["tool_errors"], 0)
                    results.append((output / "inputs.json").read_bytes())
                else:
                    self.assertIn("#char-", child.stdout)
            if results:
                self.assertEqual(results[0], results[1])
        self.assertFalse((self.root / "session.json").exists())


class IndependentExampleTests(TemporaryDirectoryTestCase):
    def test_copied_runtime_and_example_need_no_other_repository_files(self):
        isolated = self.root / "distribution"
        for name in ("harness", "examples"):
            shutil.copytree(REPO / name, isolated / name, ignore=shutil.ignore_patterns("__pycache__"))
        (self.root / ".env").write_text("EXAMPLE_SENTINEL=must-not-load\n")
        child = self.child("""
            import importlib.util, os, sys
            from pathlib import Path
            before = set(Path('.').iterdir())
            from examples.inventory_agent import create_agent, OfflineModel
            from harness import RunLimits, ToolPolicy
            assert set(Path('.').iterdir()) == before
            assert importlib.util.find_spec('agents') is None
            assert importlib.util.find_spec('evaluation') is None
            events = []
            agent = create_agent(OfflineModel(), {'CUSTOM': 19}, quiet=True, on_event=events.append,
                tool_policy=ToolPolicy(read_only=True), tool_timeout_seconds=3,
                limits=RunLimits(max_model_calls=2, max_tool_calls=1))
            result = agent.run_result('CUSTOM')
            assert result.status == 'completed' and result.answer == 'CUSTOM: 19 in stock.'
            assert result.usage.tool_errors == 0 and result.usage.model_calls == 2
            assert events[-1].kind == 'run_finished'
            assert agent.journal is None
            assert 'EXAMPLE_SENTINEL' not in os.environ
            assert 'mini_harness' not in sys.modules and 'agents' not in sys.modules
            assert not Path('workspace').exists()
            """, isolated)
        self.assertEqual(child.returncode, 0, child.stderr)

    def test_inventories_are_independent_snapshots_with_only_sku_visible_to_the_model(self):
        agents = []
        for timeout, quantity in ((None, 3), (3, 11)):
            inventory = {"WIDGET": quantity}
            model = Model(call("lookup_stock", sku="WIDGET"),
                          call("lookup_stock", sku="WIDGET", inventory={"WIDGET": 999}),
                          call("lookup_stock", sku="ABSENT"), reply("done"))
            agent = create_agent(model, inventory, quiet=True, trace_path=self.root / f"{quantity}.jsonl",
                                 tool_timeout_seconds=timeout, tool_policy=ToolPolicy(read_only=True))
            inventory["WIDGET"] = 500
            agents.append((agent, model, quantity))
        for agent, model, quantity in reversed(agents):
            result = agent.run_result("go")
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.usage.tool_errors, 2)
            self.assertEqual(json.loads(agent.messages[2]["content"][0]["content"])["quantity"], quantity)
            self.assertEqual(set(model.schemas[0][0]["input_schema"]["properties"]), {"sku"})
            self.assertIn("unexpected argument", agent.messages[4]["content"][0]["content"])

    def test_example_module_cli_handles_inline_isolated_and_unknown_skus(self):
        for sku, timeout, expected in (("WIDGET", None, 0), ("GADGET", "3", 0), ("ABSENT", None, 1)):
            args = [sys.executable, "-B", "-m", "examples.inventory_agent", "--sku", sku,
                    "--trace", str(self.root / (sku + ".jsonl"))]
            if timeout:
                args += ["--tool-timeout", timeout]
            child = subprocess.run(args, cwd=self.root, env={"PATH": os.defpath, "PYTHONPATH": str(REPO)},
                                   capture_output=True, text=True, timeout=15)
            self.assertEqual(child.returncode, expected, child.stderr)
            result = json.loads(child.stdout)
            self.assertEqual(result["usage"]["tool_errors"], int(expected != 0))
            if expected == 0:
                self.assertIn(sku + ":", result["answer"])

    def test_example_source_uses_only_stdlib_and_the_documented_public_runtime(self):
        tree = ast.parse((REPO / "examples/inventory_agent.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(alias.name in ("argparse", "json") for alias in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertIn(node.module, ("functools", "pathlib", "harness"))
            elif isinstance(node, ast.Attribute) and node.attr not in ("__name__", "__doc__"):
                self.assertFalse(node.attr.startswith("_"), node.attr)