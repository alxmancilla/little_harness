"""Application-owned data/wiring, meaningful assessment and offline capstone UX."""

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from harness import Agent, RunLimits, ToolRegistry
from lessons import capstone_checks as checks, capstone_solution as solution, lab04_capstone as lab
from tests.helpers import TemporaryDirectoryTestCase


REPO = Path(__file__).resolve().parents[1]


class LearningCapstoneTests(TemporaryDirectoryTestCase):
    def assessment(self, app):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = checks.check_capstone(app)
        return code, output.getvalue()

    def command(self, *flags):
        return subprocess.run([sys.executable, "-B", "-m", "lessons.lab04_capstone", *flags],
                              cwd=self.root, env={"PATH": os.defpath, "PYTHONPATH": str(REPO),
                              "HARNESS_PROVIDER": "invalid", "HARNESS_MAX_MODEL_CALLS": "invalid"},
                              capture_output=True, text=True, timeout=10)

    def test_reference_passes_all_nine_application_contracts(self):
        code, output = self.assessment(solution)
        self.assertEqual(code, 0, output)
        self.assertEqual(output.count("PASS:"), 9)
        self.assertNotIn("FAIL:", output)
        self.assertIn("add your own test", output)

    def test_tools_validate_semantics_and_keep_integer_cent_calculations(self):
        tools = solution.create_tools({"FREE": 1, "PAID": 9}, {"FREE": 0, "PAID": 137})
        quote = tools["quote_total"]["fn"]
        self.assertEqual(json.loads(quote("PAID", 3)), {
            "sku": "PAID", "quantity": 3, "total_cents": 411})
        self.assertEqual(json.loads(quote("FREE", 1))["total_cents"], 0)
        self.assertEqual(set(solution.create_tools({}, {})), {"lookup_stock", "quote_total"})
        for value in (0, -1, True, False, "2", 1.5, None):
            with self.subTest(quantity=value), self.assertRaises(ValueError):
                quote("PAID", value)
        for name, arguments in (("lookup_stock", ("paid",)), ("quote_total", ("ABSENT", 1))):
            with self.assertRaises(ValueError):
                tools[name]["fn"](*arguments)
        self.assertTrue(checks._snapshots(solution))

    def test_tools_have_fresh_registries_and_do_not_modify_supplied_data(self):
        stock, prices = {"CUSTOM": 4}, {"CUSTOM": 91}
        first = solution.create_tools(stock, prices)
        stock["CUSTOM"], prices["CUSTOM"] = 8, 200
        second = solution.create_tools(stock, prices)
        self.assertIsNot(first, second)
        self.assertEqual(json.loads(first["lookup_stock"]["fn"]("CUSTOM"))["quantity"], 4)
        self.assertEqual(json.loads(first["quote_total"]["fn"]("CUSTOM", 2))["total_cents"], 182)
        self.assertEqual(json.loads(second["quote_total"]["fn"]("CUSTOM", 2))["total_cents"], 400)
        self.assertEqual((stock, prices), ({"CUSTOM": 8}, {"CUSTOM": 200}))

    def test_happy_path_pairs_both_tools_and_derives_answer_from_new_data(self):
        result, messages, events = lab.run(solution.create_agent, sku="OTHER", quantity=3,
                                          stock={"OTHER": 9}, prices={"OTHER": 137})
        self.assertEqual((result.status, result.answer), ("completed", "OTHER: 3 units cost 411 cents; stock 9."))
        self.assertEqual((result.usage.model_calls, result.usage.tool_calls_attempted,
                          result.usage.tool_errors, result.usage.total_tokens), (3, 2, 0, None))
        calls = [m["content"][0] for m in messages if m["role"] == "assistant"
                 and m["content"][0]["type"] == "tool_use"]
        results = [m["content"][0] for m in messages if m["role"] == "user" and isinstance(m["content"], list)]
        self.assertEqual([c["name"] for c in calls], ["lookup_stock", "quote_total"])
        self.assertEqual([r["tool_use_id"] for r in results], [c["id"] for c in calls])
        self.assertEqual([r["is_error"] for r in results], [False, False])
        self.assertEqual([e.data["name"] for e in events if e.kind == "tool_started"],
                         ["lookup_stock", "quote_total"])
        self.assertTrue(all("content" not in event.data for event in events))

    def test_repair_rejects_bad_call_before_execution_then_preserves_error(self):
        result, messages, events = lab.run(solution.create_agent, scenario="repair")
        self.assertEqual((result.status, result.answer), ("completed", "WIDGET: 2 units cost 500 cents; stock 7."))
        self.assertEqual((result.usage.model_calls, result.usage.tool_calls_attempted,
                          result.usage.tool_errors), (4, 3, 1))
        self.assertEqual(messages[1]["content"][0]["input"], {"sku": 42})
        self.assertTrue(messages[2]["content"][0]["is_error"])
        self.assertEqual(messages[3]["content"][0]["input"], {"sku": "WIDGET"})
        self.assertNotEqual(messages[1]["content"][0]["id"], messages[3]["content"][0]["id"])
        self.assertEqual([e.data["status"] for e in events if e.kind == "tool_finished"],
                         ["not_run", "completed", "completed"])
        self.assertEqual(sum(e.kind == "tool_started" for e in events), 2)

    def test_repeat_records_four_requests_but_admits_only_three_attempts(self):
        result, messages, events = lab.run(solution.create_agent, scenario="repeat")
        self.assertEqual((result.status, result.stop_reason, result.answer), ("stopped", "max_tool_calls", ""))
        self.assertEqual((result.usage.model_calls, result.usage.tool_calls_requested,
                          result.usage.tool_calls_attempted), (4, 4, 3))
        self.assertFalse(any(e.kind == "tool_started" for e in events))
        self.assertEqual(messages[-1]["content"][0]["tool_use_id"], "lookup_stock-4")
        self.assertIn("not run", messages[-1]["content"][0]["content"])

    def test_unknown_sku_empty_stock_and_zero_quantity_fail_without_invented_data(self):
        for options, expected, errors in (
                ({"sku": "ABSENT"}, "Unable to quote that order.", 1),
                ({"sku": "GADGET"}, "GADGET: insufficient stock (0 available).", 0),
                ({"quantity": 0}, "Unable to quote that order.", 1)):
            result = lab.run(solution.create_agent, **options)[0]
            self.assertEqual((result.status, result.answer, result.usage.tool_errors),
                             ("completed", expected, errors))

    def test_policy_assessment_actually_detects_missing_dispatch_enforcement(self):
        def no_policy(model, stock, prices, *, trace_path, on_event=None):
            return Agent(model, "A prompt alone is insufficient.", tools=solution.create_tools(stock, prices),
                         quiet=True, trace_path=trace_path, on_event=on_event,
                         limits=RunLimits(max_model_calls=4, max_tool_calls=3))

        self.assertTrue(checks._policy(solution))
        self.assertFalse(checks._policy(SimpleNamespace(create_agent=no_policy)))

    def test_too_small_factory_budget_cannot_pass_recovery(self):
        def too_small(*args, **kwargs):
            agent = solution.create_agent(*args, **kwargs)
            agent.limits = RunLimits(max_model_calls=3, max_tool_calls=3)
            return agent

        self.assertFalse(checks._recovery(SimpleNamespace(create_agent=too_small)))
        result = lab.run(too_small, scenario="repair")[0]
        self.assertEqual((result.status, result.stop_reason, result.answer), ("stopped", "max_model_calls", ""))

    def test_unfinished_fixture_gets_one_failure_without_running_later_contracts(self):
        app = SimpleNamespace(create_tools=lambda *args: ToolRegistry(),
                              create_agent=mock.Mock(side_effect=AssertionError("must not run")))
        code, output = self.assessment(app)
        self.assertEqual(code, 1)
        self.assertEqual(output.count("FAIL:"), 1)
        self.assertIn("check 1/9", output)
        self.assertIn("Hint:", output)
        self.assertIn("capstone_exercises.py", output)
        app.create_agent.assert_not_called()

    def test_wrong_calculation_fails_even_when_schemas_are_valid(self):
        def wrong_tools(stock, prices):
            tools = solution.create_tools(stock, prices)
            tools["quote_total"]["fn"] = lambda sku, quantity: json.dumps({
                "sku": sku, "quantity": quantity, "total_cents": 500})
            return tools

        app = SimpleNamespace(create_tools=wrong_tools, create_agent=solution.create_agent)
        code, output = self.assessment(app)
        self.assertEqual(code, 1)
        self.assertEqual(output.count("PASS:"), 1)
        self.assertIn("check 2/9", output)

    def test_factory_cannot_ignore_caller_data_and_pass_happy_path(self):
        def hardcoded(model, stock, prices, **options):
            return solution.create_agent(model, lab.STOCK, lab.PRICES, **options)

        self.assertFalse(checks._happy(SimpleNamespace(create_agent=hardcoded)))

    def test_feedback_does_not_dump_exception_details(self):
        def broken(*args):
            raise RuntimeError("private diagnostic sentinel")

        code, output = self.assessment(SimpleNamespace(create_tools=broken))
        self.assertEqual(code, 1)
        self.assertNotIn("private diagnostic sentinel", output)
        self.assertNotIn("Traceback", output)

    def test_temporary_trace_is_cleaned_up_on_success_and_factory_error(self):
        for broken in (False, True):
            temporary = TemporaryDirectory(dir=self.root)
            self.addCleanup(temporary.cleanup)
            path = Path(temporary.name)
            with mock.patch.object(lab, "TemporaryDirectory", return_value=temporary):
                if broken:
                    with self.assertRaises(RuntimeError):
                        lab.run(mock.Mock(side_effect=RuntimeError("fixture")))
                else:
                    lab.run(solution.create_agent)
            self.assertFalse(path.exists())

    def test_cli_demos_checks_and_option_errors(self):
        for scenario in ("happy", "repair", "repeat"):
            child = self.command("--solution", "--scenario", scenario, "--messages", "--events")
            self.assertEqual(child.returncode, 0, child.stderr)
            self.assertIn("Reference mode", child.stdout)
            self.assertIn("Synthetic conversation", child.stdout)
            self.assertIn("Metadata events", child.stdout)
            self.assertIn("status: stopped" if scenario == "repeat" else "status: completed", child.stdout)
        child = self.command("--solution", "--check")
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(child.stdout.count("PASS:"), 9)
        for flags in (("--check", "--scenario", "happy"), ("--check", "--messages"),
                      ("--check", "--events"), ("--scenario", "invalid")):
            child = self.command(*flags)
            self.assertEqual(child.returncode, 2, child.stderr)
            self.assertNotIn("Traceback", child.stderr)
        self.assertEqual(self.command("--help").returncode, 0)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_exercise_selection_stays_offline_and_does_not_load_any_reference_or_old_exercises(self):
        (self.root / ".env").write_text("CAPSTONE_SENTINEL=must-not-load\n")
        child = self.child("""
            import os, socket, sys
            from pathlib import Path
            from unittest.mock import patch
            class NoReferences:
                def find_spec(self, fullname, *args):
                    if fullname in {'lessons.capstone_solution', 'lessons.solutions', 'lessons.exercises'}:
                        raise AssertionError('must not load other learner work or references')
            sys.meta_path.insert(0, NoReferences())
            before = set(Path('.').iterdir())
            with patch.object(socket, 'socket', side_effect=AssertionError('network forbidden')):
                import harness
                with patch.object(harness, 'make_model', side_effect=AssertionError('provider forbidden')):
                    from lessons import lab04_capstone as lab
                    app = lab.implementation(False)
                    lab.run(app.create_agent)
            assert 'CAPSTONE_SENTINEL' not in os.environ
            assert set(Path('.').iterdir()) == before
            assert not any(n.startswith(('agents', 'evaluation')) for n in sys.modules)
            """)
        self.assertEqual(child.returncode, 0, child.stderr)

    def test_reference_application_runs_with_copied_runtime_and_no_lesson_package(self):
        isolated = self.root / "distribution"
        shutil.copytree(REPO / "harness", isolated / "harness", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copyfile(REPO / "lessons/capstone_solution.py", isolated / "order_app.py")
        child = self.child("""
            import importlib.util, json
            from pathlib import Path
            from order_app import create_agent
            assert importlib.util.find_spec('lessons') is None
            assert importlib.util.find_spec('agents') is None
            assert not Path('trace.jsonl').exists()
            replies = iter([
                {'content': [{'type': 'tool_use', 'id': 's', 'name': 'lookup_stock', 'input': {'sku': 'NEW'}}]},
                {'content': [{'type': 'tool_use', 'id': 'q', 'name': 'quote_total', 'input': {'sku': 'NEW', 'quantity': 3}}]}])
            def model(system, messages, tools, on_text=None):
                content = messages[-1]['content']
                if isinstance(content, list) and content[0].get('tool_use_id') == 'q':
                    quote = json.loads(content[0]['content'])
                    return {'content': [{'type': 'text', 'text': str(quote['total_cents'])}]}
                return next(replies)
            events = []
            agent = create_agent(model, {'NEW': 17}, {'NEW': 9}, trace_path='trace.jsonl', on_event=events.append)
            result = agent.run_result('quote')
            assert (result.status, result.answer, result.usage.tool_errors) == ('completed', '27', 0)
            assert sum(e.kind == 'tool_started' for e in events) == 2
            """, isolated)
        self.assertEqual(child.returncode, 0, child.stderr)