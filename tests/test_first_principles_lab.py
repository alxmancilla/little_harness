"""Lab 0: an independent teaching loop, useful feedback and the runtime handoff."""

import contextlib
import io
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from unittest import mock

from lessons import checks, lab00_basics as lab, lab01_loop, solutions
from tests.helpers import TemporaryDirectoryTestCase


REPO = Path(__file__).resolve().parents[1]


class FirstPrinciplesLabTests(TemporaryDirectoryTestCase):
    def run_loop(self, model=lab.counting_model, **options):
        return solutions.run_loop(model, {"word_count": lab.word_count}, lab.TEXT, **options)

    def assessment(self, loop, **options):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = checks.check_basics(loop, **options)
        return code, output.getvalue()

    def test_plain_function_workflow_and_unchecked_failure(self):
        self.assertEqual(lab.word_count(lab.TEXT), "3")
        self.assertEqual(lab.fixed_workflow(lab.TEXT), "Word count: 3")
        self.assertIn("unchecked input reached the function", lab.unchecked_failure())
        self.assertEqual(lab.word_count(" \t\n"), "0")

    def test_reference_messages_match_the_supported_harness(self):
        report = self.run_loop()
        result, messages, _ = lab01_loop.run()
        self.assertEqual((report["status"], report["answer"], report["model_calls"]),
                         (result.status, result.answer, result.usage.model_calls))
        self.assertEqual(report["messages"], messages)
        self.assertEqual(self.run_loop()["messages"], messages)  # No shared history.

    def test_model_sees_result_before_proposing_final_text(self):
        observations = []

        def model(messages):
            observations.append(json.loads(json.dumps(messages)))
            return lab.counting_model(messages)

        report = self.run_loop(model)
        self.assertEqual(len(observations), 2)
        self.assertEqual(observations[0], [{"role": "user", "content": lab.TEXT}])
        result = observations[1][-1]["content"][0]
        self.assertEqual(result, {"type": "tool_result", "tool_use_id": "count-1",
                                  "content": "3", "is_error": False})
        self.assertEqual(report["messages"][:-1], observations[1])

    def test_invalid_input_is_repaired_without_executing_the_bad_call(self):
        counter = mock.Mock(side_effect=lab.word_count)
        report = solutions.run_loop(lab.invalid_then_repair_model,
                                    {"word_count": counter}, "one two")
        counter.assert_called_once_with(text="one two")
        self.assertEqual((report["status"], report["answer"], report["model_calls"]),
                         ("completed", "2", 3))
        errors = [m["content"][0]["is_error"] for m in report["messages"]
                  if isinstance(m["content"], list) and m["content"][0]["type"] == "tool_result"]
        self.assertEqual(errors, [True, False])

    def test_unregistered_tool_never_executes_even_if_arguments_are_valid(self):
        counter = mock.Mock(side_effect=lab.word_count)

        def model(messages):
            if len(messages) == 1:
                return {"type": "tool_use", "id": "unknown", "name": "unregistered_tool",
                        "input": {"text": lab.TEXT}}
            self.assertTrue(messages[-1]["content"][0]["is_error"])
            return {"type": "text", "text": "handled"}

        report = solutions.run_loop(model, {"word_count": counter}, lab.TEXT)
        counter.assert_not_called()
        self.assertEqual(report["answer"], "handled")

    def test_repetition_and_final_answer_both_respect_the_call_limit(self):
        for model, limit in ((lab.repeating_model, 2), (lab.counting_model, 1)):
            with self.subTest(model=model.__name__):
                counter = mock.Mock(side_effect=lab.word_count)
                report = solutions.run_loop(model, {"word_count": counter}, lab.TEXT,
                                            max_model_calls=limit)
                self.assertEqual((report["status"], report["answer"], report["model_calls"]),
                                 ("stopped", "", limit))
                self.assertEqual(counter.call_count, limit)
                self.assertEqual(report["messages"][-1]["content"][0]["type"], "tool_result")

    def test_invalid_limits_are_rejected_before_any_model_call(self):
        for limit in (0, -1, True, 1.5, "2", None):
            with self.subTest(limit=limit):
                model = mock.Mock()
                with self.assertRaises(ValueError):
                    self.run_loop(model, max_model_calls=limit)
                model.assert_not_called()

    def test_all_reference_checks_pass_and_bad_return_shapes_get_feedback(self):
        code, output = self.assessment(solutions.run_loop)
        self.assertEqual(code, 0, output)
        self.assertEqual(output.count("PASS:"), 12)
        self.assertIn("All five checkpoints complete", output)

        def broken(*args, **kwargs):
            raise RuntimeError("private diagnostic sentinel")

        for loop in (broken, lambda *a, **k: {}):
            code, output = self.assessment(loop)
            self.assertEqual(code, 1)
            self.assertEqual(output.count("FAIL:"), 1)
            self.assertIn("Focus on checkpoint 1", output)
            self.assertNotIn("Checkpoint 2/5", output)
            self.assertNotIn("private diagnostic sentinel", output)

    def test_partial_loops_pass_incremental_checkpoints_without_later_features(self):
        def partial_loop(level):
            def loop(model, tools, text, *, max_model_calls=3):
                messages = [{"role": "user", "content": text}]
                report = {"status": "stopped", "answer": "", "messages": messages, "model_calls": 0}
                for _ in range(max_model_calls):
                    block = model(messages)
                    report["model_calls"] += 1
                    messages.append({"role": "assistant", "content": [block]})
                    if block["type"] == "text":
                        report.update(status="completed", answer=block["text"])
                        return report
                    if level < 2:
                        return report
                    output = tools[block["name"]](**block["input"])
                    if level < 3:
                        return report
                    messages.append({"role": "user", "content": [{
                        "type": "tool_result", "tool_use_id": block["id"],
                        "content": output, "is_error": False}]})
                return report
            return loop

        for level in range(1, 4):
            with self.subTest(level=level):
                code, output = self.assessment(partial_loop(level), stage=level)
                self.assertEqual(code, 0, output)
                self.assertEqual(output.count("PASS:"), level)
                self.assertIn(f"--stage {level + 1}", output)
                code, output = self.assessment(partial_loop(level), stage=level + 1)
                self.assertEqual(code, 1, output)
                self.assertIn(f"Focus on checkpoint {level + 1}", output)

    def test_checkpoint_five_catches_budget_bugs_after_validation_is_complete(self):
        def wrong_repetition_limit(model, tools, text, *, max_model_calls=3):
            limit = 3 if max_model_calls == 2 else max_model_calls
            return solutions.run_loop(model, tools, text, max_model_calls=limit)

        self.assertEqual(self.assessment(wrong_repetition_limit, stage=4)[0], 0)
        code, output = self.assessment(wrong_repetition_limit)
        self.assertEqual(code, 1)
        self.assertIn("Focus on checkpoint 5", output)

    def test_invalid_limits_cannot_pass_student_assessment(self):
        def accepts_invalid_limits(model, tools, text, *, max_model_calls=3):
            if type(max_model_calls) is not int or max_model_calls < 1:
                return {}
            return solutions.run_loop(model, tools, text, max_model_calls=max_model_calls)

        code, output = self.assessment(accepts_invalid_limits)
        self.assertEqual(code, 1)
        self.assertIn("FAIL: reject invalid limits", output)

    def test_stage_selection_is_validated_before_exercise_execution(self):
        for stage in (0, 6, True, "1", 1.5):
            with self.subTest(stage=stage):
                loop = mock.Mock()
                with self.assertRaises(ValueError):
                    checks.check_basics(loop, stage=stage)
                loop.assert_not_called()

    def test_checks_catch_call_limit_and_pairing_bugs(self):
        def extra_call(model, tools, text, *, max_model_calls):
            return solutions.run_loop(model, tools, text, max_model_calls=max_model_calls + 1)

        def wrong_id(*args, **kwargs):
            report = solutions.run_loop(*args, **kwargs)
            for message in report["messages"]:
                if isinstance(message["content"], list):
                    for block in message["content"]:
                        if block["type"] == "tool_result":
                            block["tool_use_id"] = "wrong"
            return report

        for loop in (extra_call, wrong_id):
            self.assertEqual(self.assessment(loop)[0], 1)

    def test_lab_zero_import_run_and_checks_need_no_harness_or_configuration(self):
        (self.root / ".env").write_text("LEARNING_SENTINEL=must-not-load\n")
        child = self.child("""
            import contextlib, io, os, socket, sys
            from pathlib import Path
            from unittest.mock import patch
            class NoRuntime:
                def find_spec(self, fullname, *args):
                    if fullname.split('.')[0] in {'harness', 'agents', 'evaluation'}:
                        raise AssertionError('runtime import forbidden')
            sys.meta_path.insert(0, NoRuntime())
            before = set(Path('.').iterdir())
            with patch.object(socket, 'socket', side_effect=AssertionError('network forbidden')):
                from lessons import lab00_basics, support
                support.exercise_module(False)
                assert 'lessons.solutions' not in sys.modules
                with patch.object(sys, 'argv', ['lab00', '--solution', '--check']):
                    with contextlib.redirect_stdout(io.StringIO()):
                        assert lab00_basics.main() == 0
            assert 'harness' not in sys.modules
            assert 'LEARNING_SENTINEL' not in os.environ
            assert set(Path('.').iterdir()) == before
            """)
        self.assertEqual(child.returncode, 0, child.stderr)

    def test_documented_lab_zero_commands_work_outside_repository(self):
        documents = [REPO / "README.md", REPO / "LEARN.md", REPO / "lessons/00-from-functions-to-agents.md"]
        commands = {line for document in documents for line in document.read_text().splitlines()
                    if line.startswith("python3 -m lessons.lab00_basics")}
        env = {"PATH": os.defpath, "PYTHONPATH": str(REPO), "HARNESS_PROVIDER": "invalid"}
        for command in sorted(commands):
            with self.subTest(command=command):
                args = shlex.split(command)[1:]
                child = subprocess.run([sys.executable, "-B", *args], cwd=self.root, env=env,
                                       capture_output=True, text=True, timeout=10)
                expected = 1 if "--check" in args and "--solution" not in args else 0
                self.assertEqual(child.returncode, expected, child.stderr)
                self.assertNotIn("Traceback", child.stderr)
        for flags in (["--help"], ["--max-model-calls", "0"], ["--max-model-calls", "x"]):
            child = subprocess.run([sys.executable, "-B", "-m", "lessons.lab00_basics", *flags],
                                   cwd=self.root, env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(child.returncode, 0 if flags == ["--help"] else 2, child.stderr)
        self.assertEqual(list(self.root.iterdir()), [])