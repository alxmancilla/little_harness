"""Executable learning material: protocol, assessment, failure recovery and isolation."""

import contextlib
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from lessons import checks, lab01_loop, lab02_tools, lab03_failures, solutions, support
from tests.helpers import TemporaryDirectoryTestCase


REPO = Path(__file__).resolve().parents[1]


class LearningLabTests(TemporaryDirectoryTestCase):
    def assessment(self, checker, *args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = checker(*args)
        return code, output.getvalue()

    def test_loop_pairs_request_result_and_final_answer(self):
        result, messages, events = lab01_loop.run()
        self.assertEqual((result.status, result.answer), ("completed", "3"))
        self.assertEqual((result.usage.model_calls, result.usage.tool_calls_attempted,
                          result.usage.tool_errors, result.usage.total_tokens), (2, 1, 0, None))
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "user", "assistant"])
        call, response = messages[1]["content"][0], messages[2]["content"][0]
        self.assertEqual((call["name"], call["input"]), ("word_count", {"text": lab01_loop.TEXT}))
        self.assertEqual(response["tool_use_id"], call["id"])
        self.assertEqual((response["content"], response["is_error"]), ("3", False))
        self.assertEqual(messages[-1]["content"][0]["text"], result.answer)
        self.assertTrue(all("content" not in event.data for event in events))
        self.assertEqual(self.assessment(checks.check_loop, solutions.count_successful_tools, messages)[0], 0)
        self.assertEqual(self.assessment(checks.check_loop, lambda _: 0, messages)[0], 1)

    def test_annotated_trace_matches_actual_messages_events_and_summary(self):
        run = lab01_loop.run()
        document = (REPO / "lessons/annotated-trace.md").read_text()
        examples = re.findall(r"````json\n(.*?)\n````", document, re.S)
        self.assertEqual([json.loads(example) for example in examples], run[1])
        event_names = re.findall(r"^\| [0-9]+ \| `([^`]+)` \|", document, re.M)
        self.assertEqual(event_names, [event.kind for event in run[2]])
        self.assertEqual([event.sequence for event in run[2]], list(range(1, len(event_names) + 1)))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            support.show_run(run, type("Options", (), {"messages": False, "events": False})())
        summary = re.search(r"````text\n(.*?)\n````", document, re.S).group(1)
        self.assertEqual(output.getvalue().strip(), summary)

    def test_custom_tool_schema_and_unique_word_round_trip(self):
        registry = lab01_loop.create_tools(solutions.unique_word_count)
        schema = registry["unique_word_count"]["schema"]
        self.assertEqual(schema["input_schema"], {
            "type": "object", "properties": {"text": {"type": "string"}},
            "required": ["text"], "additionalProperties": False})
        self.assertTrue(registry["unique_word_count"]["read_only"])
        run = lab02_tools.run(solutions.unique_word_count)
        self.assertEqual((run[0].answer, run[0].usage.tool_errors), ("2", 0))
        self.assertEqual(run[1][1]["content"][0]["name"], "unique_word_count")
        code, output = self.assessment(checks.check_tool, solutions.unique_word_count, run)
        self.assertEqual(code, 0, output)
        self.assertEqual(output.count("PASS:"), 8)

    def test_semantically_wrong_tool_completes_but_fails_assessment(self):
        # Use an explicit faulty fixture, not the editable exercise starter.
        def total_instead_of_unique(text: str) -> str:
            return str(len(text.split()))

        run = lab02_tools.run(total_instead_of_unique)
        self.assertEqual((run[0].status, run[0].answer, run[0].usage.tool_errors), ("completed", "3", 0))
        code, output = self.assessment(checks.check_tool, total_instead_of_unique, run)
        self.assertEqual(code, 1)
        self.assertIn("FAIL: the agent uses the new tool and answers 2", output)

    def test_invalid_request_is_not_executed_and_corrected_request_is(self):
        result, messages, events = run = lab03_failures.run(solutions.next_action)
        self.assertEqual((result.status, result.answer), ("completed", "3"))
        self.assertEqual((result.usage.model_calls, result.usage.tool_calls_attempted,
                          result.usage.tool_errors), (3, 2, 1))
        calls = [m["content"][0] for m in messages if m["role"] == "assistant"
                 and m["content"][0]["type"] == "tool_use"]
        self.assertEqual([c["input"] for c in calls], [{"text": 42}, {"text": lab01_loop.TEXT}])
        results = [m["content"][0] for m in messages if isinstance(m["content"], list)
                   and m["content"][0]["type"] == "tool_result"]
        self.assertEqual([r["tool_use_id"] for r in results], [c["id"] for c in calls])
        self.assertNotEqual(calls[0]["id"], calls[1]["id"])
        self.assertEqual([r["is_error"] for r in results], [True, False])
        self.assertIn("must be string", results[0]["content"])
        finished = [event for event in events if event.kind == "tool_finished"]
        self.assertEqual([event.data["status"] for event in finished], ["not_run", "completed"])
        self.assertEqual(sum(event.kind == "tool_started" for event in events), 1)
        self.assertEqual(self.assessment(checks.check_recovery, solutions.next_action, run)[0], 0)

    def test_repeating_invalid_input_stops_at_the_budget_without_execution(self):
        repeat = lambda _: "repeat"
        result, messages, events = run = lab03_failures.run(repeat)
        self.assertEqual((result.status, result.stop_reason, result.answer), ("stopped", "max_model_calls", ""))
        self.assertEqual((result.usage.model_calls, result.usage.tool_calls_attempted,
                          result.usage.tool_errors), (3, 3, 3))
        self.assertFalse(any(event.kind == "tool_started" for event in events))
        self.assertEqual(messages[-1]["content"][0]["is_error"], True)
        self.assertEqual(self.assessment(checks.check_recovery, repeat, run)[0], 1)

    def test_finishing_on_an_error_cannot_pass_recovery_checks(self):
        finish = lambda _: "finish"
        run = lab03_failures.run(finish)
        self.assertEqual(run[0].status, "completed")
        self.assertEqual(run[0].usage.tool_errors, 1)
        self.assertEqual(self.assessment(checks.check_recovery, finish, run)[0], 1)

    def test_feedback_handles_exercise_exceptions_without_dumping_details(self):
        def broken(_):
            raise RuntimeError("private diagnostic sentinel")

        code, output = self.assessment(checks.check_loop, broken, [])
        self.assertEqual(code, 1)
        self.assertEqual(output.count("FAIL:"), 4)
        self.assertNotIn("private diagnostic sentinel", output)

    def test_temporary_trace_is_removed_after_running(self):
        temporary = TemporaryDirectory(dir=self.root)
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name)
        with mock.patch.object(support, "TemporaryDirectory", return_value=temporary):
            lab01_loop.run()
        self.assertFalse(path.exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_import_and_run_need_no_configuration_provider_or_persistent_files(self):
        (self.root / ".env").write_text("LEARNING_SENTINEL=must-not-load\n")
        child = self.child("""
            import os, socket, sys
            from pathlib import Path
            from unittest.mock import patch
            before = set(Path('.').iterdir())
            with patch.object(socket, 'socket', side_effect=AssertionError('network forbidden')):
                import harness
                with patch.object(harness, 'make_model', side_effect=AssertionError('provider forbidden')):
                    from lessons import lab01_loop, lab02_tools, lab03_failures, support
                    exercise = support.exercise_module(False)
                    assert 'lessons.solutions' not in sys.modules
                    assert set(Path('.').iterdir()) == before
                    lab01_loop.run()
                    lab02_tools.run(exercise.unique_word_count)
                    lab03_failures.run(exercise.next_action)
            assert 'LEARNING_SENTINEL' not in os.environ
            assert set(Path('.').iterdir()) == before
            assert not any(n.startswith(('agents', 'evaluation')) for n in sys.modules)
            """)
        self.assertEqual(child.returncode, 0, child.stderr)

    def test_all_lab_commands_and_solution_checks_from_another_directory(self):
        (self.root / ".env").write_text("HARNESS_PROVIDER=invalid\nHARNESS_MAX_MODEL_CALLS=invalid\n")
        env = {"PATH": os.defpath, "PYTHONPATH": str(REPO),
               "HARNESS_PROVIDER": "invalid", "HARNESS_WORKSPACE": str(self.root / "must-not-create")}
        for module in ("lab01_loop", "lab02_tools", "lab03_failures"):
            for flags in ([], ["--solution", "--check", "--messages", "--events"], ["--help"]):
                with self.subTest(module=module, flags=flags):
                    child = subprocess.run([sys.executable, "-B", "-m", "lessons." + module, *flags],
                                           cwd=self.root, env=env, capture_output=True, text=True, timeout=10)
                    self.assertEqual(child.returncode, 0, child.stderr)
                    self.assertNotIn("RuntimeWarning", child.stderr)
                    if "--check" in flags:
                        self.assertIn("PASS:", child.stdout)
                        self.assertNotIn("FAIL:", child.stdout)
                    elif not flags:
                        self.assertIn("status:", child.stdout)
        self.assertEqual({p.name for p in self.root.iterdir()}, {".env"})