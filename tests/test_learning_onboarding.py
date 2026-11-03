"""First-run success, expected exercise feedback and preview isolation."""

import contextlib
import io
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

from lessons import lab00_basics, lab03_failures, preview, solutions
from tests.helpers import TemporaryDirectoryTestCase


REPO = Path(__file__).resolve().parents[1]


class LearningOnboardingTests(TemporaryDirectoryTestCase):
    def command(self, module, *flags):
        return subprocess.run([sys.executable, "-B", "-m", "lessons." + module, *flags],
                              cwd=self.root, env={"PATH": os.defpath, "PYTHONPATH": str(REPO)},
                              capture_output=True, text=True, timeout=10)

    def test_preview_succeeds_without_loading_learner_work_or_solutions(self):
        (self.root / ".env").write_text("LEARNING_SENTINEL=must-not-load\n")
        child = self.child("""
            import contextlib, io, os, socket, sys
            from pathlib import Path
            from unittest.mock import patch
            class NoLearnerCode:
                def find_spec(self, fullname, *args):
                    if fullname in {'lessons.exercises', 'lessons.solutions'}:
                        raise AssertionError('preview must not load learner code or solutions')
            sys.meta_path.insert(0, NoLearnerCode())
            before = set(Path('.').iterdir())
            with patch.object(socket, 'socket', side_effect=AssertionError('network forbidden')):
                from lessons import preview
                output = io.StringIO()
                with patch.object(sys, 'argv', ['preview', '--messages', '--events']):
                    with contextlib.redirect_stdout(output):
                        assert preview.main() == 0
            assert 'Preview complete.' in output.getvalue()
            assert 'answer: 3' in output.getvalue()
            assert 'LEARNING_SENTINEL' not in os.environ
            assert 'lessons.exercises' not in sys.modules
            assert 'lessons.solutions' not in sys.modules
            assert set(Path('.').iterdir()) == before
            """)
        self.assertEqual(child.returncode, 0, child.stderr)

    def test_preview_never_claims_success_when_the_run_stops(self):
        stopped = lab03_failures.run(lambda _: "repeat")
        output = io.StringIO()
        with mock.patch.object(preview, "run", return_value=stopped):
            with mock.patch.object(sys, "argv", ["preview"]), contextlib.redirect_stdout(output):
                code = preview.main()
        self.assertEqual(code, 1)
        self.assertIn("did not finish as expected", output.getvalue())
        self.assertNotIn("Preview complete.", output.getvalue())

    def test_preview_commands_show_success_and_optional_observations(self):
        for flags in ((), ("--messages", "--events"), ("--help",)):
            child = self.command("preview", *flags)
            self.assertEqual(child.returncode, 0, child.stderr)
            if "--help" not in flags:
                self.assertIn("Preview complete.", child.stdout)
                self.assertIn("answer: 3", child.stdout)
                self.assertNotIn("FAIL:", child.stdout)
            if "--messages" in flags:
                self.assertIn("Synthetic conversation", child.stdout)
                self.assertIn("Metadata events", child.stdout)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_stage_one_checks_do_not_run_the_full_counting_scenario(self):
        def text_only(model, tools, text, *, max_model_calls=3):
            messages = [{"role": "user", "content": text}]
            block = model(messages)
            if block["type"] != "text":
                raise AssertionError("stage one must not ask for tool execution")
            messages.append({"role": "assistant", "content": [block]})
            return {"status": "completed", "answer": block["text"], "messages": messages, "model_calls": 1}

        output = io.StringIO()
        with mock.patch.object(solutions, "run_loop", side_effect=text_only):
            with mock.patch.object(sys, "argv", ["lab00", "--solution", "--check", "--stage", "1"]):
                with contextlib.redirect_stdout(output):
                    self.assertEqual(lab00_basics.main(), 0)
        self.assertIn("--stage 2", output.getvalue())

    def test_all_stage_commands_pass_with_reference_and_check_messages_are_explicit(self):
        for stage in range(1, 6):
            child = self.command("lab00_basics", "--solution", "--check", "--stage", str(stage))
            self.assertEqual(child.returncode, 0, child.stderr)
            self.assertIn(f"Checkpoint {stage}/5", child.stdout)
            self.assertNotIn("FAIL:", child.stdout)
        child = self.command("lab00_basics", "--solution", "--check", "--messages")
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertIn("Synthetic checkpoint conversation", child.stdout)

    def test_default_lab_zero_hides_the_deliberate_error_until_requested(self):
        child = self.command("lab00_basics")
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertIn("Exercise mode", child.stdout)
        self.assertNotIn("AttributeError", child.stdout)
        child = self.command("lab00_basics", "--solution", "--scenario", "invalid")
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertIn("Deliberate failure demonstration (caught", child.stdout)
        self.assertIn("AttributeError", child.stdout)
        self.assertIn("answer: 3", child.stdout)

    def test_stage_options_require_explicit_assessment_and_valid_range(self):
        for flags in (("--stage", "1"), ("--check", "--stage", "0"),
                      ("--check", "--stage", "6"), ("--check", "--scenario", "repeat"),
                      ("--check", "--max-model-calls", "1")):
            child = self.command("lab00_basics", *flags)
            self.assertEqual(child.returncode, 2, child.stderr)
            self.assertNotIn("Traceback", child.stderr)

    def test_other_lab_clis_label_expected_exercise_failures(self):
        for module in ("lab01_loop", "lab02_tools", "lab03_failures"):
            child = self.command(module)
            self.assertEqual(child.returncode, 0, child.stderr)
            self.assertIn("Exercise mode", child.stdout)