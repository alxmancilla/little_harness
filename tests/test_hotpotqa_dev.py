"""Offline full-development regressions; synthetic fixtures, no download or LLM."""

import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evaluation.hotpotqa import dev as d, prepare as p, run_question as runner
from tests.helpers import records


class FullDevelopmentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="hotpotqa-dev-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.rows = records()
        # Exercise the production path on a small fixture, not a large vendored file.
        patch = mock.patch.object(d, "DEV_COUNT", len(self.rows))
        patch.start()
        self.addCleanup(patch.stop)
        self.source = self.root / "input.json"
        self.output = self.root / "development"

    def prepare(self):
        self.source.write_bytes(p.json_bytes(self.rows))
        return d.prepare_dev(self.output, self.source)

    def test_complete_split_is_required_and_all_ids_are_checked(self):
        with mock.patch.object(d, "DEV_COUNT", 7405):
            with self.assertRaisesRegex(ValueError, "7405"):
                d.validate_records(self.rows)
        self.rows[-1]["_id"] = self.rows[0]["_id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_all_records_are_validated_before_any_output(self):
        for kind in ("oversized", "bad_type", "empty_context", "bad_label", "bad_question"):
            self.rows = records()
            last = self.rows[-1]
            if kind == "oversized":
                last["context"][0][1] = ["x" * 65536]
            elif kind == "bad_type":
                last["type"] = "unknown"
            elif kind == "empty_context":
                last["context"] = []
            elif kind == "bad_label":
                last["supporting_facts"][0][1] = -1
            else:
                last["question"] = "x" * 501
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.prepare()
            self.assertFalse(self.output.exists())

    def test_short_contexts_and_original_invalid_label_are_preserved_and_flagged(self):
        self.rows[-1]["context"] = self.rows[-1]["context"][:2]
        self.rows[-1]["supporting_facts"][0][1] = 902
        stats = self.prepare()
        manifest = json.loads((self.output / "manifest.json").read_bytes())
        self.assertEqual(stats["documents"], 132)
        self.assertEqual(stats["label_issues"], 1)
        self.assertEqual(manifest["candidate_count_distribution"], {"2": 1, "10": 13})
        self.assertEqual(manifest["label_issues"][0]["sentence"], 902)
        self.assertEqual(manifest["label_issues"][0]["issue"], "sentence_out_of_range")
        answers = json.loads((self.output / "answers.json").read_bytes())
        self.assertEqual(answers[-1]["supporting_facts"][0]["sentence"], 902)
        self.assertEqual(json.loads((self.output / "source.json").read_bytes()), self.rows)
        _, question, files = d.load_question(self.output, number=14)
        self.assertEqual(len(question["candidates"]), 2)
        self.assertEqual(len(files), 2)
        self.assertTrue(all(b"GOLD-LABEL-ONLY" not in value for value in files.values()))

    def test_source_bytes_provenance_and_offline_rebuild_are_deterministic(self):
        with mock.patch.object(d.urllib.request, "urlopen", side_effect=AssertionError("no network")):
            stats = self.prepare()
            second = self.root / "rebuilt"
            d.prepare_dev(second, self.output / "source.json")
        self.assertEqual(stats["questions"], len(self.rows))
        self.assertEqual(stats["documents"], 140)
        self.assertEqual({path.name for path in self.output.iterdir()},
                         {"source.json", "questions.json", "answers.json", "manifest.json"})
        for path in self.output.iterdir():
            self.assertEqual(path.read_bytes(), (second / path.name).read_bytes())
        data = self.source.read_bytes()
        manifest = json.loads((self.output / "manifest.json").read_bytes())
        self.assertEqual(manifest["snapshot_sha256"], hashlib.sha256(data).hexdigest())
        self.assertFalse(manifest["matches_pinned_source"])
        self.assertIsNone(manifest["source_url"])
        questions = json.loads((self.output / "questions.json").read_bytes())
        self.assertTrue(all(set(row) == {"id", "question", "candidates"} for row in questions))

    def test_public_download_is_pinned_bounded_and_requires_no_credentials(self):
        data = p.json_bytes(self.rows)
        digest = hashlib.sha256(data).hexdigest()
        with mock.patch.object(d, "DEV_SOURCE_SHA256", digest), mock.patch.object(
                d.urllib.request, "urlopen", return_value=io.BytesIO(data)) as fetch:
            d.prepare_dev(self.output)
        fetch.assert_called_once_with(d.DEV_SOURCE_URL, timeout=60)
        manifest = json.loads((self.output / "manifest.json").read_bytes())
        self.assertTrue(manifest["matches_pinned_source"])
        self.assertEqual(manifest["source_url"], d.DEV_SOURCE_URL)
        with mock.patch.object(d.urllib.request, "urlopen", return_value=io.BytesIO(data)):
            with self.assertRaisesRegex(ValueError, "checksum"):
                d.prepare_dev(self.root / "wrong-hash")
        with mock.patch.object(d, "MAX_INPUT_BYTES", 10):
            with self.assertRaisesRegex(ValueError, "exceeds"):
                d.read_bounded(io.BytesIO(b"x" * 11))
        self.assertFalse((self.root / "wrong-hash").exists())

    def test_existing_destinations_and_symlinks_are_never_overwritten(self):
        self.output.mkdir()
        note = self.output / "note.txt"
        note.write_text("keep")
        link = self.root / "link"
        link.symlink_to(self.root / "absent", target_is_directory=True)
        with mock.patch.object(d.urllib.request, "urlopen") as fetch:
            for target in (self.output, link):
                with self.subTest(target=target), self.assertRaises(FileExistsError):
                    d.prepare_dev(target)
            fetch.assert_not_called()
        self.assertEqual(note.read_text(), "keep")

    def test_select_by_number_or_id_is_isolated_and_titles_are_not_paths(self):
        self.rows[0]["context"][-1][0] = "../../answers.json"
        self.prepare()
        first = d.load_question(self.output)
        by_id = d.load_question(self.output, question_id="id-13")
        self.assertEqual(first[0], 1)
        self.assertEqual(by_id[0], 14)
        self.assertEqual(by_id, d.load_question(self.output, number=14))
        self.assertEqual(first[1]["question"], self.rows[0]["question"])
        self.assertEqual(set(first[1]), {"id", "question", "candidates"})
        self.assertEqual(len(first[2]), 10)
        self.assertTrue(all(name.startswith("knowledge/q01-p") and ".." not in name for name in first[2]))
        self.assertTrue(set(first[2]).isdisjoint(by_id[2]))
        for number in (0, 15, True):
            with self.subTest(number=number), self.assertRaisesRegex(ValueError, "not found"):
                d.load_question(self.output, number=number)
        with self.assertRaisesRegex(ValueError, "not found"):
            d.load_question(self.output, question_id="absent")

    def test_modified_snapshot_is_rejected_before_a_model_can_run(self):
        self.prepare()
        with (self.output / "source.json").open("ab") as target:
            target.write(b" ")
        with mock.patch.object(runner.subprocess, "run") as child:
            with self.assertRaisesRegex(ValueError, "integrity"):
                runner.run_question(self.output)
            child.assert_not_called()

    def test_offline_child_uses_only_selected_passages_and_a_fresh_workspace(self):
        self.prepare()
        original_run = subprocess.run
        captured, paths = [], []

        def child(command, **kwargs):
            paths.append(Path(kwargs["env"]["HARNESS_KNOWLEDGE"]).parent)
            self.assertEqual(len(list((paths[-1] / "knowledge").iterdir())), 10)
            self.assertFalse((paths[-1] / "workspace").exists())
            result = original_run(command, **kwargs, capture_output=True, text=True, timeout=10)
            captured.append(result)
            return result

        previous = Path.cwd()
        try:
            os.chdir(self.root)
            # Deliberately conflicting .env paths must not override isolation.
            (self.root / ".env").write_text("HARNESS_KNOWLEDGE=wrong\nHARNESS_WORKSPACE=wrong-workspace\n")
            with mock.patch.dict(os.environ, {"PATH": os.defpath, "HARNESS_TRACE_CONTENT": "1"}, clear=True), mock.patch.object(
                    runner.subprocess, "run", side_effect=child), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run_question(self.output), 0)
                self.assertEqual(runner.run_question(self.output, number=14), 0)
        finally:
            os.chdir(previous)
        self.assertTrue(all(result.returncode == 0 and "#char-" in result.stdout for result in captured))
        events = [json.loads(line) for line in (self.root / "trace.jsonl").read_text().splitlines()]
        results = [json.loads(event["output"]) for event in events if event["kind"] == "tool"]
        self.assertEqual(len(results), 2)
        self.assertTrue(all((row["searched_files"], row["skipped_files"]) == (10, 0) for row in results))
        self.assertTrue(all(passage["source"].startswith("q01-p") for passage in results[0]["results"]))
        self.assertTrue(all(passage["source"].startswith("q14-p") for passage in results[1]["results"]))
        self.assertNotIn("GOLD-LABEL-ONLY", (self.root / "trace.jsonl").read_text())
        self.assertFalse((self.root / "session.json").exists())
        self.assertFalse((self.root / "wrong-workspace").exists())
        self.assertNotEqual(paths[0], paths[1])
        self.assertTrue(all(not path.exists() for path in paths))

    def test_live_mode_is_explicit_and_child_exit_status_is_preserved(self):
        self.prepare()
        with mock.patch.object(runner.subprocess, "run", return_value=mock.Mock(returncode=9)) as child:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run_question(self.output, live=True), 9)
        command = child.call_args.args[0]
        self.assertNotIn("rag-mock", command)
        self.assertEqual(command[-1], self.rows[0]["question"])
        self.assertFalse(Path(child.call_args.kwargs["env"]["HARNESS_KNOWLEDGE"]).exists())


@unittest.skipUnless(os.environ.get("HARNESS_TEST_HOTPOTQA_DEV") == "1",
                     "opt in to installed-data checks with HARNESS_TEST_HOTPOTQA_DEV=1")
class InstalledDevelopmentTests(unittest.TestCase):
    """Optional real-data validation; never downloads data or calls a live model."""

    def test_full_snapshot_and_representative_offline_cli_runs(self):
        data = (d.DEFAULT_DEV_OUTPUT / "source.json").read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), d.DEV_SOURCE_SHA256)
        rows = json.loads(data)
        d.validate_records(rows)
        counts, issues = [], []
        for number, record in enumerate(rows, 1):
            files, _, _, _, warnings = p.build_example(record, number, full_dev=True)
            counts.append(len(files))
            issues.extend(warnings)
        self.assertEqual(sum(counts), 73700)
        self.assertEqual(sum(count < 10 for count in counts), 60)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["sentence"], 902)
        # The mirror agrees with the existing public ten-question snapshot.
        self.assertEqual(p.select_questions(rows), json.loads((p.DEFAULT_OUTPUT / "source.json").read_bytes()))
        short = next(i for i, row in enumerate(rows, 1) if len(row["context"]) == 2)
        flagged = next(i for i, row in enumerate(rows, 1) if row["_id"] == issues[0]["question_id"])
        selections = (1, 7405, short, flagged)
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="hotpotqa-real-check-") as temporary:
            root = Path(temporary)
            for number in selections:
                with self.subTest(question=number):
                    result = subprocess.run([sys.executable, "-m", "evaluation.hotpotqa.run_question",
                                             "--question", str(number)],
                                            cwd=root, env={"PATH": os.defpath, "PYTHONPATH": str(repo),
                                                           "HARNESS_TRACE_CONTENT": "1"}, capture_output=True,
                                            text=True, timeout=20)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("#char-", result.stdout)
            events = [json.loads(line) for line in (root / "trace.jsonl").read_text().splitlines()]
            results = [json.loads(event["output"]) for event in events if event["kind"] == "tool"]
            self.assertEqual([row["searched_files"] for row in results],
                             [len(rows[number - 1]["context"]) for number in selections])
            self.assertTrue(all(row["skipped_files"] == 0 and row["status"] == "ok" for row in results))
            self.assertFalse((root / "session.json").exists())


if __name__ == "__main__":
    unittest.main()