"""Offline dataset preparation tests; no network or LLM credentials required."""

import copy
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from evaluation.hotpotqa import prepare as p
from tests.helpers import records


def preview(original):
    result = []
    for index, record in enumerate(copy.deepcopy(original)):
        record["id"] = record.pop("_id")
        record["context"] = {"title": [title for title, _ in record["context"]],
                             "sentences": [sentences for _, sentences in record["context"]]}
        record["supporting_facts"] = {"title": [title for title, _ in record["supporting_facts"]],
                                      "sent_id": [sentence for _, sentence in record["supporting_facts"]]}
        result.append({"row_idx": index, "row": record, "truncated_cells": []})
    return {"rows": result}


class PreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="hotpotqa-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_selection_is_balanced_deterministic_and_keeps_source_order(self):
        original = records()
        selected = p.select_questions(original)
        self.assertEqual([row["_id"] for row in selected], [f"id-{i}" for i in range(10)])
        self.assertEqual(Counter(row["type"] for row in selected), {"bridge": 5, "comparison": 5})
        self.assertEqual(selected, p.select_questions(original))
        self.assertEqual(len(original), 14)

    def test_preview_normalizes_to_original_format_without_losing_facts(self):
        self.assertEqual(p.select_questions(preview(records())), p.select_questions(records()))

    def test_truncated_preview_or_mismatched_columns_are_rejected(self):
        payload = preview(records())
        payload["rows"][0]["truncated_cells"] = ["context"]
        with self.assertRaisesRegex(ValueError, "truncated"):
            p.select_questions(payload)
        for key, field in (("context", "title"), ("supporting_facts", "sent_id")):
            payload = preview(records())
            payload["rows"][0]["row"][key][field].pop()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "differ in length"):
                p.select_questions(payload)

    def test_insufficient_types_and_duplicate_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least five"):
            p.select_questions(records()[:9])
        original = records()
        original[1]["_id"] = original[0]["_id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            p.select_questions(original)

    def test_all_candidates_are_preserved_but_labels_are_not_retrievable(self):
        files = p.build_files(p.select_questions(records()))
        corpus = {name: content for name, content in files.items() if name.startswith("knowledge/")}
        self.assertEqual(len(corpus), 100)
        self.assertTrue(all("GOLD-LABEL-ONLY" not in text.decode() for text in corpus.values()))
        self.assertIn(b"[0] Telescope 0-9", corpus["knowledge/q01-p10.txt"])  # distractor retained
        questions = json.loads(files["questions.json"])
        self.assertTrue(all(set(row) == {"id", "question", "candidates"} for row in questions))
        self.assertEqual(len(questions[0]["candidates"]), 10)
        gold = json.loads(files["answers.json"])[0]
        self.assertEqual(gold["answer"], "GOLD-LABEL-ONLY")
        self.assertEqual(gold["supporting_facts"][1]["document"], "q01-p02.txt")
        self.assertEqual(gold["supporting_facts"][1]["sentence"], 1)

    def test_titles_never_determine_filesystem_paths(self):
        selected = p.select_questions(records())
        selected[0]["context"][9][0] = "../../outside.md"
        files = p.build_files(selected)
        self.assertEqual(len([name for name in files if name.startswith("knowledge/")]), 100)
        self.assertFalse(any(".." in name for name in files))

    def test_bad_context_supporting_facts_and_document_limits_fail_before_writing(self):
        for kind in ("context", "support", "oversized", "question"):
            selected = p.select_questions(records())
            if kind == "context":
                selected[0]["context"].pop()
            elif kind == "support":
                selected[0]["supporting_facts"][0][1] = 99
            elif kind == "oversized":
                selected[0]["context"][0][1] = ["x" * (64 * 1024)]
            else:
                selected[0]["question"] = "x" * 501
            source = self.root / "source.json"
            source.write_bytes(p.json_bytes(selected))
            output = self.root / "dataset"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                p.prepare(output, source)
            self.assertFalse(output.exists())

    def test_offline_preparation_round_trips_without_network(self):
        source = self.root / "source.json"
        source.write_bytes(p.json_bytes(records()))
        with mock.patch.object(p.urllib.request, "urlopen", side_effect=AssertionError("network forbidden")):
            stats = p.prepare(self.root / "one", source)
            p.prepare(self.root / "two", self.root / "one" / "source.json")
        self.assertEqual((stats["questions"], stats["documents"]), (10, 100))
        for path in (self.root / "one").rglob("*"):
            if path.is_file():
                self.assertEqual(path.read_bytes(), (self.root / "two" / path.relative_to(self.root / "one")).read_bytes())

    def test_public_preview_fetch_is_bounded_and_needs_no_credentials(self):
        response = io.BytesIO(p.json_bytes(preview(records())))
        with mock.patch.object(p.urllib.request, "urlopen", return_value=response) as fetch:
            self.assertEqual(p.prepare(self.root / "dataset")["questions"], 10)
        fetch.assert_called_once_with(p.SOURCE_URL, timeout=30)
        with mock.patch.object(p, "MAX_INPUT_BYTES", 10), mock.patch.object(
                p.urllib.request, "urlopen", return_value=io.BytesIO(b"x" * 11)):
            with self.assertRaisesRegex(ValueError, "exceeds"):
                p.prepare(self.root / "too-big")
        self.assertFalse((self.root / "too-big").exists())

    def test_existing_directory_and_symlink_are_never_overwritten(self):
        existing = self.root / "existing"
        existing.mkdir()
        note = existing / "note.txt"
        note.write_text("keep")
        link = self.root / "link"
        link.symlink_to(self.root / "missing", target_is_directory=True)
        with mock.patch.object(p.urllib.request, "urlopen") as fetch:
            for target in (existing, link):
                with self.subTest(target=target), self.assertRaises(FileExistsError):
                    p.prepare(target)
            fetch.assert_not_called()
        self.assertEqual(note.read_text(), "keep")

    def test_bundled_subset_rebuilds_exactly_and_satisfies_harness_limits(self):
        root = p.DEFAULT_OUTPUT
        snapshot = (root / "source.json").read_bytes()
        selected = p.select_questions(json.loads(snapshot))
        expected = p.build_files(selected)
        for name, content in expected.items():
            self.assertEqual((root / name).read_bytes(), content, name)
        manifest = json.loads(expected["manifest.json"])
        self.assertEqual(manifest["snapshot_sha256"], hashlib.sha256(snapshot).hexdigest())
        corpus = list((root / "knowledge").iterdir())
        self.assertEqual(len(corpus), 100)
        self.assertTrue(all(path.is_file() and path.suffix == ".txt" and path.stat().st_size <= 64 * 1024
                            for path in corpus))
        self.assertEqual(set(root.glob("knowledge/*.json")), set())

    def test_all_ten_bundled_questions_work_with_offline_harness_cli(self):
        repo = Path(__file__).resolve().parents[1]
        questions = json.loads((p.DEFAULT_OUTPUT / "questions.json").read_text())
        env = {"PATH": os.defpath, "PYTHONPATH": str(repo), "HARNESS_KNOWLEDGE": str(p.DEFAULT_OUTPUT / "knowledge"),
               "HARNESS_TRACE_CONTENT": "1"}  # Controlled corpus assertions require full evidence traces.
        for item in questions:
            with self.subTest(question_id=item["id"]):
                result = subprocess.run([sys.executable, "-m", "agents.teaching.cli", "rag-mock", item["question"]],
                                        cwd=self.root, env=env, text=True, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("#char-", result.stdout)
        trace = [json.loads(line) for line in (self.root / "trace.jsonl").read_text().splitlines()]
        results = [json.loads(row["output"]) for row in trace if row["kind"] == "tool"]
        self.assertEqual(len(results), 10)
        for evidence in results:
            self.assertEqual((evidence["searched_files"], evidence["skipped_files"]), (100, 0))
            self.assertEqual(evidence["status"], "ok")
        self.assertFalse((self.root / "session.json").exists())


if __name__ == "__main__":
    unittest.main()