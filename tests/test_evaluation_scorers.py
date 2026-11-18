"""Shared scorer interface, trace scorers and offline run comparison."""

import json
import subprocess
import sys

from evaluation import compare as c, scorers as s
from evaluation.support import evaluate as e
from tests.helpers import REPO, TemporaryDirectoryTestCase


@s.scorer("always")
def always(**_):
    return True


@s.scorer()
def explained(value, **_):
    return value > 1, f"value was {value}"


@s.scorer()
def broken(**_):
    raise RuntimeError("secret detail")


TRACE = [{"kind": "task"}, {"kind": "model", "stop_reason": "tool_use"},
         {"kind": "tool_policy", "name": "write_file", "allowed": False, "reason": "denied by policy"},
         {"kind": "tool", "name": "search", "is_error": False},
         {"kind": "tool", "name": "read_file", "is_error": True},
         {"kind": "model", "stop_reason": "end_turn"}]


class ScorerTests(TemporaryDirectoryTestCase):
    def test_scorers_return_values_and_rationales_and_fail_closed_without_error_text(self):
        scores = s.run_scorers((always, explained, broken), value=1)
        self.assertEqual(s.values(scores), {"always": True, "explained": False, "broken": False})
        self.assertEqual(s.rationales(scores), {"explained": "value was 1", "broken": "scorer error: RuntimeError"})

    def test_non_boolean_results_are_rejected(self):
        with self.assertRaises(TypeError):
            s.run_scorers((s.scorer("bad")(lambda **_: 1),))

    def test_trace_scorers_read_only_metadata_events(self):
        scores = s.run_scorers(s.TRACE_SCORERS, trace=TRACE)
        self.assertEqual(s.values(scores), {"trace_present": True, "tools_before_answer": True,
                                            "no_tool_errors": False, "no_policy_denials": False,
                                            "ended_normally": True})
        self.assertIn("read_file", scores["no_tool_errors"]["rationale"])
        empty = s.values(s.run_scorers(s.TRACE_SCORERS, trace=s.read_trace(self.root / "missing.jsonl")))
        self.assertFalse(empty["trace_present"] or empty["tools_before_answer"])

    def test_support_rows_record_rationales_and_separate_trace_checks(self):
        summary = e.run_evaluation(self.root / "run")
        rows = [json.loads(line) for line in (self.root / "run/results.jsonl").read_text().splitlines()]
        self.assertEqual(summary["automated_passes"], 11)  # Trace checks never change pass counts.
        self.assertEqual(summary["trace_checks_passed"]["trace_present"], 17)
        for row in rows:
            self.assertTrue(row["trace_checks"]["trace_present"])
            if not row["checks"]["required_sources_cited"]:
                self.assertIn("missing:", row["rationales"]["required_sources_cited"])
        self.assertIn("## Why checks failed", (self.root / "run/report.md").read_text())

    def test_compare_reports_flips_and_never_writes_into_runs(self):
        e.run_evaluation(self.root / "a")
        e.run_evaluation(self.root / "b")
        rows = (self.root / "b/results.jsonl").read_text().splitlines()
        first = json.loads(rows[0])
        first["checks"]["valid_output"] = not first["checks"]["valid_output"]
        (self.root / "b/results.jsonl").write_text("\n".join([json.dumps(first), *rows[1:]]) + "\n")
        before = sorted(path for path in self.root.rglob("*"))
        result = c.compare(self.root / "a", self.root / "b")
        self.assertEqual(result["shared_cases"], 17)
        self.assertIn("valid_output", result["cases"][0]["check_flips"])
        self.assertNotEqual(result["cases"][0]["change"], "same")
        self.assertIn("| c01 |", c.markdown(result))
        self.assertIn("| c01 | yes |", c.trace_report(self.root / "a"))
        self.assertEqual(sorted(self.root.rglob("*")), before)

    def test_cli_refuses_to_overwrite_an_output_file(self):
        e.run_evaluation(self.root / "a")
        (self.root / "out.md").write_text("keep")
        child = subprocess.run([sys.executable, "-B", "-m", "evaluation.compare", "--traces",
                                str(self.root / "a"), "--output", str(self.root / "out.md")],
                               cwd=REPO, capture_output=True, text=True, timeout=30)
        self.assertEqual(child.returncode, 1)
        self.assertEqual((self.root / "out.md").read_text(), "keep")
