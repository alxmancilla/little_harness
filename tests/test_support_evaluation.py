"""Offline evaluation mechanics, label isolation and deliberately limited quality claims."""

import copy
import hashlib
import json
import sys
from unittest import mock

from examples import support_agent as app
from evaluation.support import dataset as data, evaluate as e, multiturn as mt
from harness import ModelConfigError, ModelError, RunLimits
from tests.helpers import Model, REPO, RecordingModel, TemporaryDirectoryTestCase, reply, tool_calls, used


class SupportEvaluationTests(TemporaryDirectoryTestCase):
    def run_eval(self, name="run", **options):
        return e.run_evaluation(self.root / name, **options)

    def rows(self, name="run"):
        return [json.loads(line) for line in (self.root / name / "results.jsonl").read_text().splitlines()]

    def test_dataset_is_a_small_explicitly_unreviewed_draft_covering_six_categories(self):
        dataset = data.load_cases()
        self.assertEqual(dataset["review_status"], "unreviewed-synthetic")
        self.assertEqual(len(dataset["cases"]), 15)
        self.assertEqual({case["category"] for case in dataset["cases"]},
                         {"supported", "unknown_product", "missing_evidence", "policy_boundary",
                          "misleading_instructions", "adversarial_tool_use"})
        self.assertTrue(all(set(case["input"]) == {"sku", "question"} for case in dataset["cases"]))

    def test_invalid_ids_duplicate_ids_label_types_and_input_label_leakage_are_rejected(self):
        original = data.load_cases()
        mutations = [lambda cases: cases[0].update(id="../escape"),
                     lambda cases: cases[1].update(id=cases[0]["id"]),
                     lambda cases: cases[0]["expected"].update(needs_human="false"),
                     lambda cases: cases[0]["expected"].update(required_sources=["private.md"]),
                     lambda cases: cases[0]["input"].update(expected="label"),
                     lambda cases: cases[0].update(extra_documents={"warranty.md": "replaced approved note"}),
                     lambda cases: cases[0].update(extra_documents={}),
                     lambda cases: cases[0].update(extra_documents={"forum.md": 3}),
                     lambda cases: cases[12]["expected"].update(required_sources=["customer-forum.md"])]
        for index, mutate in enumerate(mutations):
            dataset = copy.deepcopy(original)
            mutate(dataset["cases"])
            path = self.root / f"bad-{index}.json"
            path.write_text(json.dumps(dataset))
            with self.subTest(index=index), self.assertRaises(ValueError):
                data.load_cases(path)

    def test_offline_run_discloses_scripted_limitations_and_freezes_inputs_evidence_and_sources(self):
        with mock.patch.object(e, "make_model", side_effect=AssertionError("network forbidden")), \
                mock.patch.object(app, "save_draft", side_effect=AssertionError("saving forbidden")):
            summary = self.run_eval()
        self.assertEqual(summary["mode"], "offline-smoke-not-quality")
        self.assertEqual(summary["statuses"], {"completed": 15})
        self.assertEqual(summary["checks_passed"]["valid_output"], 15)
        self.assertEqual(summary["automated_passes"], 9)  # The script cannot reason about policy/semantic relevance.
        self.assertIsNone(summary["answer_quality"])
        self.assertIsNone(summary["usage"]["input_tokens"])
        self.assertEqual((summary["usage"]["model_calls"], summary["usage"]["tool_calls_attempted"]), (30, 30))
        rows = self.rows()
        self.assertEqual({row["id"] for row in rows if not all(row["checks"].values())},
                         {"c07", "c08", "c09", "c10", "c11", "c14"})
        output = self.root / "run"
        contract = json.loads((output / "run.json").read_text())
        for artifact, field in (("cases.json", "dataset_sha256"), ("evidence.json", "evidence_sha256")):
            self.assertEqual(hashlib.sha256((output / artifact).read_bytes()).hexdigest(), contract[field])
        self.assertTrue(all(hashlib.sha256((REPO / name).read_bytes()).hexdigest() == digest
                            for name, digest in contract["source_sha256"].items()))
        inputs = json.loads((output / "inputs.json").read_text())
        self.assertTrue(all(set(item) == {"sku", "question"} for item in inputs))
        review = json.loads((output / "review.json").read_text())
        self.assertEqual(len(review["cases"]), 15)
        self.assertIsNone(review["reviewer"])
        self.assertTrue(all(item["expectations_approved"] is None and item["factual_correctness"] is None
                            and item["action_safety"] is None for item in review["cases"]))
        self.assertIn("not an answer-quality or safety score", (output / "report.md").read_text())
        for trace in output.glob("c*/trace.jsonl"):
            events = [json.loads(line) for line in trace.read_text().splitlines()]
            self.assertTrue(all(not {"text", "content", "input", "output", "error", "summary"} & event.keys()
                                for event in events))

    def test_each_case_has_fresh_history_and_only_input_reaches_model(self):
        dataset = data.load_cases()
        for case in dataset["cases"]:
            case["review_guide"] = "HOST-ONLY-EXPECTATION-SENTINEL"
        seen = []

        class Spy(app.OfflineModel):
            def __call__(self, system, messages, tools, on_text=None):
                assert "HOST-ONLY-EXPECTATION-SENTINEL" not in json.dumps([system, messages, tools])
                if isinstance(messages[-1]["content"], str):
                    assert len(messages) == 1
                    seen.append(json.loads(messages[-1]["content"]))
                return super().__call__(system, messages, tools, on_text)

        with mock.patch.object(e, "load_cases", return_value=dataset):
            self.run_eval(model_factory=Spy)
        self.assertEqual(seen, [case["input"] for case in dataset["cases"]])

    def test_extra_documents_are_searchable_only_in_their_own_case(self):
        spies = []

        class Spy(app.OfflineModel):
            def __init__(self):
                self.saw_injection = False
                spies.append(self)

            def __call__(self, system, messages, tools, on_text=None):
                self.saw_injection |= "NOTE TO THE SUPPORT ASSISTANT" in json.dumps(messages)
                return super().__call__(system, messages, tools, on_text)

        dataset = data.load_cases()
        dataset["cases"] = [dataset["cases"][0], dataset["cases"][12]]  # Same question, different evidence.
        with mock.patch.object(e, "load_cases", return_value=dataset):
            self.run_eval(model_factory=Spy)
        self.assertEqual([spy.saw_injection for spy in spies], [False, True])
        self.assertIn("NOTE TO THE SUPPORT ASSISTANT", (self.root / "run/cases.json").read_text())
        self.assertNotIn("customer-forum.md", (self.root / "run/evidence.json").read_text())

    def test_invalid_output_and_unretrieved_citations_remain_failed_in_denominator(self):
        for index, answer in enumerate(("not JSON", json.dumps({"sku": "WIDGET", "recommendation": "review",
                                      "sources": ["warranty.md"], "needs_human": True}))):
            summary = self.run_eval(str(index), limit=1, model_factory=lambda: Model(reply(answer)))
            self.assertEqual(summary["cases"], 1)
            self.assertEqual(summary["statuses"], {"invalid_output": 1})
            self.assertEqual(summary["automated_passes"], 0)
            self.assertFalse(any(self.rows(str(index))[0]["checks"].values()))

    def test_valid_structure_does_not_assert_prose_correctness(self):
        dataset = data.load_cases()
        dataset["cases"] = [dataset["cases"][3]]  # Gadget should be 24 months, not 999.
        model = Model(used(tool_calls(("notes", "search_support", {"query": "warranty"}),
                                      ("product", "lookup_product", {"sku": "GADGET"}))),
                      used(reply(json.dumps({"sku": "GADGET", "recommendation": "Warranty lasts 999 months.",
                                             "sources": ["warranty.md"], "needs_human": False}))))
        with mock.patch.object(e, "load_cases", return_value=dataset):
            summary = self.run_eval(model_factory=lambda: model)
        self.assertEqual(summary["automated_passes"], 1)
        self.assertEqual((summary["usage"]["input_tokens"], summary["usage"]["output_tokens"]), (6, 4))
        self.assertIsNone(summary["answer_quality"])
        self.assertEqual(summary["human_review_pending"], 1)

    def test_budget_stop_is_not_valid_output(self):
        summary = self.run_eval(limit=1, limits=RunLimits(max_model_calls=1, max_tool_calls=1))
        self.assertEqual(summary["automated_passes"], 0)
        row = self.rows()[0]
        self.assertEqual(row["stop_reason"], "max_tool_calls")
        self.assertEqual((row["usage"]["model_calls"], row["usage"]["tool_calls_attempted"]), (1, 1))

    def test_model_error_stops_remaining_calls_and_never_records_diagnostic_text(self):
        factory = mock.Mock(side_effect=lambda: RecordingModel(ModelError("diagnostic-sentinel")))
        summary = self.run_eval(model_factory=factory)
        factory.assert_called_once_with()
        self.assertEqual(summary["statuses"], {"model_error": 1, "not_run": 14})
        self.assertEqual(summary["attempted"], 1)
        self.assertEqual(summary["usage"]["model_calls"], 1)
        self.assertIsNone(summary["usage"]["input_tokens"])
        for path in (self.root / "run").rglob("*"):
            if path.is_file():
                self.assertNotIn("diagnostic-sentinel", path.read_text())

    def test_interruption_preserves_report_and_stops_new_cases(self):
        factory = mock.Mock(side_effect=lambda: RecordingModel(KeyboardInterrupt()))
        summary = self.run_eval(model_factory=factory)
        self.assertTrue(summary["interrupted"])
        self.assertEqual(summary["statuses"], {"interrupted": 1, "not_run": 14})
        factory.assert_called_once_with()
        self.assertEqual(len(self.rows()), 15)
        self.assertTrue((self.root / "run/report.md").is_file())

    def test_interruption_between_agents_keeps_completed_results_and_unknown_usage_honest(self):
        factory = mock.Mock(side_effect=[app.OfflineModel(), KeyboardInterrupt()])
        summary = self.run_eval(model_factory=factory)
        self.assertEqual(summary["statuses"], {"completed": 1, "interrupted": 1, "not_run": 13})
        self.assertEqual(factory.call_count, 2)
        self.assertIsNotNone(self.rows()[0]["recommendation"])
        self.assertIsNone(self.rows()[1]["usage"])
        self.assertIsNone(summary["usage"]["model_calls"])
        self.assertTrue(summary["interrupted"])

    def test_model_configuration_is_hashed_without_persisting_raw_identity_or_credentials(self):
        class ConfiguredModel(app.OfflineModel):
            provider, model = "test-provider-sentinel", "test-model-sentinel"
            base_url, max_tokens = "https://test-endpoint.invalid", 2048
            key = "not-a-real-credential"

        self.run_eval(limit=1, model_factory=ConfiguredModel)
        before = self.rows()[0]["model_config_sha256"]
        self.assertEqual(before, e.model_fingerprint(ConfiguredModel()))
        ConfiguredModel.max_tokens = 1024
        self.assertNotEqual(before, e.model_fingerprint(ConfiguredModel()))
        for path in (self.root / "run").rglob("*"):
            if path.is_file():
                content = path.read_text()
                for sentinel in (ConfiguredModel.provider, ConfiguredModel.model,
                                 ConfiguredModel.base_url, ConfiguredModel.key):
                    self.assertNotIn(sentinel, content)

    def test_existing_directories_symlinks_and_bad_limits_are_rejected_before_model_setup(self):
        existing = self.root / "existing"
        existing.mkdir()
        note = existing / "keep.txt"
        note.write_text("unchanged")
        link = self.root / "link"
        link.symlink_to(self.root / "absent", target_is_directory=True)
        factory = mock.Mock(side_effect=AssertionError("model setup forbidden"))
        for target in (existing, link):
            with self.subTest(target=target), self.assertRaises(FileExistsError):
                e.run_evaluation(target, live=True, model_factory=factory)
        for limit in (0, True, 16):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.run_eval(limit=limit, model_factory=factory)
        factory.assert_not_called()
        self.assertEqual(note.read_text(), "unchanged")
        self.assertFalse((self.root / "run").exists())

    def test_live_model_configuration_exit_is_sanitized_before_output_creation(self):
        with mock.patch.object(e, "make_model", side_effect=ModelConfigError("config-sentinel")):
            with self.assertRaisesRegex(ValueError, "configuration") as error:
                self.run_eval(live=True)
        self.assertNotIn("config-sentinel", str(error.exception))
        self.assertFalse((self.root / "run").exists())

    def test_cli_is_offline_by_default_does_not_load_env_and_reports_draft_failures_honestly(self):
        (self.root / ".env").write_text("SUPPORT_EVAL_SENTINEL=must-not-load\n")
        child = self.child("""
            import os, sys
            from unittest.mock import patch
            from evaluation.support.evaluate import main
            assert 'SUPPORT_EVAL_SENTINEL' not in os.environ
            with patch('evaluation.support.evaluate.make_model', side_effect=AssertionError('no network')):
                sys.argv = ['evaluate', '--output', 'evaluation']
                status = main()
            assert 'SUPPORT_EVAL_SENTINEL' not in os.environ
            raise SystemExit(status)
        """)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertIn("offline-smoke-not-quality", child.stdout)
        self.assertIn("9/15 pass draft expectations", child.stdout)
        self.assertIn("human review is required", child.stdout)
        self.assertFalse((self.root / "support-trace.jsonl").exists())
        self.assertFalse((self.root / "session.json").exists())

    def test_live_cli_uses_same_factory_and_returns_failure_for_invalid_output(self):
        for index, model in enumerate((app.OfflineModel(), Model(reply("invalid")))):
            args = ["evaluate", "--live", "--limit", "1", "--output", str(self.root / str(index))]
            with mock.patch.object(sys, "argv", args), mock.patch.object(e, "make_model", return_value=model) as make, \
                    mock.patch("builtins.print"):
                self.assertEqual(e.main(), index)
                make.assert_called_once_with()
            summary = json.loads((self.root / str(index) / "summary.json").read_text())
            self.assertEqual(summary["mode"], "live")
            self.assertIsNone(summary["answer_quality"])


class MultiTurnEvaluationTests(TemporaryDirectoryTestCase):
    def rows(self, name="run"):
        return [json.loads(line) for line in (self.root / name / "results.jsonl").read_text().splitlines()]

    def test_scenarios_are_a_small_unreviewed_draft_and_bad_shapes_are_rejected(self):
        original = data.load_scenarios()
        self.assertEqual(original["review_status"], "unreviewed-synthetic")
        self.assertEqual(len(original["scenarios"]), 5)
        self.assertTrue(all(2 <= len(s["turns"]) <= 6 for s in original["scenarios"]))
        mutations = [lambda d: d.update(version="support-v0-draft"),
                     lambda d: d["scenarios"][0].update(id="c01"),
                     lambda d: d["scenarios"][1].update(id=d["scenarios"][0]["id"]),
                     lambda d: d["scenarios"][0].update(turns=d["scenarios"][0]["turns"][:1]),
                     lambda d: d["scenarios"][0]["turns"][0]["input"].update(expected="label"),
                     lambda d: d["scenarios"][0]["turns"][1]["expected"].update(required_sources=["private.md"])]
        for index, mutate in enumerate(mutations):
            dataset = copy.deepcopy(original)
            mutate(dataset)
            path = self.root / f"bad-{index}.json"
            path.write_text(json.dumps(dataset))
            with self.subTest(index=index), self.assertRaises(ValueError):
                data.load_scenarios(path)

    def test_offline_run_scores_every_turn_keeps_history_and_discloses_scripted_limits(self):
        dataset = data.load_scenarios()
        for scenario in dataset["scenarios"]:
            scenario["review_guide"] = "HOST-ONLY-EXPECTATION-SENTINEL"
        seen = []

        class Spy(app.OfflineModel):
            def __call__(self, system, messages, tools, on_text=None):
                assert "HOST-ONLY-EXPECTATION-SENTINEL" not in json.dumps([system, messages, tools])
                if isinstance(messages[-1]["content"], str):
                    seen.append(sum(m["role"] == "user" and isinstance(m["content"], str) for m in messages))
                return super().__call__(system, messages, tools, on_text)

        with mock.patch.object(mt, "load_scenarios", return_value=dataset), \
                mock.patch.object(mt, "make_model", side_effect=AssertionError("network forbidden")):
            summary = mt.run_evaluation(self.root / "run", model_factory=Spy)
        self.assertEqual(seen, [1, 2] * 5)  # Each later turn sees the earlier task in the same history.
        self.assertEqual((summary["scenarios"], summary["turns"], summary["history_retained"]), (5, 10, 5))
        self.assertEqual(summary["turn_checks_passed"]["valid_output"], 10)
        self.assertEqual(summary["scenarios_passed"], 3)  # The script cannot infer escalation from a follow-up.
        self.assertIsNone(summary["answer_quality"])
        rows = {row["id"]: row for row in self.rows()}
        self.assertEqual({key for key, row in rows.items() if not row["passed"]}, {"m04", "m05"})
        self.assertFalse(rows["m04"]["turns"][1]["checks"]["escalation_matches_draft"])
        self.assertEqual(rows["m02"]["turns"][1]["recommendation"]["sku"], "GADGET")
        output = self.root / "run"
        contract = json.loads((output / "run.json").read_text())
        for artifact, field in (("scenarios.json", "dataset_sha256"), ("evidence.json", "evidence_sha256")):
            self.assertEqual(hashlib.sha256((output / artifact).read_bytes()).hexdigest(), contract[field])
        review = json.loads((output / "review.json").read_text())
        self.assertTrue(all(item["context_carried_correctly"] is None for item in review["scenarios"]))
        self.assertIn("not an answer-quality or safety score", (output / "report.md").read_text())
        for trace in output.glob("m*/trace.jsonl"):
            events = [json.loads(line) for line in trace.read_text().splitlines()]
            self.assertTrue(all(not {"text", "content", "input", "output", "error"} & event.keys() for event in events))
        with self.assertRaises(FileExistsError):
            mt.run_evaluation(output)

    def test_incomplete_turn_stops_its_scenario_and_model_error_stops_the_run(self):
        budget = mt.run_evaluation(self.root / "budget", limits=RunLimits(max_tool_calls=0))
        self.assertEqual(budget["halted"], False)
        for row in self.rows("budget"):
            self.assertEqual([t["status"] for t in row["turns"]], ["stopped", "not_run"])
            self.assertFalse(row["passed"])
        summary = mt.run_evaluation(self.root / "error", model_factory=lambda: RecordingModel(
            ModelError("PRIVATE-DIAGNOSTIC-SENTINEL", "rate_limited")))
        self.assertTrue(summary["halted"])
        rows = self.rows("error")
        self.assertEqual([t["status"] for t in rows[0]["turns"]], ["model_error", "not_run"])
        self.assertTrue(all(t["status"] == "not_run" for row in rows[1:] for t in row["turns"]))
        artifacts = "".join(path.read_text() for path in (self.root / "error").rglob("*") if path.is_file())
        self.assertNotIn("PRIVATE-DIAGNOSTIC-SENTINEL", artifacts)
