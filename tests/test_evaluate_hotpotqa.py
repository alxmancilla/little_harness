"""Offline scoring, permissions, budgets and isolated baseline integration tests."""

import json
import os
import subprocess
import unittest
from dataclasses import replace
from unittest import mock

from tests.helpers import (HotpotQAEvaluationTestCase, HotpotQAWorkerTestCase, RecordingModel,
                           prediction, records, reply, tool_calls)
from evaluation.hotpotqa import evaluate as e, comparison as c, scoring as s
from agents.hotpotqa import single as a
from agents.teaching.demos import ScriptedModel
from agents.teaching import TeachingConfig, create_tools
from harness import ToolRegistry, make_model
from harness import agent as h
import urllib.request



class ScoringTests(unittest.TestCase):
    def test_normalization_token_overlap_and_categorical_answers(self):
        self.assertEqual(s.normalize_answer(" The NEW-YORK, City! "), "newyork city")
        self.assertEqual(s.answer_scores("The New York City.", "New York City")["em"], 1)
        self.assertAlmostEqual(s.answer_scores("New York", "New York City")["f1"], 0.8)
        self.assertEqual(s.answer_scores("yes indeed", "yes")["f1"], 0)
        self.assertEqual(s.answer_scores("noanswer", "the answer")["f1"], 0)
        self.assertEqual(s.answer_scores("YES", "yes")["f1"], 1)
        self.assertEqual(s.answer_scores("", "")["f1"], 0)  # matches upstream, even for two empty strings

    def test_evidence_is_case_sensitive_set_based_and_joint_matches_upstream_formula(self):
        record = {"answer": "New York City", "supporting_facts": [["A", 0], ["B", 1]],
                  "context": [["A", ["one"]], ["B", ["two", "three"]]]}
        predicted = prediction("New York", [["A", 0], ["A", 0], ["C", 0]])
        result = s.score_prediction(predicted, record)
        self.assertEqual(result["sp_f1"], 0.5)
        self.assertAlmostEqual(result["joint_f1"], 0.4)
        self.assertEqual(result["joint_em"], 0)
        perfect = s.score_prediction(prediction(record["answer"], record["supporting_facts"]), record)
        self.assertTrue(all(value == 1 for value in perfect.values()))
        self.assertEqual(s.supporting_scores([["a", 0]], [["A", 0]])["f1"], 0)

    def test_invalid_original_gold_excludes_only_evidence_and_joint_metrics(self):
        record = records()[0]
        record["supporting_facts"][0][1] = 902
        result = s.score_prediction(prediction(record["answer"]), record)
        self.assertEqual(result["em"], 1)
        self.assertIsNone(result["sp_f1"])
        self.assertIsNone(result["joint_f1"])

    def test_prediction_schema_is_strict_without_silently_repairing_prose(self):
        good = prediction("yes", [["Title", 0]])
        self.assertEqual(s.parse_prediction(json.dumps(good)), good)
        self.assertEqual(s.parse_prediction(json.dumps(e.EMPTY_PREDICTION)), e.EMPTY_PREDICTION)
        invalid = ["Answer: yes", "[]", json.dumps({"answer": "yes"}),
                   json.dumps(prediction("", abstain=False)),
                   json.dumps(prediction("yes", abstain=True)),
                   json.dumps(prediction("yes", [["Title", True]])),
                   json.dumps(prediction("yes", [["Title", -1]]))]
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                s.parse_prediction(text)


class ModelConfigTests(unittest.TestCase):
    def test_metadata_records_endpoint_fingerprint_and_generation_without_url_credentials(self):
        endpoint = "https://public-placeholder@model.example.invalid/private-route?auth=public-placeholder#fragment"
        model = make_model({"HARNESS_PROVIDER": "openai", "HARNESS_MODEL": "review-model",
                              "HARNESS_BASE_URL": endpoint})
        model.max_tokens = 512
        config = a.model_config(model)
        self.assertTrue(c.known_model_config(config))
        self.assertEqual(config["generation"], {"max_output_tokens": 512, "sampling": "provider-defaults"})
        self.assertEqual(len(config["endpoint_sha256"]), 64)
        for value in (endpoint, "public-placeholder", "private-route", "fragment"):
            self.assertNotIn(value, json.dumps(config))
        model.base_url += "/changed"
        self.assertNotEqual(config["endpoint_sha256"], a.model_config(model)["endpoint_sha256"])

    def test_redacted_model_identity_is_not_treated_as_complete(self):
        model = make_model({"HARNESS_PROVIDER": "openai", "HARNESS_MODEL": "public-placeholder",
                              "OPENAI_API_KEY": "public-placeholder"})
        config = a.model_config(model)
        self.assertNotIn("public-placeholder", json.dumps(config))
        self.assertFalse(config["complete"])
        self.assertFalse(c.known_model_config(config))


class WorkerTests(HotpotQAWorkerTestCase):
    def run_worker(self, model, **kwargs):
        return a.run_agent(self.payload, model, self.trace, config=self.config, **kwargs)

    def test_core_allowlist_is_advertised_and_enforced_even_for_a_registered_tool(self):
        dangerous = mock.Mock(return_value="should not run")
        entry = {"fn": dangerous, "risky": False, "schema": {"name": "host_only"}}
        script = ScriptedModel([tool_calls(("c1", "host_only", {})), reply("done")])
        advertised = []

        def model(system, messages, tools, on_text=None):
            advertised.append(tools)
            return script(system, messages, tools, on_text)

        unrelated_registry = ToolRegistry(host_only=entry)
        with self.subTest(registered=list(unrelated_registry)):
            agent = h.Agent(model, "sys", tools={}, trace_path=self.trace)
            self.assertEqual(agent.run("go"), "done")
        dangerous.assert_not_called()
        self.assertEqual(advertised, [[], []])
        self.assertTrue(agent.messages[2]["content"][0]["is_error"])

    def test_agent_reads_actual_evidence_has_no_gold_and_reports_usage(self):
        facts = [["Observatory 0-0", 0]]
        first = tool_calls(("read1", "read_document", {"document": "q01-p01.txt"}))
        final = reply(json.dumps(prediction("desert", facts)))
        first["usage"] = {"input_tokens": 10, "output_tokens": 5}
        final["usage"] = {"input_tokens": 20, "output_tokens": 7}
        model = RecordingModel(first, final)
        result = self.run_worker(model)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["prediction"]["answer"], "desert")
        self.assertEqual(result["citation_validity"], 1)
        self.assertEqual(result["read_evidence_fraction"], 1)
        self.assertEqual(result["model_calls"], 2)
        self.assertEqual(result["tool_calls_executed"], 1)
        self.assertEqual(result["tokens"], {"input_tokens": 30, "output_tokens": 12, "complete": True})
        self.assertIsNone(result["cost_usd"])
        self.assertNotIn("GOLD-LABEL-ONLY", json.dumps(model.calls))
        self.assertIn(a.BASELINE_EVIDENCE_NOTICE, self.trace.read_text())

    def test_search_metadata_drives_sentence_reading_and_grounded_citations(self):
        verified = []

        def model(system, messages, tools, on_text=None):
            if isinstance(messages[-1]["content"], str):
                return tool_calls(("search", "search_documents", {"query": "observatory telescope"}))
            output = json.loads(messages[-1]["content"][0]["content"])
            if "results" in output:
                match = output["results"][0]
                self.assertIn(match["document"], self.payload["documents"])
                self.assertEqual(match["sentence_count"], 2)
                self.assertEqual(match["title"], self.payload["documents"][match["document"]]["title"])
                return tool_calls(("read", "read_document", {"document": match["document"],
                                                           "start_sentence": 0, "limit": 1}))
            verified.extend([[output["title"], sentence["sentence"]] for sentence in output["sentences"]])
            return reply(json.dumps(prediction("desert", verified)))

        result = self.run_worker(model)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["model_calls"], 3)
        self.assertEqual(result["tool_calls_executed"], 2)
        self.assertEqual(result["tool_errors"], 0)
        self.assertEqual(result["read_evidence_fraction"], 1)
        self.assertEqual(result["prediction"]["supporting_facts"], verified)

    def test_tool_schemas_explain_filename_and_sentence_bounds_without_changing_other_registries(self):
        teaching = create_tools(TeachingConfig(self.config.workspace, self.config.knowledge))
        original = json.dumps(teaching["search_documents"]["schema"])
        tools = a.make_tools(self.payload["documents"], set(), config=self.config)
        schema = tools["read_document"]["schema"]["input_schema"]
        self.assertEqual(schema["properties"]["document"]["enum"], sorted(self.payload["documents"]))
        self.assertEqual(schema["properties"]["start_sentence"]["minimum"], 0)
        self.assertEqual(schema["properties"]["limit"]["minimum"], 1)
        self.assertEqual(schema["properties"]["limit"]["maximum"], 3)
        self.assertFalse(schema["additionalProperties"])
        output = json.loads(tools["search_documents"]["fn"]("observatory"))
        self.assertEqual(output["notice"], a.BASELINE_EVIDENCE_NOTICE)
        self.assertNotIn("[source]", json.dumps(tools["search_documents"]["schema"]) + output["notice"])
        self.assertEqual(json.dumps(teaching["search_documents"]["schema"]), original)
        self.assertEqual(json.loads(tools["search_documents"]["fn"]("zzzzzzzz"))["status"], "no_matches")

    def test_search_metadata_preserves_json_and_ranking_at_tool_output_limit(self):
        original = json.loads(a.make_tools(self.payload["documents"], set(), config=self.config)
                              ["search_documents"]["fn"]("observatory"))
        budget = len(json.dumps(original, ensure_ascii=False)) - 100
        tools = a.make_tools(self.payload["documents"], set(), config=replace(self.config, max_tool_output=budget),
                             search_fn=lambda *_: json.dumps(original))
        encoded = tools["search_documents"]["fn"]("observatory")
        output = json.loads(encoded)
        self.assertLessEqual(len(encoded), budget)
        self.assertGreater(len(output["results"]), 0)
        self.assertLess(len(output["results"]), len(original["results"]))
        self.assertEqual(output["status"], "ok")
        self.assertEqual([row["source"] for row in output["results"]],
                         [row["source"] for row in original["results"][:len(output["results"])]])

    def test_reader_errors_support_recovery_without_extra_calls_or_erasing_errors(self):
        script = RecordingModel(
            tool_calls(("title", "read_document", {"document": "Observatory 0-0"})),
            tool_calls(("range", "read_document", {"document": "q01-p01.txt", "start_sentence": 2})),
            tool_calls(("fixed", "read_document", {"document": "q01-p01.txt", "start_sentence": 0, "limit": 1})),
            reply(json.dumps(prediction("desert", [["Observatory 0-0", 0]]))))
        result = self.run_worker(script)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["model_calls"], 4)
        self.assertEqual(result["tool_calls_executed"], 3)
        self.assertEqual(result["tool_errors"], 2)
        self.assertEqual(result["read_evidence_fraction"], 1)
        errors = [event["output"] for event in map(json.loads, self.trace.read_text().splitlines())
                  if event["kind"] == "tool" and event["is_error"]]
        self.assertIn("argument document must be one of: q01-p01.txt", errors[0])
        self.assertIn("sentence_count=2", errors[1])
        self.assertIn("0 <= start_sentence < 2", errors[1])

    def test_tools_and_execution_budget_are_enforced_for_parallel_requests(self):
        script = RecordingModel(tool_calls(("a", "calculator", {"expression": "2+2"}),
                                           ("b", "calculator", {"expression": "3+3"}),
                                           ("c", "http_get", {"url": "https://example.com"})),
                                reply(json.dumps(e.EMPTY_PREDICTION)))
        with mock.patch.object(urllib.request, "urlopen", side_effect=AssertionError("no network")):
            result = self.run_worker(script, max_tool_calls=1)
        self.assertEqual(result["tool_calls_requested"], 3)
        self.assertEqual(result["tool_calls_executed"], 1)
        self.assertEqual(result["tool_calls_rejected"], 2)
        self.assertIsNone(result["tokens"]["input_tokens"])
        self.assertFalse(result["tokens"]["complete"])

    def test_reader_cannot_escape_candidates_and_keeps_exact_sentence_ids(self):
        observed = set()
        tools = a.make_tools(self.payload["documents"], observed)
        self.assertEqual(set(tools), {"search_documents", "read_document", "calculator"})
        reader = tools["read_document"]["fn"]
        value = json.loads(reader("q01-p01.txt", 1, 1))
        self.assertEqual(value["title"], "Observatory 0-0")
        self.assertEqual(value["sentences"][0]["sentence"], 1)
        self.assertEqual(observed, {("Observatory 0-0", 1)})
        for args in (("../../source.json",), ("q01-p01.txt", 99), ("q01-p01.txt", 0, True)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                reader(*args)

    def test_reader_rejects_titles_source_ids_and_invalid_bounds_without_observing_evidence(self):
        observed = set()
        reader = a.make_tools(self.payload["documents"], observed)["read_document"]["fn"]
        invalid = [("Observatory 0-0",), ("q01-p01.txt#char-0",), (["q01-p01.txt"],),
                   ("q01-p01.txt", -1), ("q01-p01.txt", True), ("q01-p01.txt", 0, 4)]
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(ValueError):
                reader(*args)
            self.assertEqual(observed, set())

    def test_failures_and_step_exhaustion_do_not_become_valid_predictions(self):
        model = RecordingModel(tool_calls(("read", "read_document", {"document": "q01-p01.txt"})))
        result = self.run_worker(model, max_steps=1)
        self.assertEqual(result["status"], "max_steps")
        self.assertIsNone(result["prediction"])
        self.assertEqual(result["model_calls"], 1)

    def test_terminal_failure_after_truncated_tool_response_keeps_its_actual_status(self):
        for failure, expected in ((KeyboardInterrupt(), "interrupted"), (h.ModelError("failure"), "model_error")):
            with self.subTest(expected=expected):
                truncated = tool_calls(("cut", "calculator", {}))
                truncated.update(stop_reason="max_tokens", truncated_tool_ids=["cut"])
                model = RecordingModel(truncated, failure)
                result = a.run_agent(self.payload, model, self.root / f"{expected}.jsonl")
                self.assertEqual(result["status"], expected)
                self.assertEqual(result["model_calls"], 2)
                self.assertIsNone(result["prediction"])

    def test_invalid_argument_shape_consumes_one_attempt_and_can_recover(self):
        result = self.run_worker(RecordingModel(tool_calls(("bad", "read_document", [])),
                                                reply(json.dumps(e.EMPTY_PREDICTION))))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["tool_calls_executed"], 1)
        self.assertEqual(result["tool_errors"], 1)

    def test_provider_error_body_is_not_saved_and_failed_call_is_counted(self):
        result = self.run_worker(RecordingModel(h.ModelError("private-provider-detail")))
        self.assertEqual(result["status"], "model_error")
        self.assertEqual(result["model_calls"], 1)
        self.assertNotIn("private-provider-detail", self.trace.read_text() + json.dumps(result))

    def test_malformed_final_answer_is_a_visible_failure(self):
        result = self.run_worker(RecordingModel(reply("The answer is desert.")))
        self.assertEqual(result["status"], "invalid_output")
        self.assertIsNone(result["prediction"])

    def test_wrapped_valid_json_is_not_repaired_or_retried(self):
        valid = json.dumps(prediction("desert", [["Observatory 0-0", 0]]))
        fence = "`" * 3
        for index, raw in enumerate((f"{fence}json\n{valid}\n{fence}", f"Here is the answer:\n{valid}")):
            model = RecordingModel(reply(raw))
            result = a.run_agent(self.payload, model, self.root / f"wrapped-{index}.jsonl")
            self.assertEqual(result["status"], "invalid_output")
            self.assertIsNone(result["prediction"])
            self.assertEqual(result["raw_answer"], raw)
            self.assertEqual(result["model_calls"], 1)
            self.assertEqual(len(model.calls), 1)

    def test_invalid_or_unread_citations_are_not_silently_removed(self):
        facts = [["Observatory 0-0", 0], ["Invented title", 999]]
        result = self.run_worker(RecordingModel(reply(json.dumps(prediction("desert", facts)))))
        self.assertEqual(result["prediction"]["supporting_facts"], facts)
        self.assertEqual(result["citation_validity"], 0.5)
        self.assertEqual(result["read_evidence_fraction"], 0)


class BaselineIntegrationTests(HotpotQAEvaluationTestCase):
    def test_ten_question_smoke_is_isolated_frozen_and_not_an_accuracy_result(self):
        (self.root / ".env").write_text("HARNESS_KNOWLEDGE=wrong\nHARNESS_WORKSPACE=wrong-workspace\n")
        with mock.patch.dict(os.environ, {"PATH": os.defpath}, clear=True):
            summary = self.baseline()
        self.assertIsNone(summary["scores"])
        self.assertIn("not-an-accuracy-baseline", summary["mode"])
        self.assertEqual(summary["statuses"], {"completed": 10})
        self.assertEqual(summary["model_calls"], 30)
        self.assertEqual(summary["tool_calls_executed"], 20)
        output = self.root / "evaluation"
        results = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
        self.assertTrue(all(row["scores"] is None and row["prediction"]["abstain"] for row in results))
        inputs = (output / "inputs.json").read_bytes()
        self.assertNotIn(b"GOLD-LABEL-ONLY", inputs)
        self.assertNotIn(b"supporting_facts", inputs)
        self.assertEqual(summary["input_fingerprint"], e.digest(inputs))
        self.assertEqual([row["id"] for row in json.loads(inputs)], [f"id-{i}" for i in range(10)])
        for folder in sorted(output.glob("q*")):
            events = [json.loads(line) for line in (folder / "trace.jsonl").read_text().splitlines()]
            searched = json.loads(next(row["output"] for row in events if row.get("name") == "search_documents"))
            self.assertEqual((searched["searched_files"], searched["skipped_files"]), (10, 0))
        self.assertFalse((self.root / "session.json").exists())
        self.assertFalse((self.root / "wrong-workspace").exists())
        self.assertEqual(set(json.loads((output / "predictions.json").read_text())), {"answer", "sp"})

    def test_live_scoring_counts_errors_in_denominator_without_calling_a_real_model(self):
        rows = records()

        def worker(payload, files, output, **kwargs):
            self.assertTrue(kwargs["live"])
            index = int(payload["id"].split("-")[1])
            result = e.failure_result(payload["id"], "completed" if index < 3 else "invalid_output")
            if index < 3:
                result["prediction"] = prediction(rows[index]["answer"], rows[index]["supporting_facts"])
            return result

        with mock.patch.object(e, "run_worker", side_effect=worker):
            summary = self.baseline(live=True)
        self.assertEqual(summary["scores"]["em"], 0.3)
        self.assertEqual(summary["scores"]["joint_f1"], 0.3)
        self.assertEqual(summary["score_denominators"]["em"], 10)
        self.assertEqual(summary["statuses"], {"completed": 3, "invalid_output": 7})
        self.assertIsNone(summary["model_calls"])  # unknown is not falsely recorded as zero

    def test_interrupt_saves_partial_report_and_does_not_start_other_questions(self):
        with mock.patch.object(e, "run_worker", side_effect=KeyboardInterrupt) as worker:
            summary = self.baseline(live=True)
        self.assertTrue(summary["interrupted"])
        self.assertEqual(worker.call_count, 1)
        self.assertEqual(summary["statuses"], {"interrupted": 1, "not_run": 9})
        self.assertEqual(summary["scores"]["em"], 0)
        self.assertTrue((self.root / "evaluation" / "summary.json").exists())

    def test_worker_interruption_after_truncation_stops_remaining_questions(self):
        def worker(payload, files, output, **kwargs):
            truncated = tool_calls(("cut", "calculator", {}))
            truncated.update(stop_reason="max_tokens", truncated_tool_ids=["cut"])
            return a.run_agent(payload, RecordingModel(truncated, KeyboardInterrupt()), output / "trace.jsonl")

        with mock.patch.object(e, "run_worker", side_effect=worker) as run:
            summary = self.baseline(live=True)
        self.assertEqual(run.call_count, 1)
        self.assertTrue(summary["interrupted"])
        self.assertEqual(summary["statuses"], {"interrupted": 1, "not_run": 9})

    def test_timeout_is_reported_with_unknown_counts(self):
        payload, files = e.agent_input(1, records()[0])
        output = self.root / "one"
        output.mkdir()
        with mock.patch.object(e.subprocess, "run", side_effect=subprocess.TimeoutExpired("worker", 1)):
            result = e.run_worker(payload, files, output, live=False, max_steps=12, max_tool_calls=12,
                                  max_output_tokens=2048, timeout=1)
        self.assertEqual(result["status"], "timeout")
        self.assertIsNone(result["model_calls"])

    def test_output_refusals_and_limit_checks_happen_before_inference(self):
        output = self.root / "existing"
        output.mkdir()
        link = self.root / "link"
        link.symlink_to(self.root / "absent", target_is_directory=True)
        with mock.patch.object(e, "run_worker") as worker:
            for destination in (output, link):
                with self.subTest(destination=destination), self.assertRaises(FileExistsError):
                    e.run_baseline(destination, dataset=self.dataset)
            for limit in (0, True, 11):
                with self.subTest(limit=limit), self.assertRaises(ValueError):
                    self.baseline(limit=limit)
            worker.assert_not_called()


if __name__ == "__main__":
    unittest.main()