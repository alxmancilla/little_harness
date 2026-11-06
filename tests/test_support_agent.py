"""Application behaviour, strict output contracts, provenance and draft boundaries."""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from unittest import mock

from examples import support_agent as app
from harness import ToolPolicy
from tests.helpers import Model, REPO, RecordingModel, TemporaryDirectoryTestCase, reply, tool_calls
from tests.storage_fakes import FakeJournal, FakeMemoryStore


class SupportAgentTests(TemporaryDirectoryTestCase):
    def agent(self, model=None, **options):
        options.setdefault("trace_path", self.root / "trace.jsonl")
        return app.create_agent(model or app.OfflineModel(), app.load_documents(), app.PRODUCTS, **options)

    def run_app(self, question="reconnect disconnected device", sku="WIDGET", model=None):
        agent = self.agent(model)
        result = agent.run_result(json.dumps({"sku": sku, "question": question}))
        return agent, result, app.parse_recommendation(result, agent.messages, sku)

    def test_two_read_only_tools_produce_valid_cited_output_and_private_metadata_trace(self):
        agent, result, value = self.run_app()
        self.assertEqual((result.status, result.usage.model_calls, result.usage.tool_calls_attempted),
                         ("completed", 2, 2))
        self.assertEqual(value["sources"], ["connectivity.md"])
        self.assertFalse(value["needs_human"])
        self.assertIn("pairing button", value["recommendation"])
        self.assertTrue(all(tool["read_only"] for tool in agent.tools.values()))
        self.assertNotIn("pairing button", agent.trace_path.read_text())
        self.assertNotIn("reconnect disconnected device", agent.trace_path.read_text())

    def test_optional_support_memory_recall_and_approved_remember(self):
        store = FakeMemoryStore()
        store.add("customer:1", "warranty prefers email", kind="preference", created_by="application")
        model = RecordingModel(tool_calls(("a", "recall", {"query": "warranty"}),
                                          ("b", "remember", {"text": "prefers email", "kind": "preference"})),
                               reply("done"))
        approval = mock.Mock(return_value=True)
        agent = self.agent(model, memory_store=store, memory_namespace="customer:1", tool_policy=ToolPolicy(),
                           approve=approval, journal=FakeJournal())
        self.assertEqual(agent.run_result("review the customer's preferences").status, "completed")
        recalled = json.loads(model.calls[1][-1]["content"][0]["content"])
        self.assertTrue(recalled["untrusted"])
        self.assertEqual(recalled["memories"][0]["text"], "warranty prefers email")
        approval.assert_called_once()
        self.assertEqual({record.created_by for record in store.list("customer:1")}, {"model", "application"})
        self.assertEqual({action["tool_name"] for action in agent.journal.inspect()}, {"recall", "remember"})

    def test_unknown_product_or_missing_evidence_escalates_without_inventing_a_source(self):
        for question, sku in (("reconnect", "UNKNOWN"), ("penguin migration", "WIDGET")):
            with self.subTest(question=question):
                _, _, value = self.run_app(question, sku)
                self.assertTrue(value["needs_human"])
                self.assertEqual(value["sources"], [])

    def test_product_data_affects_the_answer_and_factories_keep_independent_snapshots(self):
        documents = {"custom.md": "Warranty review requires a purchase receipt."}
        products = {"CUSTOM": {"name": "Custom product", "warranty_months": 7}}
        agent = app.create_agent(app.OfflineModel(), documents, products, trace_path=self.root / "custom.jsonl")
        documents["custom.md"] = "changed"
        products["CUSTOM"]["warranty_months"] = 99
        result = agent.run_result(json.dumps({"sku": "CUSTOM", "question": "warranty"}))
        value = app.parse_recommendation(result, agent.messages, "CUSTOM")
        self.assertIn("7 months", value["recommendation"])
        self.assertIn("purchase receipt", value["recommendation"])
        self.assertEqual(value["sources"], ["custom.md"])
        other = app.create_tools(documents, products)
        self.assertIsNot(agent.tools["lookup_product"], other["lookup_product"])
        self.assertEqual(json.loads(other["lookup_product"]["fn"]("CUSTOM"))["warranty_months"], 99)

    def test_registration_exposes_only_model_arguments_and_rejects_bad_application_data(self):
        tools = self.agent().tools
        self.assertEqual(set(tools["search_support"]["schema"]["input_schema"]["properties"]), {"query"})
        self.assertEqual(set(tools["lookup_product"]["schema"]["input_schema"]["properties"]), {"sku"})
        for documents, products in (({}, app.PRODUCTS), ({"../private.md": "text"}, app.PRODUCTS),
                                    ({"ok.md": "text"}, {"SKU": {"name": "x", "warranty_months": True}})):
            with self.subTest(documents=list(documents)), self.assertRaises(ValueError):
                app.create_tools(documents, products)

    def test_strict_output_validation_rejects_prose_wrong_fields_and_invalid_citations(self):
        agent, result, value = self.run_app()
        mutations = [{**value, "extra": "field"}, {**value, "sku": "OTHER"},
                     {**value, "sources": ["unretrieved.md"]}, {**value, "sources": []},
                     {**value, "sources": ["connectivity.md", "connectivity.md"]},
                     {**value, "needs_human": "false"}, {**value, "recommendation": " "}]
        answers = ["Some prose " + result.answer, "[1]", result.answer[:-1],
                   '{"sku":"WIDGET","sku":"OTHER"}', *[json.dumps(item) for item in mutations]]
        for answer in answers:
            with self.subTest(answer=answer[:30]), self.assertRaises(ValueError):
                app.parse_recommendation(replace(result, answer=answer), agent.messages, "WIDGET")
        for status in ("stopped", "max_tokens", "refusal", "model_error", "interrupted"):
            with self.subTest(status=status), self.assertRaises(ValueError):
                app.parse_recommendation(replace(result, status=status), agent.messages, "WIDGET")

    def test_previous_task_citations_cannot_satisfy_a_new_task(self):
        agent, result, _ = self.run_app()
        agent.messages += [{"role": "user", "content": "new task"},
                           {"role": "assistant", "content": [{"type": "text", "text": result.answer}]}]
        with self.assertRaisesRegex(ValueError, "not retrieved"):
            app.parse_recommendation(result, agent.messages, "WIDGET")

    def test_claimed_success_requires_a_real_product_lookup_not_only_a_citation(self):
        agent, result, _ = self.run_app()
        messages = json.loads(json.dumps(agent.messages))
        messages[2]["content"][1]["is_error"] = True
        with self.assertRaisesRegex(ValueError, "product data"):
            app.parse_recommendation(result, messages, "WIDGET")

    def test_bad_arguments_can_recover_but_budget_exhaustion_is_not_valid_output(self):
        model = Model(tool_calls(("bad", "lookup_product", {"sku": 3})),
                      tool_calls(("notes", "search_support", {"query": "reconnect"}),
                                 ("product", "lookup_product", {"sku": "WIDGET"})),
                      reply(json.dumps({"sku": "WIDGET", "recommendation": "Check power before pairing.",
                                        "sources": ["connectivity.md"], "needs_human": False})))
        _, result, value = self.run_app(model=model)
        self.assertEqual(result.usage.tool_errors, 1)
        self.assertFalse(value["needs_human"])
        from harness import RunLimits
        agent = self.agent(limits=RunLimits(max_tool_calls=0))
        result = agent.run_result(json.dumps({"sku": "WIDGET", "question": "reconnect"}))
        with self.assertRaisesRegex(ValueError, "complete"):
            app.parse_recommendation(result, agent.messages, "WIDGET")

    def test_one_repair_turn_returns_the_validation_error_and_keeps_this_tasks_evidence(self):
        good = {"sku": "WIDGET", "recommendation": "Check power before pairing.",
                "sources": ["connectivity.md"], "needs_human": False}
        lookup = tool_calls(("notes", "search_support", {"query": "reconnect"}),
                            ("product", "lookup_product", {"sku": "WIDGET"}))
        bad = reply(json.dumps({**good, "sources": ["unretrieved.md"]}))
        task = json.dumps({"sku": "WIDGET", "question": "reconnect"})
        agent = self.agent(Model(lookup, bad, reply(json.dumps(good))))
        result, value = app.run_validated(agent, task, "WIDGET")
        self.assertEqual((result.status, value), ("completed", good))
        repairs = [m["content"] for m in agent.messages if m["role"] == "user" and isinstance(m["content"], str)][1:]
        self.assertEqual(repairs, [app.REPAIR_REQUEST.format(error="recommendation cites evidence not "
                                                                    "retrieved in this task")])
        self.assertNotIn("unretrieved.md", repairs[0])
        with self.assertRaisesRegex(ValueError, "not retrieved"):
            app.parse_recommendation(result, agent.messages, "WIDGET")
        agent = self.agent(Model(lookup, bad, bad))
        with self.assertRaisesRegex(ValueError, "not retrieved"):
            app.run_validated(agent, task, "WIDGET")
        self.assertEqual(len(agent.messages), 6)
        from harness import RunLimits
        agent = self.agent(Model(), limits=RunLimits(max_model_calls=0))
        with self.assertRaisesRegex(ValueError, "complete"):
            app.run_validated(agent, task, "WIDGET")
        self.assertEqual(len(agent.messages), 1)

    def test_draft_saving_requires_approval_and_never_overwrites(self):
        _, _, value = self.run_app()
        path = self.root / "draft.json"
        denied = mock.Mock(return_value=False)
        self.assertFalse(app.save_draft(path, value, denied))
        self.assertFalse(path.exists())
        approved = mock.Mock(return_value=True)
        self.assertTrue(app.save_draft(path, value, approved))
        self.assertEqual(json.loads(path.read_text()), value)
        self.assertEqual(json.loads(approved.call_args.args[1]["draft"]), value)
        with self.assertRaises(FileExistsError):
            app.save_draft(path, {**value, "recommendation": "changed"}, approved)
        self.assertEqual(json.loads(path.read_text()), value)

    def test_draft_saving_requires_literal_true_not_a_truthy_approval_value(self):
        _, _, value = self.run_app()
        for index, decision in enumerate((False, None, 0, 1, "false", "yes", [], {"approved": True})):
            path = self.root / f"unapproved-{index}.json"
            approve = mock.Mock(return_value=decision)
            with self.subTest(decision=decision):
                self.assertFalse(app.save_draft(path, value, approve))
                self.assertFalse(path.exists())
                approve.assert_called_once()

    def test_default_cli_never_selects_live_model_or_loads_env_and_denies_unattended_saving(self):
        child = self.child("""
            import sys
            from unittest.mock import patch
            from examples.support_agent import main
            with patch('examples.support_agent.make_model', side_effect=AssertionError('network forbidden')), \
                    patch('builtins.input', side_effect=EOFError):
                sys.argv = ['support', '--save-draft', 'draft.json']
                raise SystemExit(main())
        """)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertIn("Draft not saved", child.stdout)
        self.assertFalse((self.root / "draft.json").exists())
        self.assertEqual({p.name for p in self.root.iterdir()}, {"support-trace.jsonl"})

    def test_live_mode_only_changes_the_injected_model_and_invalid_output_cannot_be_saved(self):
        with mock.patch.object(sys, "argv", ["support", "--live", "--trace", str(self.root / "live.jsonl")]), \
                mock.patch.object(app, "make_model", return_value=app.OfflineModel()) as make, \
                mock.patch("builtins.print"):
            self.assertEqual(app.main(), 0)
            make.assert_called_once_with()
        with mock.patch.object(sys, "argv", ["support", "--save-draft", str(self.root / "bad.json"),
                                            "--trace", str(self.root / "bad-trace.jsonl")]), \
                mock.patch.object(app, "OfflineModel",
                                  return_value=Model(reply("invalid output"), reply("still invalid"))), \
                mock.patch.object(app, "save_draft") as save, mock.patch("builtins.print"):
            self.assertEqual(app.main(), 1)
            save.assert_not_called()

    def test_application_runs_with_only_copied_runtime_example_and_approved_notes(self):
        copied = self.root / "copy"
        copied.mkdir()
        shutil.copytree(REPO / "harness", copied / "harness", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(REPO / "examples", copied / "examples", ignore=shutil.ignore_patterns("__pycache__"))
        child = subprocess.run([sys.executable, "-B", "-m", "examples.support_agent", "warranty review"],
                               cwd=copied, env={"PATH": os.defpath}, capture_output=True, text=True, timeout=10)
        self.assertEqual(child.returncode, 0, child.stderr)
        value = json.loads(child.stdout)["recommendation"]
        self.assertEqual(value["sources"], ["warranty.md"])
        self.assertFalse(value["needs_human"])
