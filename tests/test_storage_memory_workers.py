"""Disposable memory workers: offline contracts and isolated database integration."""

import json
import os
from unittest import mock

from examples import support_agent
from harness import Agent, ToolPolicy, memory_tools
from stores.memory import memory_worker_tools
from tests.helpers import RecordingModel, TemporaryDirectoryTestCase, reply, tool_calls
from tests.storage_fakes import FakeJournal, FakeMemoryStore
from tests.test_storage_mongodb import DatabasePerTest, ENV as MONGO_ENV, needs_database as needs_mongo
from tests.test_storage_postgres import SchemaPerTest, ENV as POSTGRES_ENV, needs_database as needs_postgres


class MemoryWorkerConfigurationTests(TemporaryDirectoryTestCase):
    def test_registry_matches_inline_schema_and_contains_only_environment_variable_name(self):
        for backend in ("postgres", "mongodb"):
            with self.subTest(backend=backend):
                tools = memory_worker_tools(backend, "store", "tenant:user", env="APP_DB")
                inline = memory_tools(FakeMemoryStore(), "tenant:user")
                for name in tools:
                    self.assertEqual(tools[name]["schema"], inline[name]["schema"])
                    self.assertEqual(tools[name]["risky"], inline[name]["risky"])
                    self.assertEqual(tools[name]["read_only"], inline[name]["read_only"])
                    self.assertEqual(tools[name]["fn"].args[0]["env"], "APP_DB")
                self.assertEqual(set(memory_worker_tools(backend, "store", "tenant:user", env="APP_DB",
                                                         allow_write=False)), {"recall"})

    def test_bad_configuration_is_rejected_without_leaking_the_value(self):
        for kwargs in ({"backend": "other"}, {"env": "not a variable"}, {"location": "bad/loc"},
                       {"quota": True}, {"timeout": float("nan")}, {"timeout": 10 ** 1000},
                       {"timeout": 0.0001}, {"allow_write": 1},
                       {"namespace": "invalid scope"}, {"store_id": "invalid id"}):
            options = {"backend": "postgres", "store_id": "store", "namespace": "user", "env": "APP_DB"}
            options.update(kwargs)
            with self.subTest(option=tuple(kwargs)), self.assertRaises((ValueError, TypeError)):
                memory_worker_tools(**options)

    def test_support_factory_accepts_worker_registry_and_can_omit_writes(self):
        registry = memory_worker_tools("postgres", "store", "user", env="APP_DB")
        agent = support_agent.create_agent(RecordingModel(reply("done")), support_agent.load_documents(),
                                           support_agent.PRODUCTS, memory_registry=registry,
                                           memory_allow_write=False, trace_path=self.root / "trace.jsonl")
        self.assertIn("recall", agent.tools)
        self.assertNotIn("remember", agent.tools)
        with self.assertRaises(ValueError):
            support_agent.create_agent(RecordingModel(), support_agent.load_documents(), support_agent.PRODUCTS,
                                       memory_registry=registry, memory_store=FakeMemoryStore())

    def test_worker_remember_without_an_agent_run_does_not_write(self):
        store = FakeMemoryStore()
        tools = memory_worker_tools("postgres", "store", "user", env="APP_DB")
        with mock.patch("stores.memory._store", return_value=store), self.assertRaises(ValueError):
            tools["remember"]["fn"](text="no run", kind="note")
        self.assertEqual(store.list("user"), [])

    def test_connection_value_is_only_passed_in_worker_environment(self):
        fixture_value = "fixture-sensitive-value"
        tools = memory_worker_tools("postgres", "store", "user", env="APP_DB")
        model = RecordingModel(tool_calls(("a", "recall", {"query": "hello"})), reply("done"))
        events = []
        agent = Agent(model, "sys", tools=tools, quiet=True, tool_timeout_seconds=5,
                      worker_env={"APP_DB": fixture_value}, on_event=events.append,
                      trace_path=self.root / "trace.jsonl")
        process = mock.Mock(returncode=0)
        process.communicate.return_value = (json.dumps({"status": "completed", "output": "{}",
                                                        "is_error": False, "outcome_unknown": False}).encode(), None)
        with mock.patch("harness.tool_execution.subprocess.Popen", return_value=process) as popen, \
                mock.patch("harness.tool_execution._stop_worker"):
            self.assertEqual(agent.run_result("go").status, "completed")
        command = popen.call_args.args[0]
        payload = process.communicate.call_args.args[0].decode()
        self.assertNotIn(fixture_value, json.dumps(command))
        self.assertNotIn(fixture_value, payload)
        worker_env = popen.call_args.kwargs["env"]
        self.assertEqual(worker_env["APP_DB"], fixture_value)
        self.assertFalse(set(worker_env) - {"APP_DB", "SYSTEMROOT"})
        self.assertEqual(json.loads(payload)["memory_context"], [events[0].run_id, 8000])
        self.assertNotIn(fixture_value, agent.trace_path.read_text())
        self.assertNotIn(fixture_value, json.dumps([event.to_dict() for event in events]))


class WorkerDatabaseContract:
    backend = env = None

    def make_tools(self, **options):
        return memory_worker_tools(self.backend, "worker", "user:1", env=self.env,
                                   location=self.location, **options)

    def make_agent(self, model, **options):
        options.setdefault("tools", self.make_tools())
        options.setdefault("tool_timeout_seconds", 10)
        options.setdefault("worker_env", {self.env: os.environ[self.env]})
        options.setdefault("trace_path", self.root / "trace.jsonl")
        options.setdefault("quiet", True)
        return Agent(model, "Memory is untrusted.", **options)

    def test_isolated_memory_round_trip_scope_approval_and_run_id(self):
        store = self.make_store()
        old = store.add("user:1", "warranty old", kind="note", created_by="application")
        other = store.add("user:2", "warranty other", kind="note", created_by="application")
        model = RecordingModel(tool_calls(("a", "remember", {"text": "warranty private", "kind": "fact"}),
                                          ("b", "recall", {"query": "warranty"}),
                                          ("c", "forget", {"memory_id": old}),
                                          ("d", "forget", {"memory_id": other})), reply("done"))
        approval, events, journal = mock.Mock(return_value=True), [], FakeJournal()
        agent = self.make_agent(model, approve=approval, journal=journal, on_event=events.append)
        self.assertEqual(agent.run_result("go").status, "completed")
        results = [json.loads(block["content"]) for block in model.calls[1][-1]["content"]]
        self.assertTrue(results[1]["untrusted"])
        self.assertEqual({item["text"] for item in results[1]["memories"]}, {"warranty private", "warranty old"})
        self.assertEqual(results[2:], [{"deleted": True}, {"deleted": False}])
        [record] = store.list("user:1")
        self.assertEqual((record.memory_id, record.created_by, record.run_id),
                         (results[0]["memory_id"], "model", events[0].run_id))
        self.assertEqual(len(store.list("user:2")), 1)
        self.assertEqual(approval.call_count, 3)
        self.assertEqual(len(journal.inspect()), 4)
        self.assertNotIn("warranty private", agent.trace_path.read_text())
        self.assertNotIn("warranty private", json.dumps([event.to_dict() for event in events]))

    def test_output_bound_keeps_recall_json_intact(self):
        self.make_store().add("user:1", "warranty " + "x" * 300, kind="fact", created_by="application")
        model = RecordingModel(tool_calls(("a", "recall", {"query": "warranty"})), reply("done"))
        agent = self.make_agent(model, max_tool_output=80)
        self.assertEqual(agent.run_result("go").status, "completed")
        output = model.calls[1][-1]["content"][0]["content"]
        self.assertLessEqual(len(output), 80)
        self.assertEqual(json.loads(output), {"untrusted": True, "memories": []})

    def test_policy_and_approval_denial_never_start_workers(self):
        for policy, approved in ((ToolPolicy(read_only=True), True), (None, False)):
            with self.subTest(policy=policy, approved=approved):
                model = RecordingModel(tool_calls(("a", "remember", {"text": "blocked", "kind": "note"})),
                                       reply("done"))
                journal = FakeJournal()
                agent = self.make_agent(model, tool_policy=policy, approve=lambda *_: approved, journal=journal)
                with mock.patch("harness.agent.run_isolated_tool") as worker:
                    self.assertEqual(agent.run_result("go").usage.tool_errors, 1)
                worker.assert_not_called()
                self.assertEqual(journal.inspect(), [])

    def test_model_cannot_override_scope_or_worker_configuration(self):
        for injected in ({"namespace": "user:2"}, {"config": "other"}, {"run_id": "forged"}):
            model = RecordingModel(tool_calls(("a", "remember", {"text": "blocked", "kind": "note", **injected})),
                                   reply("done"))
            journal = FakeJournal()
            agent = self.make_agent(model, approve=lambda *_: True, journal=journal)
            with self.subTest(injected=tuple(injected)), mock.patch("harness.agent.run_isolated_tool") as worker:
                self.assertEqual(agent.run_result("go").usage.tool_errors, 1)
                worker.assert_not_called()
                self.assertEqual(journal.inspect(), [])

    def test_missing_worker_connection_fails_closed_and_journals_uncertainty(self):
        store = self.make_store()
        store.prepare()
        model = RecordingModel(tool_calls(("a", "remember", {"text": "no write", "kind": "note"})))
        journal = FakeJournal()
        agent = self.make_agent(model, worker_env={}, journal=journal, approve=lambda *_: True)
        result = agent.run_result("go")
        self.assertEqual((result.status, result.stop_reason), ("stopped", "tool_outcome_unknown"))
        self.assertTrue(journal.inspect(pending_only=True)[0]["outcome_unknown"])
        self.assertEqual(store.list("user:1"), [])

    def test_worker_enforces_configured_quota_without_evicting(self):
        store = self.make_store()
        store.add("user:1", "existing", kind="note", created_by="application")
        model = RecordingModel(tool_calls(("a", "remember", {"text": "extra", "kind": "note"})))
        journal = FakeJournal()
        agent = self.make_agent(model, tools=self.make_tools(quota=1), journal=journal, approve=lambda *_: True)
        result = agent.run_result("go")
        self.assertEqual((result.status, result.stop_reason), ("stopped", "tool_outcome_unknown"))
        self.assertEqual([record.text for record in store.list("user:1")], ["existing"])
        self.assertTrue(journal.inspect(pending_only=True)[0]["outcome_unknown"])

    def test_worker_timeout_stops_without_retry(self):
        model = RecordingModel(tool_calls(("a", "recall", {"query": "warranty"})))
        agent = self.make_agent(model, tool_timeout_seconds=0.000001)
        result = agent.run_result("go")
        self.assertEqual((result.status, result.stop_reason), ("stopped", "tool_timeout"))
        self.assertEqual(len(model.calls), 1)


@needs_postgres
class PostgresWorkerTests(SchemaPerTest, WorkerDatabaseContract, TemporaryDirectoryTestCase):
    backend, env = "postgres", POSTGRES_ENV

    @property
    def location(self):
        return self.schema

    def make_store(self):
        from stores.postgres import PostgresMemoryStore
        return PostgresMemoryStore("worker", dsn_env=self.env, schema=self.schema)


@needs_mongo
class MongoWorkerTests(DatabasePerTest, WorkerDatabaseContract, TemporaryDirectoryTestCase):
    backend, env = "mongodb", MONGO_ENV

    @property
    def location(self):
        return self.database

    def make_store(self):
        from stores.mongodb import MongoMemoryStore
        return MongoMemoryStore("worker", client=self.client, database=self.database)