"""Shared CLI regressions using local fake models, never live provider calls."""

import hashlib
import json
from types import SimpleNamespace
from unittest import mock

from agents.hotpotqa import multi, single, worker
from tests.helpers import TemporaryDirectoryTestCase


class WorkerMainTests(TemporaryDirectoryTestCase):
    def setUp(self):
        super().setUp()
        self.source = self.root / "input.json"
        self.payload = {"id": "question", "question": "Where?", "documents": {}}
        self.source.write_text(json.dumps(self.payload), encoding="utf-8")

    def arguments(self, output, trace):
        return ["--input", str(self.source), "--output", str(output), "--trace", str(trace)]

    def test_offline_defaults_metadata_and_exclusive_output_are_preserved(self):
        output, trace = self.root / "output.json", self.root / "trace.jsonl"
        runner = mock.Mock(side_effect=lambda *args, **kwargs: {"status": "completed"})
        smoke = mock.Mock()
        arguments = self.arguments(output, trace)
        environment = {"HARNESS_WORKSPACE": str(self.root / "workspace"),
                       "HARNESS_KNOWLEDGE": str(self.root / "knowledge")}
        with mock.patch.dict("os.environ", environment, clear=True), \
                mock.patch.object(worker, "load_env") as load, \
                mock.patch.object(worker.h, "make_model") as make_model, \
                mock.patch.object(worker, "print_model_config") as print_config:
            worker.worker_main("offline", runner, smoke, "prompt", argv=arguments)
            load.assert_called_once_with()
            make_model.assert_not_called()
            print_config.assert_not_called()
            smoke.assert_not_called()
            args, options = runner.call_args
            self.assertEqual(args, (self.payload, smoke, trace))
            self.assertEqual((options["max_steps"], options["max_tool_calls"]), (12, 12))
            self.assertEqual(options["config"].workspace, self.root / "workspace")
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["model_config"]["provider"], "offline-smoke")
            self.assertEqual(result["model_config"]["generation"]["max_output_tokens"], 2048)
            self.assertEqual(result["prompt_sha256"], hashlib.sha256(b"prompt").hexdigest())
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                worker.worker_main("offline", runner, smoke, "prompt", argv=arguments)
            self.assertEqual(output.read_bytes(), original)

    def test_both_live_entrypoints_forward_limits_and_record_effective_model_and_prompts(self):
        for module, runner_name, prompts in (
                (single, "run_agent", single.BASELINE_PROMPT),
                (multi, "run_agents", {"researcher": multi.RESEARCH_PROMPT, "reviewer": multi.REVIEW_PROMPT})):
            with self.subTest(module=module.__name__):
                output = self.root / (runner_name + ".json")
                trace = self.root / (runner_name + ".jsonl")
                model = SimpleNamespace(provider="fake-provider", model="fake-model", key="",
                                        base_url="http://localhost:8001/v1", max_tokens=1)
                arguments = self.arguments(output, trace) + ["--live", "--max-steps", "9",
                            "--max-tool-calls", "5", "--max-output-tokens", "321"]
                environment = {"HARNESS_WORKSPACE": str(self.root / "workspace"),
                               "HARNESS_KNOWLEDGE": str(self.root / "knowledge")}
                with mock.patch.dict("os.environ", environment, clear=True), \
                        mock.patch("sys.argv", ["worker", *arguments]), \
                        mock.patch.object(worker, "load_env") as load, \
                        mock.patch.object(worker.h, "make_model", return_value=model) as make_model, \
                        mock.patch.object(worker, "print_model_config") as print_config, \
                        mock.patch.object(module, runner_name, return_value={"status": "completed"}) as runner:
                    module.main()
                load.assert_called_once_with()
                make_model.assert_called_once_with()
                print_config.assert_called_once_with(model)
                self.assertEqual(model.max_tokens, 321)
                args, options = runner.call_args
                self.assertEqual(args, (self.payload, model, trace))
                self.assertEqual((options["max_steps"], options["max_tool_calls"]), (9, 5))
                self.assertEqual(options["config"].knowledge, self.root / "knowledge")
                result = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(result["model_config"], worker.model_config(model))
                expected = (hashlib.sha256(prompts.encode()).hexdigest() if isinstance(prompts, str) else
                            {role: hashlib.sha256(prompt.encode()).hexdigest() for role, prompt in prompts.items()})
                self.assertEqual(result["prompt_sha256"], expected)