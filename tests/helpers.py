"""Reusable offline fixtures; importing helpers does not initialize applications."""

import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock


REPO = Path(__file__).resolve().parents[1]


def sse_lines(*payloads):
    """Raw SSE byte lines, one data event per payload (dicts are JSON-encoded)."""
    lines = []
    for payload in payloads:
        lines += [f"data: {payload if isinstance(payload, str) else json.dumps(payload)}\n".encode(), b"\n"]
    return lines


def reply(text, stop="end_turn"):
    return {"stop_reason": stop, "content": [{"type": "text", "text": text}]}


def tool_calls(*calls):
    return {"stop_reason": "tool_use", "content": [
        {"type": "tool_use", "id": cid, "name": name, "input": args} for cid, name, args in calls]}


def used(response, input_tokens=3, output_tokens=2):
    return dict(response, usage={"input_tokens": input_tokens, "output_tokens": output_tokens})


def usage_reply(text):
    return used(reply(text), 1, 1)


def call(name, **arguments):
    return used(tool_calls(("one", name, arguments)), 1, 1)


class RecordingModel:
    """Return scripted responses (or raise them), recording detached message snapshots."""

    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def __call__(self, system, messages, tools, on_text=None):
        self.calls.append(json.loads(json.dumps(messages)))
        response = self.responses.pop(0) if self.responses else reply("done")
        if isinstance(response, BaseException):
            raise response
        return response


class Model:
    """Minimal strict script recording detached advertised schemas, not messages."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.schemas = []

    def __call__(self, system, messages, tools, on_text=None):
        self.schemas.append(json.loads(json.dumps(tools)))
        return self.responses.pop(0)


def package_echo(value: str):
    """Importable unrelated tool: its worker must not load application modules."""
    if "mini_harness" in sys.modules:
        raise RuntimeError("worker imported the legacy application")
    return value


def compaction_history(pairs=5):
    messages = [{"role": "user", "content": "task"}]
    for index in range(pairs):
        messages.append({"role": "assistant", "content": [
            {"type": "text", "text": f"step {index}"},
            {"type": "tool_use", "id": f"t{index}", "name": "read_file", "input": {"path": f"f{index}"}}]})
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"t{index}", "content": "r" * 300, "is_error": False}]})
    return messages


def records():
    """Fresh synthetic HotpotQA records with deliberately recognizable gold labels."""
    return [{"_id": f"id-{i}", "question": f"Where is observatory {i}?", "answer": "GOLD-LABEL-ONLY",
             "type": "bridge" if i % 2 else "comparison", "level": "hard",
             "context": [[f"Observatory {i}-{j}", [f"Telescope {i}-{j} is in the desert.", "Open Tuesday."]]
                         for j in range(10)],
             "supporting_facts": [[f"Observatory {i}-0", 0], [f"Observatory {i}-1", 1]]}
            for i in range(14)]


def prediction(answer="desert", facts=None, abstain=False):
    return {"answer": answer, "supporting_facts": facts or [], "abstain": abstain}


class TemporaryDirectoryTestCase(unittest.TestCase):
    """Fresh temporary root and credential-free subprocesses; contains no tests."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="harness-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def child(self, code, pythonpath=REPO, *args):
        return subprocess.run([sys.executable, "-B", "-c", textwrap.dedent(code), *args],
                              cwd=self.root, env={"PATH": os.defpath, "PYTHONPATH": str(pythonpath)},
                              capture_output=True, text=True, timeout=20)


class HarnessTestCase(TemporaryDirectoryTestCase):
    """Explicit per-test teaching tools on the unmodified public runtime."""

    def setUp(self):
        from agents.teaching import TeachingConfig, create_tools

        super().setUp()
        self.dir = self.root
        self.trace = self.dir / "trace.jsonl"
        self.config = TeachingConfig(self.root / "workspace", self.root / "knowledge")
        self.tools = create_tools(self.config)
        patcher = mock.patch("sys.stdout", new_callable=io.StringIO)
        self.stdout = patcher.start()
        self.addCleanup(patcher.stop)

    def agent(self, model, **options):
        from agents.teaching.demos import ScriptedModel
        from harness import Agent

        if isinstance(model, list):
            model = ScriptedModel(model)
        options.setdefault("approve", lambda *_: True)
        options.setdefault("tools", self.tools)
        options.setdefault("writable_roots", (self.config.workspace,))
        options.setdefault("trace_path", self.trace)
        options.setdefault("trace_include_content", True)  # Legacy full-trace assertions opt in explicitly.
        return Agent(model, "sys", **options)

    def trace_records(self):
        return [json.loads(line) for line in self.trace.read_text().splitlines()]

    def register_tool(self, fn, risky=False):
        self.tools.tool(risky=risky)(fn)


class HotpotQAWorkerTestCase(TemporaryDirectoryTestCase):
    def setUp(self):
        from agents.hotpotqa import HotpotQAConfig
        from evaluation.hotpotqa.evaluate import agent_input

        super().setUp()
        self.trace = self.root / "trace.jsonl"
        self.payload, files = agent_input(1, records()[0])
        (self.root / "knowledge").mkdir()
        for name, content in files.items():
            (self.root / name).write_bytes(content)
        self.config = HotpotQAConfig(workspace=self.root / "workspace", knowledge=self.root / "knowledge")
        patcher = mock.patch("sys.stdout", new_callable=io.StringIO)
        patcher.start()
        self.addCleanup(patcher.stop)


class HotpotQAEvaluationTestCase(TemporaryDirectoryTestCase):
    def setUp(self):
        from evaluation.hotpotqa import dev, evaluate, prepare

        super().setUp()
        previous = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)
        rows = records()
        source = self.root / "source.json"
        source.write_bytes(prepare.json_bytes(rows))
        self.demo = self.root / "demo"
        self.dataset = self.root / "development"
        prepare.prepare(self.demo, source)
        patcher = mock.patch.object(dev, "DEV_COUNT", len(rows))
        patcher.start()
        self.addCleanup(patcher.stop)
        dev.prepare_dev(self.dataset, source)
        patcher = mock.patch.object(evaluate, "TEACHING_SUBSET", self.demo)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch("sys.stdout", new_callable=io.StringIO)
        patcher.start()
        self.addCleanup(patcher.stop)

    def baseline(self, output="evaluation", **options):
        from evaluation.hotpotqa.evaluate import run_baseline
        return run_baseline(self.root / output, dataset=self.dataset, **options)