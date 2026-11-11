"""Opt-in live provider checks: real model calls that can incur small API costs.

Skipped unless HARNESS_TEST_LIVE=1. Provider settings come from the environment and,
for convenience, from .env in the current directory (shell values win). Each model in
HARNESS_TEST_LIVE_MODELS (comma-separated, e.g. "claude-sonnet-5-5,gpt-5.6-sol" for the
gateway) is checked; otherwise HARNESS_MODEL or the provider default. On a parse
failure the message lists stream event shapes (types and keys, never content).

Run: HARNESS_TEST_LIVE=1 uv run python -m unittest -v tests.test_live_providers
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agents.teaching.config import load_env
from harness import Agent, ModelError, ToolRegistry, make_model
from harness import model_clients
from harness.agent import _check_response

LIVE = os.environ.get("HARNESS_TEST_LIVE") == "1"


def live_environment():
    env = dict(os.environ)
    load_env(".env", env)
    return env


def live_models(env):
    names = [name.strip() for name in env.get("HARNESS_TEST_LIVE_MODELS", "").split(",") if name.strip()]
    return names or [env.get("HARNESS_MODEL", "")]


def shape(data):
    """Describe one stream event without its content."""
    try:
        event = json.loads(data)
    except ValueError:
        return {"non_json": len(data)}
    if not isinstance(event, dict):
        return {"json": type(event).__name__}
    described = {"type": event.get("type"), "keys": sorted(event)}
    for key in ("delta", "content_block"):
        if isinstance(event.get(key), dict):
            described[key] = {k: (v if k == "type" else type(v).__name__) for k, v in event[key].items()}
    return described


class RecordedCall:
    """Record event shapes while a real call streams, for diagnosable failures."""

    def __init__(self):
        self.shapes = []
        self._original = model_clients.sse_events

    def __enter__(self):
        def spy(lines):
            for name, data in self._original(lines):
                self.shapes.append([name, shape(data)])
                yield name, data
        self._patch = mock.patch.object(model_clients, "sse_events", spy)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        return False

    def report(self):
        return "stream event shapes:\n" + "\n".join(json.dumps(item) for item in self.shapes)


@unittest.skipUnless(LIVE, "opt in to paid live provider checks with HARNESS_TEST_LIVE=1")
class LiveProviderTests(unittest.TestCase):
    def setUp(self):
        self.env = live_environment()

    def model(self, name):
        env = dict(self.env, HARNESS_MODEL=name) if name else self.env
        return make_model(env)

    def call(self, model, *args, **kwargs):
        with RecordedCall() as recorded:
            try:
                response = model(*args, **kwargs)
                _check_response(response)
            except ModelError as error:
                self.fail(f"{model.provider}/{model.model}: {error}\n{recorded.report()}")
        return response, recorded

    def test_text_reply_streams_and_matches_the_internal_shape(self):
        for name in live_models(self.env):
            with self.subTest(model=name or "default"):
                model = self.model(name)
                deltas = []
                response, recorded = self.call(model, "Reply with one short word.",
                                               [{"role": "user", "content": "Say hello."}], [],
                                               on_text=deltas.append)
                text = "".join(b["text"] for b in response["content"] if b["type"] == "text")
                self.assertTrue(text.strip(), recorded.report())
                self.assertEqual("".join(deltas), text, "streamed deltas differ from the final text")
                self.assertEqual(response["stop_reason"], "end_turn", recorded.report())

    def test_tool_round_trip_completes(self):
        tools = ToolRegistry()
        calls = []

        @tools.tool(read_only=True)
        def add(a: int, b: int) -> str:
            """Add two integers."""
            calls.append((a, b))
            return str(a + b)

        for name in live_models(self.env):
            with self.subTest(model=name or "default"), tempfile.TemporaryDirectory() as directory:
                calls.clear()
                model = self.model(name)
                agent = Agent(model, "Use the add tool for arithmetic, then answer with the number only.",
                              tools=tools, quiet=True, max_steps=4, trace_path=Path(directory) / "trace.jsonl")
                with RecordedCall() as recorded:
                    result = agent.run_result("What is 1234 + 4321?")
                self.assertEqual(result.status, "completed", f"{result.to_dict()}\n{recorded.report()}")
                self.assertIn((1234, 4321), calls)
                self.assertIn("5555", result.answer.replace(",", ""))


if __name__ == "__main__":
    unittest.main()
