import json
import unittest

from harness import Agent, ContextConfig, ObservabilityConfig, ToolConfig, ToolRegistry
from harness.agent import ask_human
from tests.helpers import RecordingModel, TemporaryDirectoryTestCase, call, usage_reply


def registry():
    tools = ToolRegistry()

    @tools.tool(risky=True)
    def shout(text: str) -> str:
        """Return text in upper case."""
        return text.upper()

    return tools


class AgentConfigGroupTests(TemporaryDirectoryTestCase):
    def test_groups_match_individual_keywords(self):
        seen = []
        events = []
        common = dict(tools=registry(), max_steps=3)

        def run(**options):
            model = RecordingModel(call("shout", text="hi"), usage_reply("done"))
            agent = Agent(model, "system", **common, **options)
            result = agent.run_result("go")
            return agent, result

        individual, first = run(trace_path=self.root / "a.jsonl", quiet=True, on_event=events.append,
                                approve=lambda *args: seen.append(args) or True, max_tool_output=50,
                                session_path=self.root / "a.session.json", compact_at=1000)
        grouped, second = run(
            observability=ObservabilityConfig(trace_path=self.root / "b.jsonl", quiet=True, on_event=events.append),
            tool_config=ToolConfig(approve=lambda *args: seen.append(args) or True, max_tool_output=50),
            context=ContextConfig(session_path=self.root / "b.session.json", compact_at=1000))

        self.assertEqual((first.status, first.answer), (second.status, second.answer))
        self.assertEqual(seen, [("shout", {"text": "hi"})] * 2)
        self.assertEqual(json.loads((self.root / "a.session.json").read_text())["messages"],
                         json.loads((self.root / "b.session.json").read_text())["messages"])
        for name in ("quiet", "max_tool_output", "compact_at", "session_path"):
            self.assertEqual(type(getattr(grouped, name)), type(getattr(individual, name)), name)
        self.assertEqual((grouped.quiet, grouped.max_tool_output, grouped.compact_at), (True, 50, 1000))
        self.assertTrue(events)

    def test_tool_config_defaults_to_terminal_approval(self):
        agent = Agent(RecordingModel(), "s", tools={}, tool_config=ToolConfig(),
                      trace_path=self.root / "t.jsonl")
        self.assertIs(agent.approve, ask_human)
        quiet = Agent(RecordingModel(), "s", tools={}, tool_config=ToolConfig(),
                      observability=ObservabilityConfig(trace_path=self.root / "t.jsonl", quiet=True))
        self.assertFalse(quiet.approve("shout", {}))

    def test_group_conflicts_with_non_default_keyword(self):
        cases = [
            dict(observability=ObservabilityConfig(), quiet=True),
            dict(tool_config=ToolConfig(), approve=lambda *_: True),
            dict(tool_config=ToolConfig(), max_tool_output=10),
            dict(context=ContextConfig(), compact_at=5),
        ]
        for options in cases:
            with self.subTest(options=sorted(options)), self.assertRaisesRegex(TypeError, "cannot be combined"):
                Agent(RecordingModel(), "s", tools={}, **options)

    def test_default_keywords_alongside_group_are_accepted(self):
        Agent(RecordingModel(), "s", tools={}, quiet=False, compact_at=40000,
              observability=ObservabilityConfig(trace_path=self.root / "t.jsonl"), context=ContextConfig())

    def test_wrong_group_type_is_rejected(self):
        for options in (dict(observability=ToolConfig()), dict(tool_config={}), dict(context="x")):
            with self.subTest(options=sorted(options)), self.assertRaisesRegex(TypeError, "must be"):
                Agent(RecordingModel(), "s", tools={}, **options)

    def test_group_values_are_still_validated(self):
        with self.assertRaisesRegex(ValueError, "max_tool_output"):
            Agent(RecordingModel(), "s", tools={}, tool_config=ToolConfig(max_tool_output=0))
        with self.assertRaises(TypeError):
            Agent(RecordingModel(), "s", tools={}, observability=ObservabilityConfig(quiet="yes"))


if __name__ == "__main__":
    unittest.main()
