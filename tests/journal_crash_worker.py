"""Offline fixture: abruptly exit at durable action/session boundaries."""

import os
import sys
from pathlib import Path

from agents.teaching.demos import ScriptedModel
from harness import Agent, ToolRegistry
from stores.postgres import PostgresJournal


def main():
    phase, root, schema = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    tools = ToolRegistry()

    @tools.tool(risky=True)
    def action():
        with (root / "effects.txt").open("a", encoding="utf-8") as stream:
            stream.write("effect\n")
        if phase == "after_effect":
            os._exit(22)
        return "effect completed"

    model = ScriptedModel([
        {"stop_reason": "tool_use", "content": [
            {"type": "tool_use", "id": "same-provider-id", "name": "action", "input": {}}]},
        {"stop_reason": "end_turn", "content": [{"type": "text", "text": "done"}]},
    ])
    agent = Agent(model, "offline crash test", tools=tools, approve=lambda *_: True,
                    trace_path=root / "trace.jsonl", session_path=root / "session.json",
                    journal=PostgresJournal(phase, dsn_env="LITTLE_HARNESS_TEST_POSTGRES_DSN", schema=schema))
    if phase == "before_tool":
        begin = agent.journal.begin

        def crash_before_tool(*args):
            begin(*args)
            os._exit(21)

        agent.journal.begin = crash_before_tool
    elif phase == "after_finish":
        finish = agent.journal.finish

        def crash_after_finish(*args):
            finish(*args)
            os._exit(23)

        agent.journal.finish = crash_after_finish
    elif phase == "after_save":
        save = agent._save

        def crash_after_save():
            save()
            if agent.journal.inspect():
                os._exit(24)

        agent._save = crash_after_save
    agent.run_result("make one effect")
    raise AssertionError("crash point was not reached")


if __name__ == "__main__":
    main()