"""Teaching application CLI: python3 -m agents.teaching.cli."""

import os
import sys

from harness import ModelConfigError, RunLimits, SessionLoadError, ToolPolicy, make_model
from harness.model_clients import print_model_config
from harness.tool_execution import validate_tool_timeout

from .config import TeachingConfig, load_env
from .demos import DEMO_SCRIPT, SYSTEM_PROMPT, ScriptedModel, rag_demo_model
from .factory import create_agent


def main():
    try:
        load_env()
        config = TeachingConfig.from_env(os.environ)
    except (OSError, ValueError):
        sys.exit("Could not load application configuration; check .env syntax and settings.")

    def agent_factory(model, system, **options):
        return create_agent(model, config, system, **options)

    args = sys.argv[1:]
    journal_path = os.environ.get("HARNESS_JOURNAL", "").strip() or None
    read_only = os.environ.get("HARNESS_READ_ONLY", "").strip()
    if read_only not in ("", "0", "1"):
        sys.exit("Invalid HARNESS_READ_ONLY; use 1 to enable read-only tools or 0 to disable it.")
    policy = ToolPolicy(read_only=True) if read_only == "1" else None
    trace_content = os.environ.get("HARNESS_TRACE_CONTENT", "").strip()
    if trace_content not in ("", "0", "1"):
        sys.exit("Invalid HARNESS_TRACE_CONTENT; use 1 to include content or 0 for metadata only.")
    try:
        limits = RunLimits.from_env(os.environ)
        raw_timeout = os.environ.get("HARNESS_TOOL_TIMEOUT_SECONDS", "").strip()
        tool_timeout = validate_tool_timeout(float(raw_timeout)) if raw_timeout else None
    except ValueError:
        sys.exit("Invalid execution limits; check HARNESS_MAX_MODEL_CALLS, HARNESS_MAX_TOOL_CALLS, "
                 "HARNESS_MAX_TOKENS, HARNESS_MAX_SECONDS and HARNESS_TOOL_TIMEOUT_SECONDS.")
    options = dict(limits=limits, tool_timeout_seconds=tool_timeout,
                   journal_path=journal_path, tool_policy=policy, trace_include_content=trace_content == "1")
    if args and args[0] == "rag-mock":
        agent = agent_factory(rag_demo_model, SYSTEM_PROMPT, **options)
        question = " ".join(args[1:]) or "What are the Aurora Observatory visiting hours?"
        print("\n=== FINAL ===\n" + agent.run(question))
        return
    if args and args[0] == "mock":
        def auto_approve(name, args):
            print(f"\n  ⚠  {name} is risky; auto-approved in mock mode")
            return True
        agent = agent_factory(ScriptedModel(DEMO_SCRIPT), SYSTEM_PROMPT, approve=auto_approve, **options)
        print("\n=== FINAL ===\n" + agent.run("Compute 17*23+4, timestamp it, and save a report."))
        return

    try:
        model = make_model()
    except ModelConfigError as error:
        sys.exit(f"{error}. For an offline demo, run: python3 -m agents.teaching.cli mock")
    print_model_config(model)
    if args:
        agent = agent_factory(model, SYSTEM_PROMPT, **options)
        print("\n=== FINAL ===\n" + agent.run(" ".join(args)))
        return
    try:
        agent = agent_factory(model, SYSTEM_PROMPT,
                              session_path=os.environ.get("HARNESS_SESSION", "session.json"), **options)
    except SessionLoadError as error:
        sys.exit(str(error))
    print("Interactive mode. The agent remembers the conversation. Ctrl+C or 'exit' to quit.")
    while True:
        try:
            task = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if task.lower() in {"exit", "quit"}:
            break
        if task:
            print("\nagent> " + agent.run(task))


if __name__ == "__main__":
    main()