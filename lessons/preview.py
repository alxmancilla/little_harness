"""A completed offline preview before you write your own loop in Lab 0."""

import argparse

from .lab01_loop import run
from .support import show_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", action="store_true", help="show the synthetic conversation")
    parser.add_argument("--events", action="store_true", help="show metadata event names")
    args = parser.parse_args()
    print("Completed preview: no exercise code or reference solutions are loaded.")
    print("A scripted model requests a word count, receives the tool result, then answers.")
    observation = run()
    show_run(observation, args)
    result = observation[0]
    if (result.status, result.answer, result.usage.tool_errors) != ("completed", "3", 0):
        print("Preview did not finish as expected. See LEARN.md: Troubleshooting.")
        return 1
    print("\nPreview complete. You just observed two model calls and one tool execution.")
    print("This demonstrates the protocol, not live-model intelligence.")
    print("Now build the loop yourself: read lessons/00-from-functions-to-agents.md.")
    print("First checkpoint: python3 -m lessons.lab00_basics --check --stage 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())