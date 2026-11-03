"""Run one HotpotQA development question in isolation; stdlib only.

Default is an offline retrieval demo. --live explicitly opts into your configured
LLM. No batch API calls, answer scoring, or multi-agent orchestration is implied.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .dev import DEFAULT_DEV_OUTPUT, DEV_COUNT, load_question


def run_question(dataset=DEFAULT_DEV_OUTPUT, number=1, question_id=None, *, live=False):
    number, question, files = load_question(dataset, number, question_id)
    print(f"Question {number}/{DEV_COUNT} ({question['id']}): " + json.dumps(question["question"]), flush=True)
    repo = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="hotpotqa-run-") as temporary:
        root = Path(temporary)
        (root / "knowledge").mkdir()
        for name, content in files.items():
            with (root / name).open("xb") as target:
                target.write(content)
        env = dict(os.environ)
        # Environment overrides .env: neither gold nor previous runs' workspaces
        # are accessible through the child's registered filesystem tools.
        env["HARNESS_KNOWLEDGE"] = str(root / "knowledge")
        env["HARNESS_WORKSPACE"] = str(root / "workspace")
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(repo), env.get("PYTHONPATH"))))
        command = [sys.executable, "-m", "agents.teaching.cli"]
        if not live:
            command.append("rag-mock")
        command.append(question["question"])
        # Keep cwd so the harness loads the user's .env and retains trace.jsonl.
        # One-shot CLI runs never load or save a conversation session.
        return subprocess.run(command, env=env, check=False).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DEV_OUTPUT)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--question", type=int, default=1, help="1-based question number (default: 1)")
    selection.add_argument("--id", help="original HotpotQA question ID")
    parser.add_argument("--live", action="store_true", help="opt into one question using your configured LLM")
    args = parser.parse_args()
    try:
        return run_question(args.dataset, args.question, args.id, live=args.live)
    except (OSError, ValueError, KeyError, TypeError):
        parser.exit(1, "Could not run question; check dataset integrity, selection and configuration.\n")
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())