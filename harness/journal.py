"""Metadata-only action journal contract and shared recovery behavior."""

import argparse
import json
from typing import Protocol, runtime_checkable


STATUSES = {"completed", "failed", "timed_out", "outcome_unknown", "not_run"}


class JournalError(Exception):
    reason = "journal_error"

    def __init__(self):
        super().__init__("execution journal unavailable; inspect it before continuing")


class JournalRecoveryRequired(JournalError):
    reason = "journal_recovery_required"

    def __init__(self):
        Exception.__init__(self, "execution journal requires human recovery; no work was started")


# Rules shared by every ActionJournal implementation.
def check_outcome(status, outcome_unknown):
    if (status not in STATUSES or (status in {"timed_out", "outcome_unknown"} and not outcome_unknown)
            or (status in {"completed", "not_run"} and outcome_unknown)):
        raise ValueError("invalid journal outcome")


def check_inspect_limit(limit):
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("inspection limit must be between 1 and 1000")


def check_resolution(outcome, confirm):
    if not confirm or outcome not in {"completed", "failed", "not_run"}:
        raise ValueError("recovery requires confirmation and a known outcome")


def inspection_record(record):
    """Report an unfinished action as outcome_unknown, keeping what was recorded."""
    record["recorded_status"] = record["status"]
    if record["status"] == "started":
        record["status"], record["outcome_unknown"] = "outcome_unknown", True
    for name in ("risky", "run_closed", "outcome_unknown", "recovery_required"):
        record[name] = bool(record[name])
    return record


@runtime_checkable
class ActionJournal(Protocol):
    """Durable action records; every failure is JournalError or JournalRecoveryRequired."""

    def prepare(self) -> None: ...

    def assert_ready(self) -> None: ...

    def begin(self, run_id: str, tool_call_id: str, tool_name: str, risky: bool) -> str: ...

    def finish(self, action_id: str, status: str, outcome_unknown: bool = False) -> None: ...

    def close_run(self, run_id: str) -> None: ...

    def inspect(self, *, pending_only: bool = False, limit: int = 100) -> list[dict]: ...

    def resolve(self, action_id: str, outcome: str, *, confirm: bool = False) -> None: ...


def main(argv=None):
    parser = argparse.ArgumentParser(description="Use python -m stores.postgres or stores.mongodb for journal recovery.")
    parser.parse_args(argv)
    parser.exit(2, "Select a database backend: python -m stores.postgres --help or python -m stores.mongodb --help.\n")


def run_recovery_command(journal, args):
    """Shared inspect/resolve behavior for every journal command line."""
    if args.command == "inspect":
        print(json.dumps(journal.inspect(pending_only=args.pending, limit=args.limit), indent=2))
    else:
        journal.resolve(args.action_id, args.outcome, confirm=args.confirm)
        print("Recovery decision recorded; no action was executed. Review the session before continuing.")