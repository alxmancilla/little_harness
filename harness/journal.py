"""Metadata-only SQLite action journal and offline human recovery commands."""

import argparse
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable
from uuid import uuid4


APPLICATION_ID = 0x4D484A31
SCHEMA_VERSION = 1
PENDING = "a.resolution IS NULL AND (r.closed = 0 OR a.outcome_unknown = 1 OR a.status = 'started')"
STATUSES = {"completed", "failed", "timed_out", "outcome_unknown", "not_run"}


class JournalError(Exception):
    reason = "journal_error"

    def __init__(self):
        super().__init__("execution journal unavailable; inspect it before continuing")


class JournalRecoveryRequired(JournalError):
    reason = "journal_recovery_required"

    def __init__(self):
        Exception.__init__(self, "execution journal requires human recovery; no work was started")


def _now():
    return datetime.now(timezone.utc).isoformat()


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


class ExecutionJournal:
    """No arguments, outputs, exception details, credentials or automatic replay."""

    def __init__(self, path):
        try:
            self.path = Path(path).resolve()
        except (OSError, ValueError, RuntimeError):
            raise JournalError() from None

    @contextmanager
    def _connection(self, *, create=False, readonly=False):
        connection = None
        try:
            if create:
                try:
                    descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                except FileExistsError:
                    pass
                else:
                    os.close(descriptor)
            mode = "ro" if readonly else "rw"
            connection = sqlite3.connect(self.path.as_uri() + "?mode=" + mode, uri=True, timeout=2)
            connection.row_factory = sqlite3.Row
            if not readonly:
                connection.execute("PRAGMA synchronous = FULL")
            yield connection
        except (sqlite3.Error, OSError):
            raise JournalError() from None
        finally:
            if connection is not None:
                connection.close()  # rolls back any uncommitted transaction

    @staticmethod
    def _verify(connection):
        if (connection.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID
                or connection.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION):
            raise JournalError()

    def prepare(self):
        """Initialize only an empty file; never adopt an unrelated or newer database."""
        with self._connection(create=True) as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("PRAGMA application_id").fetchone()[0] == 0:
                if (connection.execute("PRAGMA user_version").fetchone()[0] != 0
                        or connection.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()):
                    raise JournalError()
                connection.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY, closed INTEGER NOT NULL DEFAULT 0)")
                connection.execute("""CREATE TABLE actions (
                    action_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, tool_call_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL, risky INTEGER NOT NULL, started_at TEXT NOT NULL,
                    finished_at TEXT, status TEXT NOT NULL, outcome_unknown INTEGER NOT NULL DEFAULT 0,
                    resolution TEXT, resolved_at TEXT)""")
                connection.execute("CREATE INDEX actions_run ON actions(run_id)")
                connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
                connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._verify(connection)
            connection.commit()

    def assert_ready(self):
        self.prepare()
        with self._connection(readonly=True) as connection:
            self._verify(connection)
            if connection.execute(f"SELECT 1 FROM actions a JOIN runs r USING (run_id) WHERE {PENDING} LIMIT 1").fetchone():
                raise JournalRecoveryRequired()

    def begin(self, run_id, tool_call_id, tool_name, risky):
        """Commit before dispatch; an unreturned begin never authorizes execution."""
        action_id = uuid4().hex
        with self._connection() as connection:
            self._verify(connection)
            connection.execute("BEGIN IMMEDIATE")
            blocked = connection.execute(f"""SELECT 1 FROM actions a JOIN runs r USING (run_id)
                WHERE a.resolution IS NULL AND (a.status = 'started' OR a.outcome_unknown = 1
                    OR (r.closed = 0 AND a.run_id != ?)) LIMIT 1""", (run_id,)).fetchone()
            if blocked:
                raise JournalRecoveryRequired()
            connection.execute("INSERT OR IGNORE INTO runs (run_id) VALUES (?)", (run_id,))
            if connection.execute("SELECT closed FROM runs WHERE run_id = ?", (run_id,)).fetchone()[0]:
                raise JournalError()
            connection.execute("""INSERT INTO actions
                (action_id, run_id, tool_call_id, tool_name, risky, started_at, status)
                VALUES (?, ?, ?, ?, ?, ?, 'started')""",
                               (action_id, run_id, tool_call_id, tool_name, bool(risky), _now()))
            connection.commit()
        return action_id

    def finish(self, action_id, status, outcome_unknown=False):
        check_outcome(status, outcome_unknown)
        with self._connection() as connection:
            self._verify(connection)
            result = connection.execute("""UPDATE actions SET status = ?, outcome_unknown = ?, finished_at = ?
                WHERE action_id = ? AND status = 'started' AND resolution IS NULL""",
                                        (status, bool(outcome_unknown), _now(), action_id))
            if result.rowcount != 1:
                raise JournalError()
            connection.commit()

    def close_run(self, run_id):
        """Called only after the harness has saved a coherent terminal conversation."""
        with self._connection() as connection:
            self._verify(connection)
            connection.execute("UPDATE runs SET closed = 1 WHERE run_id = ?", (run_id,))
            connection.commit()

    def inspect(self, *, pending_only=False, limit=100):
        """Read-only, bounded metadata; never create a file or rewrite live records."""
        check_inspect_limit(limit)
        with self._connection(readonly=True) as connection:
            self._verify(connection)
            where = "WHERE " + PENDING if pending_only else ""
            rows = connection.execute(f"""SELECT a.*, r.closed AS run_closed, ({PENDING}) AS recovery_required
                FROM actions a JOIN runs r USING (run_id) {where}
                ORDER BY a.started_at DESC, a.action_id DESC LIMIT ?""", (limit,)).fetchall()
        return [inspection_record(dict(row)) for row in rows]

    def resolve(self, action_id, outcome, *, confirm=False):
        """Human assertion, not proof, approval, undo, session repair or a retry."""
        check_resolution(outcome, confirm)
        with self._connection() as connection:
            self._verify(connection)
            connection.execute("BEGIN IMMEDIATE")
            pending = connection.execute(f"""SELECT 1 FROM actions a JOIN runs r USING (run_id)
                WHERE a.action_id = ? AND {PENDING}""", (action_id,)).fetchone()
            if not pending:
                raise ValueError("action does not require recovery")
            connection.execute("UPDATE actions SET resolution = ?, resolved_at = ? WHERE action_id = ?",
                               (outcome, _now(), action_id))
            connection.commit()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="read-only metadata; missing files are not created")
    inspect.add_argument("path", type=Path)
    inspect.add_argument("--pending", action="store_true")
    inspect.add_argument("--limit", type=int, default=100)
    resolve = commands.add_parser("resolve", help="record human recovery only; never execute an action")
    resolve.add_argument("path", type=Path)
    resolve.add_argument("action_id")
    resolve.add_argument("--outcome", required=True, choices=("completed", "failed", "not_run"))
    resolve.add_argument("--confirm", action="store_true", required=True)
    args = parser.parse_args(argv)
    try:
        run_recovery_command(ExecutionJournal(args.path), args)
    except (JournalError, ValueError, OSError):
        parser.exit(1, "Journal operation failed; check the file, action state and recovery options.\n")


def run_recovery_command(journal, args):
    """Shared inspect/resolve behavior for every journal command line."""
    if args.command == "inspect":
        print(json.dumps(journal.inspect(pending_only=args.pending, limit=args.limit), indent=2))
    else:
        journal.resolve(args.action_id, args.outcome, confirm=args.confirm)
        print("Recovery decision recorded; no action was executed. Review the session before continuing.")