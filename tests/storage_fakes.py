"""Non-durable test doubles for exercising runtime behavior without a database."""

from datetime import datetime, timezone
from threading import RLock
from uuid import uuid4

from harness import JournalError, JournalRecoveryRequired, MemoryRecord, MemoryStoreError
from harness.journal import check_inspect_limit, check_outcome, check_resolution, inspection_record
from harness.memory import _limit, _namespace, _rank, _text, _validate_add, _WORDS


class FakeJournal:
    """In-memory only: never use as an application journal or durability test."""

    def __init__(self):
        self.rows, self.closed = [], set()
        self.lock = RLock()

    def prepare(self):
        pass

    def _pending(self, row):
        return row["resolution"] is None and (row["run_id"] not in self.closed
                                               or row["outcome_unknown"] or row["status"] == "started")

    def assert_ready(self):
        with self.lock:
            if any(self._pending(row) for row in self.rows):
                raise JournalRecoveryRequired()

    def begin(self, run_id, tool_call_id, tool_name, risky):
        with self.lock:
            if run_id in self.closed:
                raise JournalError()
            if any(row["resolution"] is None and (row["status"] == "started" or row["outcome_unknown"]
                   or (row["run_id"] not in self.closed and row["run_id"] != run_id)) for row in self.rows):
                raise JournalRecoveryRequired()
            action_id = uuid4().hex
            self.rows.append({"action_id": action_id, "run_id": run_id, "tool_call_id": tool_call_id,
                              "tool_name": tool_name, "risky": bool(risky), "status": "started",
                              "outcome_unknown": False, "resolution": None, "resolved_at": None,
                              "started_at": datetime.now(timezone.utc).isoformat(), "finished_at": None})
            return action_id

    def finish(self, action_id, status, outcome_unknown=False):
        check_outcome(status, outcome_unknown)
        with self.lock:
            row = next((row for row in self.rows if row["action_id"] == action_id), None)
            if row is None or row["status"] != "started" or row["resolution"] is not None:
                raise JournalError()
            row.update(status=status, outcome_unknown=outcome_unknown, finished_at=datetime.now(timezone.utc).isoformat())

    def close_run(self, run_id):
        with self.lock:
            self.closed.add(run_id)

    def inspect(self, *, pending_only=False, limit=100):
        check_inspect_limit(limit)
        with self.lock:
            rows = [dict(row, run_closed=row["run_id"] in self.closed, recovery_required=self._pending(row))
                    for row in reversed(self.rows) if not pending_only or self._pending(row)]
            return [inspection_record(row) for row in rows[:limit]]

    def resolve(self, action_id, outcome, *, confirm=False):
        check_resolution(outcome, confirm)
        with self.lock:
            row = next((row for row in self.rows if row["action_id"] == action_id), None)
            if row is None or not self._pending(row):
                raise ValueError("action does not require recovery")
            row.update(resolution=outcome, resolved_at=datetime.now(timezone.utc).isoformat())


class FakeMemoryStore:
    """In-memory test fixture: no persistence or cross-process guarantees."""

    def __init__(self, *, quota=10000):
        self.records, self.quota, self.lock = [], quota, RLock()

    def add(self, namespace, text, *, kind, created_by, run_id=None):
        _validate_add(namespace, text, kind, created_by, run_id)
        with self.lock:
            if sum(row.namespace == namespace for row in self.records) >= self.quota:
                raise MemoryStoreError()
            memory_id = uuid4().hex
            self.records.append(MemoryRecord(memory_id, namespace, text, kind, created_by,
                                             datetime.now(timezone.utc).isoformat(), run_id))
            return memory_id

    def search(self, namespace, query, *, limit=5):
        _namespace(namespace)
        _text(query, 500, "query")
        _limit(limit, 20)
        with self.lock:
            return _rank([row for row in self.records if row.namespace == namespace],
                         set(_WORDS.findall(query.lower())), limit)

    def list(self, namespace, *, limit=100):
        _namespace(namespace)
        _limit(limit, 1000)
        with self.lock:
            rows = sorted((row for row in self.records if row.namespace == namespace), key=lambda row: row.memory_id)
            return sorted(rows, key=lambda row: row.created_at, reverse=True)[:limit]

    def delete(self, namespace, memory_id):
        _namespace(namespace)
        _text(memory_id, 200, "memory_id")
        with self.lock:
            for row in self.records:
                if row.namespace == namespace and row.memory_id == memory_id:
                    self.records.remove(row)
                    return True
            return False

    def purge(self, namespace):
        _namespace(namespace)
        with self.lock:
            before = len(self.records)
            self.records[:] = [row for row in self.records if row.namespace != namespace]
            return before - len(self.records)