"""MongoDB 9.0 session store and action journal (docs/STORAGE_SPEC.md, section 5.2).

Needs the optional extra: uv sync --extra mongodb, and a replica set (one member is
enough) because saves and journal checks use transactions. Connection settings come
from an environment variable named by the application, or an application-supplied
MongoClient; they never appear in errors, traces or command lines.

  python -m stores.mongodb init --uri-env NAME [--database NAME]
  python -m stores.mongodb journal inspect --uri-env NAME --journal-id ID [--pending]
  python -m stores.mongodb journal resolve --uri-env NAME --journal-id ID ACTION_ID --outcome O --confirm
"""

import atexit
import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from uuid import uuid4

from harness.journal import (JournalError, JournalRecoveryRequired, check_inspect_limit, check_outcome,
                             check_resolution, inspection_record)
from harness.session import (SESSION_VERSION, SessionConflictError, SessionLoadError, SessionStoreError,
                             validate_session)
from stores._common import APPLICATION, SCHEMA_VERSION, check_id, storage_main

try:
    import pymongo
    from pymongo import ASCENDING, DESCENDING, MongoClient, ReadPreference, WriteConcern
    from pymongo.errors import DuplicateKeyError, PyMongoError
    from pymongo.read_concern import ReadConcern
except ImportError:
    pymongo = None

CONNECT_TIMEOUT_MS = 10000
DATABASE_NAME = re.compile(r"[A-Za-z0-9_-]{1,63}")
STRING, BOOL, DATE = {"bsonType": "string"}, {"bsonType": "bool"}, {"bsonType": "date"}
INT = {"bsonType": ["int", "long"]}


def NULLABLE(kind):
    return {"bsonType": [kind, "null"]}


COLLECTIONS = {
    "session": {
        "lh_sessions": ({"_id": STRING, "revision": INT, "message_count": INT, "compaction": NULLABLE("string")},
                        []),
        "lh_session_messages": ({"session_id": STRING, "idx": INT, "message": STRING,
                                 "sha256": {"bsonType": "binData"}}, [([("session_id", 1), ("idx", 1)], True)])},
    "journal": {
        "lh_journals": ({"_id": STRING, "seq": INT}, []),
        "lh_journal_runs": ({"journal_id": STRING, "run_id": STRING, "closed": BOOL},
                            [([("journal_id", 1), ("run_id", 1)], True)]),
        "lh_journal_actions": ({"_id": STRING, "journal_id": STRING, "run_id": STRING, "tool_call_id": STRING,
                                "tool_name": STRING, "risky": BOOL, "started_at": DATE, "finished_at": NULLABLE("date"),
                                "status": STRING, "outcome_unknown": BOOL, "resolution": NULLABLE("string"),
                                "resolved_at": NULLABLE("date")},
                               [([("journal_id", 1), ("resolution", 1)], False)])}}


def _now():
    return datetime.now(timezone.utc)


# One MongoClient (a connection pool) per connection string per process, closed at exit.
_CLIENTS, _CLIENTS_LOCK = {}, threading.Lock()


def _shared_client(uri):
    key = hashlib.sha256(uri.encode("utf-8")).hexdigest()
    with _CLIENTS_LOCK:
        if not _CLIENTS:
            atexit.register(lambda: [client.close() for client in _CLIENTS.values()])
        if key not in _CLIENTS:
            _CLIENTS[key] = MongoClient(uri, serverSelectionTimeoutMS=CONNECT_TIMEOUT_MS,
                                        connectTimeoutMS=CONNECT_TIMEOUT_MS, tz_aware=True)
        return _CLIENTS[key]


class _MongoStorage:
    """Client, time bounds, majority-acknowledged transactions and collection ownership (ST-06, ST-07)."""

    component = None

    def __init__(self, name, *, uri_env, client, database, timeout):
        if pymongo is None:
            raise ImportError("MongoDB storage needs the optional 'mongodb' extra: uv sync --extra mongodb")
        check_id(name)
        if (uri_env is None) == (client is None):
            raise TypeError("pass exactly one of uri_env (a variable name) or client (a MongoClient)")
        if client is not None and not isinstance(client, MongoClient):
            raise TypeError("client must be a pymongo MongoClient")
        if uri_env is not None and not os.environ.get(uri_env):
            raise ValueError(f"environment variable {uri_env} is not set")
        if type(database) is not str or not DATABASE_NAME.fullmatch(database):
            raise ValueError("database must be 1-63 characters from A-Z, a-z, 0-9, '_' and '-'")
        if type(timeout) not in (int, float) or not 0 < timeout <= 300:
            raise ValueError("timeout must be between 0 and 300 seconds")
        self.name, self.database_name, self.timeout = name, database, timeout
        self._uri_env, self._client = uri_env, client
        self._ready = False

    def _error(self):
        raise NotImplementedError

    def _db(self):
        if self._client is None:
            uri = os.environ.get(self._uri_env)
            if not uri:
                raise self._error()
            self._client = _shared_client(uri)
        return self._client.get_database(self.database_name, read_concern=ReadConcern("majority"),
                                         write_concern=WriteConcern("majority", j=True),
                                         read_preference=ReadPreference.PRIMARY)

    def _run(self, operation, *, transaction=True):
        """One bounded operation. In a transaction, only the driver's labelled transient
        errors rerun it; any outcome still unknown at the time bound raises (spec 5.2)."""
        try:
            with pymongo.timeout(self.timeout):
                db = self._db()
                if not transaction:
                    return operation(db, None)
                with self._client.start_session() as session:
                    return session.with_transaction(
                        lambda s: operation(db, s), read_concern=ReadConcern("snapshot"),
                        write_concern=WriteConcern("majority", j=True), read_preference=ReadPreference.PRIMARY)
        except PyMongoError:
            raise self._error() from None

    def _check_collections(self, db, *, create):
        """Use only collections this component recorded; create them only where none exist."""
        if not db.client.admin.command("hello").get("setName"):
            raise self._error()  # Transactions need a replica set.
        record = db.lh_meta.find_one({"_id": self.component})
        if record is not None:
            if (record.get("application"), record.get("schema_version")) != (APPLICATION, SCHEMA_VERSION):
                raise self._error()
            return
        existing = set(db.list_collection_names())
        if not create or existing & set(COLLECTIONS[self.component]):
            raise self._error()
        for collection, (fields, indexes) in COLLECTIONS[self.component].items():
            db.create_collection(collection, validator={"$jsonSchema": {
                "bsonType": "object", "required": list(fields), "properties": fields}})
            for keys, unique in indexes:
                db[collection].create_index(keys, unique=unique)
        try:
            db.lh_meta.insert_one({"_id": self.component, "application": APPLICATION,
                                   "schema_version": SCHEMA_VERSION})
        except DuplicateKeyError:
            raise self._error() from None

    def ensure_schema(self, *, create=True):
        """Check this component's collections, creating them where none exist when allowed."""
        self._run(lambda db, _: self._check_collections(db, create=create), transaction=False)

    def prepare(self):
        self._run(lambda db, _: (self._check_collections(db, create=True), self._register(db)), transaction=False)
        self._ready = True

    def _register(self, db):
        pass


class MongoSessionStore(_MongoStorage):
    """A SessionStore: one document per message, holding it as a JSON string.

    Model-chosen keys such as "$where" or "a.b" never become BSON field names, and no
    session nears the 16 MiB document limit. Saves rewrite from the first changed
    message, are majority-acknowledged and journaled whatever `durable` asks for, and
    are refused with SessionConflictError if another writer saved since this handle's load.
    """

    component = "session"

    def __init__(self, session_id, *, uri_env=None, client=None, database="little_harness", timeout=5):
        super().__init__(session_id, uri_env=uri_env, client=client, database=database, timeout=timeout)
        self.revision, self._hashes = 0, []

    def __str__(self):
        return f"MongoDB session {self.name}"

    def _error(self):
        return SessionStoreError("MongoDB session operation failed; inspect the session before continuing")

    def load(self):
        if not self._ready:
            self.prepare()

        def read(db, session):
            head = db.lh_sessions.find_one({"_id": self.name}, session=session)
            rows = [] if head is None else list(db.lh_session_messages.find(
                {"session_id": self.name}, session=session).sort("idx", ASCENDING))
            return head, rows

        head, rows = self._run(read)
        if head is None:
            self.revision, self._hashes = 0, []
            return None
        try:
            if [row["idx"] for row in rows] != list(range(head["message_count"])):
                raise ValueError("stored messages are incomplete")
            data = {"version": SESSION_VERSION, "messages": [json.loads(row["message"]) for row in rows],
                    "compaction": None if head["compaction"] is None else json.loads(head["compaction"])}
        except ValueError as e:
            raise SessionLoadError(f"{self} is not a valid session ({e}); fix or delete it to start fresh") from None
        loaded = validate_session(data, self)
        self.revision, self._hashes = head["revision"], [bytes(row["sha256"]) for row in rows]
        return loaded

    def save(self, messages, compaction, *, durable=True):
        texts = [json.dumps(message) for message in messages]
        hashes = [hashlib.sha256(text.encode("utf-8")).digest() for text in texts]
        start = next((index for index, (old, new) in enumerate(zip(self._hashes, hashes)) if old != new),
                     min(len(self._hashes), len(hashes)))
        head = {"message_count": len(texts), "compaction": None if compaction is None else json.dumps(compaction)}
        conflict = f"{self} changed since it was loaded; nothing was saved"
        if not self._ready:
            self.prepare()

        def write(db, session):
            if self.revision == 0:
                try:
                    db.lh_sessions.insert_one({"_id": self.name, "revision": 1, **head}, session=session)
                except DuplicateKeyError:
                    raise SessionConflictError(conflict) from None
            elif db.lh_sessions.update_one({"_id": self.name, "revision": self.revision},
                                           {"$inc": {"revision": 1}, "$set": head}, session=session).matched_count != 1:
                raise SessionConflictError(conflict)
            db.lh_session_messages.delete_many({"session_id": self.name, "idx": {"$gte": start}}, session=session)
            if start < len(texts):
                db.lh_session_messages.insert_many(
                    [{"session_id": self.name, "idx": index, "message": texts[index], "sha256": hashes[index]}
                     for index in range(start, len(texts))], session=session)

        self._run(write)
        self.revision, self._hashes = self.revision + 1, hashes


class MongoJournal(_MongoStorage):
    """An ActionJournal with the guarantees of ExecutionJournal (FR-08, ST-03).

    begin and resolve first increment a counter on this journal's document. MongoDB
    detects only write-write conflicts, so that write is what makes concurrent calls
    check and write one at a time. Conflicts rerun within the time bound; an outcome
    still unknown then raises JournalError, so no tool is dispatched.
    """

    component = "journal"

    def __init__(self, journal_id, *, uri_env=None, client=None, database="little_harness", timeout=5):
        super().__init__(journal_id, uri_env=uri_env, client=client, database=database, timeout=timeout)

    def __str__(self):
        return f"MongoDB journal {self.name}"

    def _error(self):
        return JournalError()

    def _register(self, db):
        db.lh_journals.update_one({"_id": self.name}, {"$setOnInsert": {"seq": 0}}, upsert=True)

    def _serialize(self, db, session):
        if db.lh_journals.update_one({"_id": self.name}, {"$inc": {"seq": 1}}, session=session).matched_count != 1:
            raise JournalError()

    def _closed_runs(self, db, session, actions):
        run_ids = sorted({action["run_id"] for action in actions})
        runs = db.lh_journal_runs.find({"journal_id": self.name, "run_id": {"$in": run_ids}}, session=session)
        return {run["run_id"]: run["closed"] for run in runs}

    def _pending(self, db, session, actions):
        closed = self._closed_runs(db, session, actions)
        return [a for a in actions if a["resolution"] is None and (
            not closed.get(a["run_id"], False) or a["outcome_unknown"] or a["status"] == "started")]

    def _unresolved(self, db, session):
        return list(db.lh_journal_actions.find({"journal_id": self.name, "resolution": None}, session=session)
                    .sort([("started_at", DESCENDING), ("_id", DESCENDING)]))

    def assert_ready(self):
        self.prepare()
        if self._run(lambda db, session: self._pending(db, session, self._unresolved(db, session))):
            raise JournalRecoveryRequired()

    def begin(self, run_id, tool_call_id, tool_name, risky):
        """Commit before dispatch; an unreturned begin never authorizes execution."""
        action_id = uuid4().hex

        def write(db, session):
            self._serialize(db, session)
            unresolved = self._unresolved(db, session)
            closed = self._closed_runs(db, session, unresolved)
            if any(a["status"] == "started" or a["outcome_unknown"]
                   or (not closed.get(a["run_id"], False) and a["run_id"] != run_id) for a in unresolved):
                raise JournalRecoveryRequired()
            run = db.lh_journal_runs.find_one_and_update(
                {"journal_id": self.name, "run_id": run_id}, {"$setOnInsert": {"closed": False}}, upsert=True,
                return_document=pymongo.ReturnDocument.AFTER, session=session)
            if run["closed"]:
                raise JournalError()
            db.lh_journal_actions.insert_one({
                "_id": action_id, "journal_id": self.name, "run_id": run_id, "tool_call_id": tool_call_id,
                "tool_name": tool_name, "risky": bool(risky), "started_at": _now(), "finished_at": None,
                "status": "started", "outcome_unknown": False, "resolution": None, "resolved_at": None},
                session=session)

        self._run(write)
        return action_id

    def finish(self, action_id, status, outcome_unknown=False):
        check_outcome(status, outcome_unknown)
        updated = self._run(lambda db, _: db.lh_journal_actions.update_one(
            {"_id": action_id, "journal_id": self.name, "status": "started", "resolution": None},
            {"$set": {"status": status, "outcome_unknown": bool(outcome_unknown), "finished_at": _now()}}),
            transaction=False)
        if updated.matched_count != 1:
            raise JournalError()

    def close_run(self, run_id):
        """Called only after the harness has saved a coherent terminal conversation."""
        self._run(lambda db, _: db.lh_journal_runs.update_one({"journal_id": self.name, "run_id": run_id},
                                                              {"$set": {"closed": True}}), transaction=False)

    def inspect(self, *, pending_only=False, limit=100):
        """Read-only, bounded metadata; never creates collections, journals or records."""
        check_inspect_limit(limit)
        self.ensure_schema(create=False)

        def read(db, session):
            if db.lh_journals.find_one({"_id": self.name}, session=session) is None:
                raise JournalError()
            if pending_only:
                actions = self._pending(db, session, self._unresolved(db, session))[:limit]
            else:
                actions = list(db.lh_journal_actions.find({"journal_id": self.name}, session=session)
                               .sort([("started_at", DESCENDING), ("_id", DESCENDING)]).limit(limit))
            pending = {a["_id"] for a in self._pending(db, session, actions)}
            return actions, self._closed_runs(db, session, actions), pending

        actions, closed, pending = self._run(read)
        records = []
        for action in actions:
            record = {"action_id": action["_id"], "run_closed": closed.get(action["run_id"], False),
                      "recovery_required": action["_id"] in pending,
                      **{key: action[key] for key in ("run_id", "tool_call_id", "tool_name", "risky", "status",
                                                      "outcome_unknown", "resolution")}}
            for key in ("started_at", "finished_at", "resolved_at"):
                moment = action[key]
                record[key] = None if moment is None else (
                    moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)).isoformat()
            records.append(inspection_record(record))
        return records

    def resolve(self, action_id, outcome, *, confirm=False):
        """Human assertion, not proof, approval, undo, session repair or a retry."""
        check_resolution(outcome, confirm)

        def write(db, session):
            self._serialize(db, session)
            action = db.lh_journal_actions.find_one({"_id": action_id, "journal_id": self.name}, session=session)
            if action is None or not self._pending(db, session, [action]):
                raise ValueError("action does not require recovery")
            db.lh_journal_actions.update_one({"_id": action_id}, {"$set": {"resolution": outcome,
                                                                          "resolved_at": _now()}}, session=session)

        self._run(write)


def main(argv=None):
    storage_main(argv, description=__doc__, backend="MongoDB", env_option="--uri-env",
                 location_option="--database", default_location="little_harness",
                 session_store=lambda name, env, database: MongoSessionStore(name, uri_env=env, database=database),
                 journal=lambda name, env, database: MongoJournal(name, uri_env=env, database=database))


if __name__ == "__main__":
    main()

