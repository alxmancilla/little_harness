"""Small local storage latency probe. Uses and removes uniquely named test storage.

Set both LITTLE_HARNESS_TEST_POSTGRES_DSN and LITTLE_HARNESS_TEST_MONGODB_URI
to disposable test services; install both optional extras before running.
"""

import json
import math
import os
import platform
import statistics
import sys
import time
from uuid import uuid4

from stores import mongodb, postgres


POSTGRES_ENV = "LITTLE_HARNESS_TEST_POSTGRES_DSN"
MONGO_ENV = "LITTLE_HARNESS_TEST_MONGODB_URI"
WARMUP, SAMPLES, RECORDS = 10, 100, 100


def measure(session, journal, memory):
    session.prepare()
    journal.assert_ready()
    memory.prepare()
    messages = [{"role": "user", "content": f"message {i}"} for i in range(10)]
    session.save(messages, None, durable=True)
    for i in range(RECORDS):
        text = f"{'matching' if i < 10 else 'other'} memory item {i}"
        memory.add("benchmark", text, kind="note", created_by="application")

    durations = {key: [] for key in ("save", "begin", "search")}
    for i in range(WARMUP + SAMPLES):
        messages[-1] = {"role": "user", "content": f"changed {i}"}
        start = time.perf_counter()
        session.save(messages, None, durable=True)
        saved = time.perf_counter()
        action = journal.begin("run", f"call-{i}", "benchmark", True)
        begun = time.perf_counter()
        journal.finish(action, "completed")
        start_search = time.perf_counter()
        found = memory.search("benchmark", "matching", limit=5)
        searched = time.perf_counter()
        if len(found) != 5:
            raise RuntimeError("unexpected search results")
        if i >= WARMUP:
            durations["save"].append((saved - start) * 1000)
            durations["begin"].append((begun - saved) * 1000)
            durations["search"].append((searched - start_search) * 1000)
    journal.close_run("run")
    return {name: {"median_ms": round(statistics.median(values), 2),
                   "p95_ms": round(sorted(values)[math.ceil(len(values) * 0.95) - 1], 2)}
            for name, values in durations.items()}


def postgres_probe(name):
    created = False
    try:
        with postgres.psycopg.connect(os.environ[POSTGRES_ENV]) as connection:
            connection.execute(postgres.sql.SQL("CREATE SCHEMA {}").format(postgres.sql.Identifier(name)))
        created = True
        options = {"dsn_env": POSTGRES_ENV, "schema": name}
        return measure(postgres.PostgresSessionStore("session", **options),
                       postgres.PostgresJournal("journal", **options),
                       postgres.PostgresMemoryStore("memory", **options))
    finally:
        if created:
            with postgres.psycopg.connect(os.environ[POSTGRES_ENV]) as connection:
                connection.execute(postgres.sql.SQL("DROP SCHEMA {} CASCADE").format(postgres.sql.Identifier(name)))


def mongo_probe(name):
    client = mongodb.MongoClient(os.environ[MONGO_ENV], serverSelectionTimeoutMS=10000, tz_aware=True)
    created = False
    try:
        if name in client.list_database_names():
            raise RuntimeError("test database already exists")
        client[name].create_collection("benchmark_marker")
        created = True
        options = {"client": client, "database": name}
        return measure(mongodb.MongoSessionStore("session", **options),
                       mongodb.MongoJournal("journal", **options),
                       mongodb.MongoMemoryStore("memory", **options))
    finally:
        try:
            if created:
                client.drop_database(name)
        finally:
            client.close()


def main():
    if postgres.psycopg is None or mongodb.pymongo is None or not all(os.environ.get(e) for e in (POSTGRES_ENV, MONGO_ENV)):
        print("Install both storage extras and set both LITTLE_HARNESS_TEST connection variables.", file=sys.stderr)
        return 2
    name = "lh_bench_" + uuid4().hex
    try:
        results = {"python": platform.python_version(), "platform": platform.system(),
                   "architecture": platform.machine(), "warmup": WARMUP, "samples": SAMPLES,
                   "memory_records": RECORDS, "postgres": postgres_probe(name), "mongodb": mongo_probe(name)}
    except Exception:
        print("Benchmark or cleanup failed; check test services and permissions. "
              f"Temporary storage name: {name}", file=sys.stderr)
        return 1
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())