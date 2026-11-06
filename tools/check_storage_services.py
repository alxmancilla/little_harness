"""Check that the storage test databases are reachable; print versions only.

Reads connection settings from LITTLE_HARNESS_TEST_POSTGRES_DSN and
LITTLE_HARNESS_TEST_MONGODB_URI, never from arguments, and never prints them or
driver error text. Needs the optional extras: uv sync --extra postgres --extra mongodb
Run: python -m tools.check_storage_services
"""

import os
import sys

POSTGRES_ENV, MONGODB_ENV = "LITTLE_HARNESS_TEST_POSTGRES_DSN", "LITTLE_HARNESS_TEST_MONGODB_URI"
MAJOR_VERSIONS = {"postgres": 18, "mongodb": 9}


def check_postgres(dsn):
    import psycopg

    with psycopg.connect(dsn, connect_timeout=10) as connection:
        version = connection.info.server_version  # e.g. 180001
    return version // 10000, f"{version // 10000}.{version % 100}"


def check_mongodb(uri):
    from pymongo import MongoClient

    with MongoClient(uri, serverSelectionTimeoutMS=10000) as client:
        version = client.server_info()["version"]
        hello = client.admin.command("hello")
    if not hello.get("setName") or not hello.get("isWritablePrimary"):
        raise RuntimeError("not a writable replica set primary; transactions are unavailable")
    return int(version.split(".")[0]), version


def main():
    failed = False
    for name, env, check in (("postgres", POSTGRES_ENV, check_postgres), ("mongodb", MONGODB_ENV, check_mongodb)):
        value = os.environ.get(env)
        if not value:
            print(f"{name}: {env} is not set")
            failed = True
            continue
        try:
            major, version = check(value)
        except ImportError:
            print(f"{name}: driver not installed; run uv sync --extra {name}")
            failed = True
            continue
        except Exception as error:
            print(f"{name}: unreachable or not usable ({type(error).__name__})")
            failed = True
            continue
        expected = MAJOR_VERSIONS[name]
        print(f"{name}: {version}" + ("" if major == expected else f" (expected {expected}.x)"))
        failed |= major != expected
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
