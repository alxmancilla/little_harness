"""Start the local test databases and run the storage tests against them.

Mirrors .github/workflows/storage.yml: docker compose up, the service check, the
storage suites (failing if any skip for a missing database or driver; by-design
capability skips are allowed), then the full suite. Connection variables keep
any value already set; otherwise they use the compose.yaml defaults, honouring
LITTLE_HARNESS_POSTGRES_PORT and LITTLE_HARNESS_MONGODB_PORT from the shell or .env
(only these two are read from .env, as Docker Compose does). Needs Docker and the
optional extras: uv sync --extra postgres --extra mongodb
Run: uv run python -m tools.run_storage_tests [--no-start] [--storage-only] [--down]
"""

import argparse
import os
import subprocess
import sys
import unittest

from agents.teaching.config import load_env
from tools import check_storage_services

SUITES = ["tests.test_storage_conformance", "tests.test_storage_postgres",
          "tests.test_storage_mongodb", "tests.test_storage_memory_workers"]


def default_environment(environ=os.environ, env_file=".env"):
    # Compose reads host ports from .env, so the derived URLs must use the same values.
    file_values = {}
    load_env(env_file, file_values)
    ports = {name: environ.get(name) or file_values.get(name) or default for name, default in
             (("LITTLE_HARNESS_POSTGRES_PORT", "55432"), ("LITTLE_HARNESS_MONGODB_PORT", "57017"))}
    environ.setdefault(check_storage_services.POSTGRES_ENV,
                       f"postgresql://postgres@127.0.0.1:{ports['LITTLE_HARNESS_POSTGRES_PORT']}/postgres")
    environ.setdefault(check_storage_services.MONGODB_ENV,
                       f"mongodb://127.0.0.1:{ports['LITTLE_HARNESS_MONGODB_PORT']}/?directConnection=true")


def compose(*args):
    try:
        return subprocess.run(["docker", "compose", *args], check=False).returncode
    except FileNotFoundError:
        print("docker was not found; install Docker or start the databases and use --no-start")
        return 1


def run_tests(names=None):
    loader = unittest.defaultTestLoader
    suite = loader.loadTestsFromNames(names) if names else loader.discover(".")
    return unittest.TextTestRunner(verbosity=1).run(suite)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-start", action="store_true", help="use databases that are already running")
    parser.add_argument("--storage-only", action="store_true", help="skip the full suite")
    parser.add_argument("--down", action="store_true", help="stop the databases afterwards")
    args = parser.parse_args(argv)

    if not args.no_start and compose("up", "-d", "--wait") != 0:
        return 1
    try:
        default_environment()
        if check_storage_services.main() != 0:
            return 1
        result = run_tests(SUITES)
        if not result.wasSuccessful():
            return 1
        missing = [reason for _, reason in result.skipped if reason.startswith(("set ", "install the "))]
        if missing:
            print(f"error: {len(missing)} storage tests skipped for a missing database or driver")
            return 1
        if not args.storage_only and not run_tests().wasSuccessful():
            return 1
        return 0
    finally:
        if args.down:
            compose("down")


if __name__ == "__main__":
    sys.exit(main())
