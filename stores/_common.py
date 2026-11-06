"""Rules and the command line shared by every database adapter."""

import argparse
import re

from harness.journal import JournalError, run_recovery_command
from harness.memory import MemoryStoreError
from harness.session import SessionStoreError

IDENTIFIER = re.compile(r"[A-Za-z0-9._:-]{1,200}")
APPLICATION, SCHEMA_VERSION = "little-harness", 1


def check_id(name):
    if type(name) is not str or not IDENTIFIER.fullmatch(name):
        raise ValueError("IDs must be 1-200 characters from A-Z, a-z, 0-9, '.', '_', ':' and '-'")


def storage_main(argv, *, description, backend, env_option, location_option, default_location,
                 session_store, journal, memory_store):
    """init, journal inspect and journal resolve; connection settings only by variable name."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(env_option, dest="env", required=True,
                        help="name of the variable holding the connection string, never the string")
    common.add_argument(location_option, dest="location", default=default_location)
    journal_id = argparse.ArgumentParser(add_help=False)
    journal_id.add_argument("--journal-id", required=True)
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    areas = parser.add_subparsers(dest="area", required=True)
    areas.add_parser("init", parents=[common], help="create or check session, journal and memory storage")
    commands = areas.add_parser("journal", help="inspect or resolve journal records; never runs an action") \
        .add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", parents=[common, journal_id], help="read-only metadata")
    inspect.add_argument("--pending", action="store_true")
    inspect.add_argument("--limit", type=int, default=100)
    resolve = commands.add_parser("resolve", parents=[common, journal_id], help="record human recovery only")
    resolve.add_argument("action_id")
    resolve.add_argument("--outcome", required=True, choices=("completed", "failed", "not_run"))
    resolve.add_argument("--confirm", action="store_true", required=True)
    args = parser.parse_args(argv)
    try:
        if args.area == "init":
            for storage in (session_store, journal, memory_store):
                storage("init", args.env, args.location).ensure_schema()
            print(f"{backend} storage {args.location} is ready for sessions, journals and memory.")
        else:
            run_recovery_command(journal(args.journal_id, args.env, args.location), args)
    except (JournalError, MemoryStoreError, SessionStoreError, ValueError, TypeError, ImportError):
        parser.exit(1, f"{backend} storage operation failed; check the variable name, location, journal ID "
                       "and recovery options.\n")
