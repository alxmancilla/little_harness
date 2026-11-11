"""Session stores: full conversation history plus the compaction point. No agent state."""

import json
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

SESSION_VERSION = 2


class SessionLoadError(ValueError):
    """The stored session exists but is not a valid session; it is left unchanged."""


class SessionStoreError(OSError):
    """A session could not be read or saved; the message carries no connection details."""


class SessionConflictError(SessionStoreError):
    """Another writer saved this session since it was loaded; nothing was changed."""


@runtime_checkable
class SessionStore(Protocol):
    """One stored session. load() returns None when no session exists yet."""

    def load(self) -> tuple[list, dict | None] | None: ...

    def save(self, messages: list, compaction: dict | None, *, durable: bool) -> None: ...


def validate_session(data, source):
    """Return (messages, compaction) from a decoded version 2 session or legacy list."""
    try:
        if isinstance(data, list):
            messages, compaction = data, None
        elif isinstance(data, dict) and data.get("version") == SESSION_VERSION:
            messages, compaction = data.get("messages"), data.get("compaction")
            if not isinstance(messages, list):
                raise ValueError("expected a list of messages")
        else:
            raise ValueError(f"expected a list of messages or a version {SESSION_VERSION} session")
        if compaction is not None:
            start, summary = (compaction.get("start"), compaction.get("summary")) \
                if isinstance(compaction, dict) else (None, None)
            if (type(start) is not int or not 0 < start <= len(messages)
                    or type(summary) is not str or not summary):
                raise ValueError("invalid compaction state")
            compaction = {"start": start, "summary": summary}
    except ValueError as e:
        raise SessionLoadError(f"{source} is not a valid session ({e}); fix or delete it to start fresh") from None
    return messages, compaction


class SessionFile:
    """Load and atomically save {"version": 2, "messages": [...], "compaction": ...}.

    A legacy JSON list of messages still loads, without compaction state.
    One writer is assumed; unlike database stores, there is no conflict check.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.tmp_path = self.path.with_name(self.path.name + ".tmp")

    def __str__(self):
        return str(self.path)

    def load(self):
        """Return (messages, compaction), or None when the file does not exist."""
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except ValueError as e:
            raise SessionLoadError(f"{self.path} is not a valid session ({e}); "
                                   "fix or delete it to start fresh") from None
        return validate_session(data, self.path)

    def save(self, messages, compaction, *, durable=False):
        """Write via a temporary file; durable saves fsync the file and, on POSIX, its directory."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.tmp_path.open("w", encoding="utf-8") as stream:
            json.dump({"version": SESSION_VERSION, "messages": messages, "compaction": compaction}, stream)
            if durable:
                stream.flush()
                os.fsync(stream.fileno())
        os.replace(self.tmp_path, self.path)
        if durable and os.name == "posix":
            descriptor = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
