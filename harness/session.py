"""Session files: full conversation history plus the compaction point. No agent state."""

import json
import os
from pathlib import Path

SESSION_VERSION = 2


class SessionLoadError(ValueError):
    """The session file exists but is not a valid session; it is left unchanged."""


class SessionFile:
    """Load and atomically save {"version": 2, "messages": [...], "compaction": ...}.

    A legacy JSON list of messages still loads, without compaction state.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.tmp_path = self.path.with_name(self.path.name + ".tmp")

    def load(self):
        """Return (messages, compaction); ([], None) when the file does not exist."""
        if not self.path.exists():
            return [], None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
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
            raise SessionLoadError(f"{self.path} is not a valid session ({e}); "
                                   "fix or delete it to start fresh") from None
        return messages, compaction

    def save(self, messages, compaction, *, durable=False):
        """Write via a temporary file; durable saves fsync the file and, on POSIX, its directory."""
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
