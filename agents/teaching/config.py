"""Application configuration and opt-in .env loading, never performed on import."""

import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path


def load_env(path=".env", env=None):
    """Load simple KEY=value lines, not shell code. Existing environment values win.
    Quotes, optional 'export' and comments are supported; variable expansion is not."""
    env = os.environ if env is None else env
    path = Path(path)
    if not path.exists():
        return
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, value = line.partition("=")
        name = name.strip()
        try:
            if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ValueError
            parts = shlex.split(value, comments=True)
            if len(parts) > 1:
                raise ValueError
        except ValueError:
            raise ValueError(f"Invalid .env assignment at line {number}") from None
        env.setdefault(name, parts[0] if parts else "")


@dataclass(frozen=True)
class TeachingConfig:
    workspace: Path = Path("workspace")
    knowledge: Path = Path("knowledge")
    compact_at: int = 40000
    max_tool_output: int = 8000

    def __post_init__(self):
        object.__setattr__(self, "workspace", Path(self.workspace).resolve())
        object.__setattr__(self, "knowledge", Path(self.knowledge).resolve())
        if type(self.compact_at) is not int or self.compact_at < 0:
            raise ValueError("compact_at must be a nonnegative integer")
        if type(self.max_tool_output) is not int or self.max_tool_output <= 0:
            raise ValueError("max_tool_output must be a positive integer")

    @classmethod
    def from_env(cls, env):
        """Read only application settings; do not read .env or initialize a workspace."""
        return cls(workspace=env.get("HARNESS_WORKSPACE", "workspace"),
                   knowledge=env.get("HARNESS_KNOWLEDGE", "knowledge"),
                   compact_at=int(env.get("HARNESS_COMPACT_CHARS", "40000")))

    def tool_context(self):
        """JSON-only, application-owned values copied into each tool binding."""
        return {"workspace": str(self.workspace), "knowledge": str(self.knowledge),
                "max_tool_output": self.max_tool_output, "max_knowledge_entries": 100}