"""Optional grouped settings for Agent; each mirrors a set of individual keyword arguments.

A group replaces its keyword arguments: Agent rejects a group combined with any of
those arguments set to a non-default value. Validation still happens in Agent.
"""

from dataclasses import dataclass, fields
from typing import Any, Callable, Mapping, Optional


@dataclass(frozen=True)
class ObservabilityConfig:
    """Trace file, event callback and console output."""

    trace_path: Any = "trace.jsonl"
    trace_context: Optional[Mapping] = None
    trace_include_content: bool = False
    on_event: Optional[Callable] = None
    event_include_content: bool = False
    quiet: bool = False


@dataclass(frozen=True)
class ToolConfig:
    """Gates and limits for tool requests; the tool mapping itself stays `tools=`.

    approve=None selects the default terminal approval (ask_human).
    """

    approve: Optional[Callable] = None
    tool_policy: Any = None
    tool_admission: Optional[Callable] = None
    max_tool_output: int = 8000
    tool_timeout_seconds: Optional[float] = None
    writable_roots: tuple = ()
    worker_env: Optional[Mapping] = None


@dataclass(frozen=True)
class ContextConfig:
    """Conversation persistence and compaction thresholds."""

    compact_at: int = 40000
    compact_at_tokens: Optional[int] = None
    session_path: Any = None
    session_store: Any = None


def merge_group(name, group, group_type, given, defaults):
    """Return keyword values taken from group, or given unchanged when group is None.

    defaults maps each keyword to Agent's default; a group field of None takes the
    Agent default (this only changes approve, whose default is not None).
    """
    if group is None:
        return given
    if not isinstance(group, group_type):
        raise TypeError(f"{name} must be {group_type.__name__} or None")
    conflicts = sorted(key for key, value in given.items()
                       if value is not defaults[key] and value != defaults[key])
    if conflicts:
        raise TypeError(f"{name} cannot be combined with {', '.join(conflicts)}")
    values = {f.name: getattr(group, f.name) for f in fields(group_type)}
    return {key: defaults[key] if value is None else value for key, value in values.items()}
