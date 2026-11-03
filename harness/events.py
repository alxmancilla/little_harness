"""Immutable, versioned observation events; no application or provider dependency."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError("event data must contain JSON-compatible values")


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class RunEvent:
    """Synchronous observation, not an approval, durable record or execution command."""

    run_id: str
    sequence: int
    kind: str
    ts: str
    data: Mapping
    version: int = 1

    def __post_init__(self):
        object.__setattr__(self, "data", _freeze(self.data))

    def to_dict(self):
        """Return a detached JSON-serializable copy, including opted-in content."""
        return {"version": self.version, "run_id": self.run_id, "sequence": self.sequence,
                "kind": self.kind, "ts": self.ts, "data": _thaw(self.data)}