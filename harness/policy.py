"""Opt-in application-level dispatch policies. This is NOT an OS sandbox."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str


@dataclass(frozen=True)
class ToolPolicy:
    """Explicit read-only declarations plus per-tool argument predicates.

    Predicates get detached JSON snapshots and must return literal True. Their
    errors fail closed without exposing exception text or authorizing a retry.
    """

    read_only: bool = False
    argument_checks: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if type(self.read_only) is not bool:
            raise TypeError("read_only must be a boolean")
        if not isinstance(self.argument_checks, Mapping):
            raise TypeError("argument_checks must map tool names to callables")
        checks = dict(self.argument_checks)
        if any(type(name) is not str or not name or not callable(check) for name, check in checks.items()):
            raise ValueError("argument_checks must map nonempty tool names to callables")
        object.__setattr__(self, "argument_checks", MappingProxyType(checks))

    def validate_tools(self, tools):
        if self.argument_checks.keys() - tools.keys():
            raise ValueError("argument_checks contains a tool outside the agent allowlist")

    @staticmethod
    def snapshot(arguments):
        """Separate executed arguments from policy, approval and conversation mutations."""
        return json.loads(json.dumps(arguments, allow_nan=False))

    def check(self, name, arguments, *, declared_read_only=False):
        if self.read_only and declared_read_only is not True:
            return PolicyDecision(False, "read_only_required")
        predicate = self.argument_checks.get(name)
        if predicate is None:
            return PolicyDecision(True, "allowed")
        try:
            allowed = predicate(self.snapshot(arguments)) is True
        except Exception:
            return PolicyDecision(False, "policy_error")
        return PolicyDecision(allowed, "allowed" if allowed else "argument_rejected")