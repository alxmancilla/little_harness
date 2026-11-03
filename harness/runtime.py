"""Per-task execution limits and results; no provider or application dependencies."""

import math
import time
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class RunLimits:
    """Optional caps. Time/tokens are checked between operations, not preemptively."""

    max_model_calls: Optional[int] = None
    max_tool_calls: Optional[int] = None
    max_tokens: Optional[int] = None
    max_seconds: Optional[float] = None

    def __post_init__(self):
        for name in ("max_model_calls", "max_tool_calls", "max_tokens"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer or None")
        value = self.max_seconds
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise ValueError("max_seconds must be a finite nonnegative number or None")

    @classmethod
    def from_env(cls, env):
        """Read only the CLI's four optional limit settings; never echo bad values."""
        values = {}
        try:
            for name in ("max_model_calls", "max_tool_calls", "max_tokens", "max_seconds"):
                raw = env.get("HARNESS_" + name.upper())
                if raw is not None and raw.strip():
                    values[name] = (float if name == "max_seconds" else int)(raw)
            return cls(**values)
        except (ValueError, TypeError, AttributeError, OverflowError):
            raise ValueError("invalid execution limit settings") from None


@dataclass(frozen=True)
class RunUsage:
    model_calls: int
    tool_calls_requested: int
    tool_calls_attempted: int
    tool_errors: int
    input_tokens: Optional[int]
    output_tokens: Optional[int]

    @property
    def total_tokens(self):
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class RunResult:
    """answer is model text; text includes CLI warnings/stop messages."""

    status: str
    answer: str
    stop_reason: str
    usage: RunUsage
    elapsed_seconds: float
    text: str

    def to_dict(self):
        """Return a JSON-serializable snapshot, without conversation or tool contents."""
        return asdict(self)


class BudgetExceeded(Exception):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


class ExecutionBudget:
    """One counter set for main calls, summaries and tool attempts within a task.

    Model calls count even on failure; HTTP retries are inside a logical call.
    Token totals are trustworthy only when every call reports both usage fields.
    """

    def __init__(self, limits):
        self.limits = limits
        self.started = time.monotonic()
        self.model_calls = self.tool_calls_requested = self.tool_calls_attempted = self.tool_errors = 0
        self._tokens = {"input_tokens": 0, "output_tokens": 0}
        self._complete = {"input_tokens": True, "output_tokens": True}

    @property
    def elapsed_seconds(self):
        return time.monotonic() - self.started

    @property
    def remaining_seconds(self):
        if self.limits.max_seconds is None:
            return None
        return max(0, self.limits.max_seconds - self.elapsed_seconds)

    def snapshot(self):
        return RunUsage(self.model_calls, self.tool_calls_requested, self.tool_calls_attempted,
                        self.tool_errors, **{name: value if self._complete[name] else None
                                             for name, value in self._tokens.items()})

    def check(self, operation=None):
        if self.limits.max_seconds is not None and self.elapsed_seconds >= self.limits.max_seconds:
            raise BudgetExceeded("max_seconds")
        if self.limits.max_tokens is not None:
            total = self.snapshot().total_tokens
            if total is None:
                raise BudgetExceeded("usage_unavailable")
            # A finished answer may use the exact cap; no further work may start.
            if total > self.limits.max_tokens or (operation is not None and total >= self.limits.max_tokens):
                raise BudgetExceeded("max_total_tokens")
        if operation == "model" and self.limits.max_model_calls is not None:
            if self.model_calls >= self.limits.max_model_calls:
                raise BudgetExceeded("max_model_calls")
        if operation == "tool" and self.limits.max_tool_calls is not None:
            if self.tool_calls_attempted >= self.limits.max_tool_calls:
                raise BudgetExceeded("max_tool_calls")

    def start_model_call(self):
        self.check("model")
        self.model_calls += 1

    def record_model_usage(self, usage):
        usage = usage if isinstance(usage, dict) else {}
        for name in self._tokens:
            value = usage.get(name)
            if type(value) is int and value >= 0:
                self._tokens[name] += value
            else:
                self._complete[name] = False

    def start_tool_call(self):
        self.check("tool")
        self.tool_calls_attempted += 1

    def record_tool_results(self, results):
        self.tool_errors += sum(bool(result["is_error"]) for result in results)