"""Opt-in, disposable tool workers. Process isolation is NOT a security sandbox."""

import importlib
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import FunctionType


def validate_tool_timeout(value):
    if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
        raise ValueError("tool timeout must be a finite positive number or None")
    return value


@dataclass(frozen=True)
class ToolOutcome:
    status: str
    output: str
    is_error: bool
    outcome_unknown: bool = False


class ToolExecutionStopped(Exception):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def _descriptor(fn):
    """Do not pickle callables or fall back to executing unsupported ones inline."""
    bound_args, bound_kwargs = (), {}
    if type(fn) is partial:
        bound_args, bound_kwargs, fn = fn.args, fn.keywords, fn.func
    if not isinstance(fn, FunctionType) or fn.__qualname__ != fn.__name__:
        raise ValueError("timed tools require an importable module-level function")
    module = sys.modules.get(fn.__module__)
    if module is None or getattr(module, fn.__name__, None) is not fn:
        raise ValueError("timed tools require an importable module-level function")
    main_path = None
    if fn.__module__ == "__main__":
        path = getattr(module, "__file__", None)
        if not path or not Path(path).is_file():
            raise ValueError("timed tools defined interactively cannot be imported")
        main_path = str(Path(path).resolve())
    return {"module": fn.__module__, "function": fn.__name__, "main_path": main_path,
            "bound_args": bound_args, "bound_kwargs": bound_kwargs}


def _stop_worker(process):
    """Best-effort bounded cleanup; never mask cancellation or an uncertain result."""
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        elif process.poll() is None:
            process.terminate()
    except OSError:
        pass
    try:
        process.wait(timeout=0.2)
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        elif process.poll() is None:
            process.kill()
    except OSError:
        pass
    try:
        process.wait(timeout=1)
    except (OSError, subprocess.TimeoutExpired):
        pass


def run_isolated_tool(fn, args, timeout_seconds, max_output, *, risky=False):
    """Return a bounded result, never retry. Tool arguments travel over stdin, not argv."""
    validate_tool_timeout(timeout_seconds)
    if timeout_seconds is None:
        raise ValueError("isolated execution requires a positive timeout")
    started = time.monotonic()
    try:
        payload = {**_descriptor(fn), "args": args, "max_output": max_output,
                   "risky": risky, "sys_path": [str(path) for path in sys.path]}
        data = json.dumps(payload, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError):
        return ToolOutcome("failed", "timed tool requires an importable module-level function and JSON arguments", True)
    process = None
    try:
        env = dict(os.environ)
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve())],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   env=env, start_new_session=os.name == "posix")
        remaining = max(0, timeout_seconds - (time.monotonic() - started))
        try:
            output, _ = process.communicate(data, timeout=remaining)
        except subprocess.TimeoutExpired:
            return ToolOutcome("timed_out", "tool timed out; outcome unknown; inspect effects before retrying. "
                               "Do not retry automatically.", True, True)
        if process.returncode == 0:
            try:
                result = json.loads(output)
                if (result["status"] in ("completed", "failed") and type(result["output"]) is str
                        and type(result["is_error"]) is bool and type(result["outcome_unknown"]) is bool
                        and len(result["output"]) <= max_output + 100
                        and result["is_error"] == (result["status"] == "failed")
                        and (not result["outcome_unknown"] or result["is_error"])):
                    return ToolOutcome(**result)
            except (ValueError, KeyError, TypeError):
                pass
        return ToolOutcome("outcome_unknown", "tool worker exited without a trustworthy result; outcome unknown; "
                           "inspect effects before retrying. Do not retry automatically.", True, True)
    except OSError:
        if process is None:
            return ToolOutcome("failed", "tool worker could not start; action was not run", True)
        return ToolOutcome("outcome_unknown", "tool worker connection failed; outcome unknown; "
                           "inspect effects before retrying. Do not retry automatically.", True, True)
    finally:
        if process is not None:
            # Also clean up children left behind by a successful or crashed worker.
            _stop_worker(process)
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()


def _worker():
    # Keep a private protocol descriptor. Python/native stdout/stderr from imports
    # and tools go to /dev/null, never into the parent trace or an unbounded pipe.
    protocol_fd = os.dup(sys.stdout.fileno())
    with open(os.devnull, "w") as discard:
        os.dup2(discard.fileno(), 1)
        os.dup2(discard.fileno(), 2)
    payload = json.load(sys.stdin)
    sys.path[:] = payload["sys_path"]
    started = False
    try:
        if payload["main_path"]:
            spec = importlib.util.spec_from_file_location("_harness_tool_main", payload["main_path"])
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module(payload["module"])
        fn = getattr(module, payload["function"])
        if not isinstance(fn, FunctionType) or fn.__qualname__ != fn.__name__:
            raise ValueError("worker function is not module-level")
        started = True
        output = str(fn(*payload.get("bound_args", ()),
                        **{**payload.get("bound_kwargs", {}), **payload["args"]}))
        if len(output) > payload["max_output"]:
            output = output[:payload["max_output"]] + f"\n[truncated, {len(output)} characters total]"
        result = ToolOutcome("completed", output, False)
    except Exception:
        uncertain = started and payload["risky"]
        message = "tool execution failed" if started else "tool worker could not import the function; action was not run"
        if uncertain:
            message += "; outcome unknown; inspect effects before retrying. Do not retry automatically."
        result = ToolOutcome("failed", message, True, uncertain)
    with os.fdopen(protocol_fd, "w", encoding="utf-8") as protocol:
        json.dump(result.__dict__, protocol)


if __name__ == "__main__":
    _worker()