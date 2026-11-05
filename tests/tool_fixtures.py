"""Importable, offline-only tools for exercising disposable workers."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path


LOCAL_STATE = []


def echo(value: str):
    print("worker-noise-must-not-leak", flush=True)
    os.write(1, b"native-noise-must-not-leak\n")
    os.write(2, b"stderr-noise-must-not-leak\n")
    return value


def large_output():
    return "x" * 20000


def fail():
    raise RuntimeError("exception-detail-must-not-leak")


def change_memory():
    LOCAL_STATE.append("worker only")
    return len(LOCAL_STATE)


def read_env(name: str):
    return os.environ.get(name, "<unset>")


def record_action(path: str, mode: str = "done"):
    with Path(path).open("a", encoding="utf-8") as target:
        target.write(mode + "\n")
    if mode == "hang":
        time.sleep(60)
        with Path(path).open("a", encoding="utf-8") as target:
            target.write("late\n")
    elif mode == "fail":
        raise RuntimeError("exception-detail-must-not-leak")
    elif mode == "crash":
        os._exit(7)
    return "recorded " + mode


def stubborn_worker(path: str):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    Path(path).write_text(str(os.getpid()), encoding="utf-8")
    time.sleep(60)


def spawn_child(path: str, mode: str):
    child = subprocess.Popen([sys.executable, "-c", "import signal,time; "
                              "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"])
    Path(path).write_text(str(child.pid), encoding="utf-8")
    if mode == "hang":
        time.sleep(60)
    elif mode == "crash":
        os._exit(7)
    return "spawned"