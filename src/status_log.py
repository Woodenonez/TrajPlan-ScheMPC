"""Minimal timestamped status line, shared by the scheduler entry points and run_mpc.

Kept deliberately tiny and dependency-free so it can be imported from any layer
(main.py, the scheduler backends, run_mpc.py) without pulling in project config.
"""
import datetime


def status(msg: str) -> None:
    """Print a single timestamped status line, e.g. '[14:03:21] Scheduler executing'."""
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)
