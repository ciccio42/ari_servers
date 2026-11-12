"""Latency log of the servers, one CSV line per event (same format of ari_demo).

Columns: epoch_ns, monotonic_ns, host, pid, stage, event, extra (k=v;k=v)

- Files: <LATENCY_LOG_DIR>/latency_<date>_<host>.csv; default directory: $WORKSPACE/latency_logs
- LATENCY_LOG=False disables it. A logging error never stops the caller.
- epoch_ns compares events of different machines only with synchronized clocks;
  the durations measured on one machine (the *_s values in extra) do not need it.
"""
import os
import socket
import threading
import time
from datetime import date
from pathlib import Path

ENABLED = os.environ.get("LATENCY_LOG", "True").lower() != "false"
LOG_DIR = Path(os.environ.get("LATENCY_LOG_DIR", Path(os.environ.get("WORKSPACE", "."), "latency_logs")))
HOST = socket.gethostname()

_lock = threading.Lock()


def _clean(value) -> str:
    # commas, semicolons and new lines would break the CSV
    return str(value).replace(",", " ").replace(";", " ").replace("\n", " ")


def mark(stage: str, event: str, **extra) -> None:
    """Writes the event `stage.event` with the current time. extra: values to keep (audio length, words, ...)."""
    if not ENABLED:
        return
    epoch, mono = time.time_ns(), time.monotonic_ns()
    fields = ";".join(f"{k}={_clean(v)}" for k, v in extra.items())
    line = f"{epoch},{mono},{HOST},{os.getpid()},{stage},{event},{fields}\n"
    try:
        with _lock:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            with open(LOG_DIR / f"latency_{date.today().isoformat()}_{HOST}.csv", "a") as f:
                f.write(line)
    except Exception:
        pass
