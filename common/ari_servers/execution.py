"""No concurrent executions: one request at a time among all the servers (Whisper, LLM, RASA) of the machine.

The servers are different processes, often in different containers: they share a lock file (flock) in
ari_servers/run/, mounted in every container. A request waits for the lock, runs, and releases it; the time
waited is returned to the server, that reports it (queue_s). The lock is released also if the process dies.
An empty path disables it (each server then runs its requests one at a time only among its own threads).

Do not take it twice in the same chain: a server that, holding the lock, calls another server that takes it too,
waits for itself (e.g. RASA proxy -> Rasa -> custom action -> LLM server).
"""
import fcntl
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path


class ExclusiveExecution:

    def __init__(self, lock_path):
        self.path = Path(lock_path) if lock_path else None
        self._threads = threading.Lock()  # the threads of this process, flock is per open file
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def hold(self):
        """with execution.hold() as waited_s: ... (the work that must not overlap with the other servers)"""
        start = time.perf_counter()
        with self._threads:
            if self.path is None:
                yield time.perf_counter() - start
                return
            # read-only is enough for flock: it works even if another user (root in a container) created the file
            fd = os.open(self.path, os.O_RDONLY | os.O_CREAT, 0o666)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                yield time.perf_counter() - start
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

    def describe(self) -> str:
        return str(self.path) if self.path else "disabled (only within this server)"


def from_config() -> ExclusiveExecution:
    """The lock of servers.yaml (execution_lock, relative to WORKSPACE), or of ARI_EXECUTION_LOCK; empty: disabled."""
    from .config import shared, workspace_path
    path = shared("execution_lock", "run/execution.lock", env="ARI_EXECUTION_LOCK")
    return ExclusiveExecution(workspace_path(path) if path else None)
