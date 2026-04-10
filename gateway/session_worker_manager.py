from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from gateway.session_worker_protocol import decode_event_line


@dataclass
class WorkerProcessHandle:
    run_id: str
    process: subprocess.Popen
    command: list[str]
    cwd: str
    env: dict[str, str]
    started_at: float = field(default_factory=time.time)
    _events: "queue.Queue[dict[str, Any]]" = field(default_factory=queue.Queue, repr=False)
    _stderr: list[str] = field(default_factory=list, repr=False)
    _reader_thread: Optional[threading.Thread] = field(default=None, repr=False)

    def start_reader(self) -> None:
        if self._reader_thread is not None:
            return
        reader = threading.Thread(target=self._reader_loop, daemon=True, name=f"session-worker-{self.run_id}")
        self._reader_thread = reader
        reader.start()

    def _reader_loop(self) -> None:
        stdout = getattr(self.process, "stdout", None)
        if stdout is None:
            return
        for raw_line in iter(stdout.readline, ""):
            line = raw_line.strip()
            if not line:
                continue
            event = decode_event_line(line)
            self._events.put(event)

    def send_message(self, message: dict[str, Any]) -> None:
        stdin = getattr(self.process, "stdin", None)
        if stdin is None:
            raise RuntimeError("Worker stdin is unavailable")
        stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        stdin.flush()

    def submit_clarify_response(self, response_text: str) -> None:
        self.send_message({"type": "clarify.response", "response": str(response_text)})

    def cancel(self) -> None:
        try:
            self.send_message({"type": "control.cancel"})
        except Exception:
            pass
        try:
            self.process.terminate()
        except Exception:
            pass

    def poll_event(self, timeout: float | None = None) -> dict[str, Any]:
        return self._events.get(timeout=timeout)


class SessionWorkerManager:
    """Scaffold manager for future per-session worker-process runtime."""

    module_name = "gateway.session_worker_process"

    def __init__(self, *, python_executable: str | None = None):
        self.python_executable = python_executable or sys.executable

    def build_command(self) -> list[str]:
        return [self.python_executable, "-m", self.module_name]

    def build_env(self, *, base_env: Optional[dict[str, str]] = None, extra_env: Optional[dict[str, str]] = None) -> dict[str, str]:
        env = dict(base_env or os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        if extra_env:
            env.update({str(k): str(v) for k, v in extra_env.items()})
        return env

    def spawn(
        self,
        *,
        run_id: str,
        request_payload: dict[str, Any],
        hermes_home: str | Path | None = None,
        cwd: str | Path | None = None,
        extra_env: Optional[dict[str, str]] = None,
    ) -> WorkerProcessHandle:
        env = self.build_env(extra_env=extra_env)
        if hermes_home is not None:
            env["HERMES_HOME"] = str(Path(hermes_home).expanduser().resolve())

        working_dir = str(Path(cwd or Path.cwd()).resolve())
        proc = subprocess.Popen(
            self.build_command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=working_dir,
            env=env,
            bufsize=1,
        )
        handle = WorkerProcessHandle(
            run_id=run_id,
            process=proc,
            command=self.build_command(),
            cwd=working_dir,
            env=env,
        )
        handle.start_reader()
        handle.send_message({"type": "run.request", "run_id": run_id, "payload": request_payload})
        return handle
