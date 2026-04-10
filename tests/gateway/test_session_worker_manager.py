from __future__ import annotations

import io
from pathlib import Path

from gateway.session_worker_manager import SessionWorkerManager


class _FakeProcess:
    def __init__(self, *, stdout_lines=None):
        self.stdin = io.StringIO()
        text = "".join((stdout_lines or []))
        self.stdout = io.StringIO(text)
        self.stderr = io.StringIO("")
        self.terminated = False

    def terminate(self):
        self.terminated = True


def test_build_command_points_at_worker_entrypoint():
    manager = SessionWorkerManager(python_executable="python-test")
    assert manager.build_command() == ["python-test", "-m", "gateway.session_worker_process"]


def test_spawn_sends_initial_request_and_reads_events(monkeypatch, tmp_path):
    captured = {}
    fake_process = _FakeProcess(
        stdout_lines=[
            '{"event":"message.delta","run_id":"run_123","timestamp":1.0,"delta":"hello"}\n'
        ]
    )

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return fake_process

    monkeypatch.setattr("subprocess.Popen", _fake_popen)

    manager = SessionWorkerManager(python_executable="python-test")
    handle = manager.spawn(
        run_id="run_123",
        request_payload={"input": "hi"},
        hermes_home=tmp_path / "hermes-home",
        cwd=tmp_path,
    )

    event = handle.poll_event(timeout=1)
    assert event["event"] == "message.delta"
    assert event["delta"] == "hello"
    assert captured["cmd"] == ["python-test", "-m", "gateway.session_worker_process"]
    assert captured["kwargs"]["cwd"] == str(Path(tmp_path).resolve())
    assert captured["kwargs"]["env"]["HERMES_HOME"] == str((tmp_path / "hermes-home").resolve())
    stdin_payload = fake_process.stdin.getvalue().strip()
    assert '"type": "run.request"' in stdin_payload
    assert '"run_id": "run_123"' in stdin_payload


def test_submit_clarify_response_and_cancel_write_control_messages(monkeypatch, tmp_path):
    fake_process = _FakeProcess()
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: fake_process)

    manager = SessionWorkerManager()
    handle = manager.spawn(run_id="run_123", request_payload={"input": "hi"}, cwd=tmp_path)
    handle.submit_clarify_response("yes")
    handle.cancel()

    data = fake_process.stdin.getvalue()
    assert '"type": "clarify.response"' in data
    assert '"response": "yes"' in data
    assert '"type": "control.cancel"' in data
    assert fake_process.terminated is True
