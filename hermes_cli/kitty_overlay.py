from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Mapping, Sequence

DEFAULT_TIMEOUT_RESPONSE = (
    "The user did not provide a response within the time limit. "
    "Use your best judgement to make the choice and proceed."
)


def kitty_overlay_available(env: Mapping[str, str] | None = None) -> bool:
    env_map = env or os.environ
    return bool(
        env_map.get("KITTY_LISTEN_ON")
        and env_map.get("KITTY_WINDOW_ID")
        and shutil.which("kitten")
    )



def build_overlay_launch_command(
    *,
    listen_on: str,
    window_id: str,
    python_executable: str,
    script_path: str,
    spec_path: str,
) -> list[str]:
    return [
        "kitten",
        "@",
        "--to",
        listen_on,
        "launch",
        "--match",
        f"window_id:{window_id}",
        "--type=overlay",
        "--cwd",
        "current",
        python_executable,
        script_path,
        "--spec",
        spec_path,
    ]



def _wait_for_overlay_response(response_path: Path, timeout: int) -> str:
    deadline = time.monotonic() + max(1, timeout) + 5
    while time.monotonic() < deadline:
        if response_path.exists():
            try:
                payload = json.loads(response_path.read_text(encoding="utf-8"))
            except Exception:
                return DEFAULT_TIMEOUT_RESPONSE
            response = str(payload.get("response") or "").strip()
            return response or DEFAULT_TIMEOUT_RESPONSE
        time.sleep(0.2)
    return DEFAULT_TIMEOUT_RESPONSE



def prompt_kitty_overlay_clarify(
    question: str,
    choices: Sequence[str] | None,
    *,
    task_label: str = "background task",
    timeout: int = 120,
    env: Mapping[str, str] | None = None,
    python_executable: str | None = None,
) -> str:
    env_map = dict(env or os.environ)
    if not kitty_overlay_available(env_map):
        raise RuntimeError("Kitty overlay remote-control environment is not available.")

    listen_on = str(env_map["KITTY_LISTEN_ON"])
    window_id = str(env_map["KITTY_WINDOW_ID"])
    python_exec = python_executable or sys.executable or shutil.which("python3") or "python3"
    script_path = str(Path(__file__).with_name("kitty_overlay_prompt.py"))

    with tempfile.TemporaryDirectory(prefix="hermes-kitty-overlay-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        spec_path = tmpdir_path / "spec.json"
        response_path = tmpdir_path / "response.json"
        spec = {
            "task_label": task_label,
            "question": question,
            "choices": list(choices or []),
            "timeout": int(timeout),
            "response_path": str(response_path),
        }
        spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")

        command = build_overlay_launch_command(
            listen_on=listen_on,
            window_id=window_id,
            python_executable=python_exec,
            script_path=script_path,
            spec_path=str(spec_path),
        )
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "kitty overlay launch failed").strip()
            raise RuntimeError(detail)

        return _wait_for_overlay_response(response_path, int(timeout))
