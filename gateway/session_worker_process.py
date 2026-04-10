from __future__ import annotations

import json
import sys
from typing import Any

from gateway.session_worker_protocol import encode_event_line, make_event


"""Temporary scaffold entrypoint for future session worker processes.

This module is intentionally minimal for the Stage 1/2 substrate work.
It accepts one JSON message on stdin and emits a single structured failure event
explaining that the real worker runtime is not wired yet.
"""


def main() -> int:
    raw = sys.stdin.readline()
    if not raw:
        return 1
    try:
        message: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.stdout.write(
            encode_event_line(
                make_event("run.failed", run_id="unknown", error=f"invalid worker payload: {exc}")
            )
            + "\n"
        )
        sys.stdout.flush()
        return 1

    run_id = str(message.get("run_id") or "unknown")
    sys.stdout.write(
        encode_event_line(
            make_event(
                "run.failed",
                run_id=run_id,
                error="session worker scaffold is not wired into the runtime yet",
            )
        )
        + "\n"
    )
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
