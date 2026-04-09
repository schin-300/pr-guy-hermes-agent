#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import select
import sys
import time
from pathlib import Path

DEFAULT_TIMEOUT_RESPONSE = (
    "The user did not provide a response within the time limit. "
    "Use your best judgement to make the choice and proceed."
)



def _read_line_with_timeout(timeout: float) -> str | None:
    remaining = max(0.0, timeout)
    ready, _, _ = select.select([sys.stdin], [], [], remaining)
    if not ready:
        return None
    line = sys.stdin.readline()
    if not line:
        return None
    return line.rstrip("\r\n")



def _write_response(path: Path, response: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"response": response}, ensure_ascii=False), encoding="utf-8")



def _resolve_choice_response(raw: str, choices: list[str], timeout_deadline: float) -> str:
    cleaned = (raw or "").strip()
    if not cleaned:
        return DEFAULT_TIMEOUT_RESPONSE

    other_index = len(choices) + 1
    if cleaned.isdigit():
        numeric = int(cleaned)
        if 1 <= numeric <= len(choices):
            return choices[numeric - 1]
        if numeric == other_index:
            print()
            print("Type your answer and press Enter:")
            follow_up = _read_line_with_timeout(timeout_deadline - time.monotonic())
            cleaned_follow_up = (follow_up or "").strip()
            return cleaned_follow_up or DEFAULT_TIMEOUT_RESPONSE
    return cleaned



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hermes kitty overlay clarify prompt")
    parser.add_argument("--spec", required=True, help="Path to the prompt spec JSON file")
    args = parser.parse_args(argv)

    spec_path = Path(args.spec)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    task_label = str(spec.get("task_label") or "Hermes background task")
    question = str(spec.get("question") or "")
    choices = [str(choice).strip() for choice in (spec.get("choices") or []) if str(choice).strip()]
    timeout = int(spec.get("timeout") or 120)
    response_path = Path(spec["response_path"])
    deadline = time.monotonic() + max(1, timeout)

    print(f"{task_label} needs your input")
    print("=" * max(24, len(task_label) + 18))
    print()
    print(question)
    print()

    if choices:
        for index, choice in enumerate(choices, 1):
            print(f"  {index}. {choice}")
        print(f"  {len(choices) + 1}. Other (type your answer)")
        print()
        print(f"Respond within {timeout}s. Type a number or free-form answer, then press Enter:")
        raw = _read_line_with_timeout(deadline - time.monotonic())
        response = DEFAULT_TIMEOUT_RESPONSE if raw is None else _resolve_choice_response(raw, choices, deadline)
    else:
        print(f"Respond within {timeout}s and press Enter:")
        raw = _read_line_with_timeout(deadline - time.monotonic())
        response = (raw or "").strip() or DEFAULT_TIMEOUT_RESPONSE

    _write_response(response_path, response)
    print()
    print("Thanks — closing overlay.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
