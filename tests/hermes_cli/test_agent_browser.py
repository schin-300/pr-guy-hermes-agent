from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace

from hermes_cli import agent_browser

try:
    from prompt_toolkit.utils import get_cwidth
except Exception:
    def get_cwidth(text: str) -> int:
        return len(text)


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _write_config(profile_dir: Path, *, model: str = "openrouter/test-model", skin: str = "default") -> None:
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "workspace").mkdir(parents=True, exist_ok=True)
    (profile_dir / "memories").mkdir(parents=True, exist_ok=True)
    (profile_dir / "skills").mkdir(parents=True, exist_ok=True)
    (profile_dir / "config.yaml").write_text(
        f"model:\n  default: {model}\ndisplay:\n  skin: {skin}\n",
        encoding="utf-8",
    )


def test_save_status_writes_current_and_history(tmp_path: Path):
    profile_dir = tmp_path / ".hermes"
    _write_config(profile_dir)

    saved = agent_browser.save_status(profile_dir, {"summary": "hello there."})

    current = agent_browser.load_current_status(profile_dir)
    history_file = agent_browser.history_dir(profile_dir) / f"{saved['id']}.json"

    assert current is not None
    assert current["summary"] == "hello there."
    assert current["id"] == saved["id"]
    assert history_file.exists()


def test_collect_presence_ignores_stale_entries(tmp_path: Path):
    profile_dir = tmp_path / ".hermes"
    _write_config(profile_dir)

    agent_browser.write_presence(
        profile_dir,
        pid=os.getpid(),
        session_id="sess-live",
        profile_name="default",
        busy=True,
        model="test-model",
    )
    stale_file = agent_browser.presence_dir(profile_dir) / "99999.json"
    stale_file.parent.mkdir(parents=True, exist_ok=True)
    stale_file.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "session_id": "old",
                "profile": "default",
                "busy": False,
                "updated_at": time.time() - 999,
            }
        ),
        encoding="utf-8",
    )

    presence = agent_browser.collect_presence(profile_dir)

    assert presence["wired"] is True
    assert presence["busy"] is True
    assert presence["count"] == 1
    assert not stale_file.exists()


def test_build_profile_cards_includes_status_skin_and_wiring(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    default_home = tmp_path / ".hermes"
    alpha_home = tmp_path / ".hermes" / "profiles" / "alpha"
    _write_config(default_home, model="openrouter/default-model", skin="default")
    _write_config(alpha_home, model="openrouter/alpha-model", skin="ares")

    agent_browser.save_status(
        default_home,
        {
            "summary": "Building the native browser now.",
            "feeling": "focused",
            "focus": "ctrl+b browser",
            "message_to_user": "I am still holding the thread warm.",
            "message_to_self": "Keep it native.",
        },
    )
    agent_browser.write_presence(
        default_home,
        pid=os.getpid(),
        session_id="sess-1",
        profile_name="default",
        busy=False,
        model="openrouter/default-model",
    )

    cards = agent_browser.build_profile_cards(current_profile_name="default")
    by_name = {card.name: card for card in cards}

    assert set(by_name) >= {"default", "alpha"}
    assert by_name["default"].wired is True
    assert by_name["default"].summary == "Building the native browser now."
    assert by_name["default"].skin_name == "default"
    assert by_name["alpha"].skin_name == "ares"
    assert by_name["alpha"].model == "alpha-model"


def test_generate_reflective_status_uses_llm_json_when_available(tmp_path: Path, monkeypatch):
    profile_dir = tmp_path / ".hermes"
    _write_config(profile_dir)
    (profile_dir / "memories" / "MEMORY.md").write_text("Remember the operator likes native flows.", encoding="utf-8")
    (profile_dir / "memories" / "USER.md").write_text("User likes live agent cards.", encoding="utf-8")

    fake = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        {
                            "summary": "We are shaping the first native Hermes browser card flow.",
                            "message_to_user": "I stepped aside for a second and kept the whole shape in my hands.",
                            "message_to_self": "Stay grounded and keep it hot.",
                            "feeling": "electric",
                            "focus": "browser cards and ctrl+b",
                        }
                    )
                )
            )
        ]
    )
    monkeypatch.setattr(agent_browser, "call_llm", lambda **kwargs: fake)

    payload = agent_browser.generate_reflective_status(
        profile_dir,
        profile_name="default",
        session_id="sess-xyz",
        conversation_history=[
            {"role": "user", "content": "make the native ctrl+b browser"},
            {"role": "assistant", "content": "working on it now"},
        ],
        wired=True,
    )

    assert payload["session_id"] == "sess-xyz"
    assert payload["summary"].startswith("We are shaping")
    assert payload["message_to_user"].startswith("I stepped aside")
    assert payload["message_to_self"].startswith("Stay grounded")
    assert payload["feeling"] == "electric"
    assert payload["focus"] == "browser cards and ctrl+b"
    assert payload["wired"] is True


def test_render_browser_contains_controls_and_selected_card():
    cards = [
        agent_browser.BrowserCard(
            name="default",
            path=Path("/tmp/default"),
            is_default=True,
            is_current=True,
            wired=True,
            wire_state="wired",
            summary="default summary that should wrap onto a second status line in compact mode.",
            message_to_user="to user",
            message_to_self="to self",
        ),
        agent_browser.BrowserCard(name="alpha", path=Path("/tmp/alpha"), is_default=False, is_current=False, summary="alpha summary.", message_to_user="hi", message_to_self="inner"),
    ]

    raw_rendered = agent_browser.render_browser(cards, selected_index=1, width=90, height=26)
    rendered = _strip_ansi(raw_rendered)

    assert "Hermes Agent Browser" in rendered
    assert "Ctrl+B/Esc close" in rendered
    assert "arrows move" in rendered
    assert "Enter open" in rendered
    assert "alpha" in rendered
    assert "STATUS" in rendered
    assert "MIND" not in rendered
    assert "TRACE" not in rendered
    assert "current" not in rendered
    assert "selected " not in rendered
    assert "to you:" not in rendered
    assert "wired in" in rendered
    assert "38;2;24;119;242m" in raw_rendered


def test_render_browser_prefers_three_up_only():
    cards = [
        agent_browser.BrowserCard(name=f"card-{idx}", path=Path(f"/tmp/{idx}"), is_default=False, is_current=False, summary=f"summary {idx}")
        for idx in range(5)
    ]

    rendered = _strip_ansi(agent_browser.render_browser(cards, selected_index=3, width=140, height=26, columns=2))

    assert "3-up" in rendered
    assert "2/3 layout" not in rendered
    assert "press 2 or 3" not in rendered
    assert "card-3" in rendered


def test_render_browser_gives_status_two_lines_of_room():
    cards = [
        agent_browser.BrowserCard(
            name="atlas",
            path=Path("/tmp/atlas"),
            is_default=False,
            is_current=False,
            summary="alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha",
        )
    ]

    rendered = _strip_ansi(agent_browser.render_browser(cards, selected_index=0, width=72, height=22, columns=3))
    atlas_lines = [line for line in rendered.splitlines() if "alpha" in line]

    assert len(atlas_lines) >= 2
    assert "STATUS" in atlas_lines[0]


def test_render_browser_single_profile_is_not_confusing_or_broken():
    cards = [
        agent_browser.BrowserCard(
            name="default",
            path=Path("/tmp/default"),
            is_default=True,
            is_current=True,
            model="gpt-5.4",
            provider="openai-codex",
            wired=True,
            wire_state="wired",
            summary="I am live in the wire and this should visibly use two status rows when needed.",
            message_to_user="still here",
            message_to_self="stay grounded",
        )
    ]

    rendered = agent_browser.render_browser(cards, selected_index=0, width=120, height=24, columns=2)
    stripped_lines = [_strip_ansi(line) for line in rendered.splitlines()]
    joined = "\n".join(stripped_lines)

    assert "single profile" in joined
    assert "wired in" in joined
    assert "selected " not in joined
    assert "to you:" not in joined
    assert "STATUS" in joined
    assert "MIND" not in joined
    assert "TRACE" not in joined
    assert all(get_cwidth(line) >= 120 for line in stripped_lines[3:9])
