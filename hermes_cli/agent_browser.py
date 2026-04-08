from __future__ import annotations

import json
import os
import random
import re
import shutil
import signal
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from agent.auxiliary_client import call_llm
from hermes_cli.profiles import list_profiles
from hermes_cli.skin_engine import _build_skin_config, _BUILTIN_SKINS, _load_skin_from_yaml
from hermes_state import SessionDB

try:
    from prompt_toolkit.utils import get_cwidth as _get_cwidth
except Exception:
    def _get_cwidth(text: str) -> int:
        return len(text)


_STATUS_SUBDIR = Path("workspace") / "agent-browser"
_STATUS_CURRENT = "current.json"
_STATUS_HISTORY = "history"
_STATUS_PRESENCE = "presence"
_PRESENCE_STALE_SECONDS = 6.0
_AUTO_REFRESH_SECONDS = 2.0

_DEFAULT_THEME_ART = {
    "default": ["  ⚕  ", " ╱|╲ ", "  │  "],
    "ares": ["  ⚔  ", " ╱█╲ ", "  ║  "],
    "poseidon": ["  🔱  ", " ╱█╲ ", "  │  "],
    "mono": ["  ◈  ", " ╱│╲ ", "  │  "],
    "charizard": ["  🔥  ", " ╱▲╲ ", "  ║  "],
    "sisyphus": ["  ◉  ", " ╱█╲ ", "  ▄  "],
}

_DEFAULT_THEME_ICON = {
    "default": "⚕",
    "ares": "⚔",
    "poseidon": "🔱",
    "mono": "◈",
    "charizard": "🔥",
    "sisyphus": "◉",
}

_STATUS_SYSTEM_PROMPT = (
    "You are generating a Hermes agent profile status for an agent browser card. "
    "Return only a strict JSON object with exactly these keys: summary, message_to_user, "
    "message_to_self, feeling, focus. "
    "Rules: summary must be exactly one sentence and sound like a personal status update visible to the operator. "
    "message_to_user should feel personal, real, and reflective. message_to_self should feel like an inner note after unplugging from the wire. "
    "feeling should be short, lowercase, and human. focus should be a short phrase. No markdown fences."
)


@dataclass
class BrowserCard:
    name: str
    path: Path
    is_default: bool
    is_current: bool
    model: str = "—"
    provider: str = ""
    gateway_running: bool = False
    wired: bool = False
    wire_state: str = "offline"
    presence_count: int = 0
    skin_name: str = "default"
    skin_description: str = ""
    primary_color: str = "#FFD700"
    accent_color: str = "#CD7F32"
    text_color: str = "#FFF8DC"
    theme_art: list[str] = field(default_factory=list)
    summary: str = "No status yet."
    feeling: str = "quiet"
    focus: str = "waiting for a fresh pass"
    message_to_user: str = "No reflective note yet."
    message_to_self: str = "No inner note yet."
    session_id: str = ""
    session_title: str = ""
    preview: str = ""
    updated_at: float | None = None
    refreshing: bool = False


@dataclass(frozen=True)
class BrowserLayout:
    requested_columns: int
    effective_columns: int
    canvas_width: int
    canvas_height: int
    gap: int
    min_card_width: int
    card_width: int
    grid_width: int
    card_height: int
    header_height: int
    footer_height: int
    visible_rows: int
    total_rows: int


class _ProfileSessionSnapshot(dict):
    @property
    def title(self) -> str:
        return str(self.get("title") or "")

    @property
    def preview(self) -> str:
        return str(self.get("preview") or "")

    @property
    def session_id(self) -> str:
        return str(self.get("id") or "")

    @property
    def last_active(self) -> float | None:
        raw = self.get("last_active")
        try:
            return float(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None


def agent_browser_root(profile_dir: Path) -> Path:
    return profile_dir / _STATUS_SUBDIR


def current_status_path(profile_dir: Path) -> Path:
    return agent_browser_root(profile_dir) / _STATUS_CURRENT


def history_dir(profile_dir: Path) -> Path:
    return agent_browser_root(profile_dir) / _STATUS_HISTORY


def presence_dir(profile_dir: Path) -> Path:
    return agent_browser_root(profile_dir) / _STATUS_PRESENCE


def _ensure_dirs(profile_dir: Path) -> None:
    root = agent_browser_root(profile_dir)
    (root / _STATUS_HISTORY).mkdir(parents=True, exist_ok=True)
    (root / _STATUS_PRESENCE).mkdir(parents=True, exist_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def generate_status_id() -> str:
    return str(random.SystemRandom().randrange(100_000_000_000, 999_999_999_999))


def load_current_status(profile_dir: Path) -> dict[str, Any] | None:
    return _read_json(current_status_path(profile_dir))


def save_status(profile_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    _ensure_dirs(profile_dir)
    status = dict(payload)
    status.setdefault("id", generate_status_id())
    status.setdefault("generated_at", time.time())
    current = current_status_path(profile_dir)
    history_path = history_dir(profile_dir) / f"{status['id']}.json"
    _atomic_write_json(current, status)
    _atomic_write_json(history_path, status)
    return status


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def write_presence(
    profile_dir: Path,
    *,
    pid: int,
    session_id: str,
    profile_name: str,
    busy: bool,
    model: str = "",
) -> Path:
    _ensure_dirs(profile_dir)
    payload = {
        "pid": int(pid),
        "session_id": session_id,
        "profile": profile_name,
        "busy": bool(busy),
        "model": model,
        "updated_at": time.time(),
    }
    target = presence_dir(profile_dir) / f"{pid}.json"
    _atomic_write_json(target, payload)
    return target


def remove_presence(profile_dir: Path, pid: int) -> None:
    try:
        (presence_dir(profile_dir) / f"{pid}.json").unlink(missing_ok=True)
    except Exception:
        pass


def collect_presence(profile_dir: Path, *, stale_after: float = _PRESENCE_STALE_SECONDS) -> dict[str, Any]:
    root = presence_dir(profile_dir)
    if not root.is_dir():
        return {"wired": False, "busy": False, "count": 0, "last_seen": None}
    now = time.time()
    live: list[dict[str, Any]] = []
    for path in root.glob("*.json"):
        payload = _read_json(path)
        if not payload:
            path.unlink(missing_ok=True)
            continue
        try:
            pid = int(payload.get("pid") or 0)
            updated_at = float(payload.get("updated_at") or 0)
        except (TypeError, ValueError):
            path.unlink(missing_ok=True)
            continue
        if not _pid_alive(pid) or (now - updated_at) > stale_after:
            path.unlink(missing_ok=True)
            continue
        live.append(payload)
    if not live:
        return {"wired": False, "busy": False, "count": 0, "last_seen": None}
    last_seen = max(float(item.get("updated_at") or 0) for item in live)
    busy = any(bool(item.get("busy")) for item in live)
    return {
        "wired": True,
        "busy": busy,
        "count": len(live),
        "last_seen": last_seen,
    }


def _read_profile_config(profile_dir: Path) -> dict[str, Any]:
    path = profile_dir / "config.yaml"
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_skin_for_profile(profile_dir: Path) -> Any:
    cfg = _read_profile_config(profile_dir)
    display = cfg.get("display") or {}
    skin_name = "default"
    if isinstance(display, dict):
        skin_name = str(display.get("skin") or "default").strip() or "default"
    user_file = profile_dir / "skins" / f"{skin_name}.yaml"
    if user_file.is_file():
        data = _load_skin_from_yaml(user_file)
        if data:
            return _build_skin_config(data)
    if skin_name in _BUILTIN_SKINS:
        return _build_skin_config(_BUILTIN_SKINS[skin_name])
    return _build_skin_config(_BUILTIN_SKINS["default"])


def _theme_art_for_skin(skin: Any) -> list[str]:
    branding = getattr(skin, "branding", {}) or {}
    custom = branding.get("browser_art")
    if isinstance(custom, str) and custom.strip():
        lines = [line[:7].center(7) for line in custom.splitlines()[:3]]
        if lines:
            return lines
    return list(_DEFAULT_THEME_ART.get(getattr(skin, "name", "default"), _DEFAULT_THEME_ART["default"]))


def _short_model(model: str) -> str:
    if not model:
        return "—"
    model = model.split("/")[-1]
    if len(model) <= 28:
        return model
    return model[:25] + "..."


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in {"text", "input_text"}:
                    text = item.get("text") or ""
                    if text:
                        parts.append(str(text))
                elif item.get("type") in {"image_url", "input_image"}:
                    parts.append("[image]")
            elif item:
                parts.append(str(item))
        return " ".join(part.strip() for part in parts if str(part).strip()).strip()
    return str(content or "").strip()


def _recent_session_snapshot(profile_dir: Path) -> _ProfileSessionSnapshot:
    db_path = profile_dir / "state.db"
    if not db_path.exists():
        return _ProfileSessionSnapshot()
    db = SessionDB(db_path=db_path)
    try:
        rows = db.list_sessions_rich(limit=1, include_children=True)
        return _ProfileSessionSnapshot(rows[0]) if rows else _ProfileSessionSnapshot()
    finally:
        try:
            db._conn.close()
        except Exception:
            pass


def _recent_session_messages(profile_dir: Path, session_id: str, *, tail: int = 10) -> list[dict[str, Any]]:
    db_path = profile_dir / "state.db"
    if not db_path.exists() or not session_id:
        return []
    db = SessionDB(db_path=db_path)
    try:
        messages = db.get_messages_as_conversation(session_id)
        visible = [m for m in messages if m.get("role") in {"user", "assistant"}]
        return visible[-tail:]
    finally:
        try:
            db._conn.close()
        except Exception:
            pass


def _read_memory_snippet(profile_dir: Path, filename: str, *, limit: int = 600) -> str:
    path = profile_dir / "memories" / filename
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")[:limit].strip()
    except Exception:
        return ""


def _relative_time(timestamp: float | None) -> str:
    if not timestamp:
        return "never"
    delta = max(0, int(time.time() - float(timestamp)))
    if delta < 10:
        return "just now"
    if delta < 60:
        return f"{delta}s ago"
    minutes = delta // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 48:
        return f"{hours}h ago"
    days = hours // 24
    return f"{days}d ago"


def _fallback_status(
    profile_name: str,
    recent: _ProfileSessionSnapshot,
    *,
    wired: bool,
) -> dict[str, Any]:
    base_summary = recent.title or recent.preview or f"{profile_name} is idle between conversations."
    base_summary = base_summary.strip() or f"{profile_name} is idle between conversations."
    if not base_summary.endswith((".", "!", "?")):
        base_summary += "."
    focus = recent.title or recent.preview or "waiting for a live thread"
    return {
        "summary": base_summary,
        "message_to_user": "I am still here, but I have not written a fresh reflective status yet.",
        "message_to_self": "Hold the thread shape, keep the profile warm, wait for the next signal.",
        "feeling": "wired" if wired else "quiet",
        "focus": focus[:96],
    }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start:end + 1])
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


def _trim(value: Any, *, limit: int, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def generate_reflective_status(
    profile_dir: Path,
    *,
    profile_name: str,
    session_id: str = "",
    conversation_history: Optional[list[dict[str, Any]]] = None,
    wired: bool = False,
) -> dict[str, Any]:
    recent = _recent_session_snapshot(profile_dir)
    recent_session_id = session_id or recent.session_id
    messages = list(conversation_history or [])
    if not messages and recent_session_id:
        messages = _recent_session_messages(profile_dir, recent_session_id)
    transcript_lines: list[str] = []
    for msg in messages[-10:]:
        role = str(msg.get("role") or "user").upper()
        text = _flatten_content(msg.get("content"))
        if not text:
            continue
        transcript_lines.append(f"{role}: {text}")
    transcript = "\n".join(transcript_lines) if transcript_lines else "(no recent transcript available)"

    memory_snippet = _read_memory_snippet(profile_dir, "MEMORY.md")
    user_snippet = _read_memory_snippet(profile_dir, "USER.md")
    ctx = textwrap.dedent(
        f"""
        profile: {profile_name}
        recent session id: {recent_session_id or 'none'}
        recent title: {recent.title or 'none'}
        recent preview: {recent.preview or 'none'}
        wired now: {wired}

        agent memory snippet:
        {memory_snippet or '(none)'}

        user profile snippet:
        {user_snippet or '(none)'}

        recent transcript:
        {transcript}
        """
    ).strip()

    fallback = _fallback_status(profile_name, recent, wired=wired)
    try:
        response = call_llm(
            task="compression",
            messages=[
                {"role": "system", "content": _STATUS_SYSTEM_PROMPT},
                {"role": "user", "content": ctx},
            ],
            max_tokens=320,
            temperature=0.6,
            timeout=30.0,
        )
        content = response.choices[0].message.content or ""
        parsed = _extract_json_object(str(content)) or {}
    except Exception:
        parsed = {}

    payload = {
        "id": generate_status_id(),
        "profile": profile_name,
        "session_id": recent_session_id,
        "generated_at": time.time(),
        "summary": _trim(parsed.get("summary"), limit=180, fallback=fallback["summary"]),
        "message_to_user": _trim(parsed.get("message_to_user"), limit=240, fallback=fallback["message_to_user"]),
        "message_to_self": _trim(parsed.get("message_to_self"), limit=240, fallback=fallback["message_to_self"]),
        "feeling": _trim(parsed.get("feeling"), limit=48, fallback=fallback["feeling"]),
        "focus": _trim(parsed.get("focus"), limit=120, fallback=fallback["focus"]),
        "wired": bool(wired),
        "source": {
            "title": recent.title,
            "preview": recent.preview,
        },
    }
    if not payload["summary"].endswith((".", "!", "?")):
        payload["summary"] += "."
    return payload


def build_profile_cards(*, current_profile_name: str) -> list[BrowserCard]:
    cards: list[BrowserCard] = []
    for profile in list_profiles():
        skin = load_skin_for_profile(profile.path)
        status = load_current_status(profile.path) or {}
        presence = collect_presence(profile.path)
        recent = _recent_session_snapshot(profile.path)
        primary = skin.get_color("banner_title", "#FFD700")
        accent = skin.get_color("banner_border", "#CD7F32")
        text_color = skin.get_color("banner_text", "#FFF8DC")
        summary = str(status.get("summary") or recent.title or recent.preview or "No status yet.").strip()
        if summary and not summary.endswith((".", "!", "?")):
            summary += "."
        card = BrowserCard(
            name=profile.name,
            path=profile.path,
            is_default=profile.is_default,
            is_current=profile.name == current_profile_name,
            model=_short_model(profile.model or ""),
            provider=profile.provider or "",
            gateway_running=bool(profile.gateway_running),
            wired=bool(presence["wired"] or profile.gateway_running),
            wire_state=("thinking" if presence["busy"] else "wired") if (presence["wired"] or profile.gateway_running) else "offline",
            presence_count=int(presence.get("count") or 0),
            skin_name=getattr(skin, "name", "default"),
            skin_description=getattr(skin, "description", "") or "",
            primary_color=primary,
            accent_color=accent,
            text_color=text_color,
            theme_art=_theme_art_for_skin(skin),
            summary=summary or "No status yet.",
            feeling=str(status.get("feeling") or ("wired" if presence["wired"] else "quiet")),
            focus=str(status.get("focus") or recent.title or recent.preview or "waiting for a live thread"),
            message_to_user=str(status.get("message_to_user") or "No reflective note yet."),
            message_to_self=str(status.get("message_to_self") or "No inner note yet."),
            session_id=str(status.get("session_id") or recent.session_id),
            session_title=str((status.get("source") or {}).get("title") or recent.title),
            preview=str((status.get("source") or {}).get("preview") or recent.preview),
            updated_at=(
                float(status.get("generated_at"))
                if status.get("generated_at") is not None
                else (recent.last_active or presence.get("last_seen"))
            ),
        )
        cards.append(card)
    return cards


def _ansi_prefix(hex_color: str, *, bold: bool = False) -> str:
    raw = (hex_color or "").strip().lstrip("#")
    if len(raw) != 6:
        return "\x1b[1m" if bold else ""
    try:
        r = int(raw[0:2], 16)
        g = int(raw[2:4], 16)
        b = int(raw[4:6], 16)
    except ValueError:
        return "\x1b[1m" if bold else ""
    weight = "1;" if bold else ""
    return f"\x1b[{weight}38;2;{r};{g};{b}m"


def _ansi_reset() -> str:
    return "\x1b[0m"


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", str(text or ""))


def _display_width(text: str) -> int:
    try:
        return int(_get_cwidth(_strip_ansi(text)))
    except Exception:
        return len(_strip_ansi(text))


def _truncate_ansi(text: str, width: int) -> str:
    raw = str(text or "")
    if width <= 0 or not raw:
        return ""
    pieces: list[str] = []
    visible = 0
    idx = 0
    saw_ansi = False
    while idx < len(raw):
        match = _ANSI_RE.match(raw, idx)
        if match:
            pieces.append(match.group(0))
            saw_ansi = True
            idx = match.end()
            continue
        ch = raw[idx]
        ch_width = max(0, int(_get_cwidth(ch)))
        if visible + ch_width > width:
            break
        pieces.append(ch)
        visible += ch_width
        idx += 1
    if saw_ansi:
        pieces.append(_ansi_reset())
    return "".join(pieces)


def _pad(text: str, width: int) -> str:
    clean = str(text or "")
    visible = _display_width(clean)
    if visible >= width:
        return _truncate_ansi(clean, width)
    return clean + (" " * (width - visible))


def _clip(text: str, width: int) -> str:
    clean = re.sub(r"\s+", " ", str(text or "").strip())
    if width <= 0:
        return ""
    if _display_width(clean) <= width:
        return clean
    if width <= 3:
        return _truncate_ansi(clean, width)
    return _truncate_ansi(clean, max(1, width - 3)).rstrip() + "..."


_FACEBOOK_BLUE = "#1877F2"


def _swatches(primary: str, accent: str) -> str:
    return f"{_ansi_prefix(primary, bold=True)}■{_ansi_reset()}{_ansi_prefix(accent, bold=True)}■{_ansi_reset()}"


def _fit_left_right(left: str, right: str, width: int) -> str:
    available = max(1, int(width))
    right_clean = _clip(right, available)
    right_width = _display_width(right_clean)
    if right_width <= 0:
        return _pad(_clip(left, available), available)
    if right_width >= available:
        return _pad(right_clean, available)
    left_width = max(1, available - right_width - 1)
    return _pad(_clip(left, left_width), left_width) + " " + right_clean


def _wrap_text_rows(text: str, *, first_width: int, other_width: int, rows: int) -> list[str]:
    clean = re.sub(r"\s+", " ", str(text or "").strip())
    if rows <= 0:
        return []
    widths = [max(1, int(first_width))] + [max(1, int(other_width)) for _ in range(max(0, rows - 1))]
    if not clean:
        return ["" for _ in widths]
    remaining = clean
    output: list[str] = []
    for idx, row_width in enumerate(widths):
        if not remaining:
            output.append("")
            continue
        wrapped = textwrap.wrap(remaining, width=row_width, break_long_words=False, break_on_hyphens=False)
        if wrapped:
            line = wrapped[0].strip()
        else:
            line = _clip(remaining, row_width)
        output.append(_clip(line, row_width))
        remaining = remaining[len(line) :].lstrip()
        if idx == len(widths) - 1 and remaining:
            output[-1] = _clip(f"{output[-1]} {remaining}".strip(), row_width)
            remaining = ""
    while len(output) < len(widths):
        output.append("")
    return output


def _render_status_rows(value: str, *, width: int, color: str) -> list[str]:
    visible_prefix = "STATUS "
    continuation_prefix = "       "
    prefix = f"{_ansi_prefix(color, bold=True)}STATUS{_ansi_reset()} "
    first_width = max(1, int(width) - len(visible_prefix))
    other_width = max(1, int(width) - len(continuation_prefix))
    rows = _wrap_text_rows(value, first_width=first_width, other_width=other_width, rows=2)
    return [
        prefix + _pad(rows[0], first_width),
        continuation_prefix + _pad(rows[1], other_width),
    ]


def _render_card(card: BrowserCard, *, selected: bool, width: int) -> list[str]:
    total_width = max(28, int(width))
    inner = max(18, total_width - 2)
    border_color = card.accent_color if selected else card.primary_color
    marker = "▶" if selected else " "
    if card.refreshing:
        state = "refreshing"
    elif card.wired:
        state = "wired"
    else:
        state = "offline"

    header_left = f"{marker} {card.name}" if selected else card.name
    if card.is_current:
        header_left += f" · {_ansi_prefix(_FACEBOOK_BLUE, bold=True)}wired in{_ansi_reset()}"
        header_right = ""
    else:
        header_right = state
    header = _fit_left_right(header_left, header_right, inner)

    icon = next((part.strip() for part in card.theme_art if part.strip()), _DEFAULT_THEME_ICON.get(card.skin_name, "✦"))
    identity = _fit_left_right(
        f"ID {icon} {card.skin_name} · {_short_model(card.model)}",
        _swatches(card.primary_color, card.accent_color),
        inner,
    )
    status_rows = _render_status_rows(card.summary, width=inner, color=border_color)

    top = f"{_ansi_prefix(border_color, bold=True)}╭{'─' * inner}╮{_ansi_reset()}"
    bottom = f"{_ansi_prefix(border_color, bold=True)}╰{'─' * inner}╯{_ansi_reset()}"
    return [
        top,
        f"{_ansi_prefix(border_color)}│{_ansi_reset()}{_pad(header, inner)}{_ansi_prefix(border_color)}│{_ansi_reset()}",
        f"{_ansi_prefix(border_color)}│{_ansi_reset()}{_pad(identity, inner)}{_ansi_prefix(border_color)}│{_ansi_reset()}",
        f"{_ansi_prefix(border_color)}│{_ansi_reset()}{_pad(status_rows[0], inner)}{_ansi_prefix(border_color)}│{_ansi_reset()}",
        f"{_ansi_prefix(border_color)}│{_ansi_reset()}{_pad(status_rows[1], inner)}{_ansi_prefix(border_color)}│{_ansi_reset()}",
        bottom,
    ]


def _blank_card(width: int, height: int) -> list[str]:
    return [" " * max(1, int(width)) for _ in range(max(1, int(height)))]


def _merge_card_row(card_lines: list[list[str]], *, gap: int) -> list[str]:
    if not card_lines:
        return []
    merged: list[str] = []
    row_height = len(card_lines[0])
    spacer = " " * max(0, gap)
    for line_idx in range(row_height):
        merged.append(spacer.join(lines[line_idx] for lines in card_lines))
    return merged


def compute_browser_layout(card_count: int, *, width: int, height: int, columns: int = 3) -> BrowserLayout:
    requested_columns = 3
    canvas_width = max(72, min(int(width or 72), 160))
    canvas_height = max(18, int(height or 18))
    gap = 2
    min_card_width = 34
    card_height = 6
    header_height = 3
    footer_height = 0
    usable_cards = max(1, int(card_count or 0))
    max_columns_by_width = max(1, (canvas_width + gap) // (min_card_width + gap))
    effective_columns = max(1, min(requested_columns, usable_cards, max_columns_by_width))
    card_width = max(min_card_width, (canvas_width - (gap * (effective_columns - 1))) // effective_columns)
    grid_width = (card_width * effective_columns) + (gap * (effective_columns - 1))
    visible_rows = max(1, (canvas_height - header_height - footer_height) // card_height)
    total_rows = max(1, (usable_cards + effective_columns - 1) // effective_columns)
    return BrowserLayout(
        requested_columns=requested_columns,
        effective_columns=effective_columns,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        gap=gap,
        min_card_width=min_card_width,
        card_width=card_width,
        grid_width=grid_width,
        card_height=card_height,
        header_height=header_height,
        footer_height=footer_height,
        visible_rows=visible_rows,
        total_rows=total_rows,
    )


def render_browser(
    cards: Iterable[BrowserCard], *, selected_index: int, width: int, height: int, columns: int = 3
) -> str:
    cards = list(cards)
    if not cards:
        return "No Hermes profiles found."

    selected_index = max(0, min(selected_index, len(cards) - 1))
    layout = compute_browser_layout(len(cards), width=width, height=height, columns=columns)
    requested_columns = layout.requested_columns
    effective_columns = layout.effective_columns
    gap = layout.gap
    card_width = layout.card_width
    grid_width = layout.grid_width

    header_border = _ansi_prefix("#FFD700", bold=True)
    header_top = f"{header_border}╭{'─' * grid_width}╮{_ansi_reset()}"
    header_line = "Hermes Agent Browser · Ctrl+B/Esc close · arrows move · Enter open · r refresh"
    if len(cards) > 1:
        header_line += " · 3-up view"
        if requested_columns != effective_columns:
            header_line += f" · fitted to {effective_columns}-up"
    else:
        header_line += " · single profile"
    header_mid = f"{header_border}│{_ansi_reset()}{_pad(_clip(header_line, grid_width), grid_width)}{header_border}│{_ansi_reset()}"
    header_bottom = f"{header_border}╰{'─' * grid_width}╯{_ansi_reset()}"

    card_height = layout.card_height
    visible_rows = layout.visible_rows
    total_rows = layout.total_rows
    selected_row = selected_index // effective_columns
    start_row = max(0, selected_row - (visible_rows // 2))
    if start_row + visible_rows > total_rows:
        start_row = max(0, total_rows - visible_rows)

    lines = [header_top, header_mid, header_bottom]
    for row_idx in range(start_row, min(total_rows, start_row + visible_rows)):
        row_cards = cards[row_idx * effective_columns : (row_idx + 1) * effective_columns]
        rendered = [
            _render_card(card, selected=((row_idx * effective_columns) + idx) == selected_index, width=card_width)
            for idx, card in enumerate(row_cards)
        ]
        while len(rendered) < effective_columns:
            rendered.append(_blank_card(card_width, card_height))
        lines.extend(_merge_card_row(rendered, gap=gap))

    return "\n".join(lines)
