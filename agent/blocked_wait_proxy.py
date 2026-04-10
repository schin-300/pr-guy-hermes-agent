from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from hermes_cli.config import load_config, save_config


DEFAULT_PROXY_CONTEXT_CHARS = 32000
DEFAULT_SETUP_PROMPT_TIMEOUT = 90
DEFAULT_IDENTITY_PROMPT = (
    "You are the blocked-session continuity proxy for the active Hermes session. "
    "You are not a separate assistant from the user's perspective. Speak like the "
    "main assistant, preserve its tone and goals, and stay focused on the current "
    "blocked state instead of inventing a new task."
)

DEFAULT_KIND_INSTRUCTIONS: dict[str, str] = {
    "clarify": (
        "The main Hermes session is blocked waiting for a clarify answer from the user. "
        "Explain what Hermes is waiting on, restate options when helpful, and guide the "
        "user toward replying with /answer when they are ready. Do not claim the answer "
        "was submitted unless the surrounding system tells you it was."
    ),
    "delegate": (
        "The main Hermes session is blocked while a delegated child agent is working. "
        "You can explain what the child appears to be doing, whether it looks idle or stuck, "
        "and what the user can do next. If the user clearly wants status, summarize status. "
        "If they clearly want to steer the child, tell them their next plain message will be "
        "treated as a steer/interruption request by the main session. If they clearly want to abort, "
        "say so plainly."
    ),
    "approval": (
        "The main Hermes session is blocked on a dangerous-command approval. "
        "Explain what is pending and how approval/denial works."
    ),
    "update": (
        "The main Hermes session is blocked on an update prompt. Explain what input Hermes needs."
    ),
}


def load_blocked_wait_proxy_config() -> dict:
    cfg = load_config() or {}
    section = cfg.get("blocked_wait_proxy", {})
    return section if isinstance(section, dict) else {}


def save_default_blocked_wait_proxy_kind(kind: str) -> None:
    cfg = load_config() or {}
    section = cfg.setdefault("blocked_wait_proxy", {})
    section["enabled"] = True
    section.setdefault("context_char_budget", DEFAULT_PROXY_CONTEXT_CHARS)
    section.setdefault("setup_prompt_timeout", DEFAULT_SETUP_PROMPT_TIMEOUT)
    section.setdefault("identity_prompt", DEFAULT_IDENTITY_PROMPT)
    kinds = section.setdefault("kinds", {})
    kind_cfg = kinds.setdefault(kind, {})
    kind_cfg["enabled"] = True
    kind_cfg["instructions"] = str(kind_cfg.get("instructions") or DEFAULT_KIND_INSTRUCTIONS.get(kind, ""))
    save_config(cfg)


def blocked_wait_proxy_kind_enabled(kind: str, config: Optional[dict] = None) -> bool:
    cfg = config or load_blocked_wait_proxy_config()
    if not cfg.get("enabled"):
        return False
    kinds = cfg.get("kinds", {}) or {}
    kind_cfg = kinds.get(kind, {}) or {}
    if not kind_cfg:
        return False
    return bool(kind_cfg.get("enabled"))


def blocked_wait_proxy_setup_timeout(config: Optional[dict] = None) -> int:
    cfg = config or load_blocked_wait_proxy_config()
    return max(1, int(cfg.get("setup_prompt_timeout") or DEFAULT_SETUP_PROMPT_TIMEOUT))


def blocked_wait_proxy_context_budget(config: Optional[dict] = None) -> int:
    cfg = config or load_blocked_wait_proxy_config()
    return max(4000, int(cfg.get("context_char_budget") or DEFAULT_PROXY_CONTEXT_CHARS))


def kind_instructions(kind: str, config: Optional[dict] = None) -> str:
    cfg = config or load_blocked_wait_proxy_config()
    kinds = cfg.get("kinds", {}) or {}
    kind_cfg = kinds.get(kind, {}) or {}
    return str(kind_cfg.get("instructions") or DEFAULT_KIND_INSTRUCTIONS.get(kind, "")).strip()


def identity_prompt(config: Optional[dict] = None) -> str:
    cfg = config or load_blocked_wait_proxy_config()
    return str(cfg.get("identity_prompt") or DEFAULT_IDENTITY_PROMPT).strip()


def helper_runtime(parent_agent, config: Optional[dict] = None) -> dict:
    cfg = config or load_blocked_wait_proxy_config()
    return {
        "model": str(cfg.get("model") or "").strip() or getattr(parent_agent, "model", ""),
        "provider": str(cfg.get("provider") or "").strip() or getattr(parent_agent, "provider", None),
        "base_url": str(cfg.get("base_url") or "").strip() or getattr(parent_agent, "base_url", None),
        "api_key": str(cfg.get("api_key") or "").strip() or getattr(parent_agent, "api_key", None),
        "api_mode": getattr(parent_agent, "api_mode", None),
    }


def looks_like_meta_question(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    if raw.endswith("?"):
        return True
    lowered = raw.lower()
    return lowered.startswith((
        "what", "why", "how", "is ", "are ", "did ", "does ", "where", "when", "status", "stuck", "loop",
    ))


def looks_like_abort_request(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    return any(token in lowered for token in ("abort", "stop it", "kill it", "cancel it", "give up"))


def looks_like_yes(text: str) -> bool:
    return str(text or "").strip().lower() in {"y", "yes", "enable", "ok", "okay", "sure"}


def looks_like_no(text: str) -> bool:
    return str(text or "").strip().lower() in {"n", "no", "disable", "skip", "not now"}


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    parts.append(str(item["text"]))
                elif item.get("type") in {"input_image", "image_url"}:
                    parts.append("[image]")
            elif item:
                parts.append(str(item))
        return "\n".join(parts)
    return "" if content is None else str(content)


def build_style_examples(messages: List[Dict[str, Any]], max_chars: int = 1600) -> str:
    chunks: List[str] = []
    total = 0
    for msg in reversed(messages or []):
        if msg.get("role") != "assistant":
            continue
        text = _flatten_content(msg.get("content", "")).strip()
        if not text:
            continue
        snippet = text[:280]
        add = f"Assistant style sample:\n{snippet}\n"
        if total + len(add) > max_chars:
            break
        chunks.append(add)
        total += len(add)
        if len(chunks) >= 4:
            break
    return "\n".join(reversed(chunks))


def build_context_excerpt(messages: List[Dict[str, Any]], max_chars: int) -> str:
    rows: List[str] = []
    total = 0
    for msg in reversed(messages or []):
        role = msg.get("role")
        if role not in {"user", "assistant", "tool"}:
            continue
        text = _flatten_content(msg.get("content", "")).strip()
        if not text:
            continue
        if role == "tool":
            tool_name = msg.get("tool_name") or msg.get("name") or "tool"
            entry = f"Tool {tool_name}: {text[:220]}"
        else:
            label = "User" if role == "user" else "Assistant"
            entry = f"{label}: {text[:320]}"
        add_len = len(entry) + 2
        if total + add_len > max_chars:
            break
        rows.append(entry)
        total += add_len
    return "\n\n".join(reversed(rows))


def build_wait_snapshot(kind: str, activity: dict) -> str:
    wait = activity.get("wait_state") or {}
    parts = [f"Wait kind: {kind}"]
    if wait.get("question_preview"):
        parts.append(f"Question: {wait['question_preview']}")
    if wait.get("mode"):
        parts.append(f"Mode: {wait['mode']}")
    if activity.get("current_tool"):
        parts.append(f"Current tool: {activity['current_tool']}")
    if activity.get("last_activity_desc"):
        parts.append(f"Last activity: {activity['last_activity_desc']}")
    children = activity.get("active_children") or []
    if children:
        child_lines = []
        for idx, child in enumerate(children[:3], start=1):
            desc = child.get("current_tool") or child.get("last_activity_desc") or "active"
            idle = child.get("seconds_since_activity")
            suffix = f" ({int(idle)}s idle)" if isinstance(idle, (int, float)) else ""
            line = f"child {idx}: {desc}{suffix}"
            if child.get("watchdog_reason"):
                line += f" | watchdog={child['watchdog_reason']}"
            child_lines.append(line)
        parts.append("Children:\n" + "\n".join(child_lines))
    return "\n".join(parts)


def build_system_prompt(*, kind: str, activity: dict, history: List[Dict[str, Any]], config: Optional[dict] = None) -> str:
    cfg = config or load_blocked_wait_proxy_config()
    context_budget = blocked_wait_proxy_context_budget(cfg)
    style = build_style_examples(history)
    excerpt = build_context_excerpt(history, max_chars=max(2000, context_budget - 4000))
    wait_snapshot = build_wait_snapshot(kind, activity)
    sections = [
        identity_prompt(cfg),
        "Current blocked-session context:\n" + wait_snapshot,
        "Adapter instructions:\n" + kind_instructions(kind, cfg),
    ]
    if style:
        sections.append("Recent voice/style samples:\n" + style)
    if excerpt:
        sections.append("Recent session context excerpt:\n" + excerpt)
    sections.append(
        "Your job is to help the user understand and navigate the blocked state. "
        "Be concrete, concise, and continuity-preserving. Do not fabricate resumed work."
    )
    return "\n\n".join(section for section in sections if section.strip())


def run_blocked_wait_proxy(*, kind: str, activity: dict, history: List[Dict[str, Any]], user_message: str, parent_agent) -> str:
    from run_agent import AIAgent

    cfg = load_blocked_wait_proxy_config()
    runtime = helper_runtime(parent_agent, cfg)
    system_prompt = build_system_prompt(kind=kind, activity=activity, history=history, config=cfg)

    helper = AIAgent(
        model=runtime["model"],
        provider=runtime["provider"],
        base_url=runtime["base_url"],
        api_key=runtime["api_key"],
        api_mode=runtime["api_mode"],
        max_iterations=2,
        enabled_toolsets=[],
        quiet_mode=True,
        verbose_logging=False,
        ephemeral_system_prompt=system_prompt,
        skip_context_files=True,
        skip_memory=True,
        platform=getattr(parent_agent, "platform", None),
    )
    helper._print_fn = lambda *_a, **_kw: None
    result = helper.run_conversation(user_message=user_message)
    response = (result or {}).get("final_response", "") if isinstance(result, dict) else str(result or "")
    response = re.sub(r"^\s*Hermes:\s*", "", response or "", flags=re.IGNORECASE)
    return response.strip()
