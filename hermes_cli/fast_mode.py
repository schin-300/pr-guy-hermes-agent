"""Shared helpers for Hermes /fast mode."""

from __future__ import annotations

from typing import Optional


FAST_MODE_USAGE = "/fast [on|off|status]"
FAST_TEMP_MODE_USAGE = "/fast-temp [on|off|status]"
FAST_MODE_SUBCOMMANDS = ("on", "off", "status")


def parse_fast_command_arg(raw: str | None) -> str | None:
    """Parse a /fast argument into toggle/on/off/status.

    Returns:
        "toggle", "on", "off", "status", or None for invalid input.
    """
    if raw is None:
        return "toggle"
    value = raw.strip().lower()
    if not value:
        return "toggle"
    if value in {"on", "off", "status"}:
        return value
    return None


def normalize_service_tier(value: str | None) -> Optional[str]:
    """Normalize service-tier state to Hermes' internal values."""
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized or normalized in {"off", "none", "default", "standard"}:
        return None
    if normalized in {"fast", "priority"}:
        return "fast"
    if normalized == "flex":
        return "flex"
    return None


def responses_api_service_tier(value: str | None) -> Optional[str]:
    """Map Hermes service-tier state to OpenAI/Codex Responses payload values."""
    normalized = normalize_service_tier(value)
    if normalized == "fast":
        return "priority"
    if normalized == "flex":
        return "flex"
    return None


def fast_mode_note(
    provider: str | None = None,
    model: str | None = None,
    enabled: bool = True,
) -> str:
    """Return a short applicability note for user-facing status output."""
    if not enabled:
        return "Fast mode is off — Hermes will not send service_tier=priority until you turn it on."

    provider_norm = (provider or "").strip().lower()
    model_norm = (model or "").strip().lower().split("/")[-1]
    if provider_norm == "openai-codex" and model_norm == "gpt-5.4":
        return "Hermes will send service_tier=priority on the next Codex turn."
    if provider_norm == "openai-codex":
        return "Hermes sends service_tier=priority on Codex turns; non-gpt-5.4 models may ignore it."
    return "Hermes sends service_tier=priority on OpenAI Codex gpt-5.4 turns; other providers/models ignore it."
