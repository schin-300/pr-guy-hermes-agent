from __future__ import annotations

import re

CONTEXT_MODE_DEFAULT = "1m"
CONTEXT_MODE_USAGE = "/context-mode [1m|272k|status]"
CONTEXT_MODE_PROFILES: dict[str, dict[str, float | int]] = {
    "272k": {
        "context_length": 272_000,
        "compression_threshold": 0.95,
    },
    "1m": {
        "context_length": 1_000_000,
        "compression_threshold": 0.75,
    },
}

_SUFFIX_MULTIPLIERS = {
    "": 1,
    "k": 1_000,
    "m": 1_000_000,
}


def parse_context_limit_value(raw: str) -> int | None:
    """Parse token counts like 500000, 272k, 1m, or 500,000."""
    cleaned = (raw or "").strip().lower().replace(",", "").replace("_", "")
    if not cleaned:
        return None

    match = re.fullmatch(r"(\d+)([km]?)", cleaned)
    if not match:
        return None

    value = int(match.group(1)) * _SUFFIX_MULTIPLIERS[match.group(2)]
    return value if value > 0 else None


def parse_context_mode_arg(raw: str | None) -> str | None:
    """Parse /context-mode args into a canonical mode name."""
    cleaned = (raw or "").strip().lower()
    if not cleaned:
        return "toggle"
    if cleaned == "status":
        return "status"
    if cleaned in CONTEXT_MODE_PROFILES:
        return cleaned

    parsed_limit = parse_context_limit_value(cleaned)
    if parsed_limit is None:
        return None

    for mode_name, profile in CONTEXT_MODE_PROFILES.items():
        if parsed_limit == int(profile["context_length"]):
            return mode_name
    return None
def cycle_context_mode(current: str | None) -> str:
    """Toggle between the named session context profiles."""
    if current == "1m":
        return "272k"
    return "1m"


def detect_context_mode(
    context_length: int | None,
    compression_threshold: float | None,
) -> str | None:
    """Return the named mode matching the live context profile, if any."""
    if context_length is None or compression_threshold is None:
        return None

    for mode_name, profile in CONTEXT_MODE_PROFILES.items():
        if (
            int(profile["context_length"]) == int(context_length)
            and abs(float(profile["compression_threshold"]) - float(compression_threshold)) < 1e-9
        ):
            return mode_name
    return None
