from __future__ import annotations

import re

CONTEXT_LIMIT_PRESETS: tuple[int, int] = (272_000, 1_000_000)
CONTEXT_LIMIT_USAGE = "/context-limit [tokens]"

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


def cycle_context_limit(current: int | None) -> int:
    """Toggle between the preferred session presets."""
    low, high = CONTEXT_LIMIT_PRESETS
    return high if current == low else low
