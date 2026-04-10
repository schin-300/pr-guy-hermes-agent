"""Compact Codex auth TUI.

Workflow
- Scan every pooled Codex profile in one dense list.
- Answer: which accounts are available, limited, or deactivated right now?
- Show the short-window and long-window remaining-quota bars inline for every row.
- Let the operator remove dead profiles from Hermes auth without leaving the view.

Constraints
- Keep rows one line tall so many profiles fit on screen.
- Use the repo's terminal-native approach (curses) instead of adding a web UI.
- Separate live usage fetching/pool mutation from curses rendering.
- Support a non-interactive --smoke-test mode for validation.
"""

from __future__ import annotations

import array
import base64
import fcntl
import locale
import math
import os
import struct
import sys
import termios
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import count
from typing import Any, Callable, Optional

import httpx

from agent.credential_pool import (
    PooledCredential,
    STATUS_EXHAUSTED,
    label_from_token,
    load_pool,
)
from hermes_cli import auth as auth_mod

DEFAULT_PROVIDER = "openai-codex"
USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"
WEEKLY_RESET_GAP_SECONDS = 3 * 24 * 60 * 60
MAX_POLL_WORKERS = 8
AUTO_REFRESH_INTERVAL_SECONDS = 30.0
AUTO_REFRESH_INTERVAL_PRESETS = (30.0, 15.0, 10.0, 5.0, 2.5, 0.0, 60.0, 120.0)
KITTY_METER_MIN_WIDTH = 96
KITTY_METER_MIN_HEIGHT = 20
MAX_POOL_DRAIN_SAMPLES = 64
MAX_VISIBLE_POOL_DRAIN_POLLS = 30
REFRESH_SCOPE_ALL = "all"
_KITTY_PLACEHOLDER = "\U0010EEEE"
_KITTY_CHUNK_SIZE = 4096
_DEFAULT_CELL_PIXELS = (10, 20)
_KITTY_DIACRITIC_CODEPOINTS = (
    0x0305,
    0x030D,
    0x030E,
    0x0310,
    0x0312,
    0x033D,
    0x033E,
    0x033F,
    0x0346,
    0x034A,
    0x034B,
    0x034C,
    0x0350,
    0x0351,
    0x0352,
    0x0357,
    0x035B,
    0x0363,
    0x0364,
    0x0365,
    0x0366,
    0x0367,
    0x0368,
    0x0369,
    0x036A,
    0x036B,
    0x036C,
    0x036D,
    0x036E,
    0x036F,
    0x0483,
    0x0484,
    0x0485,
    0x0486,
    0x0487,
    0x0592,
    0x0593,
    0x0594,
    0x0595,
    0x0597,
    0x0598,
    0x0599,
    0x059C,
    0x059D,
    0x059E,
    0x059F,
    0x05A0,
    0x05A1,
    0x05A8,
    0x05A9,
    0x05AB,
    0x05AC,
    0x05AF,
    0x05C4,
    0x0610,
    0x0611,
    0x0612,
    0x0613,
    0x0614,
    0x0615,
    0x0616,
    0x0617,
    0x0657,
    0x0658,
    0x0659,
    0x065A,
    0x065B,
    0x065D,
    0x065E,
    0x06D6,
    0x06D7,
    0x06D8,
    0x06D9,
    0x06DA,
    0x06DB,
    0x06DC,
    0x06DF,
    0x06E0,
    0x06E1,
    0x06E2,
    0x06E4,
    0x06E7,
    0x06E8,
    0x06EB,
    0x06EC,
    0x0730,
    0x0732,
    0x0733,
    0x0735,
    0x0736,
    0x073A,
    0x073D,
    0x073F,
    0x0740,
    0x0741,
    0x0743,
    0x0745,
    0x0747,
    0x0749,
    0x074A,
)
_KITTY_DIACRITICS = tuple(chr(codepoint) for codepoint in _KITTY_DIACRITIC_CODEPOINTS)
_KITTY_IMAGE_IDS = count(0xC0D300, 1)
_CIRCLE_OFFSETS_BY_RADIUS: dict[int, tuple[tuple[int, int], ...]] = {}

STATE_LOADING = "loading"
STATE_AVAILABLE = "available"
STATE_LIMITED = "limited"
STATE_DEACTIVATED = "deactivated"
STATE_ERROR = "error"

SORT_AVAILABILITY = "availability"
SORT_NAME = "name"
SORT_REFRESH = "refresh"
_SORT_LABELS = {
    SORT_AVAILABILITY: "avail",
    SORT_NAME: "name",
    SORT_REFRESH: "fresh",
}

_BADGES = {
    STATE_LOADING: "LOAD",
    STATE_AVAILABLE: "AVAIL",
    STATE_LIMITED: "LIMIT",
    STATE_DEACTIVATED: "DEAD",
    STATE_ERROR: "ERR",
}


@dataclass
class CodexUsageWindow:
    label: str
    used_percent: float
    reset_at_ms: Optional[int] = None

    @property
    def available_percent(self) -> float:
        return _clamp_percent(100.0 - self.used_percent)


@dataclass
class CodexLiveStatus:
    http_status: Optional[int] = None
    email: Optional[str] = None
    plan: Optional[str] = None
    allowed: Optional[bool] = None
    limit_reached: Optional[bool] = None
    primary_window: Optional[CodexUsageWindow] = None
    secondary_window: Optional[CodexUsageWindow] = None
    error: Optional[str] = None
    deactivated: bool = False
    local_cooldown_cleared: bool = False


@dataclass
class CodexProfileSnapshot:
    index: int
    entry_id: str
    label: str
    display_name: str
    auth_badge: str
    state: str
    state_badge: str
    state_reason: str
    primary_window: Optional[CodexUsageWindow]
    secondary_window: Optional[CodexUsageWindow]
    email: Optional[str] = None
    plan: Optional[str] = None
    source: str = ""
    auth_type: str = ""
    last_refresh: Optional[str] = None
    local_status: Optional[str] = None
    local_error_code: Optional[int] = None
    http_status: Optional[int] = None
    local_cooldown_cleared: bool = False
    auth_reason: Optional[str] = None


@dataclass
class CodexPoolDrainSample:
    captured_at: float
    label: str
    rate_percent_per_hour: float
    total_drop_percent: float
    dropping_accounts: int
    compared_accounts: int
    tracked_accounts: int
    resetting_accounts: int
    average_available_percent: float
    total_available_percent: float
    total_capacity_percent: float


def _circle_offsets(radius: int) -> tuple[tuple[int, int], ...]:
    cached = _CIRCLE_OFFSETS_BY_RADIUS.get(radius)
    if cached is not None:
        return cached
    offsets: list[tuple[int, int]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                offsets.append((dx, dy))
    cached = tuple(offsets)
    _CIRCLE_OFFSETS_BY_RADIUS[radius] = cached
    return cached


def _stdout_is_tty() -> bool:
    for stream in (sys.__stdout__, sys.stdout):
        isatty = getattr(stream, "isatty", None)
        if not callable(isatty):
            continue
        try:
            if isatty():
                return True
        except OSError:
            continue
    return False


def _safe_str(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _clip(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def _short_entry_id(entry_id: str) -> str:
    if len(entry_id) <= 12:
        return entry_id
    return f"{entry_id[:6]}…{entry_id[-4:]}"


def _footer_line(message: str, *, width: int) -> str:
    shortcuts = "↑↓/jk move  a auto  n name  l fresh  u refresh  r remove  q quit"
    if not message:
        return _clip(shortcuts, width)
    return _clip(f"{message}  |  {shortcuts}", width)


def _truthy_env(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _kitty_graphics_requested(env: Optional[dict[str, str]] = None) -> bool:
    env = env or os.environ
    if _truthy_env(env.get("HERMES_AUTH_VIEW_DISABLE_KITTY_GRAPHICS")) or _truthy_env(
        env.get("HERMES_AUTH_VIEW_DISABLE_KITTY_METER")
    ):
        return False
    if _truthy_env(env.get("HERMES_AUTH_VIEW_FORCE_KITTY_GRAPHICS")) or _truthy_env(
        env.get("HERMES_AUTH_VIEW_FORCE_KITTY_METER")
    ):
        return True
    if not _stdout_is_tty():
        return False
    term = (env.get("TERM") or "").lower()
    term_program = (env.get("TERM_PROGRAM") or "").lower()
    kitty_window = (env.get("KITTY_WINDOW_ID") or "").strip()
    return bool(term == "xterm-kitty" or term_program == "kitty" or kitty_window)


def _kitty_meter_enabled_from_env(env: Optional[dict[str, str]] = None) -> bool:
    return _kitty_graphics_requested(env)


def _clamp_percent(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(100.0, numeric))


def _format_percent(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    clamped = _clamp_percent(value)
    if clamped >= 99.5:
        return "100%"
    return f"{int(round(clamped)):>3d}%"


def _format_bar(value: Optional[float], width: int) -> str:
    width = max(4, int(width))
    if value is None:
        return f"[{'·' * width}] n/a"
    clamped = _clamp_percent(value)
    filled = int(round((clamped / 100.0) * width))
    filled = max(0, min(width, filled))
    return f"[{'█' * filled}{'·' * (width - filled)}] {_format_percent(clamped)}"


def _format_bar_with_reset(window: Optional[CodexUsageWindow], width: int) -> str:
    bar = _format_bar(_window_available_percent(window), width)
    suffix = _format_reset_compact(window.reset_at_ms if window else None)
    return f"{bar} {suffix}" if suffix else bar


def _window_available_percent(window: Optional[CodexUsageWindow]) -> Optional[float]:
    if window is None:
        return None
    return window.available_percent


def _window_is_exhausted(window: Optional[CodexUsageWindow]) -> bool:
    return bool(window and window.used_percent >= 99.5)


def _collect_primary_window_state(
    snapshots: list[CodexProfileSnapshot],
) -> tuple[str, dict[str, tuple[float, Optional[int]]], int, float]:
    label = "5h"
    current_by_id: dict[str, tuple[float, Optional[int]]] = {}
    tracked_accounts = 0
    total_available_percent = 0.0
    for snapshot in snapshots:
        window = snapshot.primary_window
        available = _window_available_percent(window)
        if available is None:
            continue
        if window and window.label:
            label = window.label
        current_by_id[snapshot.entry_id] = (available, window.reset_at_ms if window else None)
        tracked_accounts += 1
        total_available_percent += available
    return label, current_by_id, tracked_accounts, total_available_percent


def _build_pool_drain_sample(
    previous_by_id: dict[str, tuple[float, Optional[int]]],
    current_snapshots: list[CodexProfileSnapshot],
    *,
    elapsed_seconds: float,
    captured_at: Optional[float] = None,
) -> Optional[CodexPoolDrainSample]:
    label, current_by_id, tracked_accounts, total_available_percent = _collect_primary_window_state(current_snapshots)
    if not current_by_id:
        return None

    compared_accounts = 0
    dropping_accounts = 0
    resetting_accounts = 0
    total_drop_percent = 0.0
    epsilon = 0.05

    if elapsed_seconds > 0 and previous_by_id:
        for entry_id, (current_available, current_reset_at_ms) in current_by_id.items():
            previous = previous_by_id.get(entry_id)
            if previous is None:
                continue
            previous_available, previous_reset_at_ms = previous
            compared_accounts += 1
            if (
                current_reset_at_ms
                and previous_reset_at_ms
                and current_reset_at_ms != previous_reset_at_ms
                and current_available > previous_available
            ):
                resetting_accounts += 1
                continue
            drop_percent = previous_available - current_available
            if drop_percent > epsilon:
                total_drop_percent += drop_percent
                dropping_accounts += 1

    rate_percent_per_hour = 0.0
    if elapsed_seconds > 0:
        rate_percent_per_hour = total_drop_percent / (elapsed_seconds / 3600.0)

    average_available_percent = total_available_percent / tracked_accounts if tracked_accounts else 0.0
    return CodexPoolDrainSample(
        captured_at=float(captured_at if captured_at is not None else time.time()),
        label=label,
        rate_percent_per_hour=rate_percent_per_hour,
        total_drop_percent=total_drop_percent,
        dropping_accounts=dropping_accounts,
        compared_accounts=compared_accounts,
        tracked_accounts=tracked_accounts,
        resetting_accounts=resetting_accounts,
        average_available_percent=average_available_percent,
        total_available_percent=total_available_percent,
        total_capacity_percent=tracked_accounts * 100.0,
    )


def _meter_border_line(width: int, *, title: str = "", top: bool = True) -> str:
    width = max(4, int(width))
    inner = width - 2
    if title:
        decorated = _clip(f" {title} ", inner)
        remaining = max(0, inner - len(decorated))
        left = remaining // 2
        right = remaining - left
        fill = "─"
        return f"{'╭' if top else '╰'}{fill * left}{decorated}{fill * right}{'╮' if top else '╯'}"
    return f"{'╭' if top else '╰'}{'─' * inner}{'╮' if top else '╯'}"


def _meter_text_line(width: int, text: str) -> str:
    inner = max(2, int(width) - 2)
    return f"│{_clip(text.ljust(inner), inner)}│"


def _resample_series(values: list[float], width: int) -> list[float]:
    width = max(1, int(width))
    if not values:
        return [0.0] * width
    if len(values) == 1:
        return [values[0]] * width
    out: list[float] = []
    last = len(values) - 1
    for idx in range(width):
        position = (idx / max(1, width - 1)) * last
        left = int(position)
        right = min(last, left + 1)
        frac = position - left
        out.append((values[left] * (1.0 - frac)) + (values[right] * frac))
    return out


def _format_poll_interval_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds <= 0.0:
        return "off"
    rounded = round(seconds)
    if math.isclose(seconds, rounded, abs_tol=1e-6):
        return f"{int(rounded)}s"
    return f"{seconds:.1f}".rstrip("0").rstrip(".") + "s"


def _next_auto_refresh_interval_seconds(current: float) -> float:
    current = max(0.0, float(current))
    for index, preset in enumerate(AUTO_REFRESH_INTERVAL_PRESETS):
        if math.isclose(current, preset, abs_tol=1e-6):
            return AUTO_REFRESH_INTERVAL_PRESETS[(index + 1) % len(AUTO_REFRESH_INTERVAL_PRESETS)]
    if current > AUTO_REFRESH_INTERVAL_PRESETS[0]:
        return AUTO_REFRESH_INTERVAL_PRESETS[0]
    for upper, lower in zip(AUTO_REFRESH_INTERVAL_PRESETS, AUTO_REFRESH_INTERVAL_PRESETS[1:]):
        if upper > current > lower:
            return lower
    return AUTO_REFRESH_INTERVAL_PRESETS[0]


def _pool_drain_plot_points_from_values(
    values: list[float],
    *,
    left: int,
    right: int,
    baseline_y: int,
    max_amplitude: int,
    min_y: int,
    max_y: int,
    max_visible_polls: int = MAX_VISIBLE_POOL_DRAIN_POLLS,
) -> list[tuple[int, int]]:
    visible = list(values[-max(1, int(max_visible_polls)) :]) if values else [0.0]
    slot_count = max(2, int(max_visible_polls))
    if len(visible) == 1:
        x_positions = [right]
    else:
        start_slot = max(0, slot_count - len(visible))
        x_span = max(0, right - left)
        x_positions = [
            left + round((x_span * (start_slot + index)) / max(1, slot_count - 1))
            for index in range(len(visible))
        ]
    peak = max(max(visible, default=0.0), 1.5)
    points: list[tuple[int, int]] = []
    for x, value in zip(x_positions, visible):
        normalized = value / peak if peak > 0 else 0.0
        y = baseline_y - int(round(normalized * max_amplitude))
        points.append((x, max(min_y, min(max_y, y))))
    return points


def _draw_text_graph_segment(
    canvas: list[list[str]],
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    endpoint_char: str,
) -> None:
    x0, y0 = start
    x1, y1 = end
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    diagonal = "╱" if y1 < y0 else "╲"
    for step in range(steps + 1):
        x = round(x0 + ((x1 - x0) * (step / steps)))
        y = round(y0 + ((y1 - y0) * (step / steps)))
        if y < 0 or y >= len(canvas) or x < 0 or x >= len(canvas[y]):
            continue
        if step == steps:
            canvas[y][x] = endpoint_char
        elif step == 0:
            canvas[y][x] = "•"
        elif y0 == y1:
            canvas[y][x] = "─"
        elif x0 == x1:
            canvas[y][x] = "│"
        else:
            canvas[y][x] = diagonal



def _render_pool_drain_meter(
    samples: list[CodexPoolDrainSample],
    *,
    width: int,
    height: int,
    auto_refresh_seconds: float,
) -> list[str]:
    width = max(28, int(width))
    height = max(6, int(height))
    latest = samples[-1] if samples else None
    label = latest.label if latest else "5h"
    poll_interval_text = _format_poll_interval_seconds(auto_refresh_seconds)
    visible_polls = min(MAX_VISIBLE_POOL_DRAIN_POLLS, len(samples))
    top = _meter_border_line(width, title=f"{label} pool burn meter", top=True)
    bottom = _meter_border_line(width, title="rate history", top=False)
    if latest is None:
        stat1 = _meter_text_line(width, "warming up — waiting for two live polls")
        stat2 = _meter_text_line(
            width,
            f"drain n/a  •  drop n/a  •  auto {poll_interval_text} all",
        )
        graph_height = max(1, height - 4)
        graph_lines = [_meter_text_line(width, "·" * max(1, width - 6)) for _ in range(graph_height)]
        return [top, stat1, stat2, *graph_lines[:graph_height], bottom][:height]

    stat1 = _meter_text_line(
        width,
        f"drain {latest.rate_percent_per_hour:.1f} pool-%/h  •  drop {latest.dropping_accounts}/{latest.compared_accounts}",
    )
    stat2 = _meter_text_line(
        width,
        (
            f"avg left {latest.average_available_percent:.0f}%  •  {visible_polls}/{MAX_VISIBLE_POOL_DRAIN_POLLS} polls  •  "
            f"auto {poll_interval_text} all"
        ),
    )

    graph_height = max(1, height - 4)
    graph_width = max(4, width - 2)
    baseline_y = graph_height - 1
    canvas = [[" " for _ in range(graph_width)] for _ in range(graph_height)]
    for x in range(graph_width):
        canvas[baseline_y][x] = "─"

    values = [max(0.0, sample.rate_percent_per_hour) for sample in samples]
    points = _pool_drain_plot_points_from_values(
        values,
        left=0,
        right=max(0, graph_width - 1),
        baseline_y=baseline_y,
        max_amplitude=max(1, graph_height - 1),
        min_y=0,
        max_y=baseline_y,
    )
    if len(points) == 1:
        x, y = points[0]
        canvas[y][x] = "●"
    else:
        for index, point in enumerate(points):
            endpoint_char = "●" if index == len(points) - 1 else "•"
            if index == 0:
                x, y = point
                canvas[y][x] = "•"
                continue
            _draw_text_graph_segment(canvas, points[index - 1], point, endpoint_char=endpoint_char)

    graph_lines = [_meter_text_line(width, "".join(row)) for row in canvas]
    return [top, stat1, stat2, *graph_lines[:graph_height], bottom][:height]


def _pool_drain_palette() -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    return (
        (2, 9, 4),
        (24, 126, 56),
        (18, 90, 38),
        (72, 255, 108),
        (243, 255, 246),
    )


def _mix_color(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    intensity: float,
) -> tuple[int, int, int]:
    intensity = max(0.0, min(1.0, intensity))
    return (
        round(start[0] + ((end[0] - start[0]) * intensity)),
        round(start[1] + ((end[1] - start[1]) * intensity)),
        round(start[2] + ((end[2] - start[2]) * intensity)),
    )


def _blend_pixel(
    pixels: bytearray,
    width_px: int,
    height_px: int,
    x: int,
    y: int,
    color: tuple[int, int, int, int],
) -> None:
    if x < 0 or y < 0 or x >= width_px or y >= height_px:
        return
    index = ((y * width_px) + x) * 4
    src_r, src_g, src_b, src_a = color
    if src_a <= 0:
        return
    if src_a >= 255:
        pixels[index] = src_r
        pixels[index + 1] = src_g
        pixels[index + 2] = src_b
        pixels[index + 3] = 255
        return
    dst_a = pixels[index + 3]
    if dst_a >= 255:
        inv_alpha = 255 - src_a
        pixels[index] = ((src_r * src_a) + (pixels[index] * inv_alpha) + 127) // 255
        pixels[index + 1] = ((src_g * src_a) + (pixels[index + 1] * inv_alpha) + 127) // 255
        pixels[index + 2] = ((src_b * src_a) + (pixels[index + 2] * inv_alpha) + 127) // 255
        pixels[index + 3] = 255
        return
    dst_r, dst_g, dst_b = pixels[index], pixels[index + 1], pixels[index + 2]
    src_alpha = src_a / 255.0
    dst_alpha = dst_a / 255.0
    out_alpha = src_alpha + (dst_alpha * (1.0 - src_alpha))
    if out_alpha <= 0:
        return
    pixels[index] = round(((src_r * src_alpha) + (dst_r * dst_alpha * (1.0 - src_alpha))) / out_alpha)
    pixels[index + 1] = round(((src_g * src_alpha) + (dst_g * dst_alpha * (1.0 - src_alpha))) / out_alpha)
    pixels[index + 2] = round(((src_b * src_alpha) + (dst_b * dst_alpha * (1.0 - src_alpha))) / out_alpha)
    pixels[index + 3] = round(out_alpha * 255)


def _stamp_pixel(
    pixels: bytearray,
    width_px: int,
    height_px: int,
    x: int,
    y: int,
    color: tuple[int, int, int, int],
    *,
    radius: int,
) -> None:
    for dx, dy in _circle_offsets(radius):
        _blend_pixel(pixels, width_px, height_px, x + dx, y + dy, color)


def _draw_polyline(
    pixels: bytearray,
    width_px: int,
    height_px: int,
    points: list[tuple[int, int]],
    color: tuple[int, int, int, int],
    *,
    radius: int,
) -> None:
    if not points:
        return
    previous_x, previous_y = points[0]
    _stamp_pixel(pixels, width_px, height_px, previous_x, previous_y, color, radius=radius)
    for x, y in points[1:]:
        steps = max(abs(x - previous_x), abs(y - previous_y), 1)
        for step in range(steps + 1):
            t = step / steps
            draw_x = round(previous_x + ((x - previous_x) * t))
            draw_y = round(previous_y + ((y - previous_y) * t))
            _stamp_pixel(pixels, width_px, height_px, draw_x, draw_y, color, radius=radius)
        previous_x, previous_y = x, y


def _draw_horizontal_band(
    pixels: bytearray,
    width_px: int,
    height_px: int,
    y: int,
    color: tuple[int, int, int, int],
    *,
    radius: int,
) -> None:
    if y < 0 or y >= height_px:
        return
    for band_y in range(max(0, y - radius), min(height_px, y + radius + 1)):
        for x in range(width_px):
            _blend_pixel(pixels, width_px, height_px, x, band_y, color)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    payload = chunk_type + data
    return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)


def _encode_png_rgba(width_px: int, height_px: int, pixels: bytearray) -> bytes:
    stride = width_px * 4
    raw = bytearray()
    for row in range(height_px):
        raw.append(0)
        start = row * stride
        raw.extend(pixels[start : start + stride])
    compressed = zlib.compress(bytes(raw), level=9)
    ihdr = struct.pack(">IIBBBBB", width_px, height_px, 8, 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", compressed) + _png_chunk(b"IEND", b"")


def _fill_pool_drain_background(pixels: bytearray, width_px: int, height_px: int) -> None:
    background, _border, _baseline, glow_color, _core = _pool_drain_palette()
    center_y = (height_px - 1) / 2
    row_stride = width_px * 4
    for y in range(height_px):
        distance = abs(y - center_y) / max(1.0, center_y)
        haze = max(0.0, 1.0 - distance)
        scanline = 0.92 if y % 2 else 1.0
        red = round((background[0] + (glow_color[0] * 0.024 * haze)) * scanline)
        green = round((background[1] + (glow_color[1] * 0.048 * haze)) * scanline)
        blue = round((background[2] + (glow_color[2] * 0.02 * haze)) * scanline)
        row = bytes((red, green, blue, 255)) * width_px
        start = y * row_stride
        pixels[start : start + row_stride] = row


def _draw_pool_drain_border(pixels: bytearray, width_px: int, height_px: int) -> None:
    _background, border, _baseline, glow_color, _core = _pool_drain_palette()
    bright = (*_mix_color(border, glow_color, 0.28), 255)
    halo = (*glow_color, 34)
    inset = 1
    for x in range(inset, width_px - inset):
        _blend_pixel(pixels, width_px, height_px, x, inset, bright)
        _blend_pixel(pixels, width_px, height_px, x, height_px - 1 - inset, bright)
        if inset + 1 < height_px:
            _blend_pixel(pixels, width_px, height_px, x, inset + 1, halo)
            _blend_pixel(pixels, width_px, height_px, x, height_px - 2 - inset, halo)
    for y in range(inset, height_px - inset):
        _blend_pixel(pixels, width_px, height_px, inset, y, bright)
        _blend_pixel(pixels, width_px, height_px, width_px - 1 - inset, y, bright)
        if inset + 1 < width_px:
            _blend_pixel(pixels, width_px, height_px, inset + 1, y, halo)
            _blend_pixel(pixels, width_px, height_px, width_px - 2 - inset, y, halo)


def _pool_drain_points(samples: list[CodexPoolDrainSample], *, width_px: int, height_px: int) -> list[tuple[int, int]]:
    left = 8
    right = max(left + 4, width_px - 9)
    baseline_y = height_px // 2
    max_amplitude = max(4, baseline_y - 8)
    values = [max(0.0, sample.rate_percent_per_hour) for sample in samples] if samples else [0.0]
    return _pool_drain_plot_points_from_values(
        values,
        left=left,
        right=right,
        baseline_y=baseline_y,
        max_amplitude=max_amplitude,
        min_y=6,
        max_y=max(6, height_px - 7),
    )


def _build_pool_drain_png_frame(
    *,
    samples: list[CodexPoolDrainSample],
    width_px: int,
    height_px: int,
) -> bytes:
    pixels = bytearray(width_px * height_px * 4)
    _fill_pool_drain_background(pixels, width_px, height_px)
    _draw_pool_drain_border(pixels, width_px, height_px)

    _background, _border, baseline, glow_color, core = _pool_drain_palette()
    baseline_y = height_px // 2
    baseline_color = _mix_color(baseline, glow_color, 0.24)
    baseline_core = _mix_color(glow_color, core, 0.12)
    _draw_horizontal_band(pixels, width_px, height_px, baseline_y, (*glow_color, 18), radius=6)
    _draw_horizontal_band(pixels, width_px, height_px, baseline_y, (*baseline_color, 46), radius=2)
    _draw_horizontal_band(pixels, width_px, height_px, baseline_y, (*baseline_core, 112), radius=0)

    points = _pool_drain_points(samples, width_px=width_px, height_px=height_px)
    outer_glow = (*glow_color, 40)
    mid_glow = (*glow_color, 96)
    trace_color = (*_mix_color(glow_color, core, 0.42), 192)
    core_color = (*core, 255)
    _draw_polyline(pixels, width_px, height_px, points, outer_glow, radius=10)
    _draw_polyline(pixels, width_px, height_px, points, mid_glow, radius=6)
    _draw_polyline(pixels, width_px, height_px, points, trace_color, radius=3)
    _draw_polyline(pixels, width_px, height_px, points, core_color, radius=1)
    if points:
        cursor_x, cursor_y = points[-1]
        _stamp_pixel(pixels, width_px, height_px, cursor_x, cursor_y, (*core, 255), radius=2)

    return _encode_png_rgba(width_px, height_px, pixels)


def _build_kitty_placeholder_grid(*, image_id: int, columns: int, rows: int) -> list[str]:
    if columns <= 0 or rows <= 0:
        return []
    if columns > len(_KITTY_DIACRITICS) or rows > len(_KITTY_DIACRITICS):
        raise ValueError("kitty placeholder grid exceeds supported diacritic range")
    return [
        "".join(
            f"{_KITTY_PLACEHOLDER}{_KITTY_DIACRITICS[row]}{_KITTY_DIACRITICS[column]}"
            for column in range(columns)
        )
        for row in range(rows)
    ]


def _build_kitty_transmission(*, image_id: int, png_bytes: bytes, columns: int, rows: int) -> str:
    encoded = base64.standard_b64encode(png_bytes).decode("ascii")
    chunks = [encoded[index : index + _KITTY_CHUNK_SIZE] for index in range(0, len(encoded), _KITTY_CHUNK_SIZE)]
    pieces: list[str] = []
    last_index = len(chunks) - 1
    for index, chunk in enumerate(chunks):
        more = 0 if index == last_index else 1
        if index == 0:
            control = f"a=T,f=100,q=2,C=1,U=1,i={image_id},c={columns},r={rows},m={more}"
        else:
            control = f"m={more}"
        pieces.append(f"\x1b_G{control};{chunk}\x1b\\")
    return "".join(pieces)


def _terminal_graphics_sequence(sequence: str) -> str:
    if not os.environ.get("TMUX"):
        return sequence
    return f"\x1bPtmux;{sequence.replace(chr(27), chr(27) * 2)}\x1b\\"


def _terminal_cell_pixels_from_fd(fd: int) -> tuple[int, int] | None:
    try:
        buffer = array.array("H", [0, 0, 0, 0])
        fcntl.ioctl(fd, termios.TIOCGWINSZ, buffer, True)
        rows, columns, width_px, height_px = buffer
        if rows > 0 and columns > 0 and width_px > 0 and height_px > 0:
            return (max(1, round(width_px / columns)), max(1, round(height_px / rows)))
    except (OSError, ValueError):
        return None
    return None


def _terminal_cell_pixels() -> tuple[int, int]:
    for stream in (sys.__stdout__, sys.stdout, sys.__stderr__, sys.stderr, sys.__stdin__, sys.stdin):
        fileno = getattr(stream, "fileno", None)
        if fileno is None:
            continue
        try:
            fd = fileno()
        except (OSError, ValueError):
            continue
        if not isinstance(fd, int) or fd < 0:
            continue
        size = _terminal_cell_pixels_from_fd(fd)
        if size is not None:
            return size
    try:
        with open("/dev/tty", "rb", buffering=0) as tty:
            size = _terminal_cell_pixels_from_fd(tty.fileno())
            if size is not None:
                return size
    except OSError:
        pass
    return _DEFAULT_CELL_PIXELS


def _image_pixel_size(columns: int, rows: int) -> tuple[int, int]:
    cell_width, cell_height = _terminal_cell_pixels()
    width = max(220, min(1600, columns * max(4, round(cell_width * 1.0))))
    height = max(96, min(540, rows * max(10, round(cell_height * 1.2))))
    return width, height


def _availability_metrics(snapshot: CodexProfileSnapshot) -> tuple[float, float]:
    values = [
        value
        for value in (
            _window_available_percent(snapshot.primary_window),
            _window_available_percent(snapshot.secondary_window),
        )
        if value is not None
    ]
    if not values:
        return 0.0, 0.0
    return min(values), sum(values)


def _limit_sort_rank(snapshot: CodexProfileSnapshot) -> tuple[int, float, float]:
    primary_left = _window_available_percent(snapshot.primary_window) or 0.0
    secondary_left = _window_available_percent(snapshot.secondary_window) or 0.0
    primary_exhausted = _window_is_exhausted(snapshot.primary_window)
    secondary_exhausted = _window_is_exhausted(snapshot.secondary_window)

    if primary_exhausted and not secondary_exhausted:
        return 0, -secondary_left, -primary_left
    if secondary_exhausted and not primary_exhausted:
        return 1, -primary_left, -secondary_left
    if primary_exhausted and secondary_exhausted:
        return 2, 0.0, 0.0
    return 3, -primary_left, -secondary_left


def _availability_sort_key(snapshot: CodexProfileSnapshot) -> tuple[Any, ...]:
    if snapshot.auth_badge == "OK" and snapshot.state == STATE_LIMITED:
        return (
            _sort_bucket(snapshot),
            *_limit_sort_rank(snapshot),
            snapshot.display_name.lower(),
            snapshot.index,
        )
    min_available, total_available = _availability_metrics(snapshot)
    return (
        _sort_bucket(snapshot),
        -min_available,
        -total_available,
        snapshot.display_name.lower(),
        snapshot.index,
    )


def _sort_bucket(snapshot: CodexProfileSnapshot) -> int:
    if snapshot.auth_badge == "OK" and snapshot.state == STATE_AVAILABLE:
        return 0
    if snapshot.auth_badge == "OK" and snapshot.state == STATE_LIMITED:
        return 1
    if snapshot.auth_badge == "ERR" or snapshot.state == STATE_ERROR:
        return 2
    if snapshot.auth_badge == "LOAD" or snapshot.state == STATE_LOADING:
        return 3
    if snapshot.auth_badge == "DEAD" or snapshot.state == STATE_DEACTIVATED:
        return 4
    return 5


def _sort_codex_profile_snapshots(
    snapshots: list[CodexProfileSnapshot],
    *,
    mode: str = SORT_AVAILABILITY,
) -> list[CodexProfileSnapshot]:
    if mode == SORT_NAME:
        ordered = sorted(
            snapshots,
            key=lambda snapshot: (
                snapshot.display_name.lower(),
                _sort_bucket(snapshot),
                snapshot.index,
            ),
        )
    elif mode == SORT_REFRESH:
        ordered = sorted(
            snapshots,
            key=lambda snapshot: (
                -(_parse_timestamp(snapshot.last_refresh).timestamp() if _parse_timestamp(snapshot.last_refresh) else 0.0),
                _sort_bucket(snapshot),
                -_availability_metrics(snapshot)[0],
                snapshot.display_name.lower(),
                snapshot.index,
            ),
        )
    else:
        ordered = sorted(snapshots, key=_availability_sort_key)
    for index, snapshot in enumerate(ordered, start=1):
        snapshot.index = index
    return ordered


def _boot_loading_text(frame: int) -> str:
    spinner = "|/-\\"
    return f"Loading {spinner[frame % len(spinner)]}"


def _format_reset_compact(reset_at_ms: Optional[int]) -> str:
    if not reset_at_ms:
        return ""
    remaining_ms = max(0, int(reset_at_ms - time.time() * 1000))
    seconds_total = remaining_ms // 1000
    minutes_total, seconds = divmod(seconds_total, 60)
    hours_total, minutes = divmod(minutes_total, 60)
    days, hours = divmod(hours_total, 24)
    if days:
        return f"@{days}d{hours}h"
    if hours:
        return f"@{hours}h{minutes}m" if minutes else f"@{hours}h"
    if minutes:
        return f"@{minutes}m"
    return "@now" if seconds_total == 0 else f"@{seconds}s"


def _format_reset(reset_at_ms: Optional[int]) -> str:
    if not reset_at_ms:
        return "unknown"
    remaining_ms = max(0, int(reset_at_ms - time.time() * 1000))
    seconds = remaining_ms // 1000
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _parse_timestamp(timestamp: Optional[str]) -> Optional[datetime]:
    if not timestamp:
        return None
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _format_timestamp(timestamp: Optional[str]) -> str:
    dt = _parse_timestamp(timestamp)
    if dt is None:
        return timestamp or "never"
    return dt.strftime("%Y-%m-%d %H:%MZ")


def _guess_display_name(entry: PooledCredential, live_email: Optional[str] = None) -> str:
    if live_email:
        return live_email
    token_label = label_from_token(entry.access_token or "", entry.label or entry.id)
    if token_label and token_label != entry.id:
        return token_label
    return entry.label or entry.id


def _format_plan(data: dict[str, Any]) -> Optional[str]:
    plan = _safe_str(data.get("plan_type"))
    credits = data.get("credits") if isinstance(data.get("credits"), dict) else None
    balance = None
    if isinstance(credits, dict) and credits.get("balance") is not None:
        try:
            balance = float(credits.get("balance"))
        except (TypeError, ValueError):
            balance = None
    if balance is not None:
        balance_text = f"${balance:.2f}"
        return f"{plan} ({balance_text})" if plan else balance_text
    return plan


def _build_window(payload: Any, *, fallback_label: str) -> Optional[CodexUsageWindow]:
    if not isinstance(payload, dict):
        return None
    reset_at = payload.get("reset_at")
    reset_at_ms = None
    if isinstance(reset_at, (int, float)):
        reset_at_ms = int(float(reset_at) * 1000)
    seconds = payload.get("limit_window_seconds")
    if isinstance(seconds, (int, float)) and float(seconds) > 0:
        label = f"{int(round(float(seconds) / 3600.0))}h"
    else:
        label = fallback_label
    return CodexUsageWindow(
        label=label,
        used_percent=_clamp_percent(payload.get("used_percent")),
        reset_at_ms=reset_at_ms,
    )


def _resolve_secondary_label(
    *,
    limit_window_seconds: Any,
    secondary_reset_at: Any,
    primary_reset_at: Any,
) -> str:
    try:
        hours = int(round(float(limit_window_seconds) / 3600.0))
    except (TypeError, ValueError):
        hours = 168

    if hours >= 168:
        return "Week"
    if hours < 24:
        return f"{hours}h"
    if isinstance(secondary_reset_at, (int, float)) and isinstance(primary_reset_at, (int, float)):
        if float(secondary_reset_at) - float(primary_reset_at) >= WEEKLY_RESET_GAP_SECONDS:
            return "Week"
    if hours == 24:
        return "Day"
    return f"{hours}h"


def _build_secondary_window(payload: Any, primary_window: Optional[dict[str, Any]]) -> Optional[CodexUsageWindow]:
    if not isinstance(payload, dict):
        return None
    label = _resolve_secondary_label(
        limit_window_seconds=payload.get("limit_window_seconds"),
        secondary_reset_at=payload.get("reset_at"),
        primary_reset_at=(primary_window or {}).get("reset_at"),
    )
    reset_at = payload.get("reset_at")
    reset_at_ms = None
    if isinstance(reset_at, (int, float)):
        reset_at_ms = int(float(reset_at) * 1000)
    return CodexUsageWindow(
        label=label,
        used_percent=_clamp_percent(payload.get("used_percent")),
        reset_at_ms=reset_at_ms,
    )


def _limit_reason(live: CodexLiveStatus) -> str:
    hit: list[str] = []
    for window in (live.primary_window, live.secondary_window):
        if _window_is_exhausted(window):
            hit.append(window.label)
    if hit:
        return f"{' + '.join(hit)} limit reached"
    if live.limit_reached:
        return "Usage limit reached"
    return "Usage blocked"


def _request_usage(
    client: httpx.Client,
    entry: PooledCredential,
) -> httpx.Response:
    headers = {
        "Authorization": f"Bearer {entry.access_token}",
        "User-Agent": "Hermes Auth View",
        "Accept": "application/json",
    }
    account_id = _safe_str(getattr(entry, "account_id", None))
    if account_id:
        headers["ChatGPT-Account-Id"] = account_id
    return client.get(USAGE_URL, headers=headers)


def _fetch_codex_live_status(
    pool,
    entry: PooledCredential,
    timeout_seconds: float,
) -> tuple[PooledCredential, CodexLiveStatus]:
    timeout = httpx.Timeout(max(5.0, float(timeout_seconds)))
    try:
        with httpx.Client(timeout=timeout) as client:
            response = _request_usage(client, entry)
            if response.status_code in {401, 403} and entry.refresh_token:
                try:
                    refreshed = auth_mod.refresh_codex_oauth_pure(
                        entry.access_token,
                        entry.refresh_token,
                        timeout_seconds=max(5.0, float(timeout_seconds)),
                    )
                    entry = (
                        pool.update_entry(
                            entry.id,
                            access_token=refreshed["access_token"],
                            refresh_token=refreshed["refresh_token"],
                            last_refresh=refreshed.get("last_refresh"),
                            last_status=None,
                            last_status_at=None,
                            last_error_code=None,
                        )
                        or entry
                    )
                    response = _request_usage(client, entry)
                except auth_mod.AuthError as exc:
                    return entry, CodexLiveStatus(
                        http_status=response.status_code,
                        error=str(exc),
                        deactivated=True,
                    )

            if response.status_code != 200:
                return entry, CodexLiveStatus(
                    http_status=response.status_code,
                    error=f"HTTP {response.status_code}",
                    deactivated=response.status_code in {401, 403},
                )

            data = response.json() if response.content else {}
            if not isinstance(data, dict):
                return entry, CodexLiveStatus(
                    http_status=200,
                    error="Usage API returned an unexpected payload.",
                )

            rate_limit = data.get("rate_limit") if isinstance(data.get("rate_limit"), dict) else {}
            primary_payload = rate_limit.get("primary_window") if isinstance(rate_limit, dict) else None
            secondary_payload = rate_limit.get("secondary_window") if isinstance(rate_limit, dict) else None
            primary_window = _build_window(primary_payload, fallback_label="5h")
            secondary_window = _build_secondary_window(secondary_payload, primary_payload)

            local_cooldown_cleared = False
            if rate_limit.get("allowed") is True and entry.last_status == STATUS_EXHAUSTED:
                updated = pool.update_entry(
                    entry.id,
                    last_status=None,
                    last_status_at=None,
                    last_error_code=None,
                )
                if updated is not None:
                    entry = updated
                    local_cooldown_cleared = True

            live_account_id = _safe_str(data.get("account_id"))
            if live_account_id and live_account_id != _safe_str(getattr(entry, "account_id", None)):
                updated = pool.update_entry(entry.id, account_id=live_account_id)
                if updated is not None:
                    entry = updated

            return entry, CodexLiveStatus(
                http_status=200,
                email=_safe_str(data.get("email")),
                plan=_format_plan(data),
                allowed=bool(rate_limit.get("allowed")) if "allowed" in rate_limit else None,
                limit_reached=bool(rate_limit.get("limit_reached")) if "limit_reached" in rate_limit else None,
                primary_window=primary_window,
                secondary_window=secondary_window,
                local_cooldown_cleared=local_cooldown_cleared,
            )
    except httpx.HTTPError as exc:
        return entry, CodexLiveStatus(error=f"Network error: {exc}")
    except Exception as exc:
        return entry, CodexLiveStatus(error=f"Unexpected error: {exc}")


def _snapshot_from_entry(
    index: int,
    entry: PooledCredential,
    live: Optional[CodexLiveStatus] = None,
) -> CodexProfileSnapshot:
    live = live or CodexLiveStatus()
    any_window_exhausted = any(
        _window_is_exhausted(window)
        for window in (live.primary_window, live.secondary_window)
    )

    if live.deactivated:
        auth_badge = "DEAD"
        auth_reason = live.error or "Token invalid or profile deactivated"
        state = STATE_DEACTIVATED
        state_badge = "n/a"
        reason = auth_reason
    elif live.error:
        auth_badge = "ERR"
        auth_reason = live.error
        state = STATE_ERROR
        state_badge = "ERR"
        reason = live.error
    elif live.allowed is False or live.limit_reached or any_window_exhausted:
        auth_badge = "OK"
        auth_reason = "Authenticated"
        state = STATE_LIMITED
        state_badge = _BADGES[state]
        reason = _limit_reason(live)
    elif live.allowed is True or live.primary_window or live.secondary_window:
        auth_badge = "OK"
        auth_reason = "Authenticated"
        state = STATE_AVAILABLE
        state_badge = _BADGES[state]
        reason = "Available"
        if live.local_cooldown_cleared:
            reason = "Available (local cooldown cleared)"
    else:
        auth_badge = "LOAD"
        auth_reason = "Loading..."
        state = STATE_LOADING
        state_badge = _BADGES[state]
        reason = "Loading..."

    display_name = _guess_display_name(entry, live.email)
    return CodexProfileSnapshot(
        index=index,
        entry_id=entry.id,
        label=entry.label,
        display_name=display_name,
        auth_badge=auth_badge,
        state=state,
        state_badge=state_badge,
        state_reason=reason,
        primary_window=live.primary_window,
        secondary_window=live.secondary_window,
        email=live.email,
        plan=live.plan,
        source=entry.source,
        auth_type=entry.auth_type,
        last_refresh=entry.last_refresh,
        local_status=entry.last_status,
        local_error_code=entry.last_error_code,
        http_status=live.http_status,
        local_cooldown_cleared=live.local_cooldown_cleared,
        auth_reason=auth_reason,
    )


FetchProfileFn = Callable[[Any, PooledCredential, float], tuple[PooledCredential, CodexLiveStatus]]


def build_codex_profile_snapshots(
    *,
    provider: str = DEFAULT_PROVIDER,
    timeout_seconds: float = 10.0,
    fetch_profile: Optional[FetchProfileFn] = None,
    entry_ids: Optional[set[str]] = None,
) -> list[CodexProfileSnapshot]:
    if provider != DEFAULT_PROVIDER:
        raise SystemExit(f"Auth view currently supports only {DEFAULT_PROVIDER}.")
    fetcher = fetch_profile or _fetch_codex_live_status
    pool = load_pool(provider)
    entries = list(pool.entries())
    if entry_ids is not None:
        allowed_ids = {str(entry_id) for entry_id in entry_ids}
        entries = [entry for entry in entries if entry.id in allowed_ids]
    if not entries:
        return []

    def _poll_one(item: tuple[int, PooledCredential]) -> CodexProfileSnapshot:
        index, entry = item
        current_entry, live = fetcher(pool, entry, timeout_seconds)
        return _snapshot_from_entry(index, current_entry, live)

    if len(entries) == 1:
        snapshots = [_poll_one((1, entries[0]))]
    else:
        max_workers = max(2, min(MAX_POLL_WORKERS, len(entries)))
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="codex-auth-view") as executor:
            snapshots = list(executor.map(_poll_one, enumerate(entries, start=1)))

    return _sort_codex_profile_snapshots(snapshots)


def build_codex_profile_placeholders(provider: str = DEFAULT_PROVIDER) -> list[CodexProfileSnapshot]:
    if provider != DEFAULT_PROVIDER:
        raise SystemExit(f"Auth view currently supports only {DEFAULT_PROVIDER}.")
    pool = load_pool(provider)
    return [_snapshot_from_entry(index, entry) for index, entry in enumerate(pool.entries(), start=1)]


def render_codex_auth_smoke_test(
    *,
    provider: str = DEFAULT_PROVIDER,
    timeout_seconds: float = 10.0,
) -> str:
    snapshots = build_codex_profile_snapshots(provider=provider, timeout_seconds=timeout_seconds)
    if not snapshots:
        return "Codex auth view: no pooled profiles found."
    lines = [f"Codex auth view: {len(snapshots)} profiles"]
    for snapshot in snapshots:
        short_label = snapshot.primary_window.label if snapshot.primary_window else "5h"
        long_label = snapshot.secondary_window.label if snapshot.secondary_window else "Week"
        short_pct = _format_percent(_window_available_percent(snapshot.primary_window))
        long_pct = _format_percent(_window_available_percent(snapshot.secondary_window))
        short_reset = _format_reset_compact(snapshot.primary_window.reset_at_ms if snapshot.primary_window else None)
        long_reset = _format_reset_compact(snapshot.secondary_window.reset_at_ms if snapshot.secondary_window else None)
        short_suffix = f" {short_reset}" if short_reset else ""
        long_suffix = f" {long_reset}" if long_reset else ""
        lines.append(
            f"{snapshot.index:>2}. {snapshot.display_name:<30} {snapshot.auth_badge:<4} {snapshot.state_badge:<5} "
            f"{long_label}_left={long_pct}{long_suffix} {short_label}_left={short_pct}{short_suffix}"
        )
    return "\n".join(lines)


class CodexAuthTui:
    def __init__(
        self,
        *,
        provider: str = DEFAULT_PROVIDER,
        timeout_seconds: float = 10.0,
        kitty_meter: Optional[bool] = None,
        auto_refresh_interval_seconds: float = AUTO_REFRESH_INTERVAL_SECONDS,
    ):
        if provider != DEFAULT_PROVIDER:
            raise SystemExit(f"Auth view currently supports only {DEFAULT_PROVIDER}.")
        self.provider = provider
        self.timeout_seconds = timeout_seconds
        self.sort_mode = SORT_AVAILABILITY
        self.snapshots: list[CodexProfileSnapshot] = []
        self.selected_index = 0
        self.message = "Loading Codex profiles..."
        self.loading = False
        self.kitty_meter_enabled = _kitty_meter_enabled_from_env() if kitty_meter is None else bool(kitty_meter)
        self.auto_refresh_interval_seconds = max(0.0, float(auto_refresh_interval_seconds))
        self._lock = threading.Lock()
        self._refresh_generation = 0
        self._selected_entry_id: Optional[str] = None
        self._remove_confirm_entry_id: Optional[str] = None
        self._remove_confirm_deadline = 0.0
        self._pool_drain_history: list[CodexPoolDrainSample] = []
        self._last_primary_available: dict[str, tuple[float, Optional[int]]] = {}
        self._last_primary_capture_at: Optional[float] = None
        self._next_auto_refresh_at = 0.0
        self._kitty_image_id = next(_KITTY_IMAGE_IDS) & 0xFFFFFF
        self._last_kitty_graph_signature: Optional[tuple[Any, ...]] = None

    def _start_refresh(
        self,
        *,
        initial: bool = False,
        show_placeholders: bool = False,
        refresh_scope: str = REFRESH_SCOPE_ALL,
    ) -> int:
        with self._lock:
            self._refresh_generation += 1
            generation = self._refresh_generation
            if show_placeholders and (not self.snapshots or initial):
                try:
                    self.snapshots = build_codex_profile_placeholders(self.provider)
                except Exception:
                    self.snapshots = []
            self.loading = True
            if initial:
                self.message = "Polling live Codex profiles..."
            else:
                self.message = "Refreshing Codex profiles..."
            self._remove_confirm_entry_id = None
            self._remove_confirm_deadline = 0.0
            selected = self._selected_snapshot_unlocked()
            self._selected_entry_id = selected.entry_id if selected else None
            return generation

    def _auto_refresh_scope_at(self, now: float) -> Optional[str]:
        with self._lock:
            if not self.kitty_meter_enabled or self.loading:
                return None
            if self.auto_refresh_interval_seconds > 0 and self._next_auto_refresh_at > 0 and now >= self._next_auto_refresh_at:
                return REFRESH_SCOPE_ALL
        return None

    def _fetch_live_snapshots(
        self,
        *,
        refresh_scope: str = REFRESH_SCOPE_ALL,
    ) -> tuple[list[CodexProfileSnapshot], str, str]:
        try:
            snapshots = build_codex_profile_snapshots(
                provider=self.provider,
                timeout_seconds=self.timeout_seconds,
            )
            message = (
                f"Loaded {len(snapshots)} live Codex profile{'s' if len(snapshots) != 1 else ''}."
                if snapshots
                else "No Codex profiles found in Hermes auth."
            )
            return snapshots, message, REFRESH_SCOPE_ALL
        except Exception as exc:
            message = f"Refresh failed: {exc}"
            return [], message, REFRESH_SCOPE_ALL

    def _apply_sort_mode(self, snapshots: list[CodexProfileSnapshot], mode: Optional[str] = None) -> list[CodexProfileSnapshot]:
        return _sort_codex_profile_snapshots(snapshots, mode=mode or self.sort_mode)

    def _set_sort_mode(self, mode: str) -> None:
        with self._lock:
            if mode not in _SORT_LABELS:
                return
            selected = self._selected_snapshot_unlocked()
            selected_id = selected.entry_id if selected else None
            self.sort_mode = mode
            self.snapshots = self._apply_sort_mode(list(self.snapshots), mode)
            if selected_id:
                for idx, snapshot in enumerate(self.snapshots):
                    if snapshot.entry_id == selected_id:
                        self.selected_index = idx
                        break
            self.selected_index = min(self.selected_index, max(0, len(self.snapshots) - 1))
            selected = self._selected_snapshot_unlocked()
            self._selected_entry_id = selected.entry_id if selected else None
            self.message = f"Sorted by {_SORT_LABELS[self.sort_mode]}."

    def _cycle_auto_refresh_interval(self) -> None:
        with self._lock:
            next_interval = _next_auto_refresh_interval_seconds(self.auto_refresh_interval_seconds)
            self.auto_refresh_interval_seconds = next_interval
            now = time.time()
            if next_interval > 0:
                self._next_auto_refresh_at = now + next_interval
            else:
                self._next_auto_refresh_at = 0.0
            self.message = f"Auto refresh set to {_format_poll_interval_seconds(next_interval)} for all Codex profiles."

    def _finish_refresh(
        self,
        generation: int,
        snapshots: list[CodexProfileSnapshot],
        message: str,
        *,
        refresh_scope: str = REFRESH_SCOPE_ALL,
    ) -> None:
        with self._lock:
            if generation != self._refresh_generation:
                return
            captured_at = time.time()
            if snapshots:
                self._record_pool_drain_sample_unlocked(snapshots, captured_at=captured_at)
            selected_id = self._selected_entry_id
            self.snapshots = self._apply_sort_mode(snapshots)
            if selected_id:
                for idx, snapshot in enumerate(self.snapshots):
                    if snapshot.entry_id == selected_id:
                        self.selected_index = idx
                        break
                else:
                    self.selected_index = min(self.selected_index, max(0, len(self.snapshots) - 1))
            else:
                self.selected_index = min(self.selected_index, max(0, len(self.snapshots) - 1))
            self.loading = False
            self.message = message
            if self.kitty_meter_enabled:
                if self.auto_refresh_interval_seconds > 0:
                    self._next_auto_refresh_at = captured_at + self.auto_refresh_interval_seconds
                else:
                    self._next_auto_refresh_at = 0.0
            selected = self._selected_snapshot_unlocked()
            self._selected_entry_id = selected.entry_id if selected else None

    def _refresh_async(self, *, refresh_scope: str = REFRESH_SCOPE_ALL) -> None:
        generation = self._start_refresh(show_placeholders=not self.snapshots, refresh_scope=refresh_scope)
        thread = threading.Thread(target=self._refresh_worker, args=(generation, refresh_scope), daemon=True)
        thread.start()

    def _refresh_blocking(self, *, initial: bool = False, refresh_scope: str = REFRESH_SCOPE_ALL) -> None:
        generation = self._start_refresh(initial=initial, show_placeholders=False, refresh_scope=refresh_scope)
        snapshots, message, actual_scope = self._fetch_live_snapshots(refresh_scope=refresh_scope)
        self._finish_refresh(generation, snapshots, message, refresh_scope=actual_scope)

    def _refresh_worker(self, generation: int, refresh_scope: str) -> None:
        snapshots, message, actual_scope = self._fetch_live_snapshots(refresh_scope=refresh_scope)
        self._finish_refresh(generation, snapshots, message, refresh_scope=actual_scope)

    def _selected_snapshot_unlocked(self) -> Optional[CodexProfileSnapshot]:
        if not self.snapshots:
            return None
        self.selected_index = max(0, min(self.selected_index, len(self.snapshots) - 1))
        return self.snapshots[self.selected_index]

    def _selected_snapshot(self) -> Optional[CodexProfileSnapshot]:
        with self._lock:
            return self._selected_snapshot_unlocked()

    def _status_counts(self) -> dict[str, int]:
        counts = {
            "auth_ok": 0,
            "dead": 0,
            "avail": 0,
            "limit": 0,
            "err": 0,
            "load": 0,
        }
        with self._lock:
            snapshots = list(self.snapshots)
        for snapshot in snapshots:
            if snapshot.auth_badge == "OK":
                counts["auth_ok"] += 1
            elif snapshot.auth_badge == "DEAD":
                counts["dead"] += 1
            elif snapshot.auth_badge == "ERR":
                counts["err"] += 1
            elif snapshot.auth_badge == "LOAD":
                counts["load"] += 1

            if snapshot.state == STATE_AVAILABLE:
                counts["avail"] += 1
            elif snapshot.state == STATE_LIMITED:
                counts["limit"] += 1
        return counts

    def _remove_selected(self) -> None:
        with self._lock:
            snapshot = self._selected_snapshot_unlocked()
            if snapshot is None:
                self.message = "No profile selected."
                return
            now = time.time()
            short_id = _short_entry_id(snapshot.entry_id)
            if self._remove_confirm_entry_id != snapshot.entry_id or now > self._remove_confirm_deadline:
                self._remove_confirm_entry_id = snapshot.entry_id
                self._remove_confirm_deadline = now + 5.0
                self.message = f"Press r again to remove [{short_id}] {snapshot.display_name}."
                return
            selected_entry_id = snapshot.entry_id
            self._remove_confirm_entry_id = None
            self._remove_confirm_deadline = 0.0

        pool = load_pool(self.provider)
        removed = pool.remove_entry(selected_entry_id)
        if removed is None:
            with self._lock:
                self.message = "Selected profile was already removed."
            self._refresh_async()
            return
        with self._lock:
            if removed is None:
                self.message = "Failed to remove selected profile."
            else:
                removed_label = snapshot.display_name if snapshot else removed.label
                self.message = f"Removed [{_short_entry_id(removed.id)}] {removed_label} from Hermes auth."
        self._refresh_async()

    def _clear_expired_confirmation(self) -> None:
        with self._lock:
            if self._remove_confirm_entry_id and time.time() > self._remove_confirm_deadline:
                self._remove_confirm_entry_id = None
                self._remove_confirm_deadline = 0.0

    def _record_pool_drain_sample_unlocked(self, snapshots: list[CodexProfileSnapshot], *, captured_at: float) -> None:
        elapsed_seconds = 0.0
        if self._last_primary_capture_at is not None:
            elapsed_seconds = max(0.0, captured_at - self._last_primary_capture_at)
        sample = _build_pool_drain_sample(
            self._last_primary_available,
            snapshots,
            elapsed_seconds=elapsed_seconds,
            captured_at=captured_at,
        )
        _, current_by_id, _, _ = _collect_primary_window_state(snapshots)
        if sample is not None:
            self._pool_drain_history.append(sample)
            if len(self._pool_drain_history) > MAX_POOL_DRAIN_SAMPLES:
                self._pool_drain_history = self._pool_drain_history[-MAX_POOL_DRAIN_SAMPLES:]
        self._last_primary_available = current_by_id
        self._last_primary_capture_at = captured_at

    def _graph_rows(self, width: int, height: int) -> int:
        if not self.kitty_meter_enabled:
            return 0
        if width < KITTY_METER_MIN_WIDTH or height < KITTY_METER_MIN_HEIGHT:
            return 0
        return 8 if height < 32 else 10

    def _graph_summary_lines(self, width: int) -> list[str]:
        with self._lock:
            latest = self._pool_drain_history[-1] if self._pool_drain_history else None
            visible_polls = min(MAX_VISIBLE_POOL_DRAIN_POLLS, len(self._pool_drain_history))
        if latest is None:
            return [
                _clip("5h token burn monitor", width - 1),
                _clip("warming up — waiting for two live polls", width - 1),
            ]
        return [
            _clip(f"{latest.label} token burn monitor", width - 1),
            _clip(
                (
                    f"drain {latest.rate_percent_per_hour:.1f} pool-%/h  •  drop {latest.dropping_accounts}/{latest.compared_accounts}  •  "
                    f"avg left {latest.average_available_percent:.0f}%  •  {visible_polls}/{MAX_VISIBLE_POOL_DRAIN_POLLS} polls  •  "
                    f"auto {_format_poll_interval_seconds(self.auto_refresh_interval_seconds)} all"
                ),
                width - 1,
            ),
        ]

    def _graph_signature_unlocked(self, columns: int, rows: int) -> tuple[Any, ...]:
        history = self._pool_drain_history[-MAX_VISIBLE_POOL_DRAIN_POLLS:]
        return (
            columns,
            rows,
            tuple((round(sample.captured_at, 3), round(sample.rate_percent_per_hour, 3)) for sample in history),
        )

    def _emit_kitty_graph(self, *, top_row: int, left_col: int, columns: int, rows: int) -> None:
        if not self.kitty_meter_enabled or columns <= 0 or rows <= 0:
            return
        with self._lock:
            history = list(self._pool_drain_history)
            signature = self._graph_signature_unlocked(columns, rows)
            needs_frame = signature != self._last_kitty_graph_signature
            if needs_frame:
                self._last_kitty_graph_signature = signature
        out = sys.__stdout__ if getattr(sys, "__stdout__", None) is not None else sys.stdout
        if out is None:
            return
        if needs_frame:
            width_px, height_px = _image_pixel_size(columns, rows)
            png_bytes = _build_pool_drain_png_frame(samples=history, width_px=width_px, height_px=height_px)
            out.write(
                _terminal_graphics_sequence(
                    _build_kitty_transmission(
                        image_id=self._kitty_image_id,
                        png_bytes=png_bytes,
                        columns=columns,
                        rows=rows,
                    )
                )
            )
        red = (self._kitty_image_id >> 16) & 0xFF
        green = (self._kitty_image_id >> 8) & 0xFF
        blue = self._kitty_image_id & 0xFF
        color_prefix = f"\x1b[38;2;{red};{green};{blue}m"
        placeholder_lines = _build_kitty_placeholder_grid(image_id=self._kitty_image_id, columns=columns, rows=rows)
        for row_offset, line in enumerate(placeholder_lines):
            out.write(f"\x1b[{top_row + row_offset + 1};{left_col + 1}H{color_prefix}{line}\x1b[39m")
        out.flush()

    def _delete_kitty_graph_image(self) -> None:
        if not self.kitty_meter_enabled:
            return
        out = sys.__stdout__ if getattr(sys, "__stdout__", None) is not None else sys.stdout
        if out is None:
            return
        out.write(_terminal_graphics_sequence(f"\x1b_Ga=d,d=I,i={self._kitty_image_id}\x1b\\"))
        out.flush()

    def _layout_sizes(self, width: int) -> tuple[int, int]:
        if width >= 120:
            return 34, 12
        if width >= 100:
            return 28, 9
        if width >= 84:
            return 20, 7
        return 14, 5

    def _label_pair(self) -> tuple[str, str]:
        short_label = "5h"
        long_label = "Week"
        with self._lock:
            snapshots = list(self.snapshots)
        for snapshot in snapshots:
            if snapshot.primary_window:
                short_label = snapshot.primary_window.label
                break
        for snapshot in snapshots:
            if snapshot.secondary_window:
                long_label = snapshot.secondary_window.label
                break
        return short_label, long_label

    def _render_row(self, snapshot: CodexProfileSnapshot, width: int, *, selected: bool = False) -> str:
        account_width, bar_width = self._layout_sizes(width)
        account = _clip(snapshot.display_name, account_width)
        week = _format_bar_with_reset(snapshot.secondary_window, bar_width)
        hour = _format_bar_with_reset(snapshot.primary_window, bar_width)
        prefix = ">" if selected else " "
        return _clip(
            f"{prefix} {snapshot.auth_badge:<4} {snapshot.state_badge:<5} {account:<{account_width}} {week} {hour}",
            width - 1,
        )

    def _detail_lines(self, width: int) -> list[str]:
        snapshot = self._selected_snapshot()
        if snapshot is None:
            return ["No Codex profiles found.", "Add one with: hermes auth add openai-codex --type oauth", ""]
        line1 = (
            f"Selected: {snapshot.display_name} [{_short_entry_id(snapshot.entry_id)}] | Auth: {snapshot.auth_badge} | "
            f"Quota: {snapshot.state_badge} | Plan: {snapshot.plan or 'unknown'} | Source: {snapshot.source}"
        )
        primary = snapshot.primary_window
        secondary = snapshot.secondary_window
        short_label = primary.label if primary else "5h"
        long_label = secondary.label if secondary else "Week"
        line2 = (
            f"{long_label} left: {_format_bar(_window_available_percent(secondary), 10)} "
            f"back {_format_reset(secondary.reset_at_ms if secondary else None)} | "
            f"{short_label} left: {_format_bar(_window_available_percent(primary), 10)} "
            f"back {_format_reset(primary.reset_at_ms if primary else None)}"
        )
        local_bits = []
        if snapshot.local_status:
            local_bits.append(snapshot.local_status)
        if snapshot.local_error_code:
            local_bits.append(str(snapshot.local_error_code))
        local_text = " ".join(local_bits) if local_bits else "clear"
        if snapshot.local_cooldown_cleared:
            local_text = "clear (auto-cleared)"
        line3 = (
            f"Hermes local: {local_text} | Last refresh: {_format_timestamp(snapshot.last_refresh)} | "
            f"Auth: {snapshot.auth_reason or 'n/a'} | Quota: {snapshot.state_reason}"
        )
        return [_clip(line1, width - 1), _clip(line2, width - 1), _clip(line3, width - 1)]

    def _draw_boot_loading(self, stdscr, frame: int = 0) -> None:
        import curses

        stdscr.erase()
        max_y, max_x = stdscr.getmaxyx()
        line = _boot_loading_text(frame)
        y = max(0, max_y // 2)
        x = max(0, (max_x - len(line)) // 2)
        attr = curses.A_BOLD if curses.has_colors() else curses.A_NORMAL
        if curses.has_colors():
            attr |= curses.color_pair(4)
        stdscr.addnstr(y, x, _clip(line, max_x - x - 1), max_x - x - 1, attr)
        stdscr.refresh()

    def _run_curses(self, stdscr) -> None:
        import curses

        locale.setlocale(locale.LC_ALL, "")
        curses.curs_set(0)
        stdscr.timeout(100)
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_GREEN, -1)
            curses.init_pair(2, curses.COLOR_YELLOW, -1)
            curses.init_pair(3, curses.COLOR_RED, -1)
            curses.init_pair(4, curses.COLOR_CYAN, -1)
            curses.init_pair(5, curses.COLOR_MAGENTA, -1)
            dim_color = 8 if getattr(curses, "COLORS", 0) > 8 else curses.COLOR_WHITE
            curses.init_pair(6, dim_color, -1)

        generation = self._start_refresh(initial=True, show_placeholders=False, refresh_scope=REFRESH_SCOPE_ALL)
        thread = threading.Thread(target=self._refresh_worker, args=(generation, REFRESH_SCOPE_ALL), daemon=True)
        thread.start()
        boot_frame = 0
        while True:
            self._draw_boot_loading(stdscr, boot_frame)
            with self._lock:
                loading = self.loading
            if not loading:
                break
            boot_frame += 1
            key = stdscr.getch()
            if key in {ord('q'), 27}:
                return

        scroll_offset = 0
        while True:
            self._clear_expired_confirmation()
            stdscr.erase()
            max_y, max_x = stdscr.getmaxyx()
            if max_y < 10 or max_x < 72:
                stdscr.addnstr(0, 0, "Terminal too small for auth view (need at least 72x10).", max_x - 1)
                stdscr.refresh()
                key = stdscr.getch()
                if key in {ord('q'), 27}:
                    return
                continue

            refresh_scope = self._auto_refresh_scope_at(time.time())
            with self._lock:
                snapshots = list(self.snapshots)
                selected_index = self.selected_index
                loading = self.loading
                message = self.message
                sort_mode = self.sort_mode

            if refresh_scope is not None:
                self._refresh_async(refresh_scope=refresh_scope)
                with self._lock:
                    snapshots = list(self.snapshots)
                    selected_index = self.selected_index
                    loading = self.loading
                    message = self.message
                    sort_mode = self.sort_mode

            counts = self._status_counts()
            short_label, long_label = self._label_pair()
            header = (
                f"Codex auth view  sort={_SORT_LABELS.get(sort_mode, sort_mode)}  total={len(snapshots)}  "
                f"ok={counts['auth_ok']}  dead={counts['dead']}  avail={counts['avail']}  limit={counts['limit']}"
            )
            if loading:
                spinner = "|/-\\"[int(time.time() * 8) % 4]
                header += f"  refreshing {spinner}"
            header_attr = curses.A_BOLD
            if curses.has_colors():
                header_attr |= curses.color_pair(4)
            stdscr.addnstr(0, 0, _clip(header, max_x - 1), max_x - 1, header_attr)

            dim_attr = curses.color_pair(6) if curses.has_colors() else curses.A_DIM
            graph_rows = self._graph_rows(max_x, max_y)
            graph_summary_top = 1
            graph_summary_lines = self._graph_summary_lines(max_x) if graph_rows else []
            for line_offset, summary_line in enumerate(graph_summary_lines, start=graph_summary_top):
                summary_attr = curses.A_BOLD if line_offset == graph_summary_top else curses.A_NORMAL
                if curses.has_colors():
                    summary_attr |= curses.color_pair(1 if line_offset > graph_summary_top else 4)
                stdscr.addnstr(line_offset, 0, summary_line, max_x - 1, summary_attr)

            graph_top = graph_summary_top + len(graph_summary_lines)
            graph_left_col = 2
            graph_columns = min(max(24, max_x - 4), len(_KITTY_DIACRITICS)) if graph_rows else 0
            col_header_y = graph_top + graph_rows
            account_width, bar_width = self._layout_sizes(max_x)
            column_width = bar_width + 14
            week_heading = _clip(f" {long_label} left / back", column_width)
            hour_heading = _clip(f" {short_label} left / back", column_width)
            col_header = _clip(
                f"  {'auth':<4} {'quota':<5} {'account':<{account_width}} {week_heading:<{column_width}} {hour_heading:<{column_width}}",
                max_x - 1,
            )
            stdscr.addnstr(col_header_y, 0, col_header, max_x - 1, dim_attr)

            body_top = col_header_y + 1
            body_bottom = max_y - 5
            visible_rows = max(1, body_bottom - body_top + 1)
            if selected_index < scroll_offset:
                scroll_offset = selected_index
            elif selected_index >= scroll_offset + visible_rows:
                scroll_offset = selected_index - visible_rows + 1
            scroll_offset = max(0, scroll_offset)

            if not snapshots:
                stdscr.addnstr(body_top, 0, _clip("  No pooled Codex profiles found.", max_x - 1), max_x - 1)
            else:
                for row_idx in range(visible_rows):
                    snapshot_idx = scroll_offset + row_idx
                    if snapshot_idx >= len(snapshots):
                        break
                    snapshot = snapshots[snapshot_idx]
                    is_selected = snapshot_idx == selected_index
                    row_text = self._render_row(snapshot, max_x, selected=is_selected)
                    attr = curses.A_NORMAL
                    if snapshot.state == STATE_AVAILABLE and curses.has_colors():
                        attr |= curses.color_pair(1)
                    elif snapshot.state == STATE_LIMITED and curses.has_colors():
                        attr |= curses.color_pair(2)
                    elif snapshot.state == STATE_DEACTIVATED and curses.has_colors():
                        attr |= curses.color_pair(3)
                    elif snapshot.state == STATE_ERROR and curses.has_colors():
                        attr |= curses.color_pair(5)
                    elif snapshot.state == STATE_LOADING and curses.has_colors():
                        attr |= curses.color_pair(6)
                    if is_selected:
                        attr |= curses.A_BOLD
                    stdscr.addnstr(body_top + row_idx, 0, row_text, max_x - 1, attr)

            detail_lines = self._detail_lines(max_x)
            for offset, line in enumerate(detail_lines, start=max_y - 4):
                stdscr.addnstr(offset, 0, line, max_x - 1)

            stdscr.addnstr(max_y - 1, 0, _footer_line(message, width=max_x - 1), max_x - 1, dim_attr)
            stdscr.refresh()
            if graph_rows and graph_columns:
                self._emit_kitty_graph(
                    top_row=graph_top,
                    left_col=graph_left_col,
                    columns=graph_columns,
                    rows=graph_rows,
                )

            key = stdscr.getch()
            if key == -1:
                continue
            if key in {ord('q'), 27}:
                return
            if key in {curses.KEY_UP, ord('k')}:
                with self._lock:
                    if self.snapshots:
                        self.selected_index = max(0, self.selected_index - 1)
                        self._selected_entry_id = self._selected_snapshot_unlocked().entry_id
                continue
            if key in {curses.KEY_DOWN, ord('j')}:
                with self._lock:
                    if self.snapshots:
                        self.selected_index = min(len(self.snapshots) - 1, self.selected_index + 1)
                        self._selected_entry_id = self._selected_snapshot_unlocked().entry_id
                continue
            if key == ord('a'):
                self._cycle_auto_refresh_interval()
                continue
            if key == ord('A'):
                self._set_sort_mode(SORT_AVAILABILITY)
                continue
            if key == ord('n'):
                self._set_sort_mode(SORT_NAME)
                continue
            if key == ord('l'):
                self._set_sort_mode(SORT_REFRESH)
                continue
            if key == ord('u'):
                self._refresh_async()
                continue
            if key == ord('r'):
                self._remove_selected()
                continue

    def run(self) -> None:
        import curses

        try:
            curses.wrapper(self._run_curses)
        finally:
            self._delete_kitty_graph_image()


def auth_view_command(args) -> None:
    provider = getattr(args, "provider", DEFAULT_PROVIDER) or DEFAULT_PROVIDER
    timeout_seconds = float(getattr(args, "timeout", 10.0) or 10.0)
    smoke_test = bool(getattr(args, "smoke_test", False))
    poll_interval_value = getattr(args, "poll_interval", AUTO_REFRESH_INTERVAL_SECONDS)
    poll_interval_seconds = AUTO_REFRESH_INTERVAL_SECONDS if poll_interval_value is None else max(0.0, float(poll_interval_value))

    if smoke_test:
        print(render_codex_auth_smoke_test(provider=provider, timeout_seconds=timeout_seconds))
        return

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise SystemExit("`hermes auth view` requires an interactive terminal. Use `--smoke-test` for CI/non-interactive checks.")

    CodexAuthTui(
        provider=provider,
        timeout_seconds=timeout_seconds,
        auto_refresh_interval_seconds=poll_interval_seconds,
    ).run()
