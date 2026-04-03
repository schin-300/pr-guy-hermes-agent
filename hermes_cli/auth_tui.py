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

import locale
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
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
    shortcuts = "↑↓/jk move  a avail  n name  l fresh  u refresh  r remove  q quit"
    if not message:
        return _clip(shortcuts, width)
    return _clip(f"{message}  |  {shortcuts}", width)


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
) -> list[CodexProfileSnapshot]:
    if provider != DEFAULT_PROVIDER:
        raise SystemExit(f"Auth view currently supports only {DEFAULT_PROVIDER}.")
    fetcher = fetch_profile or _fetch_codex_live_status
    pool = load_pool(provider)
    entries = list(pool.entries())
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
    def __init__(self, *, provider: str = DEFAULT_PROVIDER, timeout_seconds: float = 10.0):
        if provider != DEFAULT_PROVIDER:
            raise SystemExit(f"Auth view currently supports only {DEFAULT_PROVIDER}.")
        self.provider = provider
        self.timeout_seconds = timeout_seconds
        self.sort_mode = SORT_AVAILABILITY
        self.snapshots: list[CodexProfileSnapshot] = []
        self.selected_index = 0
        self.message = "Loading Codex profiles..."
        self.loading = False
        self._lock = threading.Lock()
        self._refresh_generation = 0
        self._selected_entry_id: Optional[str] = None
        self._remove_confirm_entry_id: Optional[str] = None
        self._remove_confirm_deadline = 0.0

    def _start_refresh(self, *, initial: bool = False, show_placeholders: bool = False) -> int:
        with self._lock:
            self._refresh_generation += 1
            generation = self._refresh_generation
            if show_placeholders and (not self.snapshots or initial):
                try:
                    self.snapshots = build_codex_profile_placeholders(self.provider)
                except Exception:
                    self.snapshots = []
            self.loading = True
            self.message = "Refreshing Codex profiles..." if not initial else "Polling live Codex profiles..."
            self._remove_confirm_entry_id = None
            self._remove_confirm_deadline = 0.0
            selected = self._selected_snapshot_unlocked()
            self._selected_entry_id = selected.entry_id if selected else None
            return generation

    def _fetch_live_snapshots(self) -> tuple[list[CodexProfileSnapshot], str]:
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
        except Exception as exc:
            snapshots = []
            message = f"Refresh failed: {exc}"
        return snapshots, message

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

    def _finish_refresh(self, generation: int, snapshots: list[CodexProfileSnapshot], message: str) -> None:
        with self._lock:
            if generation != self._refresh_generation:
                return
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
            selected = self._selected_snapshot_unlocked()
            self._selected_entry_id = selected.entry_id if selected else None

    def _refresh_async(self) -> None:
        generation = self._start_refresh(show_placeholders=not self.snapshots)
        thread = threading.Thread(target=self._refresh_worker, args=(generation,), daemon=True)
        thread.start()

    def _refresh_blocking(self, *, initial: bool = False) -> None:
        generation = self._start_refresh(initial=initial, show_placeholders=False)
        snapshots, message = self._fetch_live_snapshots()
        self._finish_refresh(generation, snapshots, message)

    def _refresh_worker(self, generation: int) -> None:
        snapshots, message = self._fetch_live_snapshots()
        self._finish_refresh(generation, snapshots, message)

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

        generation = self._start_refresh(initial=True, show_placeholders=False)
        thread = threading.Thread(target=self._refresh_worker, args=(generation,), daemon=True)
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

            account_width, bar_width = self._layout_sizes(max_x)
            column_width = bar_width + 14
            week_heading = _clip(f" {long_label} left / back", column_width)
            hour_heading = _clip(f" {short_label} left / back", column_width)
            col_header = _clip(
                f"  {'auth':<4} {'quota':<5} {'account':<{account_width}} {week_heading:<{column_width}} {hour_heading:<{column_width}}",
                max_x - 1,
            )
            dim_attr = curses.color_pair(6) if curses.has_colors() else curses.A_DIM
            stdscr.addnstr(1, 0, col_header, max_x - 1, dim_attr)

            body_top = 2
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

        curses.wrapper(self._run_curses)


def auth_view_command(args) -> None:
    provider = getattr(args, "provider", DEFAULT_PROVIDER) or DEFAULT_PROVIDER
    timeout_seconds = float(getattr(args, "timeout", 10.0) or 10.0)
    smoke_test = bool(getattr(args, "smoke_test", False))

    if smoke_test:
        print(render_codex_auth_smoke_test(provider=provider, timeout_seconds=timeout_seconds))
        return

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise SystemExit("`hermes auth view` requires an interactive terminal. Use `--smoke-test` for CI/non-interactive checks.")

    CodexAuthTui(provider=provider, timeout_seconds=timeout_seconds).run()
