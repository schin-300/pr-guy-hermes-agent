from __future__ import annotations

import threading
import time
from dataclasses import replace
from types import SimpleNamespace

from agent.credential_pool import PooledCredential, STATUS_EXHAUSTED
from hermes_cli.auth_tui import (
    CodexAuthTui,
    CodexLiveStatus,
    CodexProfileSnapshot,
    CodexUsageWindow,
    SORT_AVAILABILITY,
    SORT_NAME,
    SORT_REFRESH,
    STATE_AVAILABLE,
    STATE_DEACTIVATED,
    STATE_LIMITED,
    _boot_loading_text,
    _footer_line,
    auth_view_command,
    build_codex_profile_snapshots,
    render_codex_auth_smoke_test,
)



class _FakePool:
    def __init__(self, entries: list[PooledCredential]):
        self._entries = list(entries)

    def entries(self) -> list[PooledCredential]:
        return list(self._entries)

    def update_entry(self, entry_id: str, **updates):
        for idx, entry in enumerate(self._entries):
            if entry.id != entry_id:
                continue
            extra = dict(entry.extra)
            field_updates = {}
            for key, value in updates.items():
                if hasattr(entry, key):
                    field_updates[key] = value
                else:
                    extra[key] = value
            updated = replace(entry, extra=extra, **field_updates)
            self._entries[idx] = updated
            return updated
        return None

    def remove_entry(self, entry_id: str):
        for idx, entry in enumerate(self._entries):
            if entry.id != entry_id:
                continue
            return self._entries.pop(idx)
        return None


def test_build_codex_profile_snapshots_classifies_states(monkeypatch):
    available_entry = PooledCredential(
        provider="openai-codex",
        id="cred-1",
        label="available@example.com",
        auth_type="oauth",
        priority=0,
        source="manual:device_code",
        access_token="token-1",
        refresh_token="refresh-1",
        last_status=STATUS_EXHAUSTED,
        last_status_at=1711230000.0,
        last_error_code=429,
    )
    primary_exhausted_entry = PooledCredential(
        provider="openai-codex",
        id="cred-2",
        label="hour-capped@example.com",
        auth_type="oauth",
        priority=1,
        source="manual:device_code",
        access_token="token-2",
        refresh_token="refresh-2",
    )
    dead_entry = PooledCredential(
        provider="openai-codex",
        id="cred-3",
        label="dead@example.com",
        auth_type="oauth",
        priority=2,
        source="manual:device_code",
        access_token="token-3",
        refresh_token="refresh-3",
    )
    pool = _FakePool([available_entry, primary_exhausted_entry, dead_entry])
    monkeypatch.setattr("hermes_cli.auth_tui.load_pool", lambda provider: pool)

    def _fake_fetch(pool_obj, entry, timeout_seconds):
        assert timeout_seconds == 7.5
        if entry.id == "cred-1":
            updated = pool_obj.update_entry(
                entry.id,
                last_status=None,
                last_status_at=None,
                last_error_code=None,
            )
            return updated, CodexLiveStatus(
                http_status=200,
                email="available@example.com",
                plan="plus",
                allowed=True,
                limit_reached=False,
                primary_window=CodexUsageWindow(label="5h", used_percent=18, reset_at_ms=1700000000000),
                secondary_window=CodexUsageWindow(label="Week", used_percent=6, reset_at_ms=1700600000000),
                local_cooldown_cleared=True,
            )
        if entry.id == "cred-2":
            return entry, CodexLiveStatus(
                http_status=200,
                email="hour-capped@example.com",
                plan="team",
                allowed=True,
                limit_reached=False,
                primary_window=CodexUsageWindow(label="5h", used_percent=100, reset_at_ms=1700000000000),
                secondary_window=CodexUsageWindow(label="Week", used_percent=12, reset_at_ms=1700600000000),
            )
        return entry, CodexLiveStatus(http_status=401, error="HTTP 401", deactivated=True)

    snapshots = build_codex_profile_snapshots(
        provider="openai-codex",
        timeout_seconds=7.5,
        fetch_profile=_fake_fetch,
    )

    assert len(snapshots) == 3
    assert snapshots[0].auth_badge == "OK"
    assert snapshots[0].state == STATE_AVAILABLE
    assert snapshots[0].local_cooldown_cleared is True
    assert snapshots[0].local_status is None
    assert snapshots[0].primary_window is not None
    assert snapshots[0].primary_window.label == "5h"
    assert snapshots[1].auth_badge == "OK"
    assert snapshots[1].state == STATE_LIMITED
    assert snapshots[1].state_reason == "5h limit reached"
    assert snapshots[2].auth_badge == "DEAD"
    assert snapshots[2].state == STATE_DEACTIVATED
    assert snapshots[2].state_reason == "HTTP 401"


def test_build_codex_profile_snapshots_polls_in_parallel(monkeypatch):
    entries = [
        PooledCredential(
            provider="openai-codex",
            id=f"cred-{idx}",
            label=f"user{idx}@example.com",
            auth_type="oauth",
            priority=idx,
            source="manual:device_code",
            access_token=f"token-{idx}",
            refresh_token=f"refresh-{idx}",
        )
        for idx in range(4)
    ]
    pool = _FakePool(entries)
    monkeypatch.setattr("hermes_cli.auth_tui.load_pool", lambda provider: pool)

    thread_ids = set()
    thread_lock = threading.Lock()

    def _fake_fetch(pool_obj, entry, timeout_seconds):
        with thread_lock:
            thread_ids.add(threading.get_ident())
        time.sleep(0.2)
        return entry, CodexLiveStatus(
            http_status=200,
            email=entry.label,
            allowed=True,
            limit_reached=False,
            primary_window=CodexUsageWindow(label="5h", used_percent=10),
            secondary_window=CodexUsageWindow(label="Week", used_percent=10),
        )

    snapshots = build_codex_profile_snapshots(
        provider="openai-codex",
        timeout_seconds=7.5,
        fetch_profile=_fake_fetch,
    )

    assert len(snapshots) == 4
    assert len(thread_ids) >= 2


def test_build_codex_profile_snapshots_sorts_by_availability(monkeypatch):
    entries = [
        PooledCredential(
            provider="openai-codex",
            id="dead",
            label="dead@example.com",
            auth_type="oauth",
            priority=0,
            source="manual:device_code",
            access_token="token-dead",
            refresh_token="refresh-dead",
        ),
        PooledCredential(
            provider="openai-codex",
            id="limited-hour",
            label="limited-hour@example.com",
            auth_type="oauth",
            priority=1,
            source="manual:device_code",
            access_token="token-limited-hour",
            refresh_token="refresh-limited-hour",
        ),
        PooledCredential(
            provider="openai-codex",
            id="limited-week",
            label="limited-week@example.com",
            auth_type="oauth",
            priority=2,
            source="manual:device_code",
            access_token="token-limited-week",
            refresh_token="refresh-limited-week",
        ),
        PooledCredential(
            provider="openai-codex",
            id="avail-low",
            label="avail-low@example.com",
            auth_type="oauth",
            priority=3,
            source="manual:device_code",
            access_token="token-avail-low",
            refresh_token="refresh-avail-low",
        ),
        PooledCredential(
            provider="openai-codex",
            id="avail-high",
            label="avail-high@example.com",
            auth_type="oauth",
            priority=4,
            source="manual:device_code",
            access_token="token-avail-high",
            refresh_token="refresh-avail-high",
        ),
    ]
    pool = _FakePool(entries)
    monkeypatch.setattr("hermes_cli.auth_tui.load_pool", lambda provider: pool)

    def _fake_fetch(pool_obj, entry, timeout_seconds):
        if entry.id == "avail-high":
            return entry, CodexLiveStatus(
                http_status=200,
                email="avail-high@example.com",
                allowed=True,
                limit_reached=False,
                primary_window=CodexUsageWindow(label="5h", used_percent=5),
                secondary_window=CodexUsageWindow(label="Week", used_percent=10),
            )
        if entry.id == "avail-low":
            return entry, CodexLiveStatus(
                http_status=200,
                email="avail-low@example.com",
                allowed=True,
                limit_reached=False,
                primary_window=CodexUsageWindow(label="5h", used_percent=55),
                secondary_window=CodexUsageWindow(label="Week", used_percent=15),
            )
        if entry.id == "limited-hour":
            return entry, CodexLiveStatus(
                http_status=200,
                email="limited-hour@example.com",
                allowed=True,
                limit_reached=False,
                primary_window=CodexUsageWindow(label="5h", used_percent=100),
                secondary_window=CodexUsageWindow(label="Week", used_percent=5),
            )
        if entry.id == "limited-week":
            return entry, CodexLiveStatus(
                http_status=200,
                email="limited-week@example.com",
                allowed=True,
                limit_reached=False,
                primary_window=CodexUsageWindow(label="5h", used_percent=5),
                secondary_window=CodexUsageWindow(label="Week", used_percent=100),
            )
        return entry, CodexLiveStatus(http_status=401, error="HTTP 401", deactivated=True)

    snapshots = build_codex_profile_snapshots(
        provider="openai-codex",
        timeout_seconds=7.5,
        fetch_profile=_fake_fetch,
    )

    assert [snapshot.display_name for snapshot in snapshots] == [
        "avail-high@example.com",
        "avail-low@example.com",
        "limited-hour@example.com",
        "limited-week@example.com",
        "dead@example.com",
    ]
    assert [snapshot.index for snapshot in snapshots] == [1, 2, 3, 4, 5]


def test_render_codex_auth_smoke_test(monkeypatch):
    monkeypatch.setattr("hermes_cli.auth_tui.time.time", lambda: 0)
    snapshots = [
        CodexProfileSnapshot(
            index=1,
            entry_id="cred-1",
            label="available@example.com",
            display_name="available@example.com",
            auth_badge="OK",
            state=STATE_AVAILABLE,
            state_badge="AVAIL",
            state_reason="Available",
            primary_window=CodexUsageWindow(label="5h", used_percent=18, reset_at_ms=5 * 60 * 60 * 1000),
            secondary_window=CodexUsageWindow(label="Week", used_percent=6, reset_at_ms=(6 * 24 + 23) * 60 * 60 * 1000),
        ),
        CodexProfileSnapshot(
            index=2,
            entry_id="cred-2",
            label="dead@example.com",
            display_name="dead@example.com",
            auth_badge="DEAD",
            state=STATE_DEACTIVATED,
            state_badge="n/a",
            state_reason="HTTP 401",
            primary_window=None,
            secondary_window=None,
        ),
    ]
    monkeypatch.setattr("hermes_cli.auth_tui.build_codex_profile_snapshots", lambda **kwargs: snapshots)

    output = render_codex_auth_smoke_test()

    assert "Codex auth view: 2 profiles" in output
    assert "available@example.com" in output
    assert "OK   AVAIL" in output
    assert "Week_left= 94% @6d23h" in output
    assert "5h_left= 82% @5h" in output
    assert "DEAD n/a" in output


def test_boot_loading_text_is_compact_spinner_message():
    assert _boot_loading_text(0) == "Loading |"
    assert _boot_loading_text(1) == "Loading /"


def test_footer_line_prioritizes_message_visibility():
    line = _footer_line(
        "Press r again to remove [dup-b] duplicate@example.com.",
        width=72,
    )

    assert line.startswith("Press r again to remove")
    assert "duplicate@example.com" in line


def test_refresh_blocking_fetches_live_before_showing_rows(monkeypatch):
    tui = CodexAuthTui(provider="openai-codex", timeout_seconds=3.0)
    snapshots = [
        CodexProfileSnapshot(
            index=1,
            entry_id="cred-1",
            label="live@example.com",
            display_name="live@example.com",
            auth_badge="OK",
            state=STATE_AVAILABLE,
            state_badge="AVAIL",
            state_reason="Available",
            primary_window=CodexUsageWindow(label="5h", used_percent=18),
            secondary_window=CodexUsageWindow(label="Week", used_percent=6),
        )
    ]

    monkeypatch.setattr(
        "hermes_cli.auth_tui.build_codex_profile_placeholders",
        lambda provider: (_ for _ in ()).throw(AssertionError("no placeholders on initial load")),
    )
    monkeypatch.setattr(
        "hermes_cli.auth_tui.build_codex_profile_snapshots",
        lambda provider, timeout_seconds: snapshots,
    )

    tui._refresh_blocking(initial=True)

    assert tui.loading is False
    assert tui.snapshots == snapshots
    assert tui.message == "Loaded 1 live Codex profile."


def test_set_sort_mode_reorders_rows_and_preserves_selection():
    tui = CodexAuthTui(provider="openai-codex", timeout_seconds=3.0)
    tui.snapshots = [
        CodexProfileSnapshot(
            index=1,
            entry_id="b",
            label="beta@example.com",
            display_name="beta@example.com",
            auth_badge="OK",
            state=STATE_AVAILABLE,
            state_badge="AVAIL",
            state_reason="Available",
            primary_window=CodexUsageWindow(label="5h", used_percent=10),
            secondary_window=CodexUsageWindow(label="Week", used_percent=10),
            last_refresh="2026-04-03T05:00:00Z",
        ),
        CodexProfileSnapshot(
            index=2,
            entry_id="a",
            label="alpha@example.com",
            display_name="alpha@example.com",
            auth_badge="OK",
            state=STATE_AVAILABLE,
            state_badge="AVAIL",
            state_reason="Available",
            primary_window=CodexUsageWindow(label="5h", used_percent=10),
            secondary_window=CodexUsageWindow(label="Week", used_percent=10),
            last_refresh="2026-04-03T07:00:00Z",
        ),
    ]
    tui.selected_index = 1
    tui._selected_entry_id = "a"

    tui._set_sort_mode(SORT_NAME)
    assert tui.sort_mode == SORT_NAME
    assert [snapshot.display_name for snapshot in tui.snapshots] == [
        "alpha@example.com",
        "beta@example.com",
    ]
    assert tui.selected_index == 0

    tui._set_sort_mode(SORT_REFRESH)
    assert tui.sort_mode == SORT_REFRESH
    assert tui.snapshots[0].entry_id == "a"
    assert tui.selected_index == 0

    tui._set_sort_mode(SORT_AVAILABILITY)
    assert tui.sort_mode == SORT_AVAILABILITY


def test_remove_selected_removes_exact_duplicate_entry(monkeypatch):
    duplicate_a = PooledCredential(
        provider="openai-codex",
        id="dup-a",
        label="duplicate@example.com",
        auth_type="oauth",
        priority=0,
        source="manual:device_code",
        access_token="token-a",
        refresh_token="refresh-a",
    )
    duplicate_b = PooledCredential(
        provider="openai-codex",
        id="dup-b",
        label="duplicate@example.com",
        auth_type="oauth",
        priority=1,
        source="manual:device_code",
        access_token="token-b",
        refresh_token="refresh-b",
    )
    pool = _FakePool([duplicate_a, duplicate_b])
    monkeypatch.setattr("hermes_cli.auth_tui.load_pool", lambda provider: pool)

    tui = CodexAuthTui(provider="openai-codex", timeout_seconds=3.0)
    tui.snapshots = [
        CodexProfileSnapshot(
            index=1,
            entry_id="dup-a",
            label="duplicate@example.com",
            display_name="duplicate@example.com",
            auth_badge="OK",
            state=STATE_AVAILABLE,
            state_badge="AVAIL",
            state_reason="Available",
            primary_window=None,
            secondary_window=None,
        ),
        CodexProfileSnapshot(
            index=2,
            entry_id="dup-b",
            label="duplicate@example.com",
            display_name="duplicate@example.com",
            auth_badge="OK",
            state=STATE_AVAILABLE,
            state_badge="AVAIL",
            state_reason="Available",
            primary_window=None,
            secondary_window=None,
        ),
    ]
    tui.selected_index = 1
    tui._selected_entry_id = "dup-b"
    monkeypatch.setattr(tui, "_refresh_async", lambda: None)

    tui._remove_selected()
    assert "Press r again" in tui.message
    assert "dup-b" in tui.message
    assert [entry.id for entry in pool.entries()] == ["dup-a", "dup-b"]

    tui._remove_selected()
    assert [entry.id for entry in pool.entries()] == ["dup-a"]
    assert "Removed" in tui.message
    assert "dup-b" in tui.message


def test_render_row_marks_selection_without_inverting_bars(monkeypatch):
    monkeypatch.setattr("hermes_cli.auth_tui.time.time", lambda: 0)
    tui = CodexAuthTui(provider="openai-codex", timeout_seconds=3.0)
    snapshot = CodexProfileSnapshot(
        index=1,
        entry_id="cred-1",
        label="live@example.com",
        display_name="live@example.com",
        auth_badge="OK",
        state=STATE_AVAILABLE,
        state_badge="AVAIL",
        state_reason="Available",
        primary_window=CodexUsageWindow(label="5h", used_percent=18, reset_at_ms=5 * 60 * 60 * 1000),
        secondary_window=CodexUsageWindow(label="Week", used_percent=6, reset_at_ms=(6 * 24 + 23) * 60 * 60 * 1000),
    )

    plain = tui._render_row(snapshot, 120, selected=False)
    selected = tui._render_row(snapshot, 120, selected=True)

    assert plain.startswith("  OK")
    assert selected.startswith("> OK")
    assert "Week" not in selected
    assert selected.index("@6d23h") < selected.index("@5h")
    assert "[████████" in selected


def test_auth_view_command_smoke_test(capsys, monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.auth_tui.render_codex_auth_smoke_test",
        lambda **kwargs: "smoke ok",
    )

    auth_view_command(SimpleNamespace(provider="openai-codex", timeout=3.0, smoke_test=True))

    out = capsys.readouterr().out
    assert out.strip() == "smoke ok"
