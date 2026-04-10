"""Tests for tools/session_timeline_tool.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from hermes_state import SessionDB
from tools.session_timeline_tool import session_timeline


def _append(db: SessionDB, session_id: str, role: str, content: str, ts: float, *, tool_name: str | None = None) -> None:
    with patch("hermes_state.time.time", return_value=ts):
        db.append_message(session_id, role=role, content=content, tool_name=tool_name)


def _make_db(tmp_path: Path) -> SessionDB:
    return SessionDB(db_path=tmp_path / "state.db")


def test_recent_mode_returns_latest_chronological_chunk_from_lineage(tmp_path):
    db = _make_db(tmp_path)
    db.create_session("root", source="cli", model="test")
    db.create_session("child", source="cli", model="test", parent_session_id="root")

    _append(db, "root", "user", "root-1", 1000)
    _append(db, "root", "assistant", "root-2", 1001)
    _append(db, "root", "user", "root-3", 1002)
    _append(db, "child", "assistant", "child-1", 1003)
    _append(db, "child", "user", "child-2", 1004)

    payload = json.loads(session_timeline(limit=3, db=db, current_session_id="child"))

    assert payload["success"] is True
    assert payload["mode"] == "recent"
    assert payload["lineage_session_ids"] == ["root", "child"]
    assert [m["content"] for m in payload["messages"]] == ["root-3", "child-1", "child-2"]
    assert payload["messages"][0]["timestamp"] < payload["messages"][-1]["timestamp"]


def test_query_mode_prefers_more_recent_match_and_returns_contiguous_chunk(tmp_path):
    db = _make_db(tmp_path)
    db.create_session("root", source="cli", model="test")
    db.create_session("child", source="cli", model="test", parent_session_id="root")

    _append(db, "root", "user", "older lead", 1000)
    _append(db, "root", "assistant", "banana anchor old", 1001)
    _append(db, "root", "user", "older tail", 1002)
    _append(db, "child", "assistant", "newer lead", 1003)
    _append(db, "child", "user", "banana anchor new", 1004)
    _append(db, "child", "assistant", "newer tail", 1005)

    payload = json.loads(session_timeline(query="banana", limit=3, db=db, current_session_id="child"))

    assert payload["success"] is True
    assert payload["mode"] == "query"
    assert payload["selection"]["anchor_message_id"] == payload["messages"][1]["id"]
    assert payload["messages"][1]["content"] == "banana anchor new"
    assert [m["content"] for m in payload["messages"]] == ["newer lead", "banana anchor new", "newer tail"]


def test_range_mode_filters_by_timestamp(tmp_path):
    db = _make_db(tmp_path)
    db.create_session("root", source="cli", model="test")

    _append(db, "root", "user", "m1", 1000)
    _append(db, "root", "assistant", "m2", 2000)
    _append(db, "root", "user", "m3", 3000)

    payload = json.loads(
        session_timeline(after="1500", before="2500", limit=10, db=db, current_session_id="root")
    )

    assert payload["success"] is True
    assert payload["mode"] == "range"
    assert [m["content"] for m in payload["messages"]] == ["m2"]


def test_default_excludes_tool_messages_for_speed_dense_chat_chunks(tmp_path):
    db = _make_db(tmp_path)
    db.create_session("root", source="cli", model="test")

    _append(db, "root", "user", "u1", 1000)
    _append(db, "root", "tool", "tool output", 1001, tool_name="terminal")
    _append(db, "root", "assistant", "a1", 1002)

    payload = json.loads(session_timeline(limit=10, db=db, current_session_id="root"))

    assert payload["success"] is True
    assert [m["role"] for m in payload["messages"]] == ["user", "assistant"]


def test_char_budget_drops_whole_edge_messages_without_editing_content(tmp_path):
    db = _make_db(tmp_path)
    db.create_session("root", source="cli", model="test")

    _append(db, "root", "user", "A" * 200, 1000)
    _append(db, "root", "assistant", "B" * 200, 1001)
    _append(db, "root", "user", "C" * 200, 1002)

    payload = json.loads(session_timeline(limit=3, max_chars=300, db=db, current_session_id="root"))

    assert payload["success"] is True
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["content"] == "C" * 200
    assert payload["selection"]["truncated"] is True
