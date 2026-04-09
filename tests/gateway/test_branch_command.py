"""Tests for gateway /branch command regression coverage."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


@pytest.fixture
def session_db(tmp_path):
    os.environ["HERMES_HOME"] = str(tmp_path / ".hermes")
    os.makedirs(tmp_path / ".hermes", exist_ok=True)
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / ".hermes" / "test_sessions.db")
    yield db
    db.close()


def _make_event(text="/branch", platform=Platform.TELEGRAM, user_id="12345", chat_id="67890"):
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _make_runner(session_db):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._session_db = session_db
    runner._running_agents = {}
    runner.config = {"model": {"default": "anthropic/claude-sonnet-4.6"}}

    current_session_id = "20260408_091500_parent"
    session_db.create_session(
        session_id=current_session_id,
        source="telegram",
        model="anthropic/claude-sonnet-4.6",
    )
    session_db.set_session_title(current_session_id, "Gateway Session")

    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Explore option B"},
        {"role": "assistant", "content": "Sure"},
    ]
    for msg in history:
        session_db.append_message(
            session_id=current_session_id,
            role=msg["role"],
            content=msg["content"],
        )

    mock_store = MagicMock()
    mock_store.get_or_create_session.return_value = SimpleNamespace(session_id=current_session_id)
    mock_store.load_transcript.return_value = history
    mock_store.switch_session.side_effect = lambda session_key, target_session_id: SimpleNamespace(session_id=target_session_id)
    runner.session_store = mock_store
    runner._session_key_for_source = MagicMock(return_value="telegram:67890")
    runner._evict_cached_agent = MagicMock()
    return runner, current_session_id, history


def _find_child_sessions(session_db, parent_session_id):
    cursor = session_db._conn.execute(
        "SELECT id FROM sessions WHERE parent_session_id = ? ORDER BY started_at",
        (parent_session_id,),
    )
    return [row["id"] for row in cursor.fetchall()]


@pytest.mark.asyncio
async def test_gateway_branch_clones_history_and_switches_session(session_db):
    runner, parent_id, history = _make_runner(session_db)

    event = _make_event(text="/branch branch idea")
    result = await runner._handle_branch_command(event)

    child_sessions = _find_child_sessions(session_db, parent_id)
    assert len(child_sessions) == 1
    child_id = child_sessions[0]
    assert len(session_db.get_messages_as_conversation(child_id)) == len(history)
    assert session_db.get_session(child_id)["parent_session_id"] == parent_id
    runner.session_store.switch_session.assert_called_once_with("telegram:67890", child_id)
    runner._evict_cached_agent.assert_called_once_with("telegram:67890")
    assert "Branched to **branch idea**" in result
    assert parent_id in result
    assert child_id in result
