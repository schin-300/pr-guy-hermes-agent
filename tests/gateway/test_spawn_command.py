"""Tests for /spawn gateway slash command."""

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

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


def _make_event(text="/spawn", platform=Platform.TELEGRAM, user_id="12345", chat_id="67890"):
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
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_db = session_db
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._background_tasks = set()
    runner.config = {"model": {"default": "anthropic/claude-sonnet-4.6"}}

    current_session_id = "20260408_090000_parent"
    session_db.create_session(
        session_id=current_session_id,
        source="telegram",
        model="anthropic/claude-sonnet-4.6",
    )
    session_db.set_session_title(current_session_id, "Main Session")

    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "Investigate the bug"},
        {"role": "assistant", "content": "On it"},
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
    runner.session_store = mock_store

    from gateway.hooks import HookRegistry

    runner.hooks = HookRegistry()
    return runner, current_session_id, history


def _find_child_sessions(session_db, parent_session_id):
    cursor = session_db._conn.execute(
        "SELECT id FROM sessions WHERE parent_session_id = ? ORDER BY started_at",
        (parent_session_id,),
    )
    return [row["id"] for row in cursor.fetchall()]


class TestHandleSpawnCommand:
    @pytest.mark.asyncio
    async def test_no_prompt_shows_usage(self, session_db):
        runner, _, _ = _make_runner(session_db)
        event = _make_event(text="/spawn")
        result = await runner._handle_spawn_command(event)
        assert "Usage:" in result
        assert "/spawn" in result

    @pytest.mark.asyncio
    async def test_spawn_clones_child_session_without_switching(self, session_db):
        runner, parent_id, history = _make_runner(session_db)

        created_tasks = []

        def capture_task(coro, *args, **kwargs):
            coro.close()
            mock_task = MagicMock()
            created_tasks.append(mock_task)
            return mock_task

        with patch("gateway.run.asyncio.create_task", side_effect=capture_task):
            event = _make_event(text="/spawn Continue in the background")
            result = await runner._handle_spawn_command(event)

        child_sessions = _find_child_sessions(session_db, parent_id)
        assert len(child_sessions) == 1
        child_id = child_sessions[0]
        child = session_db.get_session(child_id)
        assert child["parent_session_id"] == parent_id
        assert len(session_db.get_messages_as_conversation(child_id)) == len(history)
        runner.session_store.switch_session.assert_not_called()
        assert "Spawn task started" in result
        assert parent_id in result
        assert child_id in result
        assert len(created_tasks) == 1


class TestRunSpawnTask:
    @pytest.mark.asyncio
    async def test_spawn_task_uses_child_session_and_disables_pass_session_id(self, session_db):
        runner, _, history = _make_runner(session_db)
        mock_adapter = AsyncMock()
        mock_adapter.send = AsyncMock()
        mock_adapter.extract_media = MagicMock(return_value=([], "Spawned result"))
        mock_adapter.extract_images = MagicMock(return_value=([], "Spawned result"))
        runner.adapters[Platform.TELEGRAM] = mock_adapter

        source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="12345",
            chat_id="67890",
            user_name="testuser",
        )

        with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}), \
             patch("run_agent.AIAgent") as MockAgent:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "Spawned result", "messages": []}
            MockAgent.return_value = mock_agent

            await runner._run_spawn_task(
                prompt="continue",
                source=source,
                task_id="spawn_test",
                child_session_id="20260408_090001_child",
                history_snapshot=history,
            )

        kwargs = MockAgent.call_args.kwargs
        assert kwargs["session_id"] == "20260408_090001_child"
        assert kwargs["pass_session_id"] is False
        mock_agent.run_conversation.assert_called_once_with(
            user_message="continue",
            conversation_history=history,
            task_id="spawn_test",
        )
        sent_content = mock_adapter.send.call_args.kwargs["content"]
        assert "Spawn task complete" in sent_content
        assert "20260408_090001_child" in sent_content


class TestSpawnCommandRegistration:
    def test_spawn_is_known_gateway_command(self):
        from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS

        assert "spawn" in GATEWAY_KNOWN_COMMANDS

    def test_spawn_in_cli_commands(self):
        from hermes_cli.commands import COMMANDS

        assert "/spawn" in COMMANDS
