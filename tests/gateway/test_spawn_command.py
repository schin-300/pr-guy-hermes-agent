"""Tests for /spawn gateway slash command."""

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionSource, build_session_key


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
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._background_tasks = set()
    runner.config = {"model": {"default": "anthropic/claude-sonnet-4.6"}}
    runner._session_key_for_source = MagicMock(side_effect=build_session_key)
    runner._is_user_authorized = MagicMock(return_value=True)

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

    @pytest.mark.asyncio
    async def test_spawn_prefers_running_agent_snapshot_over_transcript(self, session_db):
        runner, parent_id, history = _make_runner(session_db)

        running_history = history + [
            {"role": "user", "content": "Latest in-memory prompt"},
            {"role": "assistant", "content": "Latest in-memory answer"},
        ]
        session_key = build_session_key(_make_event().source)
        runner._running_agents[session_key] = SimpleNamespace(_session_messages=running_history)
        runner.session_store.load_transcript.return_value = history

        def capture_task(coro, *args, **kwargs):
            coro.close()
            mock_task = MagicMock()
            return mock_task

        with patch("gateway.run.asyncio.create_task", side_effect=capture_task):
            event = _make_event(text="/spawn Continue in the background")
            result = await runner._handle_spawn_command(event)

        child_sessions = _find_child_sessions(session_db, parent_id)
        assert len(child_sessions) == 1
        child_id = child_sessions[0]
        assert session_db.get_messages_as_conversation(child_id) == running_history
        runner.session_store.load_transcript.assert_not_called()
        assert child_id in result

    @pytest.mark.asyncio
    async def test_spawn_bypasses_running_agent_guard(self, session_db):
        runner, _, _ = _make_runner(session_db)
        event = _make_event(text="/spawn Continue in the background")
        session_key = build_session_key(event.source)
        running_agent = MagicMock()
        runner._running_agents[session_key] = running_agent
        runner.adapters[Platform.TELEGRAM] = SimpleNamespace(
            _pending_messages={},
            get_pending_message=MagicMock(return_value=None),
        )
        runner._handle_spawn_command = AsyncMock(return_value="spawn handled")

        result = await runner._handle_message(event)

        assert result == "spawn handled"
        runner._handle_spawn_command.assert_awaited_once_with(event)
        running_agent.interrupt.assert_not_called()
        assert runner.adapters[Platform.TELEGRAM]._pending_messages == {}

    @pytest.mark.asyncio
    async def test_spawn_bypasses_adapter_active_session_guard(self):
        source = _make_event().source
        session_key = build_session_key(source)
        handler_called_with = []

        async def fake_handler(event):
            handler_called_with.append(event)
            return "spawn handled"

        class _ConcreteAdapter(BasePlatformAdapter):
            platform = Platform.TELEGRAM

            async def connect(self):
                return None

            async def disconnect(self):
                return None

            async def send(self, chat_id, content, **kwargs):
                return None

            async def get_chat_info(self, chat_id):
                return {}

        adapter = _ConcreteAdapter(
            PlatformConfig(enabled=True, token="***"),
            Platform.TELEGRAM,
        )
        adapter.set_message_handler(fake_handler)

        sent = []

        async def fake_send_with_retry(chat_id, content, reply_to=None, metadata=None):
            sent.append(content)

        adapter._send_with_retry = fake_send_with_retry
        interrupt_event = asyncio.Event()
        adapter._active_sessions[session_key] = interrupt_event

        event = MessageEvent(
            text="/spawn Continue in the background",
            source=source,
            message_id="m1",
            message_type=MessageType.COMMAND,
        )
        await adapter.handle_message(event)

        assert handler_called_with == [event]
        assert sent == ["spawn handled"]
        assert not interrupt_event.is_set()
        assert session_key not in adapter._pending_messages


class TestRunSpawnTask:
    @pytest.mark.asyncio
    async def test_spawn_task_uses_child_session_and_disables_pass_session_id(self, session_db):
        runner, _, history = _make_runner(session_db)
        mock_adapter = AsyncMock()
        mock_adapter.send = AsyncMock(return_value=SendResult(success=True))
        mock_adapter._send_with_retry = AsyncMock(return_value=SendResult(success=True))
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
        sent_content = mock_adapter._send_with_retry.call_args.kwargs["content"]
        assert "Spawn task complete" in sent_content
        assert "20260408_090001_child" in sent_content

    @pytest.mark.asyncio
    async def test_spawn_task_persists_result_to_parent_session_context(self, session_db):
        runner, parent_id, history = _make_runner(session_db)
        runner.session_store.append_to_transcript = MagicMock()
        mock_adapter = AsyncMock()
        mock_adapter.send = AsyncMock(return_value=SendResult(success=True))
        mock_adapter._send_with_retry = AsyncMock(return_value=SendResult(success=True))
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
                parent_session_id=parent_id,
            )

        runner.session_store.append_to_transcript.assert_called_once()
        args = runner.session_store.append_to_transcript.call_args.args
        assert args[0] == parent_id
        assert "Spawn task complete" in args[1]["content"]
        assert "Spawned result" in args[1]["content"]

    @pytest.mark.asyncio
    async def test_spawn_task_sends_text_callback_even_when_response_is_media_only(self, session_db):
        runner, parent_id, history = _make_runner(session_db)
        runner.session_store.append_to_transcript = MagicMock()
        mock_adapter = AsyncMock()
        mock_adapter.send = AsyncMock(return_value=SendResult(success=True))
        mock_adapter._send_with_retry = AsyncMock(return_value=SendResult(success=True))
        mock_adapter.send_document = AsyncMock(return_value=SendResult(success=True))
        mock_adapter.extract_media = MagicMock(return_value=([("/tmp/result.txt", False)], ""))
        mock_adapter.extract_images = MagicMock(return_value=([], ""))
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
            mock_agent.run_conversation.return_value = {"final_response": "MEDIA:/tmp/result.txt", "messages": []}
            MockAgent.return_value = mock_agent

            await runner._run_spawn_task(
                prompt="continue",
                source=source,
                task_id="spawn_test",
                child_session_id="20260408_090001_child",
                history_snapshot=history,
                parent_session_id=parent_id,
            )

        sent_content = mock_adapter._send_with_retry.call_args.kwargs["content"]
        assert "Spawn task complete" in sent_content
        assert "(See attached media)" in sent_content
        mock_adapter.send_document.assert_awaited_once()


class TestSpawnCommandRegistration:
    def test_spawn_is_known_gateway_command(self):
        from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS

        assert "spawn" in GATEWAY_KNOWN_COMMANDS

    def test_spawn_in_cli_commands(self):
        from hermes_cli.commands import COMMANDS

        assert "/spawn" in COMMANDS
