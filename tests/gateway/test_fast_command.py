import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource, SessionStore


def _make_event(text="/fast", platform=Platform.TELEGRAM, user_id="12345", chat_id="67890"):
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _make_runner(tmp_path):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._show_reasoning = False
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._agent_cache = {}
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    runner._session_db = None
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    runner.session_store = SessionStore(tmp_path / "sessions", GatewayConfig())
    return runner


class _CapturingAgent:
    last_init = None

    def __init__(self, *args, **kwargs):
        type(self).last_init = dict(kwargs)
        self.tools = []
        self.service_tier = kwargs.get("service_tier")
        self.model = kwargs.get("model")
        self.provider = kwargs.get("provider")
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_api_calls = 1
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = "unknown"
        self.session_cost_source = "none"

    def run_conversation(self, user_message: str, conversation_history=None, task_id=None):
        return {
            "final_response": "ok",
            "messages": [],
            "api_calls": 1,
            "tools": [],
        }


class TestGatewayFastCommand:
    @pytest.mark.asyncio
    async def test_help_output_lists_fast_and_fast_temp(self, tmp_path):
        runner = _make_runner(tmp_path)

        result = await runner._handle_help_command(_make_event("/help"))

        assert "/fast [on|off|status]" in result
        assert "/fast-temp [on|off|status]" in result

    @pytest.mark.asyncio
    async def test_handle_fast_command_sets_session_override(self, tmp_path):
        runner = _make_runner(tmp_path)
        event = _make_event("/fast on")

        result = await runner._handle_fast_command(event)
        entry = runner.session_store.get_or_create_session(event.source)

        assert "takes effect on next message" in result
        assert entry.service_tier_override == "fast"

    @pytest.mark.asyncio
    async def test_handle_fast_temp_command_sets_session_override(self, tmp_path):
        runner = _make_runner(tmp_path)
        event = _make_event("/fast-temp on")

        result = await runner._handle_fast_temp_command(event)
        entry = runner.session_store.get_or_create_session(event.source)

        assert "takes effect on next message" in result
        assert "not saved to config" in result
        assert entry.service_tier_override == "fast"

    def test_force_new_session_clears_fast_override(self, tmp_path):
        runner = _make_runner(tmp_path)
        source = _make_event("/fast on").source

        entry = runner.session_store.get_or_create_session(source)
        runner.session_store.set_service_tier_override(entry.session_key, "fast")

        fresh_entry = runner.session_store.get_or_create_session(source, force_new=True)

        assert fresh_entry.session_id != entry.session_id
        assert fresh_entry.service_tier_override is None

    def test_switch_session_clears_fast_override(self, tmp_path):
        runner = _make_runner(tmp_path)
        source = _make_event("/fast on").source

        entry = runner.session_store.get_or_create_session(source)
        runner.session_store.set_service_tier_override(entry.session_key, "fast")

        switched_entry = runner.session_store.switch_session(entry.session_key, "older_session_abc")

        assert switched_entry is not None
        assert switched_entry.session_id == "older_session_abc"
        assert switched_entry.service_tier_override is None

    def test_run_agent_passes_service_tier_from_session(self, tmp_path, monkeypatch):
        runner = _make_runner(tmp_path)
        event = _make_event("/fast on", platform=Platform.LOCAL, chat_id="cli")
        entry = runner.session_store.get_or_create_session(event.source)
        runner.session_store.set_service_tier_override(entry.session_key, "fast")

        monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
        monkeypatch.setattr(gateway_run, "_env_path", tmp_path / ".env")
        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs",
            lambda: {
                "provider": "openai-codex",
                "api_mode": "codex_responses",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api_key": "test-key",
            },
        )
        fake_run_agent = types.ModuleType("run_agent")
        fake_run_agent.AIAgent = _CapturingAgent
        monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

        _CapturingAgent.last_init = None
        result = asyncio.run(
            runner._run_agent(
                message="ping",
                context_prompt="",
                history=[],
                source=event.source,
                session_id=entry.session_id,
                session_key=entry.session_key,
            )
        )

        assert result["final_response"] == "ok"
        assert _CapturingAgent.last_init is not None
        assert _CapturingAgent.last_init["service_tier"] == "fast"
