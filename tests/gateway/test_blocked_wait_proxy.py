from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.blocked_wait_proxy import save_default_blocked_wait_proxy_kind
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key
from hermes_cli.config import load_config


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter.resume_typing_for_chat = MagicMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._background_tasks = set()
    runner._blocked_wait_proxy_setup_pending = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    return runner


def _install_running_agent(runner, *, wait_kind=None, current_tool=None, active_children=None):
    source = _make_source()
    session_key = build_session_key(source)
    session_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = [
        {"role": "user", "content": "Please keep working."},
        {"role": "assistant", "content": "I am checking the delegated agent now."},
    ]
    running_agent = MagicMock()
    running_agent.get_activity_summary.return_value = {
        "wait_state": {"kind": wait_kind} if wait_kind else None,
        "current_tool": current_tool,
        "active_children_count": len(active_children or []),
        "active_children": active_children or [],
        "last_activity_desc": "delegate child: read_file",
        "api_call_count": 3,
        "max_iterations": 50,
    }
    runner._running_agents[session_key] = running_agent
    return session_key, running_agent


@pytest.mark.asyncio
async def test_proxy_setup_prompt_is_generic_for_blocked_wait():
    runner = _make_runner()
    session_key, _ = _install_running_agent(
        runner,
        wait_kind=None,
        current_tool="delegate_task",
        active_children=[{"current_tool": "read_file", "seconds_since_activity": 4}],
    )

    handled, response = await runner._try_handle_blocking_wait(_make_event("is it stuck?"), session_key)

    assert handled is True
    assert "blocked-session helper" in response
    assert "delegate" in response


@pytest.mark.asyncio
async def test_proxy_setup_yes_saves_default_and_proxy_runs():
    runner = _make_runner()
    session_key, _ = _install_running_agent(
        runner,
        wait_kind="clarify",
        current_tool="clarify",
    )

    handled, response = await runner._try_handle_blocking_wait(_make_event("what are you waiting on?"), session_key)
    assert handled is True
    assert "reply yes or no" in response.lower()

    handled, response = await runner._try_handle_blocking_wait(_make_event("yes"), session_key)
    assert handled is True
    assert "enabled blocked-session helper defaults" in response.lower()

    cfg = load_config()
    assert cfg["blocked_wait_proxy"]["enabled"] is True
    assert cfg["blocked_wait_proxy"]["kinds"]["clarify"]["enabled"] is True

    with patch("agent.blocked_wait_proxy.run_blocked_wait_proxy", return_value="proxy says hello") as mock_proxy:
        handled, response = await runner._try_handle_blocking_wait(_make_event("what are the options again?"), session_key)

    assert handled is True
    assert response == "proxy says hello"
    mock_proxy.assert_called_once()


@pytest.mark.asyncio
async def test_delegate_plaintext_falls_through_as_steer_when_helper_enabled():
    save_default_blocked_wait_proxy_kind("delegate")
    runner = _make_runner()
    session_key, _ = _install_running_agent(
        runner,
        wait_kind=None,
        current_tool="delegate_task",
        active_children=[{"current_tool": "read_file", "seconds_since_activity": 4}],
    )

    handled, response = await runner._try_handle_blocking_wait(
        _make_event("tell it to inspect the real stack trace path"),
        session_key,
    )

    assert handled is False
    assert response is None


@pytest.mark.asyncio
async def test_delegate_abort_request_is_handled_by_proxy_layer():
    save_default_blocked_wait_proxy_kind("delegate")
    runner = _make_runner()
    session_key, running_agent = _install_running_agent(
        runner,
        wait_kind=None,
        current_tool="delegate_task",
        active_children=[{"current_tool": "read_file", "seconds_since_activity": 4}],
    )

    handled, response = await runner._try_handle_blocking_wait(_make_event("abort it"), session_key)

    assert handled is True
    assert "aborting" in response.lower()
    running_agent.interrupt.assert_called_once()


def test_blocked_wait_proxy_uses_profile_gateway_runtime_when_configured(monkeypatch):
    from agent import blocked_wait_proxy as proxy_mod

    monkeypatch.setattr(
        proxy_mod,
        "load_blocked_wait_proxy_config",
        lambda: {
            "enabled": True,
            "launcher": "gateway",
            "profile": "helper",
            "gateway_autostart": False,
            "kinds": {"delegate": {"enabled": True, "instructions": "stay concise"}},
        },
    )

    captured = {}

    monkeypatch.setattr(
        "hermes_cli.profiles.resolve_profile_env",
        lambda profile_name: "/tmp/helper-profile",
    )

    def _fake_ensure(*, hermes_home=None, autostart=True, timeout=15.0):
        captured["ensure"] = {
            "hermes_home": Path(hermes_home).resolve() if hermes_home is not None else None,
            "autostart": autostart,
            "timeout": timeout,
        }
        return SimpleNamespace(base_url="http://127.0.0.1:8642", api_key=None)

    class _FakeProxy:
        def __init__(self, **kwargs):
            captured["proxy_kwargs"] = kwargs

        def run_conversation(self, user_message):
            captured["user_message"] = user_message
            return {"final_response": "proxy gateway answer"}

    monkeypatch.setattr("hermes_cli.gateway_session_client.ensure_gateway_session_bridge", _fake_ensure)
    monkeypatch.setattr("hermes_cli.gateway_session_client.GatewaySessionAgentProxy", _FakeProxy)

    parent_agent = SimpleNamespace(
        model="test-model",
        provider="openrouter",
        base_url=None,
        api_key=None,
        api_mode="chat_completions",
        gateway_hosted_session=False,
        platform="telegram",
    )

    response = proxy_mod.run_blocked_wait_proxy(
        kind="delegate",
        activity={"wait_state": {"kind": "delegate"}},
        history=[{"role": "assistant", "content": "Working on it."}],
        user_message="what is it doing?",
        parent_agent=parent_agent,
    )

    assert response == "proxy gateway answer"
    assert captured["ensure"]["hermes_home"] == Path("/tmp/helper-profile").resolve()
    assert captured["ensure"]["autostart"] is False
    assert captured["proxy_kwargs"]["enabled_toolsets"] == []
    assert captured["user_message"] == "what is it doing?"
