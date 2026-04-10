from types import SimpleNamespace

import pytest

from hermes_cli.gateway_session_client import GatewaySessionEndpoint
from hermes_cli.main import cmd_chat


def _args(**overrides):
    base = {
        "continue_last": None,
        "resume": None,
        "query": None,
        "model": None,
        "provider": None,
        "toolsets": None,
        "skills": None,
        "verbose": False,
        "quiet": False,
        "worktree": False,
        "checkpoints": False,
        "pass_session_id": False,
        "max_turns": None,
        "source": None,
        "yolo": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_cmd_chat_interactive_uses_gateway_without_local_provider(monkeypatch):
    captured = {}

    monkeypatch.setattr("hermes_cli.main._has_any_provider_configured", lambda: False)
    monkeypatch.setattr(
        "hermes_cli.gateway_session_client.resolve_gateway_session_endpoint",
        lambda: GatewaySessionEndpoint(base_url="http://127.0.0.1:8642", api_key=None),
    )
    monkeypatch.setattr(
        "hermes_cli.gateway_session_client.check_gateway_session_endpoint",
        lambda endpoint: True,
    )
    import cli as cli_module

    monkeypatch.setattr(
        cli_module,
        "main",
        lambda **kwargs: captured.update({"called": True, "kwargs": kwargs}),
    )

    cmd_chat(_args())

    assert captured["called"] is True
    assert captured["kwargs"].get("provider") is None


def test_cmd_chat_interactive_exits_when_gateway_unavailable(monkeypatch):
    monkeypatch.setattr("hermes_cli.main._has_any_provider_configured", lambda: True)
    monkeypatch.setattr(
        "hermes_cli.gateway_session_client.resolve_gateway_session_endpoint",
        lambda: GatewaySessionEndpoint(base_url="http://127.0.0.1:8642", api_key=None),
    )
    monkeypatch.setattr(
        "hermes_cli.gateway_session_client.check_gateway_session_endpoint",
        lambda endpoint: False,
    )

    with pytest.raises(SystemExit) as exc:
        cmd_chat(_args())

    assert exc.value.code == 1
