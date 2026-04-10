from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.gateway_session_client import (
    GatewaySessionClientError,
    GatewaySessionEndpoint,
    ensure_gateway_session_bridge,
)
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

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("hermes_cli.main._has_any_provider_configured", lambda: False)
    monkeypatch.setattr(
        "hermes_cli.gateway_session_client.ensure_gateway_session_bridge",
        lambda timeout=15.0, autostart=True: GatewaySessionEndpoint(base_url="http://127.0.0.1:8642", api_key=None),
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
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("hermes_cli.main._has_any_provider_configured", lambda: True)
    monkeypatch.setattr(
        "hermes_cli.gateway_session_client.ensure_gateway_session_bridge",
        lambda timeout=15.0, autostart=True: (_ for _ in ()).throw(GatewaySessionClientError("bridge down")),
    )

    with pytest.raises(SystemExit) as exc:
        cmd_chat(_args())

    assert exc.value.code == 1


def test_ensure_gateway_session_bridge_uses_target_profile_home(monkeypatch, tmp_path):
    profile_home = tmp_path / "helper"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text(
        "platforms:\n  api_server:\n    extra:\n      host: 127.0.0.1\n      port: 9988\n      key: profile-key\n",
        encoding="utf-8",
    )

    checks = []
    launches = {}

    def _fake_check(endpoint, timeout=1.5):
        checks.append((endpoint.base_url, endpoint.api_key, timeout))
        return len(checks) > 1

    monkeypatch.setattr("hermes_cli.gateway_session_client.check_gateway_session_endpoint", _fake_check)
    monkeypatch.setattr(
        "hermes_cli.gateway.launch_gateway_background_for_home",
        lambda hermes_home=None: launches.setdefault("home", Path(hermes_home).resolve()) or True,
    )

    endpoint = ensure_gateway_session_bridge(timeout=0.2, hermes_home=profile_home)

    assert endpoint.base_url == "http://127.0.0.1:9988"
    assert endpoint.api_key == "profile-key"
    assert launches["home"] == profile_home.resolve()
    assert checks[0][0] == "http://127.0.0.1:9988"
