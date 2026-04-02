from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cli import HermesCLI


def _make_cli():
    cli = HermesCLI.__new__(HermesCLI)
    cli.codex_service_tier = None
    cli.agent = None
    cli.model = "gpt-5.4"
    cli.provider = "openai-codex"
    return cli


class TestCliFastCommand:
    def test_process_command_routes_fast(self):
        cli = _make_cli()
        cli._handle_fast_command = MagicMock()

        result = cli.process_command("/fast on")

        assert result is True
        cli._handle_fast_command.assert_called_once_with("/fast on")

    def test_handle_fast_on_updates_live_agent(self):
        cli = _make_cli()
        cli.agent = SimpleNamespace(service_tier=None, provider="openai-codex", model="gpt-5.4")

        with patch("cli._cprint") as mock_cprint:
            cli._handle_fast_command("/fast on")

        assert cli.codex_service_tier == "fast"
        assert cli.agent.service_tier == "fast"
        printed = "\n".join(call.args[0] for call in mock_cprint.call_args_list)
        assert "Fast mode" in printed
        assert "service_tier=priority" in printed

    def test_handle_fast_status_does_not_mutate(self):
        cli = _make_cli()
        cli.codex_service_tier = "fast"

        with patch("cli._cprint") as mock_cprint:
            cli._handle_fast_command("/fast status")

        assert cli.codex_service_tier == "fast"
        printed = "\n".join(call.args[0] for call in mock_cprint.call_args_list)
        assert "ON" in printed

    def test_handle_fast_invalid_arg_shows_usage(self):
        cli = _make_cli()

        with patch("cli._cprint") as mock_cprint:
            cli._handle_fast_command("/fast turbo")

        assert cli.codex_service_tier is None
        assert any("Usage: /fast" in call.args[0] for call in mock_cprint.call_args_list)
