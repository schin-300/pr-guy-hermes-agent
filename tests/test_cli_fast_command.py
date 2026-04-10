from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cli import HermesCLI


def _make_cli():
    cli = HermesCLI.__new__(HermesCLI)
    cli.codex_service_tier = None
    cli.fast_mode_enabled = False
    cli.agent = None
    cli.model = "gpt-5.4"
    cli.provider = "openai-codex"
    cli.config = {"display": {"fast_mode": False}}
    return cli


class TestCliFastCommand:
    def test_process_command_routes_fast(self):
        cli = _make_cli()
        cli._handle_fast_command = MagicMock()

        result = cli.process_command("/fast on")

        assert result is True
        cli._handle_fast_command.assert_called_once_with("/fast on")

    def test_process_command_routes_fast_temp(self):
        cli = _make_cli()
        cli._handle_fast_temp_command = MagicMock()

        result = cli.process_command("/fast-temp on")

        assert result is True
        cli._handle_fast_temp_command.assert_called_once_with("/fast-temp on")

    def test_handle_fast_on_updates_live_agent_and_persists(self):
        cli = _make_cli()
        cli.agent = SimpleNamespace(service_tier=None, provider="openai-codex", model="gpt-5.4")

        with patch("cli._cprint") as mock_cprint, \
             patch("cli.save_config_value", return_value=True) as mock_save:
            cli._handle_fast_command("/fast on")

        assert cli.codex_service_tier == "fast"
        assert cli.fast_mode_enabled is True
        assert cli.agent.service_tier == "fast"
        assert cli.config["display"]["fast_mode"] is True
        mock_save.assert_called_once_with("display.fast_mode", True)
        printed = "\n".join(call.args[0] for call in mock_cprint.call_args_list)
        assert "Fast mode" in printed
        assert "service_tier=priority" in printed

    def test_handle_fast_off_clears_live_agent_and_persists(self):
        cli = _make_cli()
        cli.fast_mode_enabled = True
        cli.codex_service_tier = "fast"
        cli.config["display"]["fast_mode"] = True
        cli.agent = SimpleNamespace(service_tier="fast", provider="openai-codex", model="gpt-5.4")

        with patch("cli._cprint") as mock_cprint, \
             patch("cli.save_config_value", return_value=True) as mock_save:
            cli._handle_fast_command("/fast off")

        assert cli.fast_mode_enabled is False
        assert cli.codex_service_tier is None
        assert cli.agent.service_tier is None
        assert cli.config["display"]["fast_mode"] is False
        mock_save.assert_called_once_with("display.fast_mode", False)
        printed = "\n".join(call.args[0] for call in mock_cprint.call_args_list)
        assert "Fast mode" in printed
        assert "off" in printed

    def test_handle_fast_status_does_not_mutate_or_persist(self):
        cli = _make_cli()
        cli.codex_service_tier = "fast"
        cli.fast_mode_enabled = True

        with patch("cli._cprint") as mock_cprint, \
             patch("cli.save_config_value") as mock_save:
            cli._handle_fast_command("/fast status")

        assert cli.codex_service_tier == "fast"
        assert cli.fast_mode_enabled is True
        mock_save.assert_not_called()
        printed = "\n".join(call.args[0] for call in mock_cprint.call_args_list)
        assert "ON" in printed

    def test_handle_fast_invalid_arg_shows_usage(self):
        cli = _make_cli()

        with patch("cli._cprint") as mock_cprint:
            cli._handle_fast_command("/fast turbo")

        assert cli.codex_service_tier is None
        assert any("Usage: /fast" in call.args[0] for call in mock_cprint.call_args_list)

    def test_handle_fast_temp_on_updates_live_agent_without_persisting(self):
        cli = _make_cli()
        cli.agent = SimpleNamespace(service_tier=None, provider="openai-codex", model="gpt-5.4")

        with patch("cli._cprint") as mock_cprint, \
             patch("cli.save_config_value") as mock_save:
            cli._handle_fast_temp_command("/fast-temp on")

        assert cli.codex_service_tier == "fast"
        assert cli.fast_mode_enabled is False
        assert cli.agent.service_tier == "fast"
        assert cli.config["display"]["fast_mode"] is False
        mock_save.assert_not_called()
        printed = "\n".join(call.args[0] for call in mock_cprint.call_args_list)
        assert "Fast mode" in printed
        assert "not saved to config" in printed

    def test_handle_fast_temp_off_keeps_persistent_default_unchanged(self):
        cli = _make_cli()
        cli.fast_mode_enabled = True
        cli.codex_service_tier = "fast"
        cli.config["display"]["fast_mode"] = True
        cli.agent = SimpleNamespace(service_tier="fast", provider="openai-codex", model="gpt-5.4")

        with patch("cli._cprint") as mock_cprint, \
             patch("cli.save_config_value") as mock_save:
            cli._handle_fast_temp_command("/fast-temp off")

        assert cli.fast_mode_enabled is True
        assert cli.codex_service_tier is None
        assert cli.agent.service_tier is None
        assert cli.config["display"]["fast_mode"] is True
        mock_save.assert_not_called()
        printed = "\n".join(call.args[0] for call in mock_cprint.call_args_list)
        assert "Fast mode" in printed
        assert "not saved to config" in printed
