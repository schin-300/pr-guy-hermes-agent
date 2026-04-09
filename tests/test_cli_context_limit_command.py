from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cli import HermesCLI


class _FakeAgent:
    def __init__(self, context_length: int):
        self.context_compressor = SimpleNamespace(context_length=context_length)
        self.calls = []

    def set_context_length_override(self, value):
        self.calls.append(value)
        self.context_compressor.context_length = value
        return value


def _make_cli():
    cli = HermesCLI.__new__(HermesCLI)
    cli.agent = None
    cli.model = "gpt-5.4"
    cli.provider = "openai-codex"
    cli.context_length_override = None
    return cli


class TestCliContextLimitCommand:
    def test_process_command_routes_context_limit(self):
        cli = _make_cli()
        cli._handle_context_limit_command = MagicMock()

        result = cli.process_command("/context-limit 500000")

        assert result is True
        cli._handle_context_limit_command.assert_called_once_with("/context-limit 500000")

    def test_handle_context_limit_toggles_live_agent(self):
        cli = _make_cli()
        cli.agent = _FakeAgent(272_000)

        with patch("cli._cprint") as mock_cprint:
            cli._handle_context_limit_command("/context-limit")

        assert cli.context_length_override == 1_000_000
        assert cli.agent.calls == [1_000_000]
        printed = "\n".join(call.args[0] for call in mock_cprint.call_args_list)
        assert "1,000,000" in printed

    def test_handle_context_limit_sets_explicit_value(self):
        cli = _make_cli()
        cli.agent = _FakeAgent(272_000)

        with patch("cli._cprint"):
            cli._handle_context_limit_command("/context-limit 500000")

        assert cli.context_length_override == 500_000
        assert cli.agent.calls == [500_000]

    def test_handle_context_limit_invalid_arg_shows_usage(self):
        cli = _make_cli()

        with patch("cli._cprint") as mock_cprint:
            cli._handle_context_limit_command("/context-limit turbo")

        assert cli.context_length_override is None
        assert any("Usage: /context-limit" in call.args[0] for call in mock_cprint.call_args_list)
