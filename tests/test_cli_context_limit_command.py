from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cli import HermesCLI


class _FakeAgent:
    def __init__(self, context_length: int, threshold_percent: float = 0.75):
        self.context_compressor = SimpleNamespace(
            context_length=context_length,
            threshold_percent=threshold_percent,
            threshold_tokens=int(context_length * threshold_percent),
        )
        self.calls = []

    def set_context_profile(self, *, context_length=None, compression_threshold=None):
        self.calls.append(
            {
                "context_length": context_length,
                "compression_threshold": compression_threshold,
            }
        )
        if context_length is not None:
            self.context_compressor.context_length = context_length
        if compression_threshold is not None:
            self.context_compressor.threshold_percent = compression_threshold
        self.context_compressor.threshold_tokens = int(
            self.context_compressor.context_length * self.context_compressor.threshold_percent
        )
        return self.context_compressor.context_length



def _make_cli():
    cli = HermesCLI.__new__(HermesCLI)
    cli.agent = None
    cli.model = "gpt-5.4"
    cli.provider = "openai-codex"
    cli.context_length_override = 1_000_000
    cli.context_compaction_threshold = 0.75
    cli.config = {}
    cli._invalidate = MagicMock()
    return cli


class TestCliContextCommands:
    def test_process_command_routes_context_mode(self):
        cli = _make_cli()
        cli._handle_context_mode_command = MagicMock()

        result = cli.process_command("/context-mode 272k")

        assert result is True
        cli._handle_context_mode_command.assert_called_once_with("/context-mode 272k")

    def test_process_command_rejects_removed_context_limit_temp_command(self):
        cli = _make_cli()

        with patch("cli._cprint") as mock_cprint:
            cli.process_command("/context-limit-temp 500000")

        printed = "\n".join(call.args[0] for call in mock_cprint.call_args_list)
        assert "Unknown command" in printed

    def test_process_command_rejects_removed_context_limit_alias(self):
        cli = _make_cli()

        with patch("cli._cprint") as mock_cprint:
            cli.process_command("/context-limit 500000")

        printed = "\n".join(call.args[0] for call in mock_cprint.call_args_list)
        assert "Unknown command" in printed

    def test_handle_context_mode_toggle_updates_live_agent_and_compaction(self):
        cli = _make_cli()
        cli.agent = _FakeAgent(1_000_000, 0.75)

        with patch("cli._cprint") as mock_cprint:
            cli._handle_context_mode_command("/context-mode")

        assert cli.context_length_override == 272_000
        assert cli.context_compaction_threshold == 0.95
        assert cli.agent.calls == [
            {"context_length": 272_000, "compression_threshold": 0.95}
        ]
        printed = "\n".join(call.args[0] for call in mock_cprint.call_args_list)
        assert "Context mode: 272k" in printed
        assert "258,400" in printed
        assert "not saved to config" in printed

    def test_handle_context_mode_status_reports_current_mode(self):
        cli = _make_cli()
        cli.agent = _FakeAgent(1_000_000, 0.75)

        with patch("cli._cprint") as mock_cprint:
            cli._handle_context_mode_command("/context-mode status")

        printed = "\n".join(call.args[0] for call in mock_cprint.call_args_list)
        assert "Context mode: 1m" in printed
        assert "750,000" in printed

    def test_handle_context_mode_sets_one_m_threshold_to_seventy_five_percent(self):
        cli = _make_cli()
        cli.agent = _FakeAgent(272_000, 0.95)
        cli.context_length_override = 272_000
        cli.context_compaction_threshold = 0.95

        with patch("cli._cprint") as mock_cprint:
            cli._handle_context_mode_command("/context-mode 1m")

        assert cli.context_length_override == 1_000_000
        assert cli.context_compaction_threshold == 0.75
        assert cli.agent.calls == [
            {"context_length": 1_000_000, "compression_threshold": 0.75}
        ]
        printed = "\n".join(call.args[0] for call in mock_cprint.call_args_list)
        assert "Context mode: 1m" in printed
        assert "750,000" in printed
        assert "not saved to config" in printed

    def test_handle_context_mode_invalid_arg_shows_usage(self):
        cli = _make_cli()

        with patch("cli._cprint") as mock_cprint:
            cli._handle_context_mode_command("/context-mode turbo")

        assert cli.context_length_override == 1_000_000
        assert any(
            "Usage: /context-mode" in call.args[0] for call in mock_cprint.call_args_list
        )

