"""Tests for HermesCLI initialization -- catches configuration bugs
that only manifest at runtime (not in mocked unit tests)."""

import os
import queue
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_cli(env_overrides=None, config_overrides=None, **kwargs):
    """Create a HermesCLI instance with minimal mocking."""
    import importlib

    _clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": {"compact": False, "tool_progress": "all"},
        "agent": {},
        "terminal": {"env_type": "local"},
    }
    if config_overrides:
        _clean_config.update(config_overrides)
    clean_env = {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}
    if env_overrides:
        clean_env.update(env_overrides)
    prompt_toolkit_stubs = {
        "prompt_toolkit": MagicMock(),
        "prompt_toolkit.history": MagicMock(),
        "prompt_toolkit.styles": MagicMock(),
        "prompt_toolkit.patch_stdout": MagicMock(),
        "prompt_toolkit.application": MagicMock(),
        "prompt_toolkit.layout": MagicMock(),
        "prompt_toolkit.layout.processors": MagicMock(),
        "prompt_toolkit.filters": MagicMock(),
        "prompt_toolkit.layout.dimension": MagicMock(),
        "prompt_toolkit.layout.menus": MagicMock(),
        "prompt_toolkit.widgets": MagicMock(),
        "prompt_toolkit.key_binding": MagicMock(),
        "prompt_toolkit.completion": MagicMock(),
        "prompt_toolkit.formatted_text": MagicMock(),
        "prompt_toolkit.auto_suggest": MagicMock(),
    }
    with patch.dict(sys.modules, prompt_toolkit_stubs), \
         patch.dict("os.environ", clean_env, clear=False):
        import cli as _cli_mod
        _cli_mod = importlib.reload(_cli_mod)
        with patch.object(_cli_mod, "get_tool_definitions", return_value=[]), \
             patch.dict(_cli_mod.__dict__, {"CLI_CONFIG": _clean_config}):
            return _cli_mod.HermesCLI(**kwargs)


class TestMaxTurnsResolution:
    """max_turns must always resolve to a positive integer, never None."""

    def test_default_max_turns_is_integer(self):
        cli = _make_cli()
        assert isinstance(cli.max_turns, int)
        assert cli.max_turns == 90

    def test_explicit_max_turns_honored(self):
        cli = _make_cli(max_turns=25)
        assert cli.max_turns == 25

    def test_none_max_turns_gets_default(self):
        cli = _make_cli(max_turns=None)
        assert isinstance(cli.max_turns, int)
        assert cli.max_turns == 90

    def test_env_var_max_turns(self):
        """Env var is used when config file doesn't set max_turns."""
        cli_obj = _make_cli(env_overrides={"HERMES_MAX_ITERATIONS": "42"})
        assert cli_obj.max_turns == 42

    def test_legacy_root_max_turns_is_used_when_agent_key_exists_without_value(self):
        cli_obj = _make_cli(config_overrides={"agent": {}, "max_turns": 77})
        assert cli_obj.max_turns == 77

    def test_max_turns_never_none_for_agent(self):
        """The value passed to AIAgent must never be None (causes TypeError in run_conversation)."""
        cli = _make_cli()
        assert isinstance(cli.max_turns, int) and cli.max_turns == 90


class TestContextModeDefaults:
    def test_cli_defaults_to_1m_context_mode(self):
        cli = _make_cli(config_overrides={"compression": {"threshold": 0.50}})
        assert cli.context_length_override == 1_000_000
        assert cli.context_compaction_threshold == 0.75


class TestVerboseAndToolProgress:
    def test_default_verbose_is_bool(self):
        cli = _make_cli()
        assert isinstance(cli.verbose, bool)

    def test_tool_progress_mode_is_string(self):
        cli = _make_cli()
        assert isinstance(cli.tool_progress_mode, str)
        assert cli.tool_progress_mode in ("off", "new", "all", "verbose")


class TestBusyInputMode:
    def test_default_busy_input_mode_is_queue(self):
        cli = _make_cli()
        assert cli.busy_input_mode == "queue"

    def test_busy_input_mode_queue_is_honored(self):
        cli = _make_cli(config_overrides={"display": {"busy_input_mode": "queue"}})
        assert cli.busy_input_mode == "queue"

    def test_unknown_busy_input_mode_falls_back_to_queue(self):
        cli = _make_cli(config_overrides={"display": {"busy_input_mode": "bogus"}})
        assert cli.busy_input_mode == "queue"

    def test_queue_command_works_while_busy(self):
        """When agent is running, /queue should stage the prompt for the next safe boundary."""
        cli = _make_cli()
        cli._agent_running = True
        cli.process_command("/queue follow up")
        assert cli._tool_boundary_input_queue.get_nowait() == "follow up"
        assert cli._pending_input.empty()

    def test_queue_command_works_while_idle(self):
        """When agent is idle, /queue should still queue (not reject)."""
        cli = _make_cli()
        cli._agent_running = False
        cli.process_command("/queue follow up")
        assert cli._pending_input.get_nowait() == "follow up"

    def test_q_alias_works_while_idle(self):
        cli = _make_cli()
        cli._agent_running = False
        cli.process_command("/q follow up")
        assert cli._pending_input.get_nowait() == "follow up"

    def test_bare_q_shows_queue_usage_and_exit_guidance(self):
        cli = _make_cli()

        with patch("cli._cprint") as mock_cprint:
            result = cli.process_command("/q")

        assert result is True
        printed = "\n".join(str(call.args[0]) for call in mock_cprint.call_args_list)
        assert "Usage: /queue <prompt>" in printed
        assert "/quit or /exit" in printed

    def test_queue_mode_routes_busy_enter_to_tool_boundary_queue(self):
        """In queue mode, Enter while busy should stage the message for the next tool boundary."""
        cli = _make_cli(config_overrides={"display": {"busy_input_mode": "queue"}})
        cli._agent_running = True

        result = cli._submit_busy_input("follow up")

        assert result == "queue"
        assert cli._tool_boundary_input_queue.get_nowait() == "follow up"
        assert cli._pending_input.empty()
        assert cli._interrupt_queue.empty()

    def test_interrupt_mode_routes_busy_enter_to_interrupt(self):
        """In interrupt mode, Enter while busy goes to _interrupt_queue."""
        cli = _make_cli(config_overrides={"display": {"busy_input_mode": "interrupt"}})
        cli._agent_running = True

        result = cli._submit_busy_input("redirect")

        assert result == "interrupt"
        assert cli._interrupt_queue.get_nowait() == "redirect"
        assert cli._pending_input.empty()


class TestSingleQueryState:
    def test_voice_and_interrupt_state_initialized_before_run(self):
        """Single-query mode calls chat() without going through run()."""
        cli = _make_cli()
        assert cli._voice_tts is False
        assert cli._voice_mode is False
        assert cli._voice_tts_done.is_set()
        assert hasattr(cli, "_interrupt_queue")
        assert hasattr(cli, "_pending_input")


class TestBackgroundCompletionNotifications:
    def test_duplicate_completion_for_same_process_is_only_queued_once(self):
        cli = _make_cli()
        completion = {
            "session_id": "proc_dup",
            "command": "./.venv/bin/asi-headspace --host 127.0.0.1 --port 31999",
            "exit_code": -15,
            "output": "headspace listening on http://127.0.0.1:31999\n",
        }

        assert cli._queue_background_completion_notification(completion) is True
        assert cli._queue_background_completion_notification(completion) is False

        queued = cli._pending_input.get_nowait()
        assert "proc_dup" in queued
        assert cli._pending_input.empty()

    def test_distinct_completion_processes_are_both_queued(self):
        cli = _make_cli()
        first = {
            "session_id": "proc_one",
            "command": "echo one",
            "exit_code": 0,
            "output": "one",
        }
        second = {
            "session_id": "proc_two",
            "command": "echo two",
            "exit_code": 0,
            "output": "two",
        }

        assert cli._queue_background_completion_notification(first) is True
        assert cli._queue_background_completion_notification(second) is True

        queued = [cli._pending_input.get_nowait(), cli._pending_input.get_nowait()]
        assert any("proc_one" in item for item in queued)
        assert any("proc_two" in item for item in queued)
        with patch("cli._cprint"):
            with patch("tools.process_registry.process_registry") as mock_registry:
                mock_registry.completion_queue = queue.Queue()
                cli._drain_background_completion_notifications()
        assert cli._pending_input.empty()


class TestHistoryDisplay:
    def test_history_numbers_only_visible_messages_and_summarizes_tools(self, capsys):
        cli = _make_cli()
        cli.conversation_history = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1"}, {"id": "call_2"}],
            },
            {"role": "tool", "content": "tool output 1"},
            {"role": "tool", "content": "tool output 2"},
            {"role": "assistant", "content": "All set."},
            {"role": "user", "content": "A" * 250},
        ]

        cli.show_history()
        output = capsys.readouterr().out

        assert "[You #1]" in output
        assert "[Hermes #2]" in output
        assert "(requested 2 tool calls)" in output
        assert "[Tools]" in output
        assert "(2 tool messages hidden)" in output
        assert "[Hermes #3]" in output
        assert "[You #4]" in output
        assert "[You #5]" not in output
        assert "A" * 250 in output
        assert "A" * 250 + "..." not in output

    def test_history_shows_recent_sessions_when_current_chat_is_empty(self, capsys):
        cli = _make_cli()
        cli.session_id = "current"
        cli._session_db = MagicMock()
        cli._session_db.list_sessions_rich.return_value = [
            {
                "id": "current",
                "title": "Current",
                "preview": "Current preview",
                "last_active": 0,
            },
            {
                "id": "20260401_201329_d85961",
                "title": "Checking Running Hermes Agent",
                "preview": "check running gateways for hermes agent",
                "last_active": 0,
            },
        ]

        cli.show_history()
        output = capsys.readouterr().out

        assert "No messages in the current chat yet" in output
        assert "Checking Running Hermes Agent" in output
        assert "20260401_201329_d85961" in output
        assert "/resume" in output
        assert "Current preview" not in output

    def test_resume_without_target_lists_recent_sessions(self, capsys):
        cli = _make_cli()
        cli.session_id = "current"
        cli._session_db = MagicMock()
        cli._session_db.list_sessions_rich.return_value = [
            {
                "id": "current",
                "title": "Current",
                "preview": "Current preview",
                "last_active": 0,
            },
            {
                "id": "20260401_201329_d85961",
                "title": "Checking Running Hermes Agent",
                "preview": "check running gateways for hermes agent",
                "last_active": 0,
            },
        ]

        cli._handle_resume_command("/resume")
        output = capsys.readouterr().out

        assert "Recent sessions" in output
        assert "Checking Running Hermes Agent" in output
        assert "Use /resume <session id or title> to continue" in output


class TestRootLevelProviderOverride:
    """Root-level provider/base_url in config.yaml must NOT override model.provider."""

    def test_model_provider_wins_over_root_provider(self, tmp_path, monkeypatch):
        """model.provider takes priority — root-level provider is only a fallback."""
        import yaml

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config_path = hermes_home / "config.yaml"
        config_path.write_text(yaml.safe_dump({
            "provider": "opencode-go",  # stale root-level key
            "model": {
                "default": "google/gemini-3-flash-preview",
                "provider": "openrouter",  # correct canonical key
            },
        }))

        import cli
        monkeypatch.setattr(cli, "_hermes_home", hermes_home)
        cfg = cli.load_cli_config()

        assert cfg["model"]["provider"] == "openrouter"

    def test_root_provider_ignored_when_default_model_provider_exists(self, tmp_path, monkeypatch):
        """Even when model.provider is the default 'auto', root-level provider is ignored."""
        import yaml

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config_path = hermes_home / "config.yaml"
        config_path.write_text(yaml.safe_dump({
            "provider": "opencode-go",  # stale root key
            "model": {
                "default": "google/gemini-3-flash-preview",
                # no explicit model.provider — defaults provide "auto"
            },
        }))

        import cli
        monkeypatch.setattr(cli, "_hermes_home", hermes_home)
        cfg = cli.load_cli_config()

        # Root-level "opencode-go" must NOT leak through
        assert cfg["model"]["provider"] != "opencode-go"

    def test_normalize_root_model_keys_moves_to_model(self):
        """_normalize_root_model_keys migrates root keys into model section."""
        from hermes_cli.config import _normalize_root_model_keys

        config = {
            "provider": "opencode-go",
            "base_url": "https://example.com/v1",
            "model": {
                "default": "some-model",
            },
        }
        result = _normalize_root_model_keys(config)
        # Root keys removed
        assert "provider" not in result
        assert "base_url" not in result
        # Migrated into model section
        assert result["model"]["provider"] == "opencode-go"
        assert result["model"]["base_url"] == "https://example.com/v1"

    def test_normalize_root_model_keys_does_not_override_existing(self):
        """Existing model.provider is never overridden by root-level key."""
        from hermes_cli.config import _normalize_root_model_keys

        config = {
            "provider": "stale-provider",
            "model": {
                "default": "some-model",
                "provider": "correct-provider",
            },
        }
        result = _normalize_root_model_keys(config)
        assert result["model"]["provider"] == "correct-provider"
        assert "provider" not in result  # root key still cleaned up


class TestProviderResolution:
    def test_api_key_is_string_or_none(self):
        cli = _make_cli()
        assert cli.api_key is None or isinstance(cli.api_key, str)

    def test_base_url_is_string(self):
        cli = _make_cli()
        assert isinstance(cli.base_url, str)
        assert cli.base_url.startswith("http")

    def test_model_is_string(self):
        cli = _make_cli()
        assert isinstance(cli.model, str)
        assert isinstance(cli.model, str) and '/' in cli.model


class TestGatewayHostedSessionExitHelpers:
    def test_close_gateway_session_helper_closes_and_exits(self):
        cli = _make_cli(gateway_session_mode=True)
        cli.agent = SimpleNamespace(close_session=MagicMock())
        event = SimpleNamespace(app=SimpleNamespace(exit=MagicMock()))

        result = cli._close_gateway_session_and_exit(event, message="bye")

        assert result is True
        assert cli._closing_gateway_session is True
        cli.agent.close_session.assert_called_once()
        event.app.exit.assert_called_once()

    def test_open_sessions_browser_helper_detaches_running_session(self):
        cli = _make_cli(gateway_session_mode=True)
        cli._agent_running = True
        cli.agent = SimpleNamespace(detach=MagicMock())
        event = SimpleNamespace(app=SimpleNamespace(exit=MagicMock()))

        result = cli._open_sessions_browser_and_exit(event, message="browser")

        assert result is True
        assert cli._launch_sessions_browser is True
        cli.agent.detach.assert_called_once()
        event.app.exit.assert_called_once()

    def test_gateway_hosted_session_does_not_auto_close_on_plain_exit(self):
        cli = _make_cli(gateway_session_mode=True)
        cli._closing_gateway_session = False
        assert cli._should_end_local_session_on_exit() is False

    def test_gateway_hosted_session_does_close_on_explicit_ctrl_c_close(self):
        cli = _make_cli(gateway_session_mode=True)
        cli._closing_gateway_session = True
        assert cli._should_end_local_session_on_exit() is True
