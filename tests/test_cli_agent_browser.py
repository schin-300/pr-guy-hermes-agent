from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from prompt_toolkit.key_binding import KeyBindings

from hermes_cli.agent_browser import BrowserCard



def _make_cli(**kwargs):
    clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": {"compact": False, "tool_progress": "all"},
        "agent": {},
        "terminal": {"env_type": "local"},
    }
    clean_env = {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}
    prompt_toolkit_stubs = {
        "prompt_toolkit": MagicMock(),
        "prompt_toolkit.history": MagicMock(),
        "prompt_toolkit.styles": MagicMock(),
        "prompt_toolkit.patch_stdout": MagicMock(),
        "prompt_toolkit.application": MagicMock(),
        "prompt_toolkit.layout": MagicMock(),
        "prompt_toolkit.layout.processors": MagicMock(),
        "prompt_toolkit.layout.dimension": MagicMock(),
        "prompt_toolkit.layout.menus": MagicMock(),
        "prompt_toolkit.widgets": MagicMock(),
        "prompt_toolkit.key_binding": MagicMock(),
        "prompt_toolkit.completion": MagicMock(),
        "prompt_toolkit.formatted_text": MagicMock(),
        "prompt_toolkit.auto_suggest": MagicMock(),
    }
    with patch.dict(sys.modules, prompt_toolkit_stubs), patch.dict("os.environ", clean_env, clear=False):
        import cli as cli_mod

        cli_mod = importlib.reload(cli_mod)
        with patch.object(cli_mod, "get_tool_definitions", return_value=[]), patch.dict(
            cli_mod.__dict__, {"CLI_CONFIG": clean_config}
        ):
            return cli_mod.HermesCLI(**kwargs)



def _binding_for(kb: KeyBindings, key_name: str):
    for binding in kb.bindings:
        for key in binding.keys:
            value = getattr(key, "value", str(key)).lower()
            if value == key_name:
                return binding
    raise AssertionError(f"Binding not found for {key_name}")


class TestAgentBrowserHotkeys:
    def test_register_agent_browser_shortcuts_adds_core_bindings(self):
        cli = _make_cli()
        kb = KeyBindings()

        cli._register_agent_browser_shortcuts(kb)

        assert _binding_for(kb, "c-b") is not None
        assert _binding_for(kb, "left") is not None
        assert _binding_for(kb, "right") is not None
        assert _binding_for(kb, "up") is not None
        assert _binding_for(kb, "down") is not None
        assert _binding_for(kb, "r") is not None
        with pytest.raises(AssertionError):
            _binding_for(kb, "2")
        with pytest.raises(AssertionError):
            _binding_for(kb, "3")

    def test_ctrl_b_opens_browser(self):
        cli = _make_cli()
        kb = KeyBindings()
        cli._register_agent_browser_shortcuts(kb)
        event = SimpleNamespace(app=SimpleNamespace(invalidate=lambda: None))

        with patch.object(cli, "_toggle_agent_browser", return_value=True) as toggle:
            _binding_for(kb, "c-b").handler(event)

        toggle.assert_called_once_with(True)

    def test_activate_browser_selection_launches_sibling_session_without_exiting(self):
        cli = _make_cli()
        cli._agent_browser_cards = [
            BrowserCard(name="alpha", path=MagicMock(), is_default=False, is_current=False)
        ]
        cli._agent_browser_selected_index = 0

        with patch.object(cli, "_current_profile_name", return_value="default"), patch.object(
            cli, "_launch_agent_browser_target", return_value=True
        ) as launch_target, patch.object(cli, "_toggle_agent_browser", return_value=False) as toggle, patch(
            "hermes_cli.profiles.set_active_profile"
        ) as set_active:
            event_app = SimpleNamespace(exit=MagicMock(), invalidate=MagicMock())
            result = cli._activate_agent_browser_selection(event_app=event_app)

        assert result == {"action": "launch", "profile": "alpha"}
        launch_target.assert_called_once_with("alpha")
        toggle.assert_called_once_with(False)
        assert cli._should_exit is False
        set_active.assert_not_called()
        event_app.exit.assert_not_called()

    def test_activate_browser_selection_on_current_profile_just_closes(self):
        cli = _make_cli()
        cli._agent_browser_cards = [
            BrowserCard(name="default", path=MagicMock(), is_default=True, is_current=True)
        ]
        cli._agent_browser_selected_index = 0

        with patch.object(cli, "_current_profile_name", return_value="default"), patch.object(
            cli, "_toggle_agent_browser", return_value=False
        ) as toggle:
            result = cli._activate_agent_browser_selection(event_app=SimpleNamespace(exit=MagicMock()))

        assert result == {"action": "close", "profile": "default"}
        toggle.assert_called_once_with(False)

    def test_launch_agent_browser_target_opens_tmux_window(self):
        cli = _make_cli()
        fake_script = Path(tempfile.mkdtemp()) / "hermes"
        fake_script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake_script.chmod(0o755)

        def _which(name: str):
            if name == "tmux":
                return "/usr/bin/tmux"
            return None

        with patch("sys.argv", [str(fake_script)]), patch.dict("os.environ", {"TMUX": "1"}, clear=False), patch(
            "hermes_cli.profiles.get_profile_dir", return_value=Path("/tmp/profile-alpha")
        ), patch("shutil.which", side_effect=_which), patch("subprocess.Popen") as popen:
            assert cli._launch_agent_browser_target("alpha") is True

        popen.assert_called_once()
        popen_args = popen.call_args.args[0]
        popen_kwargs = popen.call_args.kwargs
        assert popen_args[:4] == ["/usr/bin/tmux", "new-window", "-n", "hermes:alpha"]
        assert "HERMES_HOME=/tmp/profile-alpha" in popen_args[-1]
        assert str(fake_script) in popen_args[-1]
        assert popen_kwargs["start_new_session"] is True

    def test_launch_agent_browser_target_falls_back_to_terminal_emulator(self):
        cli = _make_cli()
        fake_script = Path(tempfile.mkdtemp()) / "hermes"
        fake_script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake_script.chmod(0o755)

        def _which(name: str):
            if name == "x-terminal-emulator":
                return "/usr/bin/x-terminal-emulator"
            return None

        with patch("sys.argv", [str(fake_script)]), patch(
            "hermes_cli.profiles.get_profile_dir", return_value=Path("/tmp/profile-alpha")
        ), patch("shutil.which", side_effect=_which), patch("subprocess.Popen") as popen:
            assert cli._launch_agent_browser_target("alpha") is True

        popen.assert_called_once()
        popen_args = popen.call_args.args[0]
        popen_kwargs = popen.call_args.kwargs
        assert popen_args[:2] == ["/usr/bin/x-terminal-emulator", "-e"]
        assert popen_args[2:5] == ["sh", "-lc", popen_args[4]]
        assert str(fake_script) in popen_args[4]
        assert popen_kwargs["env"]["HERMES_HOME"] == "/tmp/profile-alpha"
        assert popen_kwargs["start_new_session"] is True

    def test_agent_browser_defaults_to_three_up_and_moves_by_rendered_rows(self):
        cli = _make_cli()
        cli._agent_browser_cards = [
            BrowserCard(name=f"card-{idx}", path=MagicMock(), is_default=False, is_current=False)
            for idx in range(6)
        ]
        cli._agent_browser_selected_index = 1
        cli._app = SimpleNamespace(
            output=SimpleNamespace(get_size=lambda: SimpleNamespace(columns=74, rows=30)),
            invalidate=lambda: None,
        )

        assert cli._agent_browser_columns == 3
        assert cli._set_agent_browser_columns(2) == 3

        cli._move_agent_browser_vertical(1)
        assert cli._agent_browser_selected_index == 3

        cli._move_agent_browser_vertical(-1)
        assert cli._agent_browser_selected_index == 1

    def test_browser_prompt_fragment_takes_over_when_open(self):
        cli = _make_cli()
        cli._agent_browser_open = True

        fragments = cli._get_tui_prompt_fragments()

        assert fragments == [("class:prompt-working", "▣ browse ")]
