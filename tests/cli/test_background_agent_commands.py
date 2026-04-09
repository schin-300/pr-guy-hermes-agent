import os
import threading
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from cli import HermesCLI


@pytest.fixture
def session_db(tmp_path):
    os.environ["HERMES_HOME"] = str(tmp_path / ".hermes")
    os.makedirs(tmp_path / ".hermes", exist_ok=True)
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / ".hermes" / "test_sessions.db")
    yield db
    db.close()


@pytest.fixture
def cli_instance(session_db):
    cli = MagicMock()
    cli._session_db = session_db
    cli.session_id = "20260409_010000_abc123"
    cli.model = "anthropic/claude-sonnet-4.6"
    cli.max_turns = 90
    cli.reasoning_config = {"enabled": True, "effort": "medium"}
    cli.session_start = datetime.now()
    cli._pending_title = None
    cli._resumed = False
    cli.agent = None
    cli.conversation_history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Please implement the feature."},
        {"role": "assistant", "content": "Working on it."},
    ]
    cli.enabled_toolsets = ["terminal", "file"]
    cli._providers_only = None
    cli._providers_ignore = None
    cli._providers_order = None
    cli._provider_sort = None
    cli._provider_require_params = False
    cli._provider_data_collection = None
    cli._fallback_model = None
    cli._agent_running = False
    cli._spinner_text = ""
    cli._app = None
    cli.bell_on_complete = False
    cli._background_tasks = {}
    cli._background_task_counter = 0
    cli._ensure_runtime_credentials = MagicMock(return_value=True)
    cli._resolve_turn_agent_config = MagicMock(return_value={"model": cli.model, "runtime": {}})
    cli._make_background_clarify_callback = MagicMock(return_value="CALLBACK")
    cli._remember_assistant_message = MagicMock()
    cli._show_assistant_copy_hint = MagicMock()
    cli._invalidate = MagicMock()

    session_db.create_session(
        session_id=cli.session_id,
        source="cli",
        model=cli.model,
    )
    session_db.set_session_title(cli.session_id, "Hermes Addition Session")
    return cli


class _ImmediateThread:
    def __init__(self, target=None, daemon=None, name=None, args=(), kwargs=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}
        self.daemon = daemon
        self.name = name

    def start(self):
        if self._target:
            self._target(*self._args, **self._kwargs)


def test_make_background_clarify_callback_uses_kitty_overlay():
    cli = HermesCLI.__new__(HermesCLI)
    cli._background_prompt_lock = threading.Lock()
    cli._clarify_callback = MagicMock(return_value="fallback")

    with patch("cli.kitty_overlay_available", return_value=True), \
         patch("cli.prompt_kitty_overlay_clarify", return_value="overlay answer") as overlay_prompt:
        callback = HermesCLI._make_background_clarify_callback(cli, "spawn #1")
        result = callback("Which branch should I use?", ["main", "dev"])

    assert result == "overlay answer"
    cli._clarify_callback.assert_not_called()
    overlay_prompt.assert_called_once()


def test_make_background_clarify_callback_falls_back_to_inline_prompt():
    cli = HermesCLI.__new__(HermesCLI)
    cli._background_prompt_lock = threading.Lock()
    cli._clarify_callback = MagicMock(return_value="fallback")

    with patch("cli.kitty_overlay_available", return_value=False):
        callback = HermesCLI._make_background_clarify_callback(cli, "spawn #1")
        result = callback("Which branch should I use?", ["main", "dev"])

    assert result == "fallback"
    cli._clarify_callback.assert_called_once_with("Which branch should I use?", ["main", "dev"])


def test_spawn_passes_background_clarify_callback_to_child_agent(cli_instance):
    with patch("cli.threading.Thread", _ImmediateThread), \
         patch("cli.AIAgent") as MockAgent, \
         patch("cli.ChatConsole") as MockChatConsole:
        MockAgent.return_value.run_conversation.return_value = {"final_response": ""}
        MockChatConsole.return_value.print = MagicMock()

        HermesCLI._handle_spawn_command(cli_instance, "/spawn Continue in the background")

    assert MockAgent.call_args.kwargs["clarify_callback"] == "CALLBACK"
    cli_instance._make_background_clarify_callback.assert_called_once()


def test_background_passes_background_clarify_callback_to_child_agent(cli_instance):
    with patch("cli.threading.Thread", _ImmediateThread), \
         patch("cli.AIAgent") as MockAgent, \
         patch("cli.ChatConsole") as MockChatConsole:
        MockAgent.return_value.run_conversation.return_value = {"final_response": ""}
        MockChatConsole.return_value.print = MagicMock()

        HermesCLI._handle_background_command(cli_instance, "/background Implement the feature")

    assert MockAgent.call_args.kwargs["clarify_callback"] == "CALLBACK"
    cli_instance._make_background_clarify_callback.assert_called_once()


def test_hermes_addition_and_fix_a_fork_registered():
    from hermes_cli.commands import COMMAND_REGISTRY

    assert any(command.name == "hermes-addition" for command in COMMAND_REGISTRY)
    assert any(command.name == "fix-a-fork" for command in COMMAND_REGISTRY)


def test_hermes_addition_creates_worktree_and_injects_pr_workflow(cli_instance):
    cli_instance._build_hermes_addition_prompt = HermesCLI._build_hermes_addition_prompt.__get__(cli_instance, HermesCLI)

    with patch("cli._git_repo_root", return_value="/repo"), \
         patch("cli._setup_worktree", return_value={
            "path": "/repo/.worktrees/hermes-1234",
            "branch": "hermes/hermes-1234",
            "repo_root": "/repo",
         }), \
         patch("cli._git_remote_exists", return_value=True), \
         patch("cli._git_default_base_ref", return_value="origin/main"), \
         patch("cli.threading.Thread", _ImmediateThread), \
         patch("cli.AIAgent") as MockAgent, \
         patch("cli.ChatConsole") as MockChatConsole:
        MockAgent.return_value.run_conversation.return_value = {"final_response": ""}
        MockChatConsole.return_value.print = MagicMock()

        HermesCLI._handle_hermes_addition_command(cli_instance, "/hermes-addition Fix the flaky retry path")

    user_message = MockAgent.return_value.run_conversation.call_args.kwargs["user_message"]
    assert "/repo/.worktrees/hermes-1234" in user_message
    assert "hermes/hermes-1234" in user_message
    assert "gh pr create" in user_message
    assert "fork" in user_message
    assert MockAgent.call_args.kwargs["clarify_callback"] == "CALLBACK"


def test_fix_a_fork_targets_hermes_checkout_and_separates_fork_push_from_upstream_pr(cli_instance):
    cli_instance._build_fix_a_fork_prompt = HermesCLI._build_fix_a_fork_prompt.__get__(cli_instance, HermesCLI)

    with patch("cli._git_repo_root", return_value=None), \
         patch("cli._current_hermes_checkout_root", return_value="/repo", create=True), \
         patch("cli._setup_worktree", return_value={
            "path": "/repo/.worktrees/hermes-experimental-1234",
            "branch": "experimental/hermes-experimental-1234",
            "repo_root": "/repo",
         }) as mock_setup, \
         patch("cli._git_remote_exists", return_value=True), \
         patch("cli._git_default_base_ref", return_value="origin/main"), \
         patch("cli._git_remote_url", side_effect=lambda repo_root, remote_name: {
             "fork": "https://github.com/schin-300/pr-guy-hermes-agent.git",
             "origin": "https://github.com/NousResearch/hermes-agent.git",
         }[remote_name], create=True), \
         patch("cli.threading.Thread", _ImmediateThread), \
         patch("cli.AIAgent") as MockAgent, \
         patch("cli.ChatConsole") as MockChatConsole:
        MockAgent.return_value.run_conversation.return_value = {"final_response": ""}
        MockChatConsole.return_value.print = MagicMock()

        HermesCLI._handle_fix_a_fork_command(cli_instance, "/fix-a-fork Fix the flaky retry path")

    mock_setup.assert_called_once_with(
        repo_root="/repo",
        branch_prefix="experimental",
        worktree_prefix="hermes-experimental",
    )
    user_message = MockAgent.return_value.run_conversation.call_args.kwargs["user_message"]
    assert "/repo/.worktrees/hermes-experimental-1234" in user_message
    assert "experimental/hermes-experimental-1234" in user_message
    assert "commit locally only" in user_message.lower()
    assert "push branch to your fork only" in user_message.lower()
    assert "open a polished upstream pr" in user_message.lower()
    assert "nousresearch/hermes-agent" in user_message.lower()
    assert "schin-300:experimental/hermes-experimental-1234" in user_message.lower()
    assert MockAgent.call_args.kwargs["clarify_callback"] == "CALLBACK"
