"""Tests for the /branch (/fork) command — session branching.

Verifies that:
- Branching creates a new session with copied conversation history
- The original session is preserved (ended with "branched" reason)
- Auto-generated titles use lineage numbering
- Custom branch names are used when provided
- parent_session_id links are set correctly
- Edge cases: empty conversation, missing session DB
"""

import os
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


@pytest.fixture
def session_db(tmp_path):
    """Create a real SessionDB for testing."""
    os.environ["HERMES_HOME"] = str(tmp_path / ".hermes")
    os.makedirs(tmp_path / ".hermes", exist_ok=True)
    from hermes_state import SessionDB
    db = SessionDB(db_path=tmp_path / ".hermes" / "test_sessions.db")
    yield db
    db.close()


@pytest.fixture
def cli_instance(tmp_path, session_db):
    """Create a minimal HermesCLI-like object for testing _handle_branch_command."""
    # We'll mock the CLI enough to test the branch logic without full init
    from unittest.mock import MagicMock

    cli = MagicMock()
    cli._session_db = session_db
    cli.session_id = "20260403_120000_abc123"
    cli.model = "anthropic/claude-sonnet-4.6"
    cli.max_turns = 90
    cli.reasoning_config = {"enabled": True, "effort": "medium"}
    cli.session_start = datetime.now()
    cli._pending_title = None
    cli._resumed = False
    cli.agent = None
    cli.conversation_history = [
        {"role": "user", "content": "Hello, can you help me?"},
        {"role": "assistant", "content": "Of course! How can I help?"},
        {"role": "user", "content": "Write a Python function to sort a list."},
        {"role": "assistant", "content": "def sort_list(lst): return sorted(lst)"},
    ]

    # Create the original session in the DB
    session_db.create_session(
        session_id=cli.session_id,
        source="cli",
        model=cli.model,
    )
    session_db.set_session_title(cli.session_id, "My Coding Session")

    return cli


class TestBranchCommandCLI:
    """Test the /branch command logic for the CLI."""

    def test_branch_creates_new_session(self, cli_instance, session_db):
        """Branching should create a new session in the DB."""
        from cli import HermesCLI

        # Call the real method on the mock, using the real implementation
        HermesCLI._handle_branch_command(cli_instance, "/branch")

        # Verify a new session was created
        assert cli_instance.session_id != "20260403_120000_abc123"
        new_session = session_db.get_session(cli_instance.session_id)
        assert new_session is not None

    def test_branch_copies_history(self, cli_instance, session_db):
        """Branching should copy all messages to the new session."""
        from cli import HermesCLI

        HermesCLI._handle_branch_command(cli_instance, "/branch")

        messages = session_db.get_messages_as_conversation(cli_instance.session_id)
        assert len(messages) == 4  # All 4 messages copied

    def test_branch_preserves_parent_link(self, cli_instance, session_db):
        """The new session should reference the original as parent."""
        from cli import HermesCLI
        original_id = cli_instance.session_id

        HermesCLI._handle_branch_command(cli_instance, "/branch")

        new_session = session_db.get_session(cli_instance.session_id)
        assert new_session["parent_session_id"] == original_id

    def test_branch_ends_original_session(self, cli_instance, session_db):
        """The original session should be marked as ended with 'branched' reason."""
        from cli import HermesCLI
        original_id = cli_instance.session_id

        HermesCLI._handle_branch_command(cli_instance, "/branch")

        original = session_db.get_session(original_id)
        assert original["end_reason"] == "branched"

    def test_branch_with_custom_name(self, cli_instance, session_db):
        """Custom branch name should be used as the title."""
        from cli import HermesCLI

        HermesCLI._handle_branch_command(cli_instance, "/branch refactor approach")

        title = session_db.get_session_title(cli_instance.session_id)
        assert title == "refactor approach"

    def test_branch_auto_title_lineage(self, cli_instance, session_db):
        """Without a name, branch should auto-generate a title from the parent's title."""
        from cli import HermesCLI

        HermesCLI._handle_branch_command(cli_instance, "/branch")

        title = session_db.get_session_title(cli_instance.session_id)
        assert title == "My Coding Session #2"

    def test_branch_empty_conversation(self, cli_instance, session_db):
        """Branching with no history should show an error."""
        from cli import HermesCLI
        cli_instance.conversation_history = []

        HermesCLI._handle_branch_command(cli_instance, "/branch")

        # session_id should not have changed
        assert cli_instance.session_id == "20260403_120000_abc123"

    def test_branch_no_session_db(self, cli_instance):
        """Branching without a session DB should show an error."""
        from cli import HermesCLI
        cli_instance._session_db = None

        HermesCLI._handle_branch_command(cli_instance, "/branch")

        # session_id should not have changed
        assert cli_instance.session_id == "20260403_120000_abc123"

    def test_branch_syncs_agent(self, cli_instance, session_db):
        """If an agent is active, branch should sync it to the new session."""
        from cli import HermesCLI

        agent = MagicMock()
        agent._last_flushed_db_idx = 0
        cli_instance.agent = agent

        HermesCLI._handle_branch_command(cli_instance, "/branch")

        # Agent should have been updated
        assert agent.session_id == cli_instance.session_id
        assert agent.reset_session_state.called
        assert agent._last_flushed_db_idx == 4  # len(conversation_history)

    def test_branch_sets_resumed_flag(self, cli_instance, session_db):
        """Branch should set _resumed=True to prevent auto-title generation."""
        from cli import HermesCLI

        HermesCLI._handle_branch_command(cli_instance, "/branch")

        assert cli_instance._resumed is True

    def test_fork_alias(self):
        """The /fork alias should resolve to 'branch'."""
        from hermes_cli.commands import resolve_command
        result = resolve_command("fork")
        assert result is not None
        assert result.name == "branch"


def _find_child_sessions(session_db, parent_session_id):
    cursor = session_db._conn.execute(
        "SELECT id FROM sessions WHERE parent_session_id = ? ORDER BY started_at",
        (parent_session_id,),
    )
    return [row["id"] for row in cursor.fetchall()]


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


class TestSpawnCommandCLI:
    """Test the /spawn command logic for the CLI."""

    def _configure_spawn_cli(self, cli_instance):
        cli_instance._background_task_counter = 0
        cli_instance._background_tasks = {}
        cli_instance.enabled_toolsets = ["terminal", "file"]
        cli_instance._providers_only = None
        cli_instance._providers_ignore = None
        cli_instance._providers_order = None
        cli_instance._provider_sort = None
        cli_instance._provider_require_params = False
        cli_instance._provider_data_collection = None
        cli_instance._fallback_model = None
        cli_instance._agent_running = False
        cli_instance._spinner_text = ""
        cli_instance._app = None
        cli_instance.bell_on_complete = False
        cli_instance._ensure_runtime_credentials = MagicMock(return_value=True)
        cli_instance._resolve_turn_agent_config = MagicMock(
            return_value={"model": cli_instance.model, "runtime": {}}
        )

    def test_spawn_creates_child_session_without_switching(self, cli_instance, session_db):
        from cli import HermesCLI

        self._configure_spawn_cli(cli_instance)
        original_id = cli_instance.session_id

        with patch("cli.threading.Thread", _ImmediateThread), \
             patch("cli.AIAgent") as MockAgent, \
             patch("cli.ChatConsole") as MockChatConsole:
            MockAgent.return_value.run_conversation.return_value = {"final_response": ""}
            MockChatConsole.return_value.print = MagicMock()
            HermesCLI._handle_spawn_command(cli_instance, "/spawn Continue in the background")

        child_sessions = _find_child_sessions(session_db, original_id)
        assert len(child_sessions) == 1
        child_id = child_sessions[0]
        assert cli_instance.session_id == original_id
        assert child_id != original_id
        assert session_db.get_session(original_id)["end_reason"] is None
        assert session_db.get_session(child_id)["parent_session_id"] == original_id
        assert len(session_db.get_messages_as_conversation(child_id)) == 4

    def test_spawn_runs_child_with_pass_session_id_disabled(self, cli_instance, session_db):
        from cli import HermesCLI

        self._configure_spawn_cli(cli_instance)
        original_id = cli_instance.session_id

        with patch("cli.threading.Thread", _ImmediateThread), \
             patch("cli.AIAgent") as MockAgent, \
             patch("cli.ChatConsole") as MockChatConsole:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": ""}
            MockAgent.return_value = mock_agent
            MockChatConsole.return_value.print = MagicMock()

            HermesCLI._handle_spawn_command(cli_instance, "/spawn Continue in the background")

        child_id = _find_child_sessions(session_db, original_id)[0]
        kwargs = MockAgent.call_args.kwargs
        assert kwargs["session_id"] == child_id
        assert kwargs["pass_session_id"] is False
        mock_agent.run_conversation.assert_called_once()
        assert mock_agent.run_conversation.call_args.kwargs["conversation_history"] == cli_instance.conversation_history


class TestBranchCommandDef:
    """Test the CommandDef registration for /branch and /spawn."""

    def test_branch_in_registry(self):
        """The branch command should be in the command registry."""
        from hermes_cli.commands import COMMAND_REGISTRY
        names = [c.name for c in COMMAND_REGISTRY]
        assert "branch" in names

    def test_branch_has_fork_alias(self):
        """The branch command should have 'fork' as an alias."""
        from hermes_cli.commands import COMMAND_REGISTRY
        branch = next(c for c in COMMAND_REGISTRY if c.name == "branch")
        assert "fork" in branch.aliases

    def test_branch_in_session_category(self):
        """The branch command should be in the Session category."""
        from hermes_cli.commands import COMMAND_REGISTRY
        branch = next(c for c in COMMAND_REGISTRY if c.name == "branch")
        assert branch.category == "Session"

    def test_spawn_in_registry(self):
        from hermes_cli.commands import COMMAND_REGISTRY
        names = [c.name for c in COMMAND_REGISTRY]
        assert "spawn" in names

    def test_spawn_in_session_category(self):
        from hermes_cli.commands import COMMAND_REGISTRY
        spawn = next(c for c in COMMAND_REGISTRY if c.name == "spawn")
        assert spawn.category == "Session"
