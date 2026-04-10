import queue
from unittest.mock import MagicMock, patch

from cli import HermesCLI


class _FakeThread:
    def __init__(self, alive: bool):
        self._alive = alive

    def is_alive(self):
        return self._alive


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.config = {}
    cli_obj.console = MagicMock()
    cli_obj.agent = None
    cli_obj._agent_running = True
    cli_obj._pending_input = queue.Queue()
    cli_obj._interrupt_queue = queue.Queue()
    cli_obj._busy_command_queue = queue.Queue()
    cli_obj._tool_boundary_input_queue = queue.Queue()
    cli_obj.busy_input_mode = "queue"
    cli_obj._should_exit = False
    cli_obj._active_agent_thread = _FakeThread(True)
    cli_obj._invalidate = MagicMock()
    return cli_obj


def _drain_queue(q: queue.Queue):
    items = []
    while True:
        try:
            items.append(q.get_nowait())
        except queue.Empty:
            return items


class TestCliBusySlashCommands:
    def test_busy_command_behavior_marks_fast_as_live(self):
        cli_obj = _make_cli()

        assert cli_obj._busy_command_behavior("/fast on") == "live"

    def test_busy_command_behavior_marks_queue_as_live(self):
        cli_obj = _make_cli()

        assert cli_obj._busy_command_behavior("/queue follow up") == "live"

    def test_busy_command_behavior_marks_reasoning_show_live_and_reasoning_effort_deferred(self):
        cli_obj = _make_cli()

        assert cli_obj._busy_command_behavior("/reasoning show") == "live"
        assert cli_obj._busy_command_behavior("/reasoning high") == "defer"

    def test_busy_command_behavior_honors_unique_prefix_for_live_command(self):
        cli_obj = _make_cli()

        assert cli_obj._busy_command_behavior("/ver") == "live"

    def test_busy_command_behavior_defers_exact_quick_command_shadowing_live_prefix(self):
        cli_obj = _make_cli()
        cli_obj.config = {"quick_commands": {"ver": {"type": "alias", "target": "help"}}}

        assert cli_obj._busy_command_behavior("/ver") == "defer"

    def test_submit_busy_input_routes_live_safe_slash_command_to_busy_queue(self):
        cli_obj = _make_cli()

        with patch("cli._cprint"):
            result = cli_obj._submit_busy_input("/fast on")

        assert result == "live"
        assert _drain_queue(cli_obj._busy_command_queue) == ["/fast on"]
        assert _drain_queue(cli_obj._pending_input) == []
        assert _drain_queue(cli_obj._interrupt_queue) == []

    def test_submit_busy_input_defers_unsafe_slash_command_to_pending_queue(self):
        cli_obj = _make_cli()

        with patch("cli._cprint"):
            result = cli_obj._submit_busy_input("/model gpt-5.4")

        assert result == "defer"
        assert _drain_queue(cli_obj._busy_command_queue) == []
        assert _drain_queue(cli_obj._pending_input) == ["/model gpt-5.4"]
        assert _drain_queue(cli_obj._interrupt_queue) == []

    def test_submit_busy_input_routes_queue_command_to_live_busy_queue(self):
        cli_obj = _make_cli()

        with patch("cli._cprint"):
            result = cli_obj._submit_busy_input("/queue follow up")

        assert result == "live"
        assert _drain_queue(cli_obj._busy_command_queue) == ["/queue follow up"]
        assert _drain_queue(cli_obj._pending_input) == []
        assert _drain_queue(cli_obj._interrupt_queue) == []

    def test_submit_busy_input_preserves_non_command_interrupt_behavior(self):
        cli_obj = _make_cli()
        cli_obj.busy_input_mode = "interrupt"

        with patch("cli._cprint"):
            result = cli_obj._submit_busy_input("please stop and do this instead")

        assert result == "interrupt"
        assert _drain_queue(cli_obj._busy_command_queue) == []
        assert _drain_queue(cli_obj._pending_input) == []
        assert _drain_queue(cli_obj._interrupt_queue) == ["please stop and do this instead"]

    def test_submit_busy_input_queue_mode_stages_followup_for_next_tool_boundary(self):
        cli_obj = _make_cli()
        cli_obj.busy_input_mode = "queue"

        with patch("cli._cprint"):
            result = cli_obj._submit_busy_input("follow up after the tool call")

        assert result == "queue"
        assert _drain_queue(cli_obj._tool_boundary_input_queue) == ["follow up after the tool call"]
        assert _drain_queue(cli_obj._pending_input) == []
        assert _drain_queue(cli_obj._interrupt_queue) == []

    def test_submit_busy_empty_enter_promotes_queued_followup_to_interrupt(self):
        cli_obj = _make_cli()
        cli_obj.busy_input_mode = "queue"
        cli_obj.agent = MagicMock()
        cli_obj.agent._interrupt_requested = False
        cli_obj._tool_boundary_input_queue.put("queued follow up")

        with patch("cli._cprint"):
            result = cli_obj._submit_busy_empty_enter()

        assert result == "interrupt"
        cli_obj.agent.interrupt.assert_called_once_with("queued follow up")
        assert _drain_queue(cli_obj._tool_boundary_input_queue) == []

    def test_busy_followup_indicator_lines_show_enter_to_send_now_affordance(self):
        cli_obj = _make_cli()
        cli_obj._tool_boundary_input_queue.put("hello from the queued followup")

        lines = cli_obj._busy_followup_indicator_lines()

        assert any("next tool call" in line for line in lines)
        assert any("Enter again" in line for line in lines)
        assert any("hello from the queued followup" in line for line in lines)

    def test_submit_busy_input_defers_live_command_when_active_thread_has_already_finished(self):
        cli_obj = _make_cli()
        cli_obj._active_agent_thread = _FakeThread(False)

        with patch("cli._cprint"):
            result = cli_obj._submit_busy_input("/fast on")

        assert result == "defer"
        assert _drain_queue(cli_obj._busy_command_queue) == []
        assert _drain_queue(cli_obj._pending_input) == ["/fast on"]

    def test_process_live_busy_commands_executes_enqueued_commands(self):
        cli_obj = _make_cli()
        cli_obj._busy_command_queue.put("/fast on")
        cli_obj.process_command = MagicMock(return_value=True)

        cli_obj._process_live_busy_commands()

        cli_obj.process_command.assert_called_once_with("/fast on")
        assert _drain_queue(cli_obj._busy_command_queue) == []

    def test_process_live_busy_commands_contains_handler_exceptions(self):
        cli_obj = _make_cli()
        cli_obj._busy_command_queue.put("/fast on")
        cli_obj.process_command = MagicMock(side_effect=RuntimeError("boom"))

        with patch("cli._cprint") as mock_cprint:
            cli_obj._process_live_busy_commands()

        assert _drain_queue(cli_obj._busy_command_queue) == []
        assert cli_obj._should_exit is False
        assert any("boom" in str(call.args[0]) for call in mock_cprint.call_args_list)

    def test_agent_busy_placeholder_mentions_enter_queue_behavior(self):
        cli_obj = _make_cli()

        assert "Enter to queue after the next tool call" in cli_obj._agent_busy_placeholder_text()

    def test_queue_command_while_busy_stages_tool_boundary_input(self):
        cli_obj = _make_cli()
        cli_obj.agent = MagicMock()
        cli_obj.agent._executing_tools = True
        cli_obj.agent._current_tool = "terminal"

        with patch("cli._cprint"):
            cli_obj.process_command("/queue follow up")

        assert _drain_queue(cli_obj._tool_boundary_input_queue) == ["follow up"]
        assert _drain_queue(cli_obj._pending_input) == []

    def test_q_alias_while_busy_stages_tool_boundary_input(self):
        cli_obj = _make_cli()
        cli_obj.agent = MagicMock()
        cli_obj.agent._executing_tools = True
        cli_obj.agent._current_tool = "terminal"

        with patch("cli._cprint"):
            cli_obj.process_command("/q follow up")

        assert _drain_queue(cli_obj._tool_boundary_input_queue) == ["follow up"]
        assert _drain_queue(cli_obj._pending_input) == []

    def test_on_tool_complete_promotes_queued_followup_to_interrupt(self):
        cli_obj = _make_cli()
        cli_obj.agent = MagicMock()
        cli_obj.agent._interrupt_requested = False
        cli_obj._inline_diffs_enabled = False
        cli_obj._pending_edit_snapshots = {}
        cli_obj._tool_boundary_input_queue.put("follow up")

        cli_obj._on_tool_complete("call_1", "terminal", {"command": "echo hi"}, "ok")

        cli_obj.agent.interrupt.assert_called_once_with("follow up")
        assert _drain_queue(cli_obj._tool_boundary_input_queue) == []

    def test_flush_tool_boundary_queue_to_pending_preserves_followups(self):
        cli_obj = _make_cli()
        cli_obj._tool_boundary_input_queue.put("first")
        cli_obj._tool_boundary_input_queue.put("second")

        flushed = cli_obj._flush_tool_boundary_queue_to_pending()

        assert flushed == ["first", "second"]
        assert _drain_queue(cli_obj._pending_input) == ["first", "second"]
        assert _drain_queue(cli_obj._tool_boundary_input_queue) == []
