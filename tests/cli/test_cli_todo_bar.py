from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cli import HermesCLI
from tools.todo_tool import TODO_SNAPSHOT_MARKER, TodoStore, build_todo_snapshot_message


def _make_cli(model: str = "anthropic/claude-sonnet-4-20250514"):
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.model = model
    cli_obj.agent = None
    cli_obj._clarify_state = None
    cli_obj._approval_state = None
    cli_obj._sudo_state = None
    cli_obj._secret_state = None
    cli_obj._todo_plan_signature = ""
    cli_obj._todo_plan_updated_at = 0.0
    cli_obj._todo_plan_refresh_timer = None
    cli_obj._inline_diffs_enabled = False
    cli_obj._pending_edit_snapshots = {}
    cli_obj._app = None
    cli_obj._local_todo_store = None
    cli_obj._plan_popup_state = None
    cli_obj.conversation_history = []
    cli_obj._session_db = None
    cli_obj.session_id = "session-1"
    cli_obj._invalidate = MagicMock()
    cli_obj.console = MagicMock()
    cli_obj._resumed = False
    cli_obj.resume_display = "full"
    cli_obj.max_turns = 20
    cli_obj.reasoning_config = None
    return cli_obj


def _attach_todos(cli_obj, items):
    store = TodoStore()
    store.write(items)
    cli_obj.agent = SimpleNamespace(_todo_store=store)
    return cli_obj


def _task_items(count: int):
    return [
        {"id": f"step-{idx}", "content": f"Step {idx}", "status": "pending", "kind": "task"}
        for idx in range(count)
    ]


class TestCLITodoPlanWidget:
    def test_build_todo_plan_lines_returns_empty_without_todos(self):
        cli_obj = _make_cli()

        assert cli_obj._build_todo_plan_lines(width=100) == []
        assert cli_obj._get_todo_plan_height() == 0

    def test_inline_plan_widget_hidden_while_clarify_prompt_active(self):
        cli_obj = _attach_todos(_make_cli(), _task_items(3))
        cli_obj._clarify_state = {"question": "Need your input"}

        assert cli_obj._should_show_inline_plan_widget() is False

    def test_inline_plan_widget_hidden_while_approval_prompt_active(self):
        cli_obj = _attach_todos(_make_cli(), _task_items(3))
        cli_obj._approval_state = {"command": "rm -rf /tmp/demo"}

        assert cli_obj._should_show_inline_plan_widget() is False

    def test_inline_plan_widget_visible_when_no_modal_prompt_is_active(self):
        cli_obj = _attach_todos(_make_cli(), _task_items(3))

        assert cli_obj._should_show_inline_plan_widget() is True

    def test_build_todo_plan_lines_show_plain_tasks_and_more_rows(self):
        cli_obj = _attach_todos(_make_cli(), _task_items(20))
        cli_obj._mark_todo_plan_updated(cli_obj._get_todo_items())

        lines = cli_obj._build_todo_plan_lines(width=120)
        texts = [text for _style, text in lines]

        assert texts[0] == "• Updated Plan"
        assert any("Starting with: Step 0" in text for text in texts)
        assert any("☐ Step 0" in text for text in texts)
        assert any("☐ Step 4" in text for text in texts)
        assert any("… +2 more lines" in text for text in texts)

    def test_build_plan_popup_lines_show_plain_tasks_and_more_rows(self):
        cli_obj = _attach_todos(_make_cli(), _task_items(20))
        cli_obj._plan_popup_state = {"selected": 10, "mode": "browse"}

        lines = cli_obj._build_plan_popup_lines(width=120)
        texts = [text for _style, text in lines]

        assert texts[0].startswith("↑/↓ select · a add task")
        assert any("esc close" in text for text in texts)
        assert any("❯ ☐ Step 10" in text for text in texts)
        assert any("☐ Step 7" in text for text in texts)
        assert not any("[review-loop]" in text.lower() for text in texts)

    def test_plan_popup_commit_entry_adds_plain_task_and_persists_snapshot(self):
        cli_obj = _attach_todos(
            _make_cli(),
            [{"id": "inspect", "content": "Inspect code paths", "status": "pending", "kind": "task"}],
        )
        cli_obj._plan_popup_state = {"selected": 0, "mode": "add_task"}

        committed = cli_obj._plan_popup_commit_entry("Follow up on the remaining edge cases")

        assert committed is True
        items = cli_obj._get_todo_items()
        assert items[-1]["content"] == "Follow up on the remaining edge cases"
        assert items[-1]["kind"] == "task"
        assert "parent_id" not in items[-1]
        assert cli_obj.conversation_history[-1]["role"] == "user"
        assert cli_obj.conversation_history[-1]["content"].startswith(TODO_SNAPSHOT_MARKER)
        assert cli_obj._plan_popup_state["mode"] == "browse"

    def test_plan_popup_set_selected_status_keeps_single_in_progress(self):
        cli_obj = _attach_todos(
            _make_cli(),
            [
                {"id": "one", "content": "First", "status": "in_progress", "kind": "task"},
                {"id": "two", "content": "Second", "status": "pending", "kind": "task"},
            ],
        )
        cli_obj._plan_popup_state = {"selected": 1, "mode": "browse"}

        cli_obj._plan_popup_set_selected_status("in_progress")

        items = cli_obj._get_todo_items()
        assert items[0]["status"] == "pending"
        assert items[1]["status"] == "in_progress"

    def test_on_tool_complete_marks_plan_updated_and_persists_snapshot(self):
        cli_obj = _attach_todos(
            _make_cli(),
            [{"id": "build", "content": "Implement popup", "status": "in_progress", "kind": "task"}],
        )

        cli_obj._on_tool_complete(
            "call_123",
            "todo",
            {
                "todos": [
                    {"id": "build", "content": "Implement popup", "status": "in_progress", "kind": "task"},
                ]
            },
            "{}",
        )

        assert cli_obj._todo_plan_signature
        assert cli_obj._todo_plan_updated_at > 0
        assert cli_obj.conversation_history[-1]["content"].startswith(TODO_SNAPSHOT_MARKER)
        cli_obj._invalidate.assert_called_once()

    def test_dedupe_todo_snapshot_messages_keeps_latest_only(self):
        cli_obj = _make_cli()
        first = build_todo_snapshot_message([
            {"id": "one", "content": "First", "status": "pending", "kind": "task"}
        ])
        second = build_todo_snapshot_message([
            {"id": "two", "content": "Second", "status": "in_progress", "kind": "task"}
        ])

        deduped = cli_obj._dedupe_todo_snapshot_messages(
            [
                {"role": "user", "content": "hello"},
                {"role": "user", "content": first},
                {"role": "assistant", "content": "noted"},
                {"role": "user", "content": second},
            ]
        )

        snapshot_messages = [
            message for message in deduped
            if isinstance(message.get("content"), str)
            and message["content"].startswith(TODO_SNAPSHOT_MARKER)
        ]
        assert len(snapshot_messages) == 1
        assert "Second" in snapshot_messages[0]["content"]

    def test_get_todo_store_hydrates_from_snapshot_history(self):
        cli_obj = _make_cli()
        cli_obj.conversation_history = [
            {
                "role": "user",
                "content": build_todo_snapshot_message(
                    [{"id": "two", "content": "Second", "status": "in_progress", "kind": "task"}]
                ),
            }
        ]

        store = cli_obj._get_todo_store()

        assert store is not None
        assert store.read()[0]["content"] == "Second"

    def test_persist_plan_snapshot_replaces_prior_db_snapshot(self):
        cli_obj = _make_cli()
        cli_obj._session_db = SimpleNamespace(_conn=MagicMock(), append_message=MagicMock())

        cli_obj._persist_plan_snapshot(
            [{"id": "two", "content": "Second", "status": "in_progress", "kind": "task"}]
        )

        cli_obj._session_db._conn.execute.assert_called_once()
        cli_obj._session_db.append_message.assert_called_once()

    def test_preload_resumed_session_excludes_snapshot_messages_from_counts(self):
        cli_obj = _make_cli()
        cli_obj._resumed = True
        cli_obj._session_db = SimpleNamespace(
            get_session=lambda _sid: {"title": "Saved plan"},
            get_messages_as_conversation=lambda _sid: [
                {"role": "user", "content": "hello"},
                {
                    "role": "user",
                    "content": build_todo_snapshot_message(
                        [{"id": "two", "content": "Second", "status": "in_progress", "kind": "task"}]
                    ),
                },
                {"role": "assistant", "content": "hi"},
            ],
            _conn=MagicMock(),
        )

        loaded = cli_obj._preload_resumed_session()

        assert loaded is True
        printed = "\n".join(str(call.args[0]) for call in cli_obj.console.print.call_args_list)
        assert "1 user message" in printed
        assert "2 total messages" in printed

    def test_display_resumed_history_hides_snapshot_messages(self):
        cli_obj = _make_cli()
        cli_obj.conversation_history = [
            {"role": "user", "content": "hello"},
            {
                "role": "user",
                "content": build_todo_snapshot_message(
                    [{"id": "two", "content": "Second", "status": "in_progress", "kind": "task"}]
                ),
            },
            {"role": "assistant", "content": "hi"},
        ]

        cli_obj._display_resumed_history()

        panel = cli_obj.console.print.call_args[0][0]
        assert TODO_SNAPSHOT_MARKER not in panel.renderable.plain

    def test_handle_resume_command_rehydrates_agent_todos_and_dedupes_snapshots(self):
        cli_obj = _make_cli()
        existing_store = TodoStore()
        existing_store.write([{"id": "old", "content": "Old", "status": "pending", "kind": "task"}])
        cli_obj.agent = SimpleNamespace(
            session_id="session-1",
            service_tier="priority",
            _last_flushed_db_idx=0,
            _todo_store=existing_store,
            reset_session_state=MagicMock(),
            set_context_length_override=MagicMock(),
            _invalidate_system_prompt=MagicMock(),
        )
        cli_obj._session_db = SimpleNamespace(
            get_session=lambda sid: {"title": "Saved plan"} if sid == "target-session" else None,
            get_messages_as_conversation=lambda sid: [
                {"role": "user", "content": "hello"},
                {
                    "role": "user",
                    "content": build_todo_snapshot_message(
                        [{"id": "one", "content": "First", "status": "pending", "kind": "task"}]
                    ),
                },
                {
                    "role": "user",
                    "content": build_todo_snapshot_message(
                        [{"id": "two", "content": "Second", "status": "in_progress", "kind": "task"}]
                    ),
                },
                {"role": "assistant", "content": "hi"},
            ],
            reopen_session=MagicMock(),
            end_session=MagicMock(),
        )

        with patch("hermes_cli.main._resolve_session_by_name_or_id", return_value="target-session"):
            cli_obj._handle_resume_command("/resume target-session")

        snapshot_messages = [
            msg for msg in cli_obj.conversation_history if cli_obj._is_todo_snapshot_message(msg)
        ]
        assert cli_obj.session_id == "target-session"
        assert len(snapshot_messages) == 1
        assert cli_obj.agent._todo_store.read()[0]["content"] == "Second"
        cli_obj.agent.reset_session_state.assert_called_once()

    def test_handle_branch_command_keeps_agent_todos_and_copies_one_snapshot(self):
        cli_obj = _make_cli()
        branch_store = TodoStore()
        branch_store.write([{"id": "two", "content": "Second", "status": "in_progress", "kind": "task"}])
        cli_obj.agent = SimpleNamespace(
            session_id="session-1",
            session_start=None,
            _last_flushed_db_idx=0,
            _todo_store=branch_store,
            reset_session_state=MagicMock(),
            _invalidate_system_prompt=MagicMock(),
        )
        cli_obj.conversation_history = [
            {"role": "user", "content": "hello"},
            {
                "role": "user",
                "content": build_todo_snapshot_message(
                    [{"id": "one", "content": "First", "status": "pending", "kind": "task"}]
                ),
            },
            {
                "role": "user",
                "content": build_todo_snapshot_message(
                    [{"id": "two", "content": "Second", "status": "in_progress", "kind": "task"}]
                ),
            },
        ]
        cli_obj._session_db = SimpleNamespace(
            end_session=MagicMock(),
            create_session=MagicMock(),
            append_message=MagicMock(),
            set_session_title=MagicMock(),
        )

        cli_obj._handle_branch_command("/branch branch-plan")

        copied_snapshot_messages = [
            call.kwargs["content"]
            for call in cli_obj._session_db.append_message.call_args_list
            if isinstance(call.kwargs.get("content"), str)
            and call.kwargs["content"].startswith(TODO_SNAPSHOT_MARKER)
        ]
        assert len(copied_snapshot_messages) == 1
        assert cli_obj.agent._todo_store.read()[0]["content"] == "Second"
        cli_obj.agent.reset_session_state.assert_called_once()
