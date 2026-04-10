"""Tests for the todo tool module."""

import json

from tools.todo_tool import TODO_SNAPSHOT_MARKER, TodoStore, build_todo_snapshot_message, todo_tool


class TestWriteAndRead:
    def test_write_replaces_list(self):
        store = TodoStore()
        items = [
            {"id": "1", "content": "First task", "status": "pending"},
            {"id": "2", "content": "Second task", "status": "in_progress"},
        ]
        result = store.write(items)
        assert len(result) == 2
        assert result[0]["id"] == "1"
        assert result[1]["status"] == "in_progress"

    def test_read_returns_copy(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "Task", "status": "pending"}])
        items = store.read()
        items[0]["content"] = "MUTATED"
        assert store.read()[0]["content"] == "Task"

    def test_normalizes_review_loop_compat_data_to_plain_task(self):
        store = TodoStore()
        store.write([
            {
                "id": "review",
                "content": "Review popup",
                "status": "pending",
                "kind": "review_loop",
                "success_criteria": "Popup persists edits.",
                "reviewer_profile": "gpt-5.4 reviewer",
                "reviewer_prompt": "Compare expected vs actual behavior",
                "attempt_count": 2,
                "parent_id": "outer",
            }
        ])
        item = store.read()[0]
        assert item == {
            "id": "review",
            "content": "Review popup",
            "status": "pending",
            "kind": "task",
        }


class TestHasItems:
    def test_empty_store(self):
        store = TodoStore()
        assert store.has_items() is False

    def test_non_empty_store(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "x", "status": "pending"}])
        assert store.has_items() is True

    def test_has_active_items(self):
        store = TodoStore()
        assert store.has_active_items() is False
        store.write([{"id": "1", "content": "x", "status": "completed"}])
        assert store.has_active_items() is False
        store.write([{"id": "2", "content": "y", "status": "in_progress"}])
        assert store.has_active_items() is True


class TestFormatForInjection:
    def test_empty_returns_none(self):
        store = TodoStore()
        assert store.format_for_injection() is None

    def test_non_empty_shows_plain_active_tasks_only(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "Completed thing", "status": "completed"},
            {"id": "2", "content": "Queued task", "status": "pending"},
            {"id": "3", "content": "Working", "status": "in_progress", "kind": "review_loop"},
        ])
        text = store.format_for_injection()
        assert "Completed thing" not in text
        assert "Queued task" in text
        assert "Working" in text
        assert "review_loop" not in text
        assert "Scratch-like" not in text
        assert "- [ ] 2. Queued task" in text
        assert "- [>] 3. Working" in text


class TestMergeMode:
    def test_update_existing_by_id(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "Original", "status": "pending"},
        ])
        store.write(
            [{"id": "1", "status": "completed"}],
            merge=True,
        )
        items = store.read()
        assert len(items) == 1
        assert items[0]["status"] == "completed"
        assert items[0]["content"] == "Original"

    def test_merge_appends_new(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "First", "status": "pending"}])
        store.write(
            [{"id": "2", "content": "Second", "status": "pending"}],
            merge=True,
        )
        items = store.read()
        assert len(items) == 2


class TestSnapshotMessage:
    def test_build_todo_snapshot_message_has_marker(self):
        text = build_todo_snapshot_message([
            {"id": "1", "content": "Task", "status": "pending", "kind": "task"}
        ])
        assert text.startswith(TODO_SNAPSHOT_MARKER)
        data = json.loads(text.split("\n", 1)[1])
        assert data["todos"][0]["content"] == "Task"


class TestTodoToolFunction:
    def test_read_mode(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "Task", "status": "pending"}])
        result = json.loads(todo_tool(store=store))
        assert result["summary"]["total"] == 1
        assert result["summary"]["pending"] == 1

    def test_write_mode(self):
        store = TodoStore()
        result = json.loads(todo_tool(
            todos=[{"id": "1", "content": "New", "status": "in_progress"}],
            store=store,
        ))
        assert result["summary"]["in_progress"] == 1

    def test_no_store_returns_error(self):
        result = json.loads(todo_tool())
        assert "error" in result
