#!/usr/bin/env python3
"""
Todo Tool Module - lightweight planning / task tracking.

The todo tool stores a flat in-memory task list on the AIAgent instance so the
agent can keep track of work when the user explicitly wants a plan/todo. The
state is re-injected after context compression and persisted via snapshot
messages when the CLI edits the board directly.

Design:
- Single `todo` tool: provide `todos` to write, omit to read
- Every call returns the full current list
- Keep the schema intentionally small and flat
- Legacy review-loop snapshots are normalized to plain tasks for compatibility
"""

import json
from typing import Any, Dict, List, Optional


VALID_STATUSES = {"pending", "in_progress", "completed", "cancelled"}
TODO_SNAPSHOT_MARKER = "[PLAN BOARD SNAPSHOT]"


class TodoStore:
    """In-memory todo list. One instance per AIAgent/session."""

    def __init__(self):
        self._items: List[Dict[str, str]] = []

    def write(self, todos: List[Dict[str, Any]], merge: bool = False) -> List[Dict[str, str]]:
        """Write todos and return the full current list."""
        if not merge:
            self._items = [self._validate(t) for t in todos]
            return self.read()

        existing = {item["id"]: item.copy() for item in self._items}
        for todo in todos:
            item_id = str(todo.get("id", "")).strip()
            if not item_id:
                continue

            if item_id in existing:
                current = existing[item_id].copy()
                if "content" in todo:
                    content = str(todo.get("content", "") or "").strip()
                    current["content"] = content or current.get("content") or "(no description)"
                if "status" in todo:
                    status = str(todo.get("status", "") or "").strip().lower()
                    if status in VALID_STATUSES:
                        current["status"] = status
                existing[item_id] = self._validate(current)
            else:
                validated = self._validate(todo)
                existing[validated["id"]] = validated
                self._items.append(validated)

        seen = set()
        rebuilt = []
        for item in self._items:
            current = existing.get(item["id"], item)
            if current["id"] in seen:
                continue
            rebuilt.append(current)
            seen.add(current["id"])
        self._items = rebuilt
        return self.read()

    def read(self) -> List[Dict[str, str]]:
        return [item.copy() for item in self._items]

    def has_items(self) -> bool:
        return bool(self._items)

    def has_active_items(self) -> bool:
        return any(item.get("status") in {"pending", "in_progress"} for item in self._items)

    def format_for_injection(self) -> Optional[str]:
        if not self._items:
            return None

        markers = {
            "completed": "[x]",
            "in_progress": "[>]",
            "pending": "[ ]",
            "cancelled": "[~]",
        }
        active_items = [item for item in self._items if item["status"] in {"pending", "in_progress"}]
        if not active_items:
            return None

        lines = [
            "[Your active task list was preserved across context compression]",
            "Use this list sparingly; prefer working directly unless the user explicitly wanted a plan/todo or approved one.",
        ]
        for item in active_items:
            item_id = str(item.get("id") or "").strip() or "?"
            marker = markers.get(str(item.get("status") or "pending"), "[?]")
            lines.append(f"- {marker} {item_id}. {item.get('content', '(no description)')}")
        return "\n".join(lines)

    @staticmethod
    def _validate(item: Dict[str, Any]) -> Dict[str, str]:
        item_id = str(item.get("id", "")).strip() or "?"
        content = str(item.get("content", "")).strip() or "(no description)"
        status = str(item.get("status", "pending") or "pending").strip().lower()
        if status not in VALID_STATUSES:
            status = "pending"
        return {
            "id": item_id,
            "content": content,
            "status": status,
            # Keep `kind` in the serialized shape for compatibility with existing
            # CLI rendering and snapshot messages, but collapse everything to a
            # plain task now that review loops are removed.
            "kind": "task",
        }


def build_todo_snapshot_message(items: List[Dict[str, Any]]) -> str:
    """Serialize a todo snapshot into a user-message marker for persistence."""
    payload = json.dumps({"todos": items}, ensure_ascii=False)
    return f"{TODO_SNAPSHOT_MARKER}\n{payload}"


def todo_tool(
    todos: Optional[List[Dict[str, Any]]] = None,
    merge: bool = False,
    store: Optional[TodoStore] = None,
) -> str:
    """Single entry point for the todo tool."""
    if store is None:
        return tool_error("TodoStore not initialized")

    if todos is not None:
        items = store.write(todos, merge)
    else:
        items = store.read()

    pending = sum(1 for i in items if i["status"] == "pending")
    in_progress = sum(1 for i in items if i["status"] == "in_progress")
    completed = sum(1 for i in items if i["status"] == "completed")
    cancelled = sum(1 for i in items if i["status"] == "cancelled")

    return json.dumps(
        {
            "todos": items,
            "summary": {
                "total": len(items),
                "pending": pending,
                "in_progress": in_progress,
                "completed": completed,
                "cancelled": cancelled,
            },
        },
        ensure_ascii=False,
    )


def check_todo_requirements() -> bool:
    """Todo tool has no external requirements -- always available."""
    return True


TODO_SCHEMA = {
    "name": "todo",
    "description": (
        "Manage your task list for the current session. Use this sparingly because "
        "plans/todos add token overhead. Before creating or replacing a plan/task "
        "list, ask the user whether they want one unless they explicitly requested "
        "it. Prefer working directly when the task is simple or the user declines. "
        "Call with no parameters to read the current list.\n\n"
        "Writing:\n"
        "- Provide 'todos' array to create/update items\n"
        "- merge=false (default): replace the entire list with a fresh plan\n"
        "- merge=true: update existing items by id, add any new ones\n\n"
        "Each item: {id: string, content: string, status: pending|in_progress|completed|cancelled, kind?: task}\n"
        "List order is priority. Only ONE item in_progress at a time.\n"
        "Mark items completed immediately when done. If something fails, cancel it and add a revised item.\n\n"
        "Always returns the full current list."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "Task items to write. Omit to read current list.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Unique item identifier"},
                        "content": {"type": "string", "description": "Task description"},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed", "cancelled"],
                            "description": "Current status",
                        },
                        "kind": {
                            "type": "string",
                            "enum": ["task"],
                            "description": "Optional item kind. Only plain tasks are supported.",
                        },
                    },
                    "required": ["id", "content", "status"],
                },
            },
            "merge": {
                "type": "boolean",
                "description": "true: update existing items by id, add new ones. false (default): replace the entire list.",
                "default": False,
            },
        },
        "required": [],
    },
}


from tools.registry import registry, tool_error

registry.register(
    name="todo",
    toolset="todo",
    schema=TODO_SCHEMA,
    handler=lambda args, **kw: todo_tool(
        todos=args.get("todos"),
        merge=args.get("merge", False),
        store=kw.get("store"),
    ),
    check_fn=check_todo_requirements,
    emoji="📋",
)
