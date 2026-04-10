"""Raw current-session lineage retrieval for long-context continuity.

This tool is deliberately *not* a cross-session summarizer. It reads the active
session's compression lineage and returns raw chronological transcript chunks.

Design goals:
- current-session only (root -> current compression chain)
- recent-first by default
- query-centered contiguous chunk retrieval when a query is provided
- date-range browsing for explicit chronology lookups
- no LLM in the hot path
- no per-message rewriting/truncation; only whole-message chunk selection
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from tools.registry import registry, tool_error, tool_result

DEFAULT_LIMIT = 32
MAX_LIMIT = 120
DEFAULT_MAX_CHARS = 24_000
MAX_MAX_CHARS = 80_000


def _clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _parse_timestamp(raw: Any) -> float | None:
    if raw in (None, ""):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _format_timestamp(ts: Any) -> str | None:
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(ts)).isoformat(timespec="seconds")
    except Exception:
        return str(ts)


def _message_size(msg: dict[str, Any]) -> int:
    content = msg.get("content") or ""
    return len(content) + 64


def _fit_messages_by_chars(
    messages: list[dict[str, Any]],
    *,
    max_chars: int,
    anchor_index: int | None = None,
) -> tuple[list[dict[str, Any]], bool, dict[str, Any] | None]:
    if not messages or max_chars <= 0:
        return messages, False, None

    total = sum(_message_size(msg) for msg in messages)
    if total <= max_chars:
        return messages, False, None

    if anchor_index is None or anchor_index < 0 or anchor_index >= len(messages):
        selected: list[dict[str, Any]] = []
        used = 0
        for msg in reversed(messages):
            size = _message_size(msg)
            if selected and used + size > max_chars:
                break
            if not selected and size > max_chars:
                selected = [msg]
                used = size
                break
            selected.append(msg)
            used += size
        selected.reverse()
        return selected, True, {
            "dropped_from": "start",
            "dropped_messages": max(0, len(messages) - len(selected)),
            "returned_chars": used,
        }

    left = anchor_index
    right = anchor_index
    used = _message_size(messages[anchor_index])
    if used > max_chars:
        return [messages[anchor_index]], True, {
            "dropped_from": "edges",
            "dropped_messages": max(0, len(messages) - 1),
            "returned_chars": used,
            "anchor_only": True,
        }

    while True:
        candidates: list[tuple[str, int, int]] = []
        if left - 1 >= 0:
            candidates.append(("left", left - 1, abs((left - 1) - anchor_index)))
        if right + 1 < len(messages):
            candidates.append(("right", right + 1, abs((right + 1) - anchor_index)))
        if not candidates:
            break
        candidates.sort(key=lambda item: (item[2], 0 if item[0] == "right" else 1))
        expanded = False
        for side, idx, _distance in candidates:
            size = _message_size(messages[idx])
            if used + size > max_chars:
                continue
            used += size
            if side == "left":
                left = idx
            else:
                right = idx
            expanded = True
            break
        if not expanded:
            break

    selected = messages[left : right + 1]
    return selected, True, {
        "dropped_from": "edges",
        "dropped_messages": max(0, len(messages) - len(selected)),
        "returned_chars": used,
        "anchor_message_index": anchor_index,
    }


def _serialize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for msg in messages:
        item = {
            "id": msg.get("id"),
            "session_id": msg.get("session_id"),
            "timestamp": _format_timestamp(msg.get("timestamp")),
            "role": msg.get("role"),
            "content": msg.get("content") or "",
        }
        if msg.get("tool_name"):
            item["tool_name"] = msg.get("tool_name")
        if msg.get("tool_call_id"):
            item["tool_call_id"] = msg.get("tool_call_id")
        serialized.append(item)
    return serialized


def session_timeline(
    query: str | None = None,
    before: Any = None,
    after: Any = None,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
    include_tools: bool = False,
    max_chars: int = DEFAULT_MAX_CHARS,
    db=None,
    current_session_id: str | None = None,
) -> str:
    """Return raw contiguous transcript chunks from the active session lineage."""
    if db is None:
        return tool_error("Session database not available.", success=False)
    if not current_session_id:
        return tool_error("Current session ID not available.", success=False)

    limit = _clamp_int(limit, DEFAULT_LIMIT, 1, MAX_LIMIT)
    offset = max(0, int(offset or 0))
    max_chars = _clamp_int(max_chars, DEFAULT_MAX_CHARS, 256, MAX_MAX_CHARS)

    lineage_ids = db.get_session_lineage_ids(current_session_id)
    if not lineage_ids:
        return tool_error("Could not resolve a lineage for the current session.", success=False)

    query_text = (query or "").strip()
    mode = "recent"
    selection: dict[str, Any] = {
        "recent_first": True,
        "contiguous_chunk": True,
        "lineage_depth": len(lineage_ids),
    }

    if query_text:
        mode = "query"
        matches = db.search_lineage_messages(
            current_session_id,
            query_text,
            include_tools=include_tools,
            limit=8,
            offset=offset,
        )
        if not matches:
            return tool_result(
                success=True,
                mode=mode,
                lineage_session_ids=lineage_ids,
                query=query_text,
                count=0,
                messages=[],
                message="No matching raw transcript chunk found in the current session lineage.",
            )
        anchor = matches[0]
        before_count = limit // 2
        after_count = max(0, limit - before_count - 1)
        raw_messages = db.get_lineage_window_around_message(
            current_session_id,
            message_id=int(anchor["id"]),
            before=before_count,
            after=after_count,
            include_tools=include_tools,
        )
        anchor_index = next((i for i, msg in enumerate(raw_messages) if msg.get("id") == anchor.get("id")), None)
        raw_messages, was_truncated, truncation = _fit_messages_by_chars(
            raw_messages,
            max_chars=max_chars,
            anchor_index=anchor_index,
        )
        selection.update(
            {
                "query": query_text,
                "query_match_offset": offset,
                "anchor_message_id": anchor.get("id"),
                "anchor_timestamp": _format_timestamp(anchor.get("timestamp")),
                "anchor_snippet": anchor.get("snippet"),
                "truncated": was_truncated,
                "truncation": truncation,
            }
        )
    else:
        before_ts = _parse_timestamp(before)
        after_ts = _parse_timestamp(after)
        if before not in (None, "") and before_ts is None:
            return tool_error(f"Could not parse 'before' timestamp: {before!r}", success=False)
        if after not in (None, "") and after_ts is None:
            return tool_error(f"Could not parse 'after' timestamp: {after!r}", success=False)

        if before_ts is not None or after_ts is not None:
            mode = "range"
            raw_messages = db.get_lineage_messages_between(
                current_session_id,
                after_ts=after_ts,
                before_ts=before_ts,
                limit=limit,
                include_tools=include_tools,
            )
            raw_messages, was_truncated, truncation = _fit_messages_by_chars(
                raw_messages,
                max_chars=max_chars,
            )
            selection.update(
                {
                    "before": _format_timestamp(before_ts),
                    "after": _format_timestamp(after_ts),
                    "truncated": was_truncated,
                    "truncation": truncation,
                }
            )
        else:
            raw_messages = db.get_lineage_recent_messages(
                current_session_id,
                limit=limit,
                offset=offset,
                include_tools=include_tools,
            )
            raw_messages, was_truncated, truncation = _fit_messages_by_chars(
                raw_messages,
                max_chars=max_chars,
            )
            selection.update(
                {
                    "message_offset": offset,
                    "truncated": was_truncated,
                    "truncation": truncation,
                }
            )

    serialized = _serialize_messages(raw_messages)
    return tool_result(
        success=True,
        mode=mode,
        lineage_session_ids=lineage_ids,
        count=len(serialized),
        include_tools=bool(include_tools),
        messages=serialized,
        first_timestamp=serialized[0]["timestamp"] if serialized else None,
        last_timestamp=serialized[-1]["timestamp"] if serialized else None,
        selection=selection,
        performance={
            "path": "sqlite-lineage-only",
            "ranking": "recent-first" if mode == "recent" else "fts5-bm25-then-recency",
            "llm_free": True,
            "raw_transcript": True,
        },
    )


def check_session_timeline_requirements() -> bool:
    """The session timeline tool only needs the built-in session store."""
    return True


SESSION_TIMELINE_SCHEMA = {
    "name": "session_timeline",
    "description": (
        "Fast raw transcript lookup for the current session lineage only. "
        "Use this when you need deep current-chat history without lossy summaries. "
        "It returns contiguous chronological chunks from the unedited session lineage, "
        "defaults to a recent-first scan, supports query-centered chunk retrieval via FTS5, "
        "and supports explicit date-range browsing with before/after timestamps."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Optional text query. When provided, the tool finds the best current-session lineage match and returns a surrounding raw chunk.",
            },
            "before": {
                "type": "string",
                "description": "Optional upper timestamp bound (ISO datetime or unix seconds). Use with after to browse by date.",
            },
            "after": {
                "type": "string",
                "description": "Optional lower timestamp bound (ISO datetime or unix seconds). Use with before to browse by date.",
            },
            "limit": {
                "type": "integer",
                "description": "Chunk size in whole messages. Defaults to 32. Larger values scan fatter contiguous portions.",
                "minimum": 1,
                "maximum": MAX_LIMIT,
                "default": DEFAULT_LIMIT,
            },
            "offset": {
                "type": "integer",
                "description": "Skip count. In recent mode it skips newest messages before selecting the chunk. In query mode it skips ranked matches before anchoring the chunk.",
                "minimum": 0,
                "default": 0,
            },
            "include_tools": {
                "type": "boolean",
                "description": "Whether to include tool result messages. Default false for faster, denser chat retrieval.",
                "default": False,
            },
            "max_chars": {
                "type": "integer",
                "description": "Soft output budget in characters. The tool trims by dropping whole edge messages, never by editing individual messages.",
                "minimum": 256,
                "maximum": MAX_MAX_CHARS,
                "default": DEFAULT_MAX_CHARS,
            },
        },
        "additionalProperties": False,
    },
}


registry.register(
    name="session_timeline",
    toolset="session_search",
    schema=SESSION_TIMELINE_SCHEMA,
    handler=lambda args, **kw: session_timeline(
        query=args.get("query"),
        before=args.get("before"),
        after=args.get("after"),
        limit=args.get("limit", DEFAULT_LIMIT),
        offset=args.get("offset", 0),
        include_tools=args.get("include_tools", False),
        max_chars=args.get("max_chars", DEFAULT_MAX_CHARS),
        db=kw.get("db"),
        current_session_id=kw.get("current_session_id"),
    ),
    check_fn=check_session_timeline_requirements,
    emoji="🧭",
    max_result_size_chars=100_000,
)
