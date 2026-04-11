from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from agent.session_events import SessionEvent, build_session_event

logger = logging.getLogger(__name__)


@dataclass
class _RunSubscriber:
    queue: "asyncio.Queue[Optional[dict[str, Any]]]"
    loop: "asyncio.AbstractEventLoop"


@dataclass
class HostedRun:
    run_id: str
    session_id: str
    created_at: float = field(default_factory=time.time)
    status: str = "running"
    events: list[SessionEvent] = field(default_factory=list)
    subscribers: list[_RunSubscriber] = field(default_factory=list)
    agent: Any = None
    agent_ref: Optional[list[Any]] = None
    task: Optional["asyncio.Task[Any]"] = None
    usage: dict[str, int] = field(default_factory=dict)
    final_response: str = ""
    error: Optional[str] = None
    reasoning_parts: list[str] = field(default_factory=list)
    reasoning_completed_text: Optional[str] = None
    finished: bool = False
    cancel_requested: bool = False
    cancel_reason: Optional[str] = None


@dataclass
class HostedSession:
    session_id: str
    created_at: float = field(default_factory=time.time)
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    active_run_id: Optional[str] = None
    closed: bool = False
    attachments: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_active_at: float = field(default_factory=time.time)


class _AgentRef(list):
    def __init__(self, on_assign: Callable[[Any], None]):
        super().__init__([None])
        self._on_assign = on_assign

    def __setitem__(self, index, value):  # type: ignore[override]
        super().__setitem__(index, value)
        if index == 0:
            self._on_assign(value)


class SessionHost:
    """Host many persistent chat sessions behind one runtime owner.

    The host owns session state, run state, and canonical event fanout. The
    caller supplies a single async ``run_callable`` that performs one turn using
    the provided callbacks.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, HostedSession] = {}
        self._runs: dict[str, HostedRun] = {}

    def get_session(self, session_id: str) -> Optional[HostedSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def get_run(self, run_id: str) -> Optional[HostedRun]:
        with self._lock:
            return self._runs.get(run_id)

    @staticmethod
    def _clean_session_metadata(metadata: Optional[dict[str, Any]]) -> dict[str, Any]:
        clean: dict[str, Any] = {}
        for key, value in dict(metadata or {}).items():
            if value in (None, ""):
                continue
            clean[str(key)] = value
        return clean

    @staticmethod
    def _message_preview(content: Any) -> str:
        if isinstance(content, str):
            return " ".join(content.strip().split())
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    text = str(part.get("text") or "").strip()
                    if text:
                        text_parts.append(text)
            return " ".join(" ".join(text_parts).split())
        return " ".join(str(content or "").strip().split())

    def _session_summary_locked(self, session: HostedSession) -> dict[str, Any]:
        active_run = self._runs.get(session.active_run_id) if session.active_run_id else None
        run_is_live = bool(active_run is not None and not active_run.finished)
        preview = ""
        for message in reversed(session.conversation_history):
            if not isinstance(message, dict):
                continue
            preview = self._message_preview(message.get("content"))
            if preview:
                break
        title = str(session.metadata.get("title") or session.metadata.get("label") or "").strip() or None
        status = "running" if run_is_live else ("attached" if session.attachments else "detached")
        return {
            "id": session.session_id,
            "title": title,
            "preview": preview,
            "source": "live",
            "last_active": session.last_active_at,
            "status": status,
            "active_run_id": session.active_run_id if run_is_live else None,
            "attached_clients": len(session.attachments),
            "model": session.metadata.get("model"),
            "provider": session.metadata.get("provider"),
        }

    def attach_session(
        self,
        session_id: str,
        *,
        client_id: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        now = time.time()
        clean_meta = self._clean_session_metadata(metadata)
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = HostedSession(session_id=session_id)
                self._sessions[session_id] = session
            session.closed = False
            session.metadata.update(clean_meta)
            existing = dict(session.attachments.get(client_id, {}))
            session.attachments[client_id] = {
                "client_id": client_id,
                "attached_at": existing.get("attached_at", now),
                "last_seen_at": now,
                **clean_meta,
            }
            session.last_active_at = now
            return self._session_summary_locked(session)

    def detach_session(self, session_id: str, *, client_id: str) -> bool:
        now = time.time()
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            session.attachments.pop(client_id, None)
            session.last_active_at = now
        return True

    def list_sessions(self, *, live_only: bool = False, limit: Optional[int] = None) -> list[dict[str, Any]]:
        with self._lock:
            rows: list[dict[str, Any]] = []
            for session in self._sessions.values():
                if session.closed:
                    continue
                summary = self._session_summary_locked(session)
                rows.append(summary)

        status_rank = {"running": 0, "attached": 1, "detached": 2}
        rows.sort(key=lambda row: (status_rank.get(str(row.get("status") or "detached"), 99), -(float(row.get("last_active") or 0.0))))
        if limit is not None:
            try:
                limit = max(int(limit), 0)
            except (TypeError, ValueError):
                limit = 0
            rows = rows[:limit]
        return rows

    def active_run_count(self) -> int:
        with self._lock:
            return sum(1 for run in self._runs.values() if not run.finished)

    def purge_finished_runs(self, *, older_than_seconds: float) -> int:
        cutoff = time.time() - max(float(older_than_seconds), 0.0)
        removed = []
        with self._lock:
            for run_id, run in list(self._runs.items()):
                if run.finished and run.created_at <= cutoff:
                    removed.append(run_id)
                    self._runs.pop(run_id, None)
        return len(removed)

    def close_session(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            session.closed = True
            session.attachments.clear()
            session.last_active_at = time.time()
            active_run_id = session.active_run_id
        if active_run_id:
            self.cancel_run(active_run_id, reason="Session closed")
        return True

    def cancel_run(self, run_id: str, reason: str = "Cancelled") -> bool:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return False
            run.cancel_requested = True
            run.cancel_reason = reason
            agent = run.agent
            if agent is None and run.agent_ref:
                try:
                    agent = run.agent_ref[0]
                except Exception:
                    agent = None
        if agent is not None and hasattr(agent, "interrupt"):
            try:
                agent.interrupt(reason)
            except Exception:
                logger.debug("Failed to interrupt run %s", run_id, exc_info=True)
        return True

    def subscribe_run(self, run_id: str, *, loop: "asyncio.AbstractEventLoop") -> "asyncio.Queue[Optional[dict[str, Any]]]":
        queue: "asyncio.Queue[Optional[dict[str, Any]]]" = asyncio.Queue()
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError(run_id)
            subscriber = _RunSubscriber(queue=queue, loop=loop)
            run.subscribers.append(subscriber)
            backlog = [event.to_dict() for event in run.events]
            finished = run.finished
        for item in backlog:
            queue.put_nowait(item)
        if finished:
            queue.put_nowait(None)
        return queue

    async def start_run(
        self,
        *,
        session_id: str,
        user_message: str,
        run_callable: Callable[..., Awaitable[tuple[dict[str, Any], dict[str, int], Any]]],
        conversation_history: Optional[list[dict[str, Any]]] = None,
        instructions: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> str:
        created_session = False
        session_meta = dict(metadata or {})
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = HostedSession(session_id=session_id)
                self._sessions[session_id] = session
                created_session = True
            if session_meta:
                session.metadata.update(self._clean_session_metadata(session_meta))
            if session.active_run_id:
                active = self._runs.get(session.active_run_id)
                if active is not None and not active.finished:
                    raise RuntimeError(f"Session already has an active run: {session.active_run_id}")
            if conversation_history is not None:
                session.conversation_history = [msg for msg in conversation_history if isinstance(msg, dict)]
            session.closed = False
            session.last_active_at = time.time()
            resolved_run_id = run_id or f"run_{uuid.uuid4().hex}"
            run = HostedRun(run_id=resolved_run_id, session_id=session_id)
            self._runs[resolved_run_id] = run
            session.active_run_id = resolved_run_id

        if created_session:
            self._emit(
                resolved_run_id,
                build_session_event(
                    "session.created",
                    session_id=session_id,
                    run_id=resolved_run_id,
                    payload=session_meta,
                ),
            )

        self._emit(
            resolved_run_id,
            build_session_event(
                "run.started",
                session_id=session_id,
                run_id=resolved_run_id,
                payload={"user_message": user_message, **session_meta},
            ),
        )

        def _on_agent_assigned(agent: Any) -> None:
            cancel_reason = None
            with self._lock:
                current_run = self._runs.get(resolved_run_id)
                if current_run is None:
                    return
                current_run.agent = agent
                cancel_reason = current_run.cancel_reason if current_run.cancel_requested else None
            if cancel_reason and agent is not None and hasattr(agent, "interrupt"):
                try:
                    agent.interrupt(cancel_reason)
                except Exception:
                    logger.debug("Failed to interrupt just-assigned agent for run %s", resolved_run_id, exc_info=True)

        agent_ref: list[Any] = _AgentRef(_on_agent_assigned)
        with self._lock:
            self._runs[resolved_run_id].agent_ref = agent_ref

        async def _runner() -> None:
            reasoning_seen = False

            def _append_reasoning(text: str) -> None:
                nonlocal reasoning_seen
                clean = str(text or "")
                if not clean:
                    return
                reasoning_seen = True
                with self._lock:
                    current_run = self._runs.get(resolved_run_id)
                    if current_run is not None:
                        current_run.reasoning_parts.append(clean)
                self._emit(
                    resolved_run_id,
                    build_session_event(
                        "reasoning.delta",
                        session_id=session_id,
                        run_id=resolved_run_id,
                        text=clean,
                    ),
                )

            def _on_tool_progress(event_type: str, name: str = None, preview: str = None, args: Any = None, **kwargs) -> None:
                if event_type == "tool.started":
                    self._emit(
                        resolved_run_id,
                        build_session_event(
                            "tool.started",
                            session_id=session_id,
                            run_id=resolved_run_id,
                            tool=name,
                            preview=preview,
                            args=args if isinstance(args, dict) else {},
                        ),
                    )
                    return
                if event_type == "tool.completed":
                    self._emit(
                        resolved_run_id,
                        build_session_event(
                            "tool.completed",
                            session_id=session_id,
                            run_id=resolved_run_id,
                            tool=name,
                            duration=round(float(kwargs.get("duration", 0) or 0), 3),
                            error=bool(kwargs.get("is_error", False)),
                        ),
                    )
                    return
                if event_type in {"reasoning.available", "_thinking"}:
                    text = preview or name or ""
                    _append_reasoning(text)
                    return
                if event_type == "subagent_progress":
                    self._emit(
                        resolved_run_id,
                        build_session_event(
                            "subagent.progress",
                            session_id=session_id,
                            run_id=resolved_run_id,
                            tool=name,
                            text=preview or "",
                        ),
                    )

            def _on_message_delta(delta: Optional[str]) -> None:
                if delta is None:
                    return
                text = str(delta or "")
                if not text:
                    return
                self._emit(
                    resolved_run_id,
                    build_session_event(
                        "message.delta",
                        session_id=session_id,
                        run_id=resolved_run_id,
                        delta=text,
                    ),
                )

            def _on_reasoning_delta(text: str) -> None:
                _append_reasoning(text)

            def _on_tool_generating(tool_name: str) -> None:
                if not tool_name:
                    return
                self._emit(
                    resolved_run_id,
                    build_session_event(
                        "tool.generating",
                        session_id=session_id,
                        run_id=resolved_run_id,
                        tool=tool_name,
                    ),
                )

            try:
                result, usage, agent = await run_callable(
                    session_id=session_id,
                    user_message=user_message,
                    conversation_history=conversation_history,
                    instructions=instructions,
                    stream_delta_callback=_on_message_delta,
                    reasoning_callback=_on_reasoning_delta,
                    tool_progress_callback=_on_tool_progress,
                    tool_gen_callback=_on_tool_generating,
                    agent_ref=agent_ref,
                )
                with self._lock:
                    current_run = self._runs.get(resolved_run_id)
                    if current_run is not None:
                        current_run.agent = agent_ref[0] or agent
                        current_run.usage = dict(usage or {})

                result = result or {}
                if not isinstance(result, dict):
                    result = {"final_response": str(result)}
                messages = result.get("messages") if isinstance(result.get("messages"), list) else None
                if messages is not None:
                    with self._lock:
                        session = self._sessions.get(session_id)
                        if session is not None:
                            session.conversation_history = [msg for msg in messages if isinstance(msg, dict)]

                final_reasoning = str(result.get("last_reasoning") or "").strip()
                if final_reasoning:
                    with self._lock:
                        current_run = self._runs.get(resolved_run_id)
                        if current_run is not None:
                            current_run.reasoning_completed_text = final_reasoning
                    self._emit(
                        resolved_run_id,
                        build_session_event(
                            "reasoning.completed",
                            session_id=session_id,
                            run_id=resolved_run_id,
                            text=final_reasoning,
                        ),
                    )
                elif reasoning_seen:
                    with self._lock:
                        current_run = self._runs.get(resolved_run_id)
                        reasoning_text = "".join(current_run.reasoning_parts) if current_run else ""
                    if reasoning_text:
                        self._emit(
                            resolved_run_id,
                            build_session_event(
                                "reasoning.completed",
                                session_id=session_id,
                                run_id=resolved_run_id,
                                text=reasoning_text,
                            ),
                        )

                final_response = str(result.get("final_response") or "")
                if final_response:
                    self._emit(
                        resolved_run_id,
                        build_session_event(
                            "message.completed",
                            session_id=session_id,
                            run_id=resolved_run_id,
                            content=final_response,
                        ),
                    )

                if result.get("interrupted"):
                    error_text = str(result.get("interrupt_message") or result.get("error") or "Interrupted")
                    with self._lock:
                        current_run = self._runs.get(resolved_run_id)
                        if current_run is not None:
                            current_run.status = "cancelled"
                            current_run.error = error_text
                    self._emit(
                        resolved_run_id,
                        build_session_event(
                            "run.cancelled",
                            session_id=session_id,
                            run_id=resolved_run_id,
                            error=error_text,
                        ),
                    )
                elif result.get("failed"):
                    error_text = str(result.get("error") or "Hosted run failed")
                    with self._lock:
                        current_run = self._runs.get(resolved_run_id)
                        if current_run is not None:
                            current_run.status = "failed"
                            current_run.error = error_text
                    self._emit(
                        resolved_run_id,
                        build_session_event(
                            "run.failed",
                            session_id=session_id,
                            run_id=resolved_run_id,
                            error=error_text,
                        ),
                    )
                else:
                    with self._lock:
                        current_run = self._runs.get(resolved_run_id)
                        if current_run is not None:
                            current_run.status = "completed"
                            current_run.final_response = final_response
                    self._emit(
                        resolved_run_id,
                        build_session_event(
                            "run.completed",
                            session_id=session_id,
                            run_id=resolved_run_id,
                            output=final_response,
                            usage=dict(usage or {}),
                        ),
                    )
            except Exception as exc:
                logger.exception("Hosted run %s failed", resolved_run_id)
                error_text = str(exc)
                with self._lock:
                    current_run = self._runs.get(resolved_run_id)
                    if current_run is not None:
                        current_run.status = "failed"
                        current_run.error = error_text
                self._emit(
                    resolved_run_id,
                    build_session_event(
                        "run.failed",
                        session_id=session_id,
                        run_id=resolved_run_id,
                        error=error_text,
                    ),
                )
            finally:
                self._finish_run(resolved_run_id)

        task = asyncio.create_task(_runner())
        with self._lock:
            run = self._runs[resolved_run_id]
            run.task = task
        return resolved_run_id

    def _emit(self, run_id: str, event: SessionEvent) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            run.events.append(event)
            session = self._sessions.get(run.session_id)
            if session is not None:
                session.last_active_at = float(event.timestamp)
            subscribers = list(run.subscribers)
        event_payload = event.to_dict()
        for subscriber in subscribers:
            try:
                subscriber.loop.call_soon_threadsafe(subscriber.queue.put_nowait, event_payload)
            except Exception:
                logger.debug("Failed to fan out session event %s", event.event, exc_info=True)

    def _finish_run(self, run_id: str) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            run.finished = True
            subscribers = list(run.subscribers)
            session = self._sessions.get(run.session_id)
            if session is not None and session.active_run_id == run_id:
                session.active_run_id = None
                session.last_active_at = time.time()
        for subscriber in subscribers:
            try:
                subscriber.loop.call_soon_threadsafe(subscriber.queue.put_nowait, None)
            except Exception:
                logger.debug("Failed to finish run subscriber for %s", run_id, exc_info=True)
