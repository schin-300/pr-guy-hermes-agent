from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

import requests

from gateway.config import Platform, load_gateway_config
from gateway.platforms.api_server import DEFAULT_HOST, DEFAULT_PORT
from run_agent import message_content_to_text

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class HostedSessionClientError(RuntimeError):
    """Raised when the local hosted-session gateway bridge is unavailable or fails."""


@dataclass(frozen=True)
class HostedSessionEndpoint:
    base_url: str
    api_key: Optional[str] = None


def resolve_hosted_session_endpoint() -> HostedSessionEndpoint:
    config = load_gateway_config()
    platform_cfg = config.platforms.get(Platform.API_SERVER)
    extra = dict(getattr(platform_cfg, "extra", {}) or {}) if platform_cfg else {}

    host = str(os.getenv("API_SERVER_HOST") or extra.get("host") or DEFAULT_HOST)
    port = int(os.getenv("API_SERVER_PORT") or extra.get("port") or DEFAULT_PORT)
    api_key = os.getenv("API_SERVER_KEY") or extra.get("key") or None
    return HostedSessionEndpoint(base_url=f"http://{host}:{port}", api_key=api_key)


def check_hosted_session_endpoint(endpoint: HostedSessionEndpoint, timeout: float = 1.5) -> bool:
    try:
        response = requests.get(f"{endpoint.base_url}/health", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return False
    return payload.get("status") == "ok"


def _launch_gateway_background() -> None:
    env = os.environ.copy()
    env.setdefault("API_SERVER_ENABLED", "true")
    cmd = [
        sys.executable,
        "-m",
        "hermes_cli.main",
        "gateway",
        "run",
        "--replace",
        "--quiet",
    ]
    subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


def ensure_hosted_session_bridge(timeout: float = 15.0, autostart: bool = True) -> HostedSessionEndpoint:
    endpoint = resolve_hosted_session_endpoint()
    if check_hosted_session_endpoint(endpoint):
        return endpoint
    if not autostart:
        raise HostedSessionClientError("Hosted session bridge is not running")

    _launch_gateway_background()

    deadline = time.time() + max(timeout, 1.0)
    while time.time() < deadline:
        if check_hosted_session_endpoint(endpoint):
            return endpoint
        time.sleep(0.25)

    raise HostedSessionClientError(
        f"Hosted session bridge did not become ready at {endpoint.base_url} within {timeout:.1f}s"
    )


class HostedSessionAgentProxy:
    """AIAgent-compatible client that runs turns on the hosted gateway runtime."""

    def __init__(
        self,
        *,
        endpoint: HostedSessionEndpoint,
        session_id: str,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_mode: Optional[str] = None,
        enabled_toolsets: Optional[list[str]] = None,
        service_tier: Optional[str] = None,
        context_length_override: Optional[int] = None,
        compression_threshold: float = 0.5,
        verbose_logging: bool = False,
        quiet_mode: bool = True,
        ephemeral_system_prompt: Optional[str] = None,
        tool_progress_callback: Optional[Callable[..., Any]] = None,
        reasoning_callback: Optional[Callable[[str], Any]] = None,
        tool_gen_callback: Optional[Callable[[str], Any]] = None,
        http_session: Optional[requests.Session] = None,
    ):
        self.endpoint = endpoint
        self.session_id = session_id
        self.session_start = None
        self.model = model
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self.api_mode = api_mode or "chat_completions"
        self.enabled_toolsets = list(enabled_toolsets or [])
        self.service_tier = service_tier
        self.context_length_override = context_length_override
        self.verbose_logging = verbose_logging
        self.quiet_mode = quiet_mode
        self.ephemeral_system_prompt = ephemeral_system_prompt
        self.tool_progress_callback = tool_progress_callback
        self.reasoning_callback = reasoning_callback
        self.tool_gen_callback = tool_gen_callback
        self.http_session = http_session or requests.Session()

        self.context_compressor = SimpleNamespace(
            context_length=context_length_override,
            threshold_percent=compression_threshold,
            on_session_reset=lambda: None,
        )
        self.gateway_hosted_session = True
        self.compression_enabled = False
        self._checkpoint_mgr = SimpleNamespace(enabled=False)
        self._active_children: list[Any] = []
        self._interrupt_requested = False
        self._interrupt_message: Optional[str] = None
        self._active_run_id: Optional[str] = None
        self._active_events_response: Any = None
        self._active_lock = threading.Lock()
        self._last_flushed_db_idx = 0
        self.tools: list[Any] = []
        self.valid_tool_names: set[str] = set()
        self.client_id = f"cli_{uuid.uuid4().hex}"
        self._attached = False

        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_total_tokens = 0
        self.session_api_calls = 0
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = "unknown"
        self.session_cost_source = "none"

        self.attach_session()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.endpoint.api_key:
            headers["Authorization"] = f"Bearer {self.endpoint.api_key}"
        return headers

    def _normalized_history(self, conversation_history: Optional[list[dict[str, Any]]]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for message in conversation_history or []:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "")
            if role not in {"user", "assistant"}:
                continue
            normalized.append({
                "role": role,
                "content": message_content_to_text(message.get("content", "")),
            })
        return normalized

    def _runtime_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "api_mode": self.api_mode,
            "toolsets": list(self.enabled_toolsets or []),
            "service_tier": self.service_tier,
            "context_length_override": self.context_length_override,
        }
        return {k: v for k, v in payload.items() if v not in (None, "", [])}

    def _session_registration_payload(self) -> dict[str, Any]:
        payload = {
            "client_id": self.client_id,
            "client_kind": "cli",
            "model": self.model,
            "provider": self.provider,
        }
        return {k: v for k, v in payload.items() if v not in (None, "", [])}

    def attach_session(self) -> None:
        response = self.http_session.post(
            f"{self.endpoint.base_url}/v1/sessions/{self.session_id}/attach",
            json=self._session_registration_payload(),
            headers=self._headers(),
            timeout=10,
        )
        response.raise_for_status()
        self._attached = True

    def detach_session(self) -> None:
        if not self._attached:
            return
        response = self.http_session.post(
            f"{self.endpoint.base_url}/v1/sessions/{self.session_id}/detach",
            json={"client_id": self.client_id},
            headers=self._headers(),
            timeout=10,
        )
        response.raise_for_status()
        self._attached = False

    def list_live_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        response = self.http_session.get(
            f"{self.endpoint.base_url}/v1/sessions/live",
            headers=self._headers(),
            timeout=10,
            params={"limit": limit},
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("sessions") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            return []
        return [dict(row) for row in rows if isinstance(row, dict) and row.get("id")]

    def switch_session(self, new_session_id: str) -> None:
        target_id = str(new_session_id or "").strip()
        if not target_id or target_id == self.session_id:
            return
        if self._attached:
            self.detach_session()
        self.session_id = target_id
        self.attach_session()

    def _start_run(self, *, user_message: str, conversation_history: list[dict[str, str]]) -> tuple[str, str]:
        payload: dict[str, Any] = {
            "input": user_message,
            "session_id": self.session_id,
            "conversation_history": conversation_history,
        }
        if self.ephemeral_system_prompt:
            payload["instructions"] = self.ephemeral_system_prompt
        payload.update(self._runtime_payload())

        response = self.http_session.post(
            f"{self.endpoint.base_url}/v1/runs",
            json=payload,
            headers=self._headers(),
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        run_id = data.get("run_id")
        session_id = data.get("session_id") or self.session_id
        if not run_id:
            raise HostedSessionClientError("Hosted bridge did not return a run_id")
        self.session_id = str(session_id)
        return str(run_id), str(session_id)

    def _emit_tool_progress(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.tool_progress_callback is None:
            return
        try:
            self.tool_progress_callback(
                event_type,
                payload.get("tool"),
                payload.get("preview") or payload.get("text"),
                payload.get("args"),
                duration=payload.get("duration"),
                is_error=payload.get("error", False),
            )
        except Exception:
            logger.debug("Hosted tool progress callback failed", exc_info=True)

    @staticmethod
    def _normalize_event_payload(event: dict[str, Any]) -> dict[str, Any]:
        """Accept both canonical nested payload events and older flattened events."""
        nested = event.get("payload")
        payload = dict(nested) if isinstance(nested, dict) else {}
        for key in ("delta", "content", "text", "tool", "preview", "args", "duration", "error", "output", "usage"):
            if key not in payload and key in event:
                payload[key] = event.get(key)
        return payload

    def run_conversation(
        self,
        *,
        user_message: str,
        conversation_history: Optional[list[dict[str, Any]]] = None,
        stream_callback: Optional[Callable[[str], Any]] = None,
        task_id: Optional[str] = None,
        persist_user_message: Any = None,
        **_: Any,
    ) -> dict[str, Any]:
        del task_id
        if not self._attached:
            self.attach_session()
        normalized_history = self._normalized_history(conversation_history)
        visible_user_message = message_content_to_text(
            persist_user_message if persist_user_message is not None else user_message
        )
        wire_user_message = message_content_to_text(user_message)
        final_reasoning = ""
        streamed_chunks: list[str] = []
        response_previewed = False
        final_response = ""
        failed = False
        interrupted = False
        error_message: Optional[str] = None
        usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        run_id, _ = self._start_run(user_message=wire_user_message, conversation_history=normalized_history)
        events_response = self.http_session.get(
            f"{self.endpoint.base_url}/v1/runs/{run_id}/events",
            headers=self._headers(),
            stream=True,
            timeout=300,
        )
        events_response.raise_for_status()

        with self._active_lock:
            self._active_run_id = run_id
            self._active_events_response = events_response
            self._interrupt_requested = False
            self._interrupt_message = None

        try:
            for raw_line in events_response.iter_lines(decode_unicode=True):
                if raw_line is None:
                    continue
                line = raw_line.strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                payload_text = line.split(":", 1)[1].strip()
                if not payload_text:
                    continue
                event = json.loads(payload_text)
                event_type = str(event.get("event") or "")
                payload = self._normalize_event_payload(event)

                if event_type == "message.delta":
                    delta = str(payload.get("delta") or "")
                    if delta:
                        streamed_chunks.append(delta)
                        if stream_callback is not None:
                            stream_callback(delta)
                            response_previewed = True
                elif event_type == "message.completed":
                    final_response = str(payload.get("content") or final_response or "")
                elif event_type in {"reasoning.delta", "reasoning.completed", "reasoning.available"}:
                    reasoning_text = str(payload.get("text") or "")
                    if reasoning_text:
                        final_reasoning = reasoning_text
                        if self.reasoning_callback is not None:
                            self.reasoning_callback(reasoning_text)
                elif event_type == "tool.generating":
                    tool_name = str(payload.get("tool") or "")
                    if tool_name and self.tool_gen_callback is not None:
                        self.tool_gen_callback(tool_name)
                elif event_type in {"tool.started", "tool.completed", "subagent.progress"}:
                    self._emit_tool_progress(event_type, payload)
                elif event_type == "run.completed":
                    final_response = str(payload.get("output") or final_response or "")
                    raw_usage = payload.get("usage") or {}
                    usage = {
                        "input_tokens": int(raw_usage.get("input_tokens") or 0),
                        "output_tokens": int(raw_usage.get("output_tokens") or 0),
                        "total_tokens": int(raw_usage.get("total_tokens") or 0),
                    }
                elif event_type == "run.failed":
                    failed = True
                    error_message = str(payload.get("error") or "Hosted run failed")
                elif event_type == "run.cancelled":
                    interrupted = True
                    if self._interrupt_message is not None:
                        error_message = self._interrupt_message
                    else:
                        error_message = str(payload.get("error") or "Interrupted")

            if not final_response:
                final_response = "".join(streamed_chunks)
            if self._interrupt_requested:
                interrupted = True
                error_message = self._interrupt_message or error_message or "Interrupted"

        except Exception as exc:
            if self._interrupt_requested:
                interrupted = True
                error_message = self._interrupt_message or "Interrupted"
            else:
                logger.debug("Hosted event stream failed for %s: %s", run_id, exc, exc_info=True)
                raise HostedSessionClientError(str(exc)) from exc
        finally:
            try:
                events_response.close()
            except Exception:
                pass
            with self._active_lock:
                self._active_run_id = None
                self._active_events_response = None

        self.session_input_tokens += usage["input_tokens"]
        self.session_output_tokens += usage["output_tokens"]
        self.session_prompt_tokens += usage["input_tokens"]
        self.session_completion_tokens += usage["output_tokens"]
        self.session_total_tokens += usage["total_tokens"]
        self.session_api_calls += 1

        messages = list(normalized_history)
        messages.append({"role": "user", "content": visible_user_message})
        if final_response:
            messages.append({"role": "assistant", "content": final_response})

        result = {
            "final_response": final_response,
            "messages": messages,
            "api_calls": 1,
            "completed": not failed and not interrupted,
            "failed": failed,
            "interrupted": interrupted,
            "response_previewed": response_previewed,
        }
        if final_reasoning:
            result["last_reasoning"] = final_reasoning
        if error_message:
            result["error"] = error_message
        if interrupted and self._interrupt_message is not None:
            result["interrupt_message"] = self._interrupt_message
        return result

    def interrupt(self, message: Optional[str] = None) -> None:
        with self._active_lock:
            self._interrupt_requested = True
            self._interrupt_message = message
            run_id = self._active_run_id
            response = self._active_events_response

        if run_id:
            try:
                cancel_response = self.http_session.post(
                    f"{self.endpoint.base_url}/v1/runs/{run_id}/cancel",
                    headers=self._headers(),
                    timeout=10,
                )
                cancel_response.raise_for_status()
            except Exception:
                logger.debug("Hosted run cancel failed for %s", run_id, exc_info=True)
        if response is not None:
            try:
                response.close()
            except Exception:
                pass

    def close_session(self) -> None:
        with self._active_lock:
            session_id = self.session_id
        self.interrupt("Session closed by user")
        response = self.http_session.post(
            f"{self.endpoint.base_url}/v1/sessions/{session_id}/close",
            headers=self._headers(),
            timeout=10,
        )
        response.raise_for_status()
        self._attached = False

    def reset_session_state(self) -> None:
        self.session_total_tokens = 0
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_api_calls = 0
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = "unknown"
        self.session_cost_source = "none"
        if hasattr(self, "context_compressor") and self.context_compressor and hasattr(self.context_compressor, "on_session_reset"):
            self.context_compressor.on_session_reset()

    def set_context_length_override(self, context_length: Optional[int]) -> int:
        self.context_length_override = int(context_length) if context_length else None
        self.context_compressor.context_length = self.context_length_override
        return self.context_length_override or 0

    def _invalidate_system_prompt(self) -> None:
        return None

    def flush_memories(self, messages: Optional[list[dict[str, Any]]] = None, min_turns: Optional[int] = None) -> None:
        del messages, min_turns
        return None

    def switch_model(
        self,
        *,
        new_model: Optional[str] = None,
        new_provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_mode: Optional[str] = None,
    ) -> None:
        if new_model:
            self.model = new_model
        if new_provider:
            self.provider = new_provider
        if api_key:
            self.api_key = api_key
        if base_url:
            self.base_url = base_url
        if api_mode:
            self.api_mode = api_mode

    def set_context_profile(self, *, context_length: Optional[int] = None, compression_threshold: Optional[float] = None) -> Optional[int]:
        if context_length is not None:
            self.context_length_override = context_length
            self.context_compressor.context_length = context_length
        if compression_threshold is not None:
            self.context_compressor.threshold_percent = compression_threshold
        return self.context_length_override
