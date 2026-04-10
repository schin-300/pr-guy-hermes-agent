from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

import requests

from gateway.config import Platform, load_gateway_config
from gateway.platforms.api_server import DEFAULT_HOST, DEFAULT_PORT
from hermes_cli.config import get_hermes_home

logger = logging.getLogger(__name__)


def message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    parts.append(str(item["text"]))
                elif item.get("content"):
                    parts.append(str(item["content"]))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return "" if content is None else str(content)


class GatewaySessionClientError(RuntimeError):
    """Raised when the local gateway session bridge is unavailable or fails."""


@dataclass(frozen=True)
class GatewaySessionEndpoint:
    base_url: str
    api_key: Optional[str] = None


def _normalize_hermes_home(hermes_home: str | Path | None) -> Optional[Path]:
    if hermes_home is None or str(hermes_home).strip() == "":
        return None
    return Path(hermes_home).expanduser().resolve()


def _load_profile_api_server_settings(hermes_home: Path) -> tuple[str, int, Optional[str]]:
    host = DEFAULT_HOST
    port = DEFAULT_PORT
    api_key: Optional[str] = None

    gateway_json = hermes_home / "gateway.json"
    if gateway_json.exists():
        try:
            data = json.loads(gateway_json.read_text(encoding="utf-8")) or {}
            platform_cfg = dict((data.get("platforms") or {}).get(Platform.API_SERVER.value, {}) or {})
            extra = dict(platform_cfg.get("extra") or {})
            host = str(extra.get("host") or host)
            port = int(extra.get("port") or port)
            api_key = extra.get("key") or api_key
        except Exception:
            logger.debug("Failed to read %s", gateway_json, exc_info=True)

    config_yaml = hermes_home / "config.yaml"
    if config_yaml.exists():
        try:
            import yaml

            yaml_cfg = yaml.safe_load(config_yaml.read_text(encoding="utf-8")) or {}
            platform_cfg = dict((yaml_cfg.get("platforms") or {}).get(Platform.API_SERVER.value, {}) or {})
            extra = dict(platform_cfg.get("extra") or {})
            host = str(extra.get("host") or host)
            port = int(extra.get("port") or port)
            api_key = extra.get("key") or api_key
        except Exception:
            logger.debug("Failed to read %s", config_yaml, exc_info=True)

    return host, port, api_key


def resolve_gateway_session_endpoint(hermes_home: str | Path | None = None) -> GatewaySessionEndpoint:
    """Resolve the loopback API-server endpoint used for gateway-backed CLI sessions."""
    target_home = _normalize_hermes_home(hermes_home)
    current_home = get_hermes_home().resolve()
    if target_home is None or target_home == current_home:
        config = load_gateway_config()
        platform_cfg = config.platforms.get(Platform.API_SERVER)
        extra = dict(getattr(platform_cfg, "extra", {}) or {})
        host = str(os.getenv("API_SERVER_HOST") or extra.get("host") or DEFAULT_HOST)
        port = int(os.getenv("API_SERVER_PORT") or extra.get("port") or DEFAULT_PORT)
        api_key = os.getenv("API_SERVER_KEY") or extra.get("key") or None
        return GatewaySessionEndpoint(base_url=f"http://{host}:{port}", api_key=api_key)

    host, port, api_key = _load_profile_api_server_settings(target_home)
    return GatewaySessionEndpoint(base_url=f"http://{host}:{port}", api_key=api_key)



def check_gateway_session_endpoint(endpoint: GatewaySessionEndpoint, timeout: float = 1.5) -> bool:
    """Return True when the local gateway API server responds to /health."""
    try:
        response = requests.get(f"{endpoint.base_url}/health", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return False
    return payload.get("status") == "ok"


def ensure_gateway_session_bridge(
    timeout: float = 15.0,
    autostart: bool = True,
    hermes_home: str | Path | None = None,
) -> GatewaySessionEndpoint:
    """Ensure the chosen profile's gateway-backed session bridge is running and reachable."""
    target_home = _normalize_hermes_home(hermes_home)
    endpoint = resolve_gateway_session_endpoint(target_home)
    if check_gateway_session_endpoint(endpoint):
        return endpoint
    if not autostart:
        raise GatewaySessionClientError("Gateway session bridge is not running")

    from hermes_cli.gateway import launch_gateway_background_for_home

    if not launch_gateway_background_for_home(target_home or get_hermes_home()):
        raise GatewaySessionClientError("Failed to launch the Hermes gateway in the background")

    deadline = time.time() + max(timeout, 1.0)
    while time.time() < deadline:
        if check_gateway_session_endpoint(endpoint):
            return endpoint
        time.sleep(0.25)

    raise GatewaySessionClientError(
        f"Gateway session bridge did not become ready at {endpoint.base_url} within {timeout:.1f}s"
    )


class GatewaySessionAgentProxy:
    """Small AIAgent-compatible proxy that routes CLI turns through the gateway API server."""

    def __init__(
        self,
        *,
        endpoint: GatewaySessionEndpoint,
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
        clarify_callback: Optional[Callable[[str, Optional[list[str]]], Any]] = None,
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
        self.clarify_callback = clarify_callback
        self.http_session = http_session or requests.Session()

        self.context_compressor = SimpleNamespace(
            context_length=context_length_override,
            threshold_percent=compression_threshold,
        )
        self.gateway_hosted_session = True
        self.compression_enabled = False
        self._checkpoint_mgr = SimpleNamespace(enabled=False)
        self._active_children: list[Any] = []
        self._interrupt_requested = False
        self._detach_requested = False
        self._interrupt_message: Optional[str] = None
        self._active_run_id: Optional[str] = None
        self._active_events_response: Any = None
        self._active_lock = threading.Lock()
        self._last_flushed_db_idx = 0
        self.tools: list[Any] = []
        self.valid_tool_names: set[str] = set()

        self.session_cache_write_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_api_calls = 0

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

    def _start_run(self, *, user_message: str, conversation_history: list[dict[str, str]]) -> str:
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
        if not run_id:
            raise GatewaySessionClientError("Gateway did not return a run_id")
        return str(run_id)

    def _submit_clarify_response(self, run_id: str, response_text: str) -> None:
        response = self.http_session.post(
            f"{self.endpoint.base_url}/v1/runs/{run_id}/clarify",
            json={"response": response_text},
            headers=self._headers(),
            timeout=30,
        )
        response.raise_for_status()

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
        normalized_history = self._normalized_history(conversation_history)
        visible_user_message = message_content_to_text(
            persist_user_message if persist_user_message is not None else user_message
        )
        wire_user_message = message_content_to_text(user_message)
        reasoning_text = ""
        streamed_chunks: list[str] = []
        final_response = ""
        failed = False
        interrupted = False
        detached = False
        error_message: Optional[str] = None
        usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        run_id = self._start_run(user_message=wire_user_message, conversation_history=normalized_history)
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
            self._detach_requested = False
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

                if event_type == "message.delta":
                    delta = str(event.get("delta") or "")
                    if delta:
                        streamed_chunks.append(delta)
                        if stream_callback is not None:
                            stream_callback(delta)
                elif event_type == "reasoning.available":
                    reasoning_text = str(event.get("text") or "")
                    if reasoning_text and self.reasoning_callback is not None:
                        self.reasoning_callback(reasoning_text)
                elif event_type in {"tool.started", "subagent.heartbeat", "subagent.warning"}:
                    if self.tool_progress_callback is not None:
                        self.tool_progress_callback(
                            event_type,
                            event.get("tool"),
                            event.get("preview"),
                            None,
                        )
                elif event_type == "tool.completed":
                    if self.tool_progress_callback is not None:
                        self.tool_progress_callback(
                            "tool.completed",
                            event.get("tool"),
                            None,
                            None,
                            duration=event.get("duration"),
                            is_error=event.get("error", False),
                        )
                elif event_type == "agent.activity":
                    if self.tool_progress_callback is not None:
                        self.tool_progress_callback(
                            "agent.activity",
                            None,
                            None,
                            None,
                            activity=event.get("activity") or {},
                        )
                elif event_type == "clarify.request":
                    question = str(event.get("question") or "").strip()
                    raw_choices = event.get("choices") or []
                    choices = [str(choice) for choice in raw_choices] if isinstance(raw_choices, list) else None
                    try:
                        if self.clarify_callback is not None:
                            clarify_result = self.clarify_callback(question, choices)
                        else:
                            clarify_result = (
                                "The interactive client could not collect a clarify response. "
                                "Use your best judgement to proceed."
                            )
                    except Exception as exc:
                        logger.debug("Gateway clarify callback failed: %s", exc, exc_info=True)
                        clarify_result = (
                            "The interactive client could not collect a clarify response. "
                            "Use your best judgement to proceed."
                        )
                    self._submit_clarify_response(run_id, str(clarify_result or "").strip())
                elif event_type == "run.completed":
                    final_response = str(event.get("output") or "")
                    raw_usage = event.get("usage") or {}
                    usage = {
                        "input_tokens": int(raw_usage.get("input_tokens") or 0),
                        "output_tokens": int(raw_usage.get("output_tokens") or 0),
                        "total_tokens": int(raw_usage.get("total_tokens") or 0),
                    }
                elif event_type == "run.failed":
                    failed = True
                    error_message = str(event.get("error") or "Gateway run failed")
                elif event_type == "run.cancelled":
                    interrupted = True
                    error_message = self._interrupt_message or str(event.get("reason") or "Interrupted")

            if self._detach_requested:
                detached = True
                final_response = ""
            elif not final_response:
                final_response = "".join(streamed_chunks)
            if self._interrupt_requested:
                interrupted = True
                error_message = self._interrupt_message or error_message or "Interrupted"

        except Exception as exc:
            if self._detach_requested:
                detached = True
            elif self._interrupt_requested:
                interrupted = True
                error_message = self._interrupt_message or "Interrupted"
            else:
                logger.debug("Gateway event stream failed for %s: %s", run_id, exc, exc_info=True)
                raise GatewaySessionClientError(str(exc)) from exc
        finally:
            try:
                events_response.close()
            except Exception:
                pass
            with self._active_lock:
                self._active_run_id = None
                self._active_events_response = None

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
            "completed": not failed and not interrupted and not detached,
            "failed": failed,
            "interrupted": interrupted,
            "detached": detached,
            "response_previewed": bool(streamed_chunks),
        }
        if reasoning_text:
            result["last_reasoning"] = reasoning_text
        if error_message:
            result["error"] = error_message
        if interrupted:
            result["interrupt_message"] = error_message
        return result

    def interrupt(self, message: Optional[str] = None) -> None:
        with self._active_lock:
            self._interrupt_requested = True
            self._detach_requested = False
            self._interrupt_message = message or "Interrupted"
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
                logger.debug("Gateway run cancel failed for %s", run_id, exc_info=True)
        if response is not None:
            try:
                response.close()
            except Exception:
                pass

    def detach(self) -> None:
        with self._active_lock:
            self._detach_requested = True
            response = self._active_events_response

        if response is not None:
            try:
                response.close()
            except Exception:
                pass

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

    def set_context_length_override(self, context_length: Optional[int] = None) -> Optional[int]:
        return self.set_context_profile(context_length=context_length)

    def reset_session_state(self) -> None:
        self._interrupt_requested = False
        self._detach_requested = False
        self._interrupt_message = None

    def flush_memories(self, conversation_history: Optional[list[dict[str, Any]]] = None) -> None:
        del conversation_history

    def shutdown_memory_provider(self, conversation_history: Optional[list[dict[str, Any]]] = None) -> None:
        del conversation_history

    def _invalidate_system_prompt(self) -> None:
        return None

    def _persist_session(self, conversation_history: Optional[list[dict[str, Any]]] = None, *args: Any, **kwargs: Any) -> None:
        del conversation_history, args, kwargs

    @staticmethod
    def _summarize_api_error(exc: Exception) -> str:
        return str(exc)[:300]
