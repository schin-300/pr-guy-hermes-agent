import json

from hermes_cli.hosted_session_client import (
    HostedSessionAgentProxy,
    HostedSessionEndpoint,
)


class _FakeResponse:
    def __init__(self, json_payload=None, lines=None):
        self._json_payload = json_payload or {}
        self._lines = list(lines or [])

    def raise_for_status(self):
        return None

    def json(self):
        return self._json_payload

    def iter_lines(self, decode_unicode=True):
        del decode_unicode
        for line in self._lines:
            yield line

    def close(self):
        return None


class _FakeSession:
    def __init__(self, run_response, event_response, live_sessions_response=None):
        self.run_response = run_response
        self.event_response = event_response
        self.live_sessions_response = live_sessions_response or _FakeResponse({"sessions": []})
        self.posts = []
        self.gets = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        if url.endswith("/attach") or url.endswith("/detach") or url.endswith("/close") or url.endswith("/cancel"):
            return _FakeResponse({"ok": True})
        return self.run_response

    def get(self, url, headers=None, stream=False, timeout=None, params=None):
        self.gets.append({"url": url, "headers": headers, "stream": stream, "timeout": timeout, "params": params})
        if url.endswith("/v1/sessions/live"):
            return self.live_sessions_response
        return self.event_response


def test_hosted_session_proxy_maps_canonical_events_to_cli_callbacks():
    streamed = []
    reasoning = []
    tool_events = []
    tool_gen = []

    events = [
        'data: ' + json.dumps({"event": "session.created", "session_id": "sess_1", "run_id": "run_1", "timestamp": 1, "payload": {}}),
        'data: ' + json.dumps({"event": "run.started", "session_id": "sess_1", "run_id": "run_1", "timestamp": 2, "payload": {"user_message": "hi"}}),
        'data: ' + json.dumps({"event": "tool.generating", "session_id": "sess_1", "run_id": "run_1", "timestamp": 3, "payload": {"tool": "write_file"}}),
        'data: ' + json.dumps({"event": "tool.started", "session_id": "sess_1", "run_id": "run_1", "timestamp": 4, "payload": {"tool": "write_file", "preview": "notes.txt", "args": {"path": "notes.txt"}}}),
        'data: ' + json.dumps({"event": "reasoning.delta", "session_id": "sess_1", "run_id": "run_1", "timestamp": 5, "payload": {"text": "thinking..."}}),
        'data: ' + json.dumps({"event": "message.delta", "session_id": "sess_1", "run_id": "run_1", "timestamp": 6, "payload": {"delta": "hello"}}),
        'data: ' + json.dumps({"event": "subagent.progress", "session_id": "sess_1", "run_id": "run_1", "timestamp": 7, "payload": {"tool": "delegate_task", "text": "child 1/2"}}),
        'data: ' + json.dumps({"event": "tool.completed", "session_id": "sess_1", "run_id": "run_1", "timestamp": 8, "payload": {"tool": "write_file", "duration": 0.2, "error": False}}),
        'data: ' + json.dumps({"event": "message.completed", "session_id": "sess_1", "run_id": "run_1", "timestamp": 9, "payload": {"content": "hello world"}}),
        'data: ' + json.dumps({"event": "run.completed", "session_id": "sess_1", "run_id": "run_1", "timestamp": 10, "payload": {"output": "hello world", "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}}}),
        ': stream closed',
    ]

    fake_session = _FakeSession(
        run_response=_FakeResponse({"run_id": "run_1", "session_id": "sess_1", "status": "started"}),
        event_response=_FakeResponse(lines=events),
    )

    proxy = HostedSessionAgentProxy(
        endpoint=HostedSessionEndpoint(base_url="http://127.0.0.1:8642", api_key=None),
        session_id="sess_1",
        model="gpt-test",
        provider="openai",
        tool_progress_callback=lambda *args, **kwargs: tool_events.append((args, kwargs)),
        reasoning_callback=lambda text: reasoning.append(text),
        tool_gen_callback=lambda tool: tool_gen.append(tool),
        http_session=fake_session,
    )

    result = proxy.run_conversation(
        user_message="hi",
        conversation_history=[{"role": "assistant", "content": "previous"}],
        stream_callback=lambda delta: streamed.append(delta),
    )

    assert streamed == ["hello"]
    assert reasoning == ["thinking..."]
    assert tool_gen == ["write_file"]
    assert result["final_response"] == "hello world"
    assert result["last_reasoning"] == "thinking..."
    assert result["response_previewed"] is True
    assert proxy.session_total_tokens == 3
    assert any(args[0] == "tool.started" for args, _ in tool_events)
    assert any(args[0] == "tool.completed" for args, _ in tool_events)
    assert any(args[0] == "subagent.progress" for args, _ in tool_events)


def test_hosted_session_proxy_accepts_flattened_legacy_event_shape():
    streamed = []
    reasoning = []
    tool_events = []

    events = [
        'data: ' + json.dumps({"event": "tool.started", "tool": "read_file", "preview": "README.md", "args": {"path": "README.md"}}),
        'data: ' + json.dumps({"event": "message.delta", "delta": "Hello"}),
        'data: ' + json.dumps({"event": "reasoning.available", "text": "Thinking"}),
        'data: ' + json.dumps({"event": "tool.completed", "tool": "read_file", "duration": 0.1, "error": False}),
        'data: ' + json.dumps({"event": "run.completed", "output": "Hello!", "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}}),
        ': stream closed',
    ]

    fake_session = _FakeSession(
        run_response=_FakeResponse({"run_id": "run_legacy", "session_id": "sess_legacy", "status": "started"}),
        event_response=_FakeResponse(lines=events),
    )

    proxy = HostedSessionAgentProxy(
        endpoint=HostedSessionEndpoint(base_url="http://127.0.0.1:8642", api_key=None),
        session_id="sess_legacy",
        tool_progress_callback=lambda *args, **kwargs: tool_events.append((args, kwargs)),
        reasoning_callback=lambda text: reasoning.append(text),
        http_session=fake_session,
    )

    result = proxy.run_conversation(
        user_message="legacy",
        conversation_history=[],
        stream_callback=lambda delta: streamed.append(delta),
    )

    assert streamed == ["Hello"]
    assert reasoning == ["Thinking"]
    assert result["final_response"] == "Hello!"
    assert result["last_reasoning"] == "Thinking"
    assert result["response_previewed"] is True
    assert proxy.session_total_tokens == 5
    assert any(args[0] == "tool.started" for args, _ in tool_events)
    assert any(args[0] == "tool.completed" for args, _ in tool_events)


def test_hosted_session_proxy_plain_interrupt_does_not_create_redirect_message():
    events = [
        'data: ' + json.dumps({"event": "run.cancelled", "payload": {"error": "Interrupted"}}),
        ': stream closed',
    ]
    fake_session = _FakeSession(
        run_response=_FakeResponse({"run_id": "run_plain_interrupt", "session_id": "sess_1", "status": "started"}),
        event_response=_FakeResponse(lines=events),
    )
    proxy = HostedSessionAgentProxy(
        endpoint=HostedSessionEndpoint(base_url="http://127.0.0.1:8642", api_key=None),
        session_id="sess_1",
        http_session=fake_session,
    )
    proxy._active_run_id = "run_plain_interrupt"
    proxy.interrupt()
    result = proxy.run_conversation(user_message="ignored", conversation_history=[])
    assert result.get("interrupted") is True
    assert result.get("interrupt_message") is None
    assert result.get("error") == "Interrupted"


def test_hosted_session_proxy_interrupt_posts_cancel_and_close_session_posts_close():
    fake_session = _FakeSession(
        run_response=_FakeResponse({"run_id": "run_1", "session_id": "sess_1", "status": "started"}),
        event_response=_FakeResponse(lines=[]),
    )
    proxy = HostedSessionAgentProxy(
        endpoint=HostedSessionEndpoint(base_url="http://127.0.0.1:8642", api_key=None),
        session_id="sess_1",
        http_session=fake_session,
    )
    proxy._active_run_id = "run_1"
    proxy.interrupt("stop")
    assert fake_session.posts[-1]["url"].endswith("/v1/runs/run_1/cancel")

    proxy.close_session()
    assert fake_session.posts[-1]["url"].endswith("/v1/sessions/sess_1/close")


def test_hosted_session_proxy_registers_live_session_and_can_switch_between_live_sessions():
    fake_session = _FakeSession(
        run_response=_FakeResponse({"run_id": "run_1", "session_id": "sess_1", "status": "started"}),
        event_response=_FakeResponse(lines=[]),
        live_sessions_response=_FakeResponse({
            "sessions": [
                {"id": "sess_1", "title": "Current", "preview": "", "source": "live", "last_active": 10},
                {"id": "sess_2", "title": "Other", "preview": "", "source": "live", "last_active": 9},
            ]
        }),
    )
    proxy = HostedSessionAgentProxy(
        endpoint=HostedSessionEndpoint(base_url="http://127.0.0.1:8642", api_key=None),
        session_id="sess_1",
        http_session=fake_session,
    )

    assert fake_session.posts[0]["url"].endswith("/v1/sessions/sess_1/attach")
    assert fake_session.posts[0]["json"]["client_id"] == proxy.client_id

    rows = proxy.list_live_sessions(limit=25)
    assert [row["id"] for row in rows] == ["sess_1", "sess_2"]
    assert fake_session.gets[-1]["params"] == {"limit": 25}

    proxy.switch_session("sess_2")
    assert proxy.session_id == "sess_2"
    assert fake_session.posts[-2]["url"].endswith("/v1/sessions/sess_1/detach")
    assert fake_session.posts[-1]["url"].endswith("/v1/sessions/sess_2/attach")


def test_hosted_session_proxy_exposes_cli_compat_methods_for_session_switching():
    fake_session = _FakeSession(
        run_response=_FakeResponse({"run_id": "run_1", "session_id": "sess_1", "status": "started"}),
        event_response=_FakeResponse(lines=[]),
    )
    proxy = HostedSessionAgentProxy(
        endpoint=HostedSessionEndpoint(base_url="http://127.0.0.1:8642", api_key=None),
        session_id="sess_1",
        http_session=fake_session,
    )

    proxy.session_total_tokens = 99
    proxy.session_input_tokens = 50
    proxy.session_output_tokens = 49
    proxy.session_api_calls = 3
    assert proxy.set_context_length_override(12345) == 12345
    assert proxy.context_compressor.context_length == 12345

    proxy.reset_session_state()

    assert proxy.session_total_tokens == 0
    assert proxy.session_input_tokens == 0
    assert proxy.session_output_tokens == 0
    assert proxy.session_api_calls == 0
    proxy.flush_memories([{"role": "user", "content": "hi"}], min_turns=0)
    proxy._invalidate_system_prompt()
