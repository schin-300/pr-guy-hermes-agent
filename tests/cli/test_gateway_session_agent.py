import json
import threading
import time

from hermes_cli.gateway_session_client import (
    GatewaySessionAgentProxy,
    GatewaySessionEndpoint,
)


class _FakeResponse:
    def __init__(self, payload=None, lines=None, status_code=200):
        self._payload = payload or {}
        self._lines = list(lines or [])
        self.status_code = status_code
        self.closed = False

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def iter_lines(self, decode_unicode=True):
        for line in self._lines:
            yield line if decode_unicode else line.encode()

    def close(self):
        self.closed = True


class _BlockingResponse(_FakeResponse):
    def __init__(self, payload=None, status_code=200):
        super().__init__(payload=payload, lines=[], status_code=status_code)
        self._started = threading.Event()

    def iter_lines(self, decode_unicode=True):
        self._started.set()
        while not self.closed:
            time.sleep(0.01)
        return


class _FakeSession:
    def __init__(self, *, run_response, event_response, cancel_response=None):
        self.run_response = run_response
        self.event_response = event_response
        self.cancel_response = cancel_response or _FakeResponse({"status": "cancelling"})
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        if url.endswith("/cancel"):
            return self.cancel_response
        return self.run_response

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self.event_response


def test_run_conversation_streams_gateway_events_and_updates_usage():
    events = [
        'data: {"event":"reasoning.available","text":"thinking..."}',
        'data: {"event":"tool.started","tool":"search_files","preview":"searching"}',
        'data: {"event":"message.delta","delta":"Hi"}',
        'data: {"event":"message.delta","delta":" there"}',
        'data: {"event":"run.completed","output":"Hi there","usage":{"input_tokens":3,"output_tokens":4,"total_tokens":7}}',
        ': stream closed',
    ]
    fake_session = _FakeSession(
        run_response=_FakeResponse({"run_id": "run_123", "status": "started"}),
        event_response=_FakeResponse(lines=events),
    )
    streamed = []
    tool_events = []
    reasoning = []
    proxy = GatewaySessionAgentProxy(
        endpoint=GatewaySessionEndpoint(base_url="http://127.0.0.1:8642", api_key=None),
        session_id="sess_1",
        model="gpt-test",
        provider="openai",
        api_mode="chat_completions",
        enabled_toolsets=["terminal", "file"],
        http_session=fake_session,
        tool_progress_callback=lambda event_type, name=None, preview=None, args=None, **kwargs: tool_events.append((event_type, name, preview)),
        reasoning_callback=lambda text: reasoning.append(text),
    )

    result = proxy.run_conversation(
        user_message="Hello",
        conversation_history=[{"role": "assistant", "content": "Earlier"}],
        stream_callback=lambda delta: streamed.append(delta),
    )

    assert streamed == ["Hi", " there"]
    assert reasoning == ["thinking..."]
    assert tool_events == [("tool.started", "search_files", "searching")]
    assert result["final_response"] == "Hi there"
    assert result["messages"] == [
        {"role": "assistant", "content": "Earlier"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    assert proxy.session_prompt_tokens == 3
    assert proxy.session_completion_tokens == 4
    assert proxy.session_total_tokens == 7
    assert proxy.session_api_calls == 1

    method, url, kwargs = fake_session.calls[0]
    assert method == "POST"
    assert url.endswith("/v1/runs")
    assert kwargs["json"]["session_id"] == "sess_1"
    assert kwargs["json"]["conversation_history"] == [{"role": "assistant", "content": "Earlier"}]
    assert kwargs["json"]["toolsets"] == ["terminal", "file"]


def test_interrupt_posts_run_cancel_and_closes_stream():
    fake_stream = _FakeResponse(lines=[])
    fake_session = _FakeSession(
        run_response=_FakeResponse({"run_id": "run_123", "status": "started"}),
        event_response=fake_stream,
    )
    proxy = GatewaySessionAgentProxy(
        endpoint=GatewaySessionEndpoint(base_url="http://127.0.0.1:8642", api_key=None),
        session_id="sess_1",
        http_session=fake_session,
    )
    proxy._active_run_id = "run_123"
    proxy._active_events_response = fake_stream

    proxy.interrupt("new message")

    assert fake_stream.closed is True
    assert fake_session.calls[-1][0] == "POST"
    assert fake_session.calls[-1][1].endswith("/v1/runs/run_123/cancel")


def test_detach_closes_stream_without_cancelling_run():
    fake_stream = _BlockingResponse()
    fake_session = _FakeSession(
        run_response=_FakeResponse({"run_id": "run_123", "status": "started"}),
        event_response=fake_stream,
    )
    proxy = GatewaySessionAgentProxy(
        endpoint=GatewaySessionEndpoint(base_url="http://127.0.0.1:8642", api_key=None),
        session_id="sess_1",
        http_session=fake_session,
    )
    result_box = {}

    def _run():
        result_box["result"] = proxy.run_conversation(
            user_message="Hello",
            conversation_history=[],
        )

    thread = threading.Thread(target=_run)
    thread.start()
    fake_stream._started.wait(timeout=1)

    proxy.detach()
    thread.join(timeout=2)

    assert thread.is_alive() is False
    assert fake_stream.closed is True
    assert result_box["result"]["detached"] is True
    assert not any(call[1].endswith("/cancel") for call in fake_session.calls)


def test_close_session_posts_close_endpoint_and_cancels_active_run():
    fake_stream = _FakeResponse(lines=[])
    fake_session = _FakeSession(
        run_response=_FakeResponse({"run_id": "run_123", "status": "started"}),
        event_response=fake_stream,
    )
    proxy = GatewaySessionAgentProxy(
        endpoint=GatewaySessionEndpoint(base_url="http://127.0.0.1:8642", api_key=None),
        session_id="sess_1",
        http_session=fake_session,
    )
    proxy._active_run_id = "run_123"
    proxy._active_events_response = fake_stream

    proxy.close_session()

    urls = [call[1] for call in fake_session.calls if call[0] == "POST"]
    assert any(url.endswith("/v1/runs/run_123/cancel") for url in urls)
    assert any(url.endswith("/v1/sessions/sess_1/close") for url in urls)
