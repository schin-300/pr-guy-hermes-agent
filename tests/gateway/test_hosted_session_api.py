import asyncio
import json
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware, security_headers_middleware


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    app.router.add_post("/v1/runs/{run_id}/cancel", adapter._handle_cancel_run)
    app.router.add_get("/v1/sessions/live", adapter._handle_live_sessions)
    app.router.add_post("/v1/sessions/{session_id}/attach", adapter._handle_attach_session)
    app.router.add_post("/v1/sessions/{session_id}/detach", adapter._handle_detach_session)
    app.router.add_post("/v1/sessions/{session_id}/close", adapter._handle_close_session)
    return app


def _parse_sse_events(body: str) -> list[dict]:
    events = []
    for line in body.splitlines():
        if line.startswith("data: {"):
            events.append(json.loads(line[len("data: "):]))
    return events


class _InterruptableAgent:
    def __init__(self):
        self.interrupts = []

    def interrupt(self, message=None):
        self.interrupts.append(message)


@pytest.mark.asyncio
async def test_runs_stream_canonical_hosted_session_events():
    adapter = _make_adapter()
    app = _create_app(adapter)

    async def _mock_run_agent(**kwargs):
        kwargs["tool_gen_callback"]("write_file")
        kwargs["tool_progress_callback"]("tool.started", "write_file", "notes.txt", {"path": "notes.txt"})
        kwargs["reasoning_callback"]("thinking...")
        kwargs["stream_delta_callback"]("hello")
        kwargs["tool_progress_callback"]("subagent_progress", "delegate_task", "child 1/2")
        kwargs["tool_progress_callback"]("tool.completed", "write_file", None, None, duration=0.2, is_error=False)
        return (
            {
                "final_response": "hello world",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello world"},
                ],
                "last_reasoning": "thinking...done",
                "api_calls": 1,
            },
            {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        )

    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
            resp = await cli.post("/v1/runs", json={"input": "hi", "session_id": "sess_1"})
            assert resp.status == 202
            data = await resp.json()
            run_id = data["run_id"]
            assert data["session_id"] == "sess_1"

            events_resp = await cli.get(f"/v1/runs/{run_id}/events")
            assert events_resp.status == 200
            body = await events_resp.text()
            events = _parse_sse_events(body)

    names = [event["event"] for event in events]
    assert names[0] == "session.created"
    assert names[1] == "run.started"
    assert "tool.generating" in names
    assert "tool.started" in names
    assert "reasoning.delta" in names
    assert "subagent.progress" in names
    assert "tool.completed" in names
    assert "message.completed" in names
    assert names[-1] == "run.completed"


@pytest.mark.asyncio
async def test_second_run_reuses_session_without_reemitting_session_created():
    adapter = _make_adapter()
    app = _create_app(adapter)

    async def _mock_run_agent(**kwargs):
        return (
            {
                "final_response": "ok",
                "messages": [
                    {"role": "user", "content": kwargs["user_message"]},
                    {"role": "assistant", "content": "ok"},
                ],
            },
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )

    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
            first = await cli.post("/v1/runs", json={"input": "one", "session_id": "sess_reuse"})
            first_run_id = (await first.json())["run_id"]
            first_events = _parse_sse_events(await (await cli.get(f"/v1/runs/{first_run_id}/events")).text())

            second = await cli.post("/v1/runs", json={"input": "two", "session_id": "sess_reuse"})
            second_run_id = (await second.json())["run_id"]
            second_events = _parse_sse_events(await (await cli.get(f"/v1/runs/{second_run_id}/events")).text())

    assert first_events[0]["event"] == "session.created"
    assert second_events[0]["event"] == "run.started"
    assert all(event["event"] != "session.created" for event in second_events)


@pytest.mark.asyncio
async def test_cancel_run_endpoint_interrupts_hosted_agent():
    adapter = _make_adapter()
    app = _create_app(adapter)
    agent = _InterruptableAgent()

    async def _mock_run_agent(**kwargs):
        kwargs["agent_ref"][0] = agent
        await asyncio.sleep(0.2)
        return ({"interrupted": True, "interrupt_message": "Cancelled by client"}, {},)

    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
            resp = await cli.post("/v1/runs", json={"input": "stop", "session_id": "sess_cancel"})
            run_id = (await resp.json())["run_id"]
            cancel = await cli.post(f"/v1/runs/{run_id}/cancel")
            assert cancel.status == 200
            payload = await cancel.json()
            assert payload["ok"] is True

        await asyncio.sleep(0.05)

    assert agent.interrupts == ["Cancelled by client"]


@pytest.mark.asyncio
async def test_close_session_endpoint_returns_404_for_missing_session():
    adapter = _make_adapter()
    app = _create_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/v1/sessions/missing/close")
        assert resp.status == 404


@pytest.mark.asyncio
async def test_live_session_endpoints_keep_detached_sessions_visible_until_closed():
    adapter = _make_adapter()
    app = _create_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        attach = await cli.post(
            "/v1/sessions/sess_live/attach",
            json={"client_id": "cli_1", "model": "gpt-test", "provider": "openai"},
        )
        assert attach.status == 200
        attach_payload = await attach.json()
        assert attach_payload["session"]["id"] == "sess_live"
        assert attach_payload["session"]["attached_clients"] == 1

        live = await cli.get("/v1/sessions/live")
        assert live.status == 200
        rows = (await live.json())["sessions"]
        assert [row["id"] for row in rows] == ["sess_live"]
        assert rows[0]["status"] == "attached"

        detach = await cli.post("/v1/sessions/sess_live/detach", json={"client_id": "cli_1"})
        assert detach.status == 200

        live_after = await cli.get("/v1/sessions/live")
        rows_after = (await live_after.json())["sessions"]
        assert [row["id"] for row in rows_after] == ["sess_live"]
        assert rows_after[0]["status"] == "detached"

        close = await cli.post("/v1/sessions/sess_live/close")
        assert close.status == 200

        live_closed = await cli.get("/v1/sessions/live")
        assert (await live_closed.json())["sessions"] == []
