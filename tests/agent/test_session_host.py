import asyncio

import pytest

from agent.session_host import SessionHost


@pytest.mark.asyncio
async def test_session_host_replays_backlog_and_finishes_run():
    host = SessionHost()

    async def _fake_run(**kwargs):
        kwargs["tool_gen_callback"]("write_file")
        kwargs["tool_progress_callback"]("tool.started", "write_file", "notes.txt", {"path": "notes.txt"})
        kwargs["reasoning_callback"]("thinking...")
        kwargs["stream_delta_callback"]("hello")
        kwargs["tool_progress_callback"]("tool.completed", "write_file", None, None, duration=0.2, is_error=False)
        return (
            {
                "final_response": "hello world",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello world"},
                ],
                "last_reasoning": "thinking...done",
            },
            {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            object(),
        )

    run_id = await host.start_run(session_id="sess_1", user_message="hi", run_callable=_fake_run)
    queue = host.subscribe_run(run_id, loop=asyncio.get_running_loop())

    events = []
    while True:
        item = await asyncio.wait_for(queue.get(), timeout=1)
        if item is None:
            break
        events.append(item)

    event_names = [item["event"] for item in events]
    assert event_names[0] == "session.created"
    assert event_names[1] == "run.started"
    assert "tool.generating" in event_names
    assert "tool.started" in event_names
    assert "reasoning.delta" in event_names
    assert "message.delta" in event_names
    assert "tool.completed" in event_names
    assert "reasoning.completed" in event_names
    assert "message.completed" in event_names
    assert event_names[-1] == "run.completed"

    session = host.get_session("sess_1")
    assert session is not None
    assert session.conversation_history[-1]["content"] == "hello world"

    run = host.get_run(run_id)
    assert run is not None
    assert run.finished is True
    assert run.status == "completed"


@pytest.mark.asyncio
async def test_session_host_cancel_run_interrupts_agent():
    host = SessionHost()
    interrupted = asyncio.Event()

    class _Agent:
        def interrupt(self, message=None):
            interrupted.set()

    async def _fake_run(**kwargs):
        if kwargs.get("agent_ref") is not None:
            kwargs["agent_ref"][0] = _Agent()
        await asyncio.sleep(0.2)
        return ({"interrupted": True, "interrupt_message": "stop"}, {}, kwargs["agent_ref"][0])

    run_id = await host.start_run(session_id="sess_cancel", user_message="stop", run_callable=_fake_run)
    assert host.cancel_run(run_id, reason="stop") is True
    await asyncio.wait_for(interrupted.wait(), timeout=1)


@pytest.mark.asyncio
async def test_session_host_emits_subagent_progress():
    host = SessionHost()

    async def _fake_run(**kwargs):
        kwargs["tool_progress_callback"]("subagent_progress", "delegate_task", "child 1/2")
        return ({"final_response": "done", "messages": []}, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, object())

    run_id = await host.start_run(session_id="sess_sub", user_message="go", run_callable=_fake_run)
    queue = host.subscribe_run(run_id, loop=asyncio.get_running_loop())
    events = []
    while True:
        item = await asyncio.wait_for(queue.get(), timeout=1)
        if item is None:
            break
        events.append(item)

    subagent_events = [event for event in events if event["event"] == "subagent.progress"]
    assert len(subagent_events) == 1
    assert subagent_events[0]["payload"]["tool"] == "delegate_task"
    assert subagent_events[0]["payload"]["text"] == "child 1/2"


def test_session_host_lists_detached_sessions_until_explicitly_closed():
    host = SessionHost()

    attached = host.attach_session(
        "sess_live",
        client_id="cli_1",
        metadata={"title": "Live Session", "model": "gpt-test", "provider": "openai"},
    )

    assert attached["id"] == "sess_live"
    assert attached["status"] == "attached"
    assert attached["attached_clients"] == 1

    rows = host.list_sessions(live_only=True)
    assert [row["id"] for row in rows] == ["sess_live"]
    assert rows[0]["title"] == "Live Session"

    assert host.detach_session("sess_live", client_id="cli_1") is True
    rows_after_detach = host.list_sessions(live_only=True)
    assert [row["id"] for row in rows_after_detach] == ["sess_live"]
    assert rows_after_detach[0]["status"] == "detached"

    assert host.close_session("sess_live") is True
    assert host.list_sessions(live_only=True) == []
