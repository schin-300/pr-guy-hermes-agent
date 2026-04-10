from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _build_agent(*, context_length_override: int = 272_000) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            model="test/model",
            base_url="https://example.invalid/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            context_length_override=context_length_override,
        )
    agent.client = MagicMock()
    return agent


def test_set_context_length_override_updates_live_compressor_and_runtime():
    agent = _build_agent()

    new_context = agent.set_context_length_override(1_000_000)

    assert new_context == 1_000_000
    assert agent.context_length_override == 1_000_000
    assert agent.context_compressor.context_length == 1_000_000
    assert agent.context_compressor.threshold_tokens == int(
        1_000_000 * agent.context_compressor.threshold_percent
    )
    assert agent.context_compressor.tail_token_budget == int(
        agent.context_compressor.threshold_tokens * agent.context_compressor.summary_target_ratio
    )
    assert agent._primary_runtime["compressor_context_length"] == 1_000_000



def test_set_context_profile_updates_threshold_and_runtime():
    agent = _build_agent()

    new_context = agent.set_context_profile(
        context_length=1_000_000,
        compression_threshold=0.95,
    )

    assert new_context == 1_000_000
    assert agent.context_length_override == 1_000_000
    assert agent.context_compressor.context_length == 1_000_000
    assert agent.context_compressor.threshold_percent == 0.95
    assert agent.context_compressor.threshold_tokens == 950_000
    assert agent.context_compressor.tail_token_budget == int(950_000 * agent.context_compressor.summary_target_ratio)
    assert agent._primary_runtime["compressor_context_length"] == 1_000_000
    assert agent._primary_runtime["compressor_threshold_percent"] == 0.95


def test_set_context_profile_updates_compression_mode():
    agent = _build_agent()

    agent.set_context_profile(
        context_length=1_000_000,
        compression_threshold=0.95,
        compression_mode="timeline",
    )

    assert agent.context_compressor.mode == "timeline"
    assert agent._primary_runtime["compressor_mode"] == "timeline"


def test_switch_model_preserves_context_length_override():
    agent = _build_agent(context_length_override=500_000)

    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch.object(agent, "_create_openai_client", return_value=MagicMock()),
    ):
        agent.switch_model(
            new_model="gpt-5",
            new_provider="openai-codex",
            api_key="test-key-1234567890",
            base_url="https://chatgpt.com/backend-api/codex",
            api_mode="codex_responses",
        )

    assert agent.context_length_override == 500_000
    assert agent.context_compressor.context_length == 500_000
    assert agent._primary_runtime["compressor_context_length"] == 500_000
