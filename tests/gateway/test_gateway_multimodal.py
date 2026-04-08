import sys
import types

import pytest


sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

from gateway.run import GatewayRunner


@pytest.mark.asyncio
async def test_prepare_user_message_with_images_uses_native_multimodal_for_codex(tmp_path):
    runner = object.__new__(GatewayRunner)
    img = tmp_path / "shot.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")

    result = await runner._prepare_user_message_with_images(
        "Describe this",
        [str(img)],
        {
            "runtime": {
                "provider": "openai-codex",
                "api_mode": "codex_responses",
                "base_url": "https://chatgpt.com/backend-api/codex",
            }
        },
    )

    assert isinstance(result, list)
    assert result[0]["type"] == "text"
    assert result[1]["type"] == "image_url"
    assert result[1]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_prepare_user_message_with_images_uses_vision_fallback_for_text_only_route(tmp_path):
    runner = object.__new__(GatewayRunner)
    img = tmp_path / "shot.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")

    async def _fake_enrich(message_text, image_paths):
        assert message_text == "Describe this"
        assert image_paths == [str(img)]
        return "fallback text"

    runner._enrich_message_with_vision = _fake_enrich

    result = await runner._prepare_user_message_with_images(
        "Describe this",
        [str(img)],
        {
            "runtime": {
                "provider": "openrouter",
                "api_mode": "chat_completions",
                "base_url": "https://openrouter.ai/api/v1",
            }
        },
    )

    assert result == "fallback text"