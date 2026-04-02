from hermes_cli.fast_mode import (
    FAST_MODE_USAGE,
    fast_mode_note,
    normalize_service_tier,
    parse_fast_command_arg,
    responses_api_service_tier,
)


class TestFastModeHelpers:
    def test_parse_fast_command_args(self):
        assert parse_fast_command_arg("") == "toggle"
        assert parse_fast_command_arg(None) == "toggle"
        assert parse_fast_command_arg("on") == "on"
        assert parse_fast_command_arg("off") == "off"
        assert parse_fast_command_arg("status") == "status"
        assert parse_fast_command_arg("weird") is None

    def test_normalize_service_tier(self):
        assert normalize_service_tier(None) is None
        assert normalize_service_tier("priority") == "fast"
        assert normalize_service_tier("fast") == "fast"
        assert normalize_service_tier("flex") == "flex"
        assert normalize_service_tier("off") is None

    def test_responses_api_mapping(self):
        assert responses_api_service_tier("fast") == "priority"
        assert responses_api_service_tier("priority") == "priority"
        assert responses_api_service_tier("flex") == "flex"
        assert responses_api_service_tier(None) is None

    def test_fast_mode_note(self):
        assert "service_tier=priority" in fast_mode_note()
        assert "next Codex turn" in fast_mode_note("openai-codex", "gpt-5.4")
        assert "will not send" in fast_mode_note(enabled=False)
        assert FAST_MODE_USAGE.startswith("/fast")
