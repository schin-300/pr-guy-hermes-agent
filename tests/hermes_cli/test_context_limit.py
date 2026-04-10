from hermes_cli.context_limit import (
    CONTEXT_MODE_DEFAULT,
    CONTEXT_MODE_PROFILES,
    CONTEXT_MODE_USAGE,
    cycle_context_mode,
    detect_context_mode,
    parse_context_limit_value,
    parse_context_mode_arg,
)


def test_parse_context_limit_accepts_plain_integer():
    assert parse_context_limit_value("500000") == 500_000



def test_parse_context_limit_accepts_suffixes_and_separators():
    assert parse_context_limit_value("272k") == 272_000
    assert parse_context_limit_value("1m") == 1_000_000
    assert parse_context_limit_value("500,000") == 500_000



def test_parse_context_limit_rejects_invalid_values():
    assert parse_context_limit_value("") is None
    assert parse_context_limit_value("turbo") is None
    assert parse_context_limit_value("0") is None



def test_context_mode_constants_are_documented():
    assert CONTEXT_MODE_DEFAULT == "1m"
    assert CONTEXT_MODE_USAGE == "/context-mode [1m|272k|status]"



def test_context_mode_profiles_use_expected_compaction_thresholds():
    assert CONTEXT_MODE_PROFILES["1m"]["compression_threshold"] == 0.75
    assert CONTEXT_MODE_PROFILES["272k"]["compression_threshold"] == 0.95



def test_parse_context_mode_accepts_named_modes_and_status():
    assert parse_context_mode_arg("") == "toggle"
    assert parse_context_mode_arg(None) == "toggle"
    assert parse_context_mode_arg("status") == "status"
    assert parse_context_mode_arg("1m") == "1m"
    assert parse_context_mode_arg("272k") == "272k"
    assert parse_context_mode_arg("1,000,000") == "1m"
    assert parse_context_mode_arg("272000") == "272k"



def test_parse_context_mode_rejects_unknown_values():
    assert parse_context_mode_arg("500k") is None
    assert parse_context_mode_arg("turbo") is None



def test_cycle_context_mode_switches_between_profiles():
    assert cycle_context_mode("1m") == "272k"
    assert cycle_context_mode("272k") == "1m"
    assert cycle_context_mode("custom") == "1m"



def test_detect_context_mode_matches_known_profiles_and_custom_values():
    one_m = CONTEXT_MODE_PROFILES["1m"]
    two_seventy_two = CONTEXT_MODE_PROFILES["272k"]

    assert detect_context_mode(
        one_m["context_length"],
        one_m["compression_threshold"],
    ) == "1m"
    assert detect_context_mode(
        two_seventy_two["context_length"],
        two_seventy_two["compression_threshold"],
    ) == "272k"
    assert detect_context_mode(500_000, one_m["compression_threshold"]) is None
