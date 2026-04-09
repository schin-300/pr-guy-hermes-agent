from hermes_cli.context_limit import (
    CONTEXT_LIMIT_PRESETS,
    cycle_context_limit,
    parse_context_limit_value,
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


def test_cycle_context_limit_switches_between_presets():
    low, high = CONTEXT_LIMIT_PRESETS
    assert cycle_context_limit(low) == high
    assert cycle_context_limit(high) == low
    assert cycle_context_limit(500_000) == low
