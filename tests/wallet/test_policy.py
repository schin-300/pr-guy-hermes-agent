"""Tests for wallet.policy — transaction policy engine."""

import time
from decimal import Decimal

import pytest

from wallet.policy import (
    PolicyEngine,
    PolicyResult,
    PolicyVerdict,
    TxRequest,
    AGENT_WALLET_DEFAULTS,
)


def _make_tx(**overrides) -> TxRequest:
    defaults = {
        "wallet_id": "w_test",
        "wallet_type": "agent",
        "chain": "ethereum-sepolia",
        "to_address": "0x1234567890abcdef1234567890abcdef12345678",
        "amount": Decimal("0.01"),
        "symbol": "ETH",
    }
    defaults.update(overrides)
    return TxRequest(**defaults)


class TestBasicPolicy:
    def test_allow_small_agent_tx(self):
        engine = PolicyEngine()
        tx = _make_tx(amount=Decimal("0.001"))
        result = engine.evaluate(tx)
        assert result.verdict == PolicyVerdict.ALLOW

    def test_block_over_spending_limit(self):
        engine = PolicyEngine({"spending_limit": {"max_native": "0.5"}})
        tx = _make_tx(amount=Decimal("1.0"))
        result = engine.evaluate(tx)
        assert result.verdict == PolicyVerdict.BLOCK
        assert "spending_limit" in result.failed

    def test_require_approval_above_threshold(self):
        engine = PolicyEngine({"require_approval": {"above_native": "0.1"}})
        tx = _make_tx(amount=Decimal("0.5"), wallet_type="agent")
        result = engine.evaluate(tx)
        assert result.verdict == PolicyVerdict.REQUIRE_APPROVAL

    def test_user_wallet_always_requires_approval(self):
        engine = PolicyEngine()
        tx = _make_tx(wallet_type="user", amount=Decimal("0.001"))
        result = engine.evaluate(tx)
        # User wallet defaults require approval for any amount
        assert result.verdict == PolicyVerdict.REQUIRE_APPROVAL


class TestRateLimit:
    def test_rate_limit_blocks_after_max(self):
        engine = PolicyEngine({"rate_limit": {"max_txns": 2, "window_seconds": 3600}})
        tx = _make_tx()

        # First two should pass
        r1 = engine.evaluate(tx)
        engine.record_transaction(tx)
        r2 = engine.evaluate(tx)
        engine.record_transaction(tx)

        # Third should be blocked
        r3 = engine.evaluate(tx)
        assert r3.verdict == PolicyVerdict.BLOCK
        assert "rate_limit" in r3.failed


class TestCooldown:
    def test_cooldown_blocks_rapid_txs(self):
        engine = PolicyEngine({"cooldown": {"min_seconds": 60}})
        tx = _make_tx()

        # First is fine
        engine.record_transaction(tx)

        # Immediate second should be blocked
        result = engine.evaluate(tx)
        assert result.verdict == PolicyVerdict.BLOCK
        assert "cooldown" in result.failed


class TestRecipientPolicies:
    def test_allowed_recipients_blocks_unknown(self):
        engine = PolicyEngine({
            "allowed_recipients": {"addresses": ["0xAAAA"]},
        })
        tx = _make_tx(to_address="0xBBBB")
        result = engine.evaluate(tx)
        assert result.verdict == PolicyVerdict.BLOCK

    def test_allowed_recipients_passes_known(self):
        addr = "0x1234567890abcdef1234567890abcdef12345678"
        engine = PolicyEngine({
            "allowed_recipients": {"addresses": [addr]},
        })
        tx = _make_tx(to_address=addr)
        # Agent defaults may still require approval, but shouldn't be BLOCKED
        result = engine.evaluate(tx)
        assert result.verdict != PolicyVerdict.BLOCK or result.failed != "allowed_recipients"

    def test_blocked_recipients(self):
        bad = "0xBADBADBADBADBADBADBADBADBADBADBADBADBADBA"
        engine = PolicyEngine({
            "blocked_recipients": {"addresses": [bad]},
        })
        tx = _make_tx(to_address=bad)
        result = engine.evaluate(tx)
        assert result.verdict == PolicyVerdict.BLOCK
        assert "blocked_recipients" in result.failed


class TestFreezeKillSwitch:
    def test_freeze_blocks_everything(self):
        engine = PolicyEngine()
        engine.freeze()
        tx = _make_tx(amount=Decimal("0.0001"))
        result = engine.evaluate(tx)
        assert result.verdict == PolicyVerdict.BLOCK
        assert "frozen" in result.reason.lower()

    def test_unfreeze_resumes(self):
        engine = PolicyEngine()
        engine.freeze()
        engine.unfreeze()
        tx = _make_tx(amount=Decimal("0.0001"))
        result = engine.evaluate(tx)
        assert result.verdict != PolicyVerdict.BLOCK or result.failed != "freeze"


class TestDailyLimit:
    def test_daily_limit_blocks_aggregate(self):
        engine = PolicyEngine({"daily_limit": {"max_native": "0.05"}})
        tx = _make_tx(amount=Decimal("0.03"))

        # First tx: 0.03 of 0.05 limit
        r1 = engine.evaluate(tx)
        assert r1.verdict != PolicyVerdict.BLOCK or r1.failed != "daily_limit"
        engine.record_transaction(tx)

        # Second tx: 0.03 more would be 0.06 > 0.05
        r2 = engine.evaluate(tx)
        assert r2.verdict == PolicyVerdict.BLOCK
        assert "daily_limit" in r2.failed
