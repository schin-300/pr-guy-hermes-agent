"""Tests for the startup allowlist warning check in gateway/run.py."""

import os
from unittest.mock import patch

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import should_warn_missing_allowlists


class TestAllowlistStartupCheck:
    def test_no_user_facing_platforms_suppresses_warning(self):
        config = GatewayConfig(platforms={Platform.API_SERVER: PlatformConfig(enabled=True)})
        with patch.dict(os.environ, {}, clear=True):
            assert should_warn_missing_allowlists(config) is False

    def test_user_facing_platform_without_allowlist_warns(self):
        config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
        with patch.dict(os.environ, {}, clear=True):
            assert should_warn_missing_allowlists(config) is True

    def test_signal_group_allowed_users_suppresses_warning(self):
        config = GatewayConfig(platforms={Platform.SIGNAL: PlatformConfig(enabled=True)})
        with patch.dict(os.environ, {"SIGNAL_GROUP_ALLOWED_USERS": "user1"}, clear=True):
            assert should_warn_missing_allowlists(config) is False

    def test_telegram_allow_all_users_suppresses_warning(self):
        config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
        with patch.dict(os.environ, {"TELEGRAM_ALLOW_ALL_USERS": "true"}, clear=True):
            assert should_warn_missing_allowlists(config) is False

    def test_gateway_allow_all_users_suppresses_warning(self):
        config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
        with patch.dict(os.environ, {"GATEWAY_ALLOW_ALL_USERS": "yes"}, clear=True):
            assert should_warn_missing_allowlists(config) is False
