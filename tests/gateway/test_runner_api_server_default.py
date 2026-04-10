from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner


def test_gateway_runner_enables_api_server_by_default():
    runner = GatewayRunner(GatewayConfig())

    assert Platform.API_SERVER in runner.config.platforms
    assert runner.config.platforms[Platform.API_SERVER].enabled is True
