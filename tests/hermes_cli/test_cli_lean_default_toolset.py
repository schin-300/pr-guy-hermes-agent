from unittest.mock import patch

from hermes_cli.tools_config import PLATFORMS, _get_platform_tools
from model_tools import get_tool_definitions


def test_cli_platform_defaults_to_lean_toolset():
    assert PLATFORMS["cli"]["default_toolset"] == "hermes-cli-lean"


def test_unconfigured_cli_uses_lean_leaf_toolsets_without_skills():
    with patch("hermes_cli.tools_config._get_plugin_toolset_keys", return_value=set()):
        enabled = _get_platform_tools({}, "cli", include_default_mcp_servers=False)

    assert enabled == {"terminal", "file", "todo", "memory", "session_search", "clarify"}

    tools = get_tool_definitions(enabled_toolsets=sorted(enabled), quiet_mode=True)
    tool_names = {tool["function"]["name"] for tool in tools}

    assert "terminal" in tool_names
    assert "read_file" in tool_names
    assert "todo" in tool_names
    assert "memory" in tool_names
    assert "session_search" in tool_names
    assert "clarify" in tool_names

    assert "skills_list" not in tool_names
    assert "skill_view" not in tool_names
    assert "skill_manage" not in tool_names
    assert "browser_navigate" not in tool_names
    assert "delegate_task" not in tool_names
