"""Tests for tools/hermes_docs_lookup_tool.py — public-upstream retrieval and ranking."""

import json
from textwrap import dedent
from unittest.mock import patch

from tools.hermes_docs_lookup_tool import hermes_docs_lookup_tool


PUBLIC_TREE = [
    {"path": "website/docs/user-guide/features/plugins.md", "type": "blob"},
    {"path": "website/docs/user-guide/features/skins.md", "type": "blob"},
    {"path": "website/docs/user-guide/features/personality.md", "type": "blob"},
    {"path": "website/docs/user-guide/features/memory-providers.md", "type": "blob"},
    {"path": "hermes_cli/plugins.py", "type": "blob"},
    {"path": "model_tools.py", "type": "blob"},
    {"path": "RELEASE_v0.4.0.md", "type": "blob"},
]

PUBLIC_CONTENT = {
    "website/docs/user-guide/features/plugins.md": dedent(
        """
        ---
        title: Plugins
        ---
        # Plugins

        Hermes has a plugin system for adding custom tools, hooks, and integrations without modifying core code.

        ## Quick overview
        Drop a directory into ~/.hermes/plugins/ with a plugin.yaml and Python code.
        """
    ).strip(),
    "website/docs/user-guide/features/skins.md": dedent(
        """
        # Skins & Themes

        Skins control the visual presentation of the Hermes CLI.
        Personality changes tone and wording.
        """
    ).strip(),
    "website/docs/user-guide/features/personality.md": dedent(
        """
        # Personality & SOUL.md

        Personality controls how Hermes sounds, while skins control the visual presentation.
        """
    ).strip(),
    "website/docs/user-guide/features/memory-providers.md": dedent(
        """
        # Memory Providers

        Hermes Agent ships with 8 external memory provider plugins.
        """
    ).strip(),
    "hermes_cli/plugins.py": dedent(
        '''
        class PluginManager:
            """Central manager that discovers, loads, and invokes plugins."""

        def discover_plugins():
            pass
        '''
    ).strip(),
    "model_tools.py": dedent(
        """
        # Plugin tool discovery (user/project/pip plugins)
        from hermes_cli.plugins import discover_plugins
        discover_plugins()
        """
    ).strip(),
    "RELEASE_v0.4.0.md": dedent(
        """
        # Release v0.4.0

        - hermes plugins install/remove/list commands
        - Slash command registration for plugins
        """
    ).strip(),
}


def _mock_fetch_text(path: str, *, prefer_cache: bool = True) -> str:
    return PUBLIC_CONTENT[path]


class TestHermesDocsLookupTool:
    def test_prefers_public_docs_page_over_code_and_release_notes(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        with (
            patch("tools.hermes_docs_lookup_tool._fetch_upstream_tree", return_value=PUBLIC_TREE),
            patch("tools.hermes_docs_lookup_tool._fetch_upstream_text", side_effect=_mock_fetch_text),
        ):
            data = json.loads(hermes_docs_lookup_tool("how do plugins work", scope="docs_and_code", limit=3))

        assert data["success"] is True
        assert data["corpus"] == "public_hermes_main"
        assert data["results"][0]["path"] == "website/docs/user-guide/features/plugins.md"
        assert data["results"][0]["source_kind"] == "public_docs"
        assert data["results"][0]["heading"] == "Plugins"
        assert "plugin system" in data["results"][0]["snippet"].lower()
        assert data["results"][0]["url"].startswith("https://hermes-agent.nousresearch.com/docs/")

    def test_returns_public_code_result_for_symbol_query_when_scope_includes_code(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        with (
            patch("tools.hermes_docs_lookup_tool._fetch_upstream_tree", return_value=PUBLIC_TREE),
            patch("tools.hermes_docs_lookup_tool._fetch_upstream_text", side_effect=_mock_fetch_text),
        ):
            data = json.loads(hermes_docs_lookup_tool("discover_plugins", scope="docs_and_code", limit=5))

        assert data["success"] is True
        assert any(
            r["path"] == "hermes_cli/plugins.py" and r["source_kind"] == "public_code"
            for r in data["results"]
        )

    def test_handles_concept_query_that_maps_to_multiple_docs_pages(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        with (
            patch("tools.hermes_docs_lookup_tool._fetch_upstream_tree", return_value=PUBLIC_TREE),
            patch("tools.hermes_docs_lookup_tool._fetch_upstream_text", side_effect=_mock_fetch_text),
        ):
            data = json.loads(hermes_docs_lookup_tool("difference between personality and skin", limit=5))

        assert data["success"] is True
        top_paths = [r["path"] for r in data["results"][:3]]
        assert "website/docs/user-guide/features/skins.md" in top_paths
        assert "website/docs/user-guide/features/personality.md" in top_paths

    def test_invalid_scope_returns_error_json(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        data = json.loads(hermes_docs_lookup_tool("plugins", scope="nope"))
        assert "error" in data
