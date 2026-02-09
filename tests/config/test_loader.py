"""
Tests for FlowAgents Config Loader
"""

import pytest
import tempfile
from pathlib import Path

from flowagents.config import ConfigLoader


class TestConfigLoaderFileName:
    """Tests for config file name"""

    def test_loads_flowagents_yaml(self):
        """Verify the loader loads flowagents.yaml"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "flowagents.yaml"
            config_file.write_text("""
llm:
  default: main
  routing: main
  providers:
    main:
      provider: openai
      model: gpt-4
      api_key: test-key
""")

            loader = ConfigLoader(config_dir=tmpdir)
            loader.load()

            assert loader.get_default_llm_name() == "main"
            assert loader.get_routing_llm_name() == "main"


class TestConfigLoaderLLM:
    """Tests for LLM configuration loading"""

    def test_load_llm_providers(self):
        """Test loading LLM providers from config"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "flowagents.yaml"
            config_file.write_text("""
llm:
  default: main
  routing: quick
  providers:
    main:
      provider: openai
      model: gpt-4o
      api_key: sk-main
      temperature: 0.7
    quick:
      provider: openai
      model: gpt-4o-mini
      api_key: sk-quick
""")

            loader = ConfigLoader(config_dir=tmpdir)
            loader.load()

            assert loader.get_default_llm_name() == "main"
            assert loader.get_routing_llm_name() == "quick"

            providers = loader.get_all_llm_providers()
            assert "main" in providers
            assert "quick" in providers
            assert providers["main"].model == "gpt-4o"
            assert providers["quick"].model == "gpt-4o-mini"

    def test_llm_default_must_exist_in_providers(self):
        """Test that default LLM must exist in providers"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "flowagents.yaml"
            config_file.write_text("""
llm:
  default: nonexistent
  providers:
    main:
      provider: openai
      model: gpt-4
      api_key: test
""")

            loader = ConfigLoader(config_dir=tmpdir)

            with pytest.raises(ValueError, match="default.*not found"):
                loader.load()

    def test_llm_routing_must_exist_in_providers(self):
        """Test that routing LLM must exist in providers"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "flowagents.yaml"
            config_file.write_text("""
llm:
  default: main
  routing: nonexistent
  providers:
    main:
      provider: openai
      model: gpt-4
      api_key: test
""")

            loader = ConfigLoader(config_dir=tmpdir)

            with pytest.raises(ValueError, match="routing.*not found"):
                loader.load()


class TestConfigLoaderTools:
    """Tests for tool configuration loading"""

    def test_load_tools(self):
        """Test loading tool definitions"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "flowagents.yaml"
            config_file.write_text("""
tools:
  search:
    description: Search the web
    module: myapp.tools
    function: search
    parameters:
      type: object
      properties:
        query:
          type: string
      required:
        - query
""")

            loader = ConfigLoader(config_dir=tmpdir)
            loader.load()

            tools = loader.get_all_tools()
            assert "search" in tools
            assert tools["search"].description == "Search the web"


class TestConfigLoaderMCP:
    """Tests for MCP server configuration loading"""

    def test_load_mcp_servers(self):
        """Test loading MCP server definitions"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "flowagents.yaml"
            config_file.write_text("""
mcp_servers:
  filesystem:
    transport: stdio
    command: npx
    args:
      - "-y"
      - "@anthropic/mcp-server-filesystem"
    enabled: true
""")

            loader = ConfigLoader(config_dir=tmpdir)
            loader.load()

            mcp = loader.get_all_mcp_servers()
            assert "filesystem" in mcp
            assert mcp["filesystem"].command == "npx"
            assert mcp["filesystem"].enabled is True
