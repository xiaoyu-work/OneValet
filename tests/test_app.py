"""Tests for onevalet.app — config loading and env var mapping"""

import os
import pytest
import tempfile

from onevalet.app import OneValet, _load_config


# =========================================================================
# _load_config — env var substitution
# =========================================================================


class TestLoadConfig:

    def test_substitutes_env_vars(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TEST_DB_URL", "postgresql://localhost/test")
        config_file = tmp_path / "config.yaml"
        config_file.write_text("database: ${TEST_DB_URL}\n")
        cfg = _load_config(str(config_file))
        assert cfg["database"] == "postgresql://localhost/test"

    def test_multiple_substitutions(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VAR_A", "aaa")
        monkeypatch.setenv("VAR_B", "bbb")
        config_file = tmp_path / "config.yaml"
        config_file.write_text("a: ${VAR_A}\nb: ${VAR_B}\n")
        cfg = _load_config(str(config_file))
        assert cfg["a"] == "aaa"
        assert cfg["b"] == "bbb"

    def test_missing_env_var_raises(self, monkeypatch, tmp_path):
        monkeypatch.delenv("NONEXISTENT_VAR_12345", raising=False)
        config_file = tmp_path / "config.yaml"
        config_file.write_text("key: ${NONEXISTENT_VAR_12345}\n")
        with pytest.raises(ValueError, match="NONEXISTENT_VAR_12345"):
            _load_config(str(config_file))

    def test_no_substitution_needed(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("key: plain_value\n")
        cfg = _load_config(str(config_file))
        assert cfg["key"] == "plain_value"

    def test_inline_substitution(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOST", "myhost")
        config_file = tmp_path / "config.yaml"
        config_file.write_text("url: https://${HOST}.openai.azure.com/\n")
        cfg = _load_config(str(config_file))
        assert cfg["url"] == "https://myhost.openai.azure.com/"

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            _load_config("/nonexistent/path/config.yaml")

    def test_nested_yaml_structure(self, monkeypatch, tmp_path):
        monkeypatch.setenv("API_KEY", "sk-test")
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "llm:\n  provider: openai\n  api_key: ${API_KEY}\n"
        )
        cfg = _load_config(str(config_file))
        assert cfg["llm"]["api_key"] == "sk-test"


# =========================================================================
# _load_credentials_to_env — flat credential loading
# =========================================================================


class TestCredentialsLoading:
    """Test the flat credentials → env var loading."""

    @pytest.mark.asyncio
    async def test_load_credentials(self, monkeypatch):
        monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
        monkeypatch.delenv("WEATHER_API_KEY", raising=False)

        app = OneValet.__new__(OneValet)
        app._config = {"credentials": {
            "COMPOSIO_API_KEY": "test-composio-key",
            "WEATHER_API_KEY": "test-weather-key",
        }}
        await app._load_credentials_to_env()

        assert os.environ.get("COMPOSIO_API_KEY") == "test-composio-key"
        assert os.environ.get("WEATHER_API_KEY") == "test-weather-key"
        monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
        monkeypatch.delenv("WEATHER_API_KEY", raising=False)

    @pytest.mark.asyncio
    async def test_empty_credentials_no_side_effects(self, monkeypatch):
        monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)

        app = OneValet.__new__(OneValet)
        app._config = {"credentials": {}}
        await app._load_credentials_to_env()

        assert os.environ.get("COMPOSIO_API_KEY") is None

    @pytest.mark.asyncio
    async def test_empty_value_skipped(self, monkeypatch):
        monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)

        app = OneValet.__new__(OneValet)
        app._config = {"credentials": {"COMPOSIO_API_KEY": ""}}
        await app._load_credentials_to_env()

        assert os.environ.get("COMPOSIO_API_KEY") is None

    @pytest.mark.asyncio
    async def test_setdefault_does_not_overwrite(self, monkeypatch):
        """Env vars already set take precedence over config values."""
        monkeypatch.setenv("COMPOSIO_API_KEY", "from-env")

        app = OneValet.__new__(OneValet)
        app._config = {"credentials": {"COMPOSIO_API_KEY": "from-config"}}
        await app._load_credentials_to_env()

        assert os.environ.get("COMPOSIO_API_KEY") == "from-env"
        monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
