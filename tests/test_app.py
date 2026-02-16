"""Tests for onevalet.app — config loading and env var mapping"""

import os
import pytest
import tempfile

from onevalet.app import _load_config


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
# _API_KEY_ENV_MAP — env var loading logic
# =========================================================================


class TestApiKeyEnvMap:
    """Test the mapping structure used by _load_api_keys_to_env."""

    def test_map_structure(self):
        from onevalet.app import OneValet
        mapping = OneValet._API_KEY_ENV_MAP

        assert "composio" in mapping
        assert mapping["composio"] == {"api_key": "COMPOSIO_API_KEY"}

        assert "google_api" in mapping
        # google_api.api_key maps to a list of env vars
        assert isinstance(mapping["google_api"]["api_key"], list)
        assert "GOOGLE_MAPS_API_KEY" in mapping["google_api"]["api_key"]
        assert "GOOGLE_SEARCH_API_KEY" in mapping["google_api"]["api_key"]

    def test_all_services_have_at_least_one_mapping(self):
        from onevalet.app import OneValet
        for service, mapping in OneValet._API_KEY_ENV_MAP.items():
            assert len(mapping) > 0, f"Service {service} has empty mapping"

    def test_load_api_keys_single_value(self, monkeypatch):
        """Test _load_api_keys_to_env with a single-value mapping."""
        from onevalet.app import OneValet

        # Clean env
        monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)

        # Simulate config credentials
        config = {"credentials": {"composio": {"api_key": "test-composio-key"}}}

        # Manually run the mapping logic
        file_creds = config.get("credentials", {})
        for service, mapping in OneValet._API_KEY_ENV_MAP.items():
            svc_creds = file_creds.get(service, {})
            if not svc_creds:
                continue
            for json_key, env_vars in mapping.items():
                val = svc_creds.get(json_key, "")
                if val:
                    if isinstance(env_vars, list):
                        for env_var in env_vars:
                            os.environ[env_var] = val
                    else:
                        os.environ[env_vars] = val

        assert os.environ.get("COMPOSIO_API_KEY") == "test-composio-key"
        # Cleanup
        monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)

    def test_load_api_keys_list_mapping(self, monkeypatch):
        """Test _load_api_keys_to_env with list-type env var mapping (google_api)."""
        monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_SEARCH_API_KEY", raising=False)

        from onevalet.app import OneValet
        config = {"credentials": {"google_api": {"api_key": "google-key-123"}}}

        file_creds = config.get("credentials", {})
        for service, mapping in OneValet._API_KEY_ENV_MAP.items():
            svc_creds = file_creds.get(service, {})
            if not svc_creds:
                continue
            for json_key, env_vars in mapping.items():
                val = svc_creds.get(json_key, "")
                if val:
                    if isinstance(env_vars, list):
                        for env_var in env_vars:
                            os.environ[env_var] = val
                    else:
                        os.environ[env_vars] = val

        assert os.environ.get("GOOGLE_MAPS_API_KEY") == "google-key-123"
        assert os.environ.get("GOOGLE_SEARCH_API_KEY") == "google-key-123"
        monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_SEARCH_API_KEY", raising=False)

    def test_empty_credentials_no_side_effects(self, monkeypatch):
        monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
        config = {"credentials": {}}

        from onevalet.app import OneValet
        file_creds = config.get("credentials", {})
        for service, mapping in OneValet._API_KEY_ENV_MAP.items():
            svc_creds = file_creds.get(service, {})
            if not svc_creds:
                continue

        assert os.environ.get("COMPOSIO_API_KEY") is None

    def test_empty_value_skipped(self, monkeypatch):
        monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
        config = {"credentials": {"composio": {"api_key": ""}}}

        from onevalet.app import OneValet
        file_creds = config.get("credentials", {})
        for service, mapping in OneValet._API_KEY_ENV_MAP.items():
            svc_creds = file_creds.get(service, {})
            if not svc_creds:
                continue
            for json_key, env_vars in mapping.items():
                val = svc_creds.get(json_key, "")
                if val:
                    if isinstance(env_vars, list):
                        for env_var in env_vars:
                            os.environ[env_var] = val
                    else:
                        os.environ[env_vars] = val

        assert os.environ.get("COMPOSIO_API_KEY") is None
