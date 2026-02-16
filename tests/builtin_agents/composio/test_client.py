"""Tests for onevalet.builtin_agents.composio.client — pure logic only"""

import pytest

from onevalet.builtin_agents.composio.client import ComposioClient


# =========================================================================
# format_action_result
# =========================================================================


class TestFormatActionResult:

    def test_successful_with_dict_data(self):
        data = {
            "successful": True,
            "data": {"id": "123", "name": "test-repo", "status": "open"},
        }
        result = ComposioClient.format_action_result(data)
        assert "id: 123" in result
        assert "name: test-repo" in result
        assert "status: open" in result

    def test_successfull_typo_key(self):
        """Composio API uses 'successfull' (with double l) in responses."""
        data = {
            "successfull": True,
            "data": {"channel": "#general"},
        }
        result = ComposioClient.format_action_result(data)
        assert "channel: #general" in result

    def test_both_keys_present(self):
        data = {
            "successful": True,
            "successfull": True,
            "data": {"key": "val"},
        }
        result = ComposioClient.format_action_result(data)
        assert "key: val" in result

    def test_long_value_truncated(self):
        data = {
            "successful": True,
            "data": {"body": "x" * 500},
        }
        result = ComposioClient.format_action_result(data)
        assert "..." in result
        # Truncated to 297 + "..."
        body_line = [line for line in result.split("\n") if "body:" in line][0]
        # "  body: " + 297 + "..." = total
        assert len(body_line) <= 310

    def test_empty_data_dict(self):
        data = {"successful": True, "data": {}}
        result = ComposioClient.format_action_result(data)
        assert result == "{}"

    def test_non_dict_data(self):
        data = {"successful": True, "data": "plain string"}
        result = ComposioClient.format_action_result(data)
        assert result == "plain string"

    def test_error_response(self):
        data = {"successful": False, "error": "Rate limit exceeded"}
        result = ComposioClient.format_action_result(data)
        assert result == "Error: Rate limit exceeded"

    def test_error_with_message_key(self):
        data = {"successful": False, "message": "Not found"}
        result = ComposioClient.format_action_result(data)
        assert result == "Error: Not found"

    def test_error_unknown(self):
        data = {"successful": False}
        result = ComposioClient.format_action_result(data)
        assert result == "Error: Unknown error"

    def test_neither_success_key(self):
        data = {"data": {"key": "val"}}
        result = ComposioClient.format_action_result(data)
        assert result == "Error: Unknown error"


# =========================================================================
# __init__ / _headers
# =========================================================================


class TestComposioClientInit:

    def test_explicit_api_key(self):
        client = ComposioClient(api_key="my-key")
        assert client.api_key == "my-key"
        assert client._headers["x-api-key"] == "my-key"

    def test_env_fallback(self, monkeypatch):
        monkeypatch.setenv("COMPOSIO_API_KEY", "env-key")
        client = ComposioClient()
        assert client.api_key == "env-key"

    def test_no_key_empty_string(self, monkeypatch):
        monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
        client = ComposioClient()
        assert client.api_key == ""

    def test_headers_structure(self):
        client = ComposioClient(api_key="test")
        headers = client._headers
        assert headers["Content-Type"] == "application/json"
        assert headers["x-api-key"] == "test"
