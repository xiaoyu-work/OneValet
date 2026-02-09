"""
Tests for FlowAgent Memory System (mem0-based).

Tests use mocking since mem0 requires external services.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from flowagents.memory import (
    MemoryConfig,
    RecallResult,
    StoreResult,
    MemoryManager,
    MemoryMixin,
    configure_memory,
)


# ============================================================================
# Test MemoryConfig
# ============================================================================

class TestMemoryConfig:
    """Tests for MemoryConfig"""

    def test_default_config(self):
        """Test default configuration"""
        config = MemoryConfig()

        assert config.enabled is False
        assert config.use_platform is False
        assert config.auto_recall is True
        assert config.auto_store is True

    def test_from_bool(self):
        """Test creating config from boolean"""
        config = MemoryConfig.from_dict(True)

        assert config.enabled is True

    def test_from_dict(self):
        """Test creating config from dictionary"""
        config = MemoryConfig.from_dict({
            "enabled": True,
            "use_platform": True,
            "api_key": "test-key",
            "remember_fields": ["email", "name"],
        })

        assert config.enabled is True
        assert config.use_platform is True
        assert config.api_key == "test-key"
        assert config.remember_fields == ["email", "name"]

    def test_should_remember(self):
        """Test field filtering"""
        config = MemoryConfig(
            remember_fields=["email", "name"],
            exclude_fields=["password"]
        )

        assert config.should_remember("email") is True
        assert config.should_remember("name") is True
        assert config.should_remember("phone") is False  # Not in remember_fields

        # Test exclude
        config2 = MemoryConfig(exclude_fields=["password"])
        assert config2.should_remember("email") is True
        assert config2.should_remember("password") is False

        # Test internal fields
        assert config2.should_remember("_internal") is False

    def test_to_mem0_config(self):
        """Test converting to mem0 config format"""
        config = MemoryConfig(
            llm_provider="openai",
            llm_model="gpt-4",
            vector_store_provider="qdrant",
            vector_store_config={"host": "localhost", "port": 6333}
        )

        mem0_config = config.to_mem0_config()

        assert mem0_config["llm"]["provider"] == "openai"
        assert mem0_config["llm"]["config"]["model"] == "gpt-4"
        assert mem0_config["vector_store"]["provider"] == "qdrant"


# ============================================================================
# Test RecallResult and StoreResult
# ============================================================================

class TestRecallResult:
    """Tests for RecallResult"""

    def test_empty_result(self):
        """Test empty recall result"""
        result = RecallResult()

        assert result.any_recalled is False
        assert result.memories == []
        assert result.recalled_fields == {}

    def test_with_memories(self):
        """Test recall result with memories"""
        result = RecallResult(
            memories=[{"id": "1", "memory": "User email is test@example.com"}],
            recalled_fields={"email": "test@example.com"}
        )

        assert result.any_recalled is True
        assert len(result.memories) == 1


class TestStoreResult:
    """Tests for StoreResult"""

    def test_success(self):
        """Test successful store"""
        result = StoreResult(stored_count=2, memory_ids=["id1", "id2"])

        assert result.success is True
        assert result.stored_count == 2

    def test_with_errors(self):
        """Test store with errors"""
        result = StoreResult(errors=["Failed to store"])

        assert result.success is False


# ============================================================================
# Test MemoryManager (with mocked mem0)
# ============================================================================

class TestMemoryManagerDisabled:
    """Tests for MemoryManager when disabled"""

    def test_disabled_add(self):
        """Test add when disabled"""
        config = MemoryConfig(enabled=False)
        manager = MemoryManager(config)

        result = manager.add_text("test", user_id="user_1")

        assert result.stored_count == 0

    def test_disabled_search(self):
        """Test search when disabled"""
        config = MemoryConfig(enabled=False)
        manager = MemoryManager(config)

        result = manager.search("test", user_id="user_1")

        assert result.any_recalled is False

    def test_disabled_auto_recall(self):
        """Test auto_recall when disabled"""
        config = MemoryConfig(enabled=False)
        manager = MemoryManager(config)

        result = manager.auto_recall("user_1", ["email", "name"])

        assert result.any_recalled is False

    def test_disabled_auto_store(self):
        """Test auto_store when disabled"""
        config = MemoryConfig(enabled=False)
        manager = MemoryManager(config)

        result = manager.auto_store("user_1", "TestAgent", {"email": "test@example.com"})

        assert result.stored_count == 0


class TestMemoryManagerMocked:
    """Tests for MemoryManager with mocked mem0"""

    @patch('flowagents.memory.manager.MemoryManager._init_mem0')
    def test_add_text_platform(self, mock_init):
        """Test add_text with platform version"""
        config = MemoryConfig(enabled=True, use_platform=True, api_key="test-key")
        manager = MemoryManager(config)

        # Mock the client
        manager._client = Mock()
        manager._client.add.return_value = {
            "results": [{"id": "mem_123", "memory": "test"}]
        }

        result = manager.add_text("User email is test@example.com", user_id="user_1")

        assert result.stored_count == 1
        assert "mem_123" in result.memory_ids
        manager._client.add.assert_called_once()

    @patch('flowagents.memory.manager.MemoryManager._init_mem0')
    def test_add_messages(self, mock_init):
        """Test add with conversation messages"""
        config = MemoryConfig(enabled=True, use_platform=True, api_key="test-key")
        manager = MemoryManager(config)

        manager._client = Mock()
        manager._client.add.return_value = {
            "results": [{"id": "mem_456"}]
        }

        messages = [
            {"role": "user", "content": "My email is alice@example.com"},
            {"role": "assistant", "content": "Got it, I'll remember your email."}
        ]

        result = manager.add(messages, user_id="user_1", agent_type="TestAgent")

        assert result.stored_count == 1
        manager._client.add.assert_called_once()

    @patch('flowagents.memory.manager.MemoryManager._init_mem0')
    def test_search(self, mock_init):
        """Test search"""
        config = MemoryConfig(enabled=True, use_platform=True, api_key="test-key")
        manager = MemoryManager(config)

        manager._client = Mock()
        manager._client.search.return_value = {
            "results": [
                {"id": "mem_1", "memory": "User email is alice@example.com", "score": 0.9}
            ]
        }

        result = manager.search("email", user_id="user_1")

        assert result.any_recalled is True
        assert len(result.memories) == 1
        manager._client.search.assert_called_once()

    @patch('flowagents.memory.manager.MemoryManager._init_mem0')
    def test_get_all(self, mock_init):
        """Test get_all"""
        config = MemoryConfig(enabled=True, use_platform=True, api_key="test-key")
        manager = MemoryManager(config)

        manager._client = Mock()
        manager._client.get_all.return_value = {
            "results": [
                {"id": "mem_1", "memory": "Memory 1"},
                {"id": "mem_2", "memory": "Memory 2"},
            ]
        }

        result = manager.get_all(user_id="user_1")

        assert len(result.memories) == 2

    @patch('flowagents.memory.manager.MemoryManager._init_mem0')
    def test_delete(self, mock_init):
        """Test delete"""
        config = MemoryConfig(enabled=True, use_platform=True, api_key="test-key")
        manager = MemoryManager(config)

        manager._client = Mock()

        result = manager.delete("mem_123")

        assert result is True
        manager._client.delete.assert_called_once_with("mem_123")

    @patch('flowagents.memory.manager.MemoryManager._init_mem0')
    def test_delete_all(self, mock_init):
        """Test delete_all"""
        config = MemoryConfig(enabled=True, use_platform=True, api_key="test-key")
        manager = MemoryManager(config)

        manager._client = Mock()

        result = manager.delete_all(user_id="user_1")

        assert result is True
        manager._client.delete_all.assert_called_once()

    @patch('flowagents.memory.manager.MemoryManager._init_mem0')
    def test_auto_recall(self, mock_init):
        """Test auto_recall"""
        config = MemoryConfig(enabled=True, use_platform=True, api_key="test-key")
        manager = MemoryManager(config)

        manager._client = Mock()
        manager._client.search.return_value = {
            "results": [
                {"id": "mem_1", "memory": "User's email is alice@example.com"}
            ]
        }

        result = manager.auto_recall("user_1", ["email", "name"])

        assert result.any_recalled is True
        assert "email" in result.recalled_fields

    @patch('flowagents.memory.manager.MemoryManager._init_mem0')
    def test_auto_store(self, mock_init):
        """Test auto_store"""
        config = MemoryConfig(enabled=True, use_platform=True, api_key="test-key")
        manager = MemoryManager(config)

        manager._client = Mock()
        manager._client.add.return_value = {
            "results": [{"id": "mem_new"}]
        }

        result = manager.auto_store(
            user_id="user_1",
            agent_type="SendEmailAgent",
            fields={"email": "alice@example.com", "name": "Alice"}
        )

        assert result.stored_count == 1
        manager._client.add.assert_called_once()

    @patch('flowagents.memory.manager.MemoryManager._init_mem0')
    def test_auto_store_filters_fields(self, mock_init):
        """Test that auto_store respects remember_fields config"""
        config = MemoryConfig(
            enabled=True,
            use_platform=True,
            api_key="test-key",
            remember_fields=["email"]  # Only remember email
        )
        manager = MemoryManager(config)

        manager._client = Mock()
        manager._client.add.return_value = {"results": [{"id": "mem_1"}]}

        manager.auto_store(
            user_id="user_1",
            agent_type="TestAgent",
            fields={"email": "test@example.com", "password": "secret123"}
        )

        # Should only store email, not password
        call_args = manager._client.add.call_args
        stored_text = call_args[0][0]  # First positional argument
        assert "email" in stored_text.lower()
        assert "password" not in stored_text.lower()


# ============================================================================
# Test MemoryMixin
# ============================================================================

class TestMemoryMixin:
    """Tests for MemoryMixin"""

    @patch('flowagents.memory.manager.MemoryManager._init_mem0')
    def test_mixin_recall(self, mock_init):
        """Test mixin auto-recall"""

        class TestAgent(MemoryMixin):
            def __init__(self):
                self.user_id = "user_1"
                self.agent_type = "TestAgent"
                self.required_fields = []

        config = MemoryConfig(enabled=True, use_platform=True, api_key="test-key")
        manager = MemoryManager(config)
        manager._client = Mock()
        manager._client.search.return_value = {"results": []}

        agent = TestAgent()
        agent.set_memory_manager(manager)

        result = agent._auto_recall_fields(["email"])

        assert isinstance(result, RecallResult)

    @patch('flowagents.memory.manager.MemoryManager._init_mem0')
    def test_mixin_store(self, mock_init):
        """Test mixin auto-store"""

        class TestAgent(MemoryMixin):
            def __init__(self):
                self.user_id = "user_1"
                self.agent_type = "TestAgent"
                self.collected_fields = {"email": "test@example.com"}

        config = MemoryConfig(enabled=True, use_platform=True, api_key="test-key")
        manager = MemoryManager(config)
        manager._client = Mock()
        manager._client.add.return_value = {"results": [{"id": "mem_1"}]}

        agent = TestAgent()
        agent.set_memory_manager(manager)

        result = agent._auto_store_fields()

        assert isinstance(result, StoreResult)
        manager._client.add.assert_called_once()

    def test_mixin_no_manager(self):
        """Test mixin without manager set"""

        class TestAgent(MemoryMixin):
            def __init__(self):
                self.user_id = "user_1"

        agent = TestAgent()

        result = agent._auto_recall_fields(["email"])
        assert result.any_recalled is False

        result = agent._auto_store_fields()
        assert result.success is True  # No errors, just empty


# ============================================================================
# Test Self-Hosted Version
# ============================================================================

class TestMemoryManagerSelfHosted:
    """Tests for self-hosted mem0 version"""

    @patch('flowagents.memory.manager.MemoryManager._init_mem0')
    def test_self_hosted_add(self, mock_init):
        """Test add with self-hosted version"""
        config = MemoryConfig(
            enabled=True,
            use_platform=False,
            vector_store_provider="qdrant",
            vector_store_config={"host": "localhost", "port": 6333}
        )
        manager = MemoryManager(config)

        # Mock the Memory object
        manager._memory = Mock()
        manager._memory.add.return_value = {
            "results": [{"id": "mem_local_1"}]
        }

        result = manager.add_text("Test memory", user_id="user_1")

        assert result.stored_count == 1
        manager._memory.add.assert_called_once()

    @patch('flowagents.memory.manager.MemoryManager._init_mem0')
    def test_self_hosted_search(self, mock_init):
        """Test search with self-hosted version"""
        config = MemoryConfig(enabled=True, use_platform=False)
        manager = MemoryManager(config)

        manager._memory = Mock()
        manager._memory.search.return_value = [
            {"id": "mem_1", "memory": "Test memory"}
        ]

        result = manager.search("test", user_id="user_1")

        assert result.any_recalled is True
        manager._memory.search.assert_called_once()


# ============================================================================
# Test configure_memory
# ============================================================================

class TestConfigureMemory:
    """Tests for configure_memory function"""

    @patch('flowagents.memory.manager.MemoryManager._init_mem0')
    def test_configure_memory(self, mock_init):
        """Test global memory configuration"""
        from flowagents.memory import get_memory_manager

        config = MemoryConfig(enabled=True, use_platform=True, api_key="test")
        manager = configure_memory(config)

        assert manager is not None
        assert get_memory_manager() is manager


# ============================================================================
# Test Error Handling
# ============================================================================

class TestErrorHandling:
    """Tests for error handling"""

    @patch('flowagents.memory.manager.MemoryManager._init_mem0')
    def test_add_error(self, mock_init):
        """Test error handling in add"""
        config = MemoryConfig(enabled=True, use_platform=True, api_key="test-key")
        manager = MemoryManager(config)

        manager._client = Mock()
        manager._client.add.side_effect = Exception("Connection failed")

        result = manager.add_text("test", user_id="user_1")

        assert result.success is False
        assert len(result.errors) == 1

    @patch('flowagents.memory.manager.MemoryManager._init_mem0')
    def test_search_error_returns_empty(self, mock_init):
        """Test that search errors return empty result"""
        config = MemoryConfig(enabled=True, use_platform=True, api_key="test-key")
        manager = MemoryManager(config)

        manager._client = Mock()
        manager._client.search.side_effect = Exception("Search failed")

        result = manager.search("test", user_id="user_1")

        # Should return empty result, not raise
        assert result.any_recalled is False

    @patch('flowagents.memory.manager.MemoryManager._init_mem0')
    def test_delete_error(self, mock_init):
        """Test error handling in delete"""
        config = MemoryConfig(enabled=True, use_platform=True, api_key="test-key")
        manager = MemoryManager(config)

        manager._client = Mock()
        manager._client.delete.side_effect = Exception("Delete failed")

        result = manager.delete("mem_123")

        assert result is False
