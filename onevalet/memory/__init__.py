"""
OneValet Long-Term Memory - Mem0-based memory for agents

This module provides:
- Automatic memory recall and storage using mem0
- Support for both platform and self-hosted mem0
- YAML configuration for zero-code setup

Example (Platform):
    from onevalet.memory import MemoryManager, MemoryConfig

    config = MemoryConfig(
        enabled=True,
        use_platform=True,
        api_key="your-mem0-api-key"
    )
    manager = MemoryManager(config)

    # Add memory
    manager.add_text("User's email is alice@example.com", user_id="user_1")

    # Search memory
    result = manager.search("email", user_id="user_1")

Example (Self-hosted):
    config = MemoryConfig(
        enabled=True,
        use_platform=False,
        vector_store_provider="qdrant",
        vector_store_config={"host": "localhost", "port": 6333}
    )
    manager = MemoryManager(config)

Example YAML:
    memory:
      enabled: true
      use_platform: true
      api_key: ${MEM0_API_KEY}

    # Or self-hosted:
    memory:
      enabled: true
      use_platform: false
      vector_store_provider: qdrant
      vector_store_config:
        host: localhost
        port: 6333
"""

from .models import (
    MemoryConfig,
    RecallResult,
    StoreResult,
)

from .manager import (
    MemoryManager,
    MemoryMixin,
    get_memory_manager,
    configure_memory,
)

__all__ = [
    # Models
    "MemoryConfig",
    "RecallResult",
    "StoreResult",
    # Manager
    "MemoryManager",
    "MemoryMixin",
    "get_memory_manager",
    "configure_memory",
]
