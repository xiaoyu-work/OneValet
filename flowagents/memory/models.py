"""
FlowAgent Long-Term Memory Models - Data structures for mem0-based memory

This module defines:
- MemoryConfig: Configuration for mem0 memory system
- RecallResult: Result from memory recall
- StoreResult: Result from memory store
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from enum import Enum


@dataclass
class MemoryConfig:
    """Configuration for mem0 memory system"""
    # Enable/disable
    enabled: bool = False

    # Mem0 settings
    api_key: Optional[str] = None  # For platform version
    use_platform: bool = False  # True = platform, False = self-hosted

    # Self-hosted config
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    embedder_provider: str = "openai"
    embedder_model: str = "text-embedding-3-small"

    # Vector store config (for self-hosted)
    vector_store_provider: str = "qdrant"
    vector_store_config: Dict[str, Any] = field(default_factory=lambda: {
        "collection_name": "flowagent_memory",
        "host": "localhost",
        "port": 6333,
    })

    # Fields to remember
    remember_fields: Optional[List[str]] = None  # None = all fields
    exclude_fields: Optional[List[str]] = None

    # Auto behaviors
    auto_recall: bool = True
    auto_store: bool = True

    @classmethod
    def from_dict(cls, d: Union[bool, Dict[str, Any]]) -> "MemoryConfig":
        """Create from dictionary or boolean"""
        if isinstance(d, bool):
            return cls(enabled=d)

        return cls(
            enabled=d.get("enabled", True),
            api_key=d.get("api_key"),
            use_platform=d.get("use_platform", False),
            llm_provider=d.get("llm_provider", "openai"),
            llm_model=d.get("llm_model", "gpt-4o-mini"),
            embedder_provider=d.get("embedder_provider", "openai"),
            embedder_model=d.get("embedder_model", "text-embedding-3-small"),
            vector_store_provider=d.get("vector_store_provider", "qdrant"),
            vector_store_config=d.get("vector_store_config", {}),
            remember_fields=d.get("remember_fields"),
            exclude_fields=d.get("exclude_fields"),
            auto_recall=d.get("auto_recall", True),
            auto_store=d.get("auto_store", True),
        )

    def should_remember(self, field_name: str) -> bool:
        """Check if a field should be remembered"""
        if field_name.startswith('_'):
            return False
        if self.exclude_fields and field_name in self.exclude_fields:
            return False
        if self.remember_fields:
            return field_name in self.remember_fields
        return True

    def to_mem0_config(self) -> Dict[str, Any]:
        """Convert to mem0 config format"""
        return {
            "llm": {
                "provider": self.llm_provider,
                "config": {
                    "model": self.llm_model,
                }
            },
            "embedder": {
                "provider": self.embedder_provider,
                "config": {
                    "model": self.embedder_model,
                }
            },
            "vector_store": {
                "provider": self.vector_store_provider,
                "config": self.vector_store_config,
            }
        }


@dataclass
class RecallResult:
    """Result from memory recall operation"""
    # Recalled memories as list of dicts
    memories: List[Dict[str, Any]] = field(default_factory=list)

    # Extracted key-value pairs
    recalled_fields: Dict[str, Any] = field(default_factory=dict)

    # Search query used
    query: Optional[str] = None

    @property
    def any_recalled(self) -> bool:
        """Check if any memories were found"""
        return len(self.memories) > 0


@dataclass
class StoreResult:
    """Result from memory store operation"""
    stored_count: int = 0
    memory_ids: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if store was successful"""
        return len(self.errors) == 0
