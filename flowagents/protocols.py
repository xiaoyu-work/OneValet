"""
FlowAgent Protocols - Abstract interfaces for dependency injection

These protocols define the contracts that external implementations must fulfill.
This allows FlowAgent to be framework-agnostic and work with any LLM provider.
"""

from typing import Protocol, List, Dict, Any, Optional, runtime_checkable


@runtime_checkable
class LLMClientProtocol(Protocol):
    """
    Abstract interface for LLM clients

    Implement this protocol to integrate any LLM provider (OpenAI, Anthropic, etc.)

    Example:
        class MyLLMClient:
            async def chat_completion(
                self,
                messages: List[Dict[str, Any]],
                tools: Optional[List[Dict]] = None,
                config: Optional[Dict] = None
            ) -> Any:
                # Your implementation
                response = await openai.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    tools=tools
                )
                return response
    """

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Call LLM for chat completion

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool schemas (OpenAI format)
            config: Optional configuration (model, temperature, etc.)

        Returns:
            Response object with choices[0].message structure
            (OpenAI-compatible format)
        """
        ...


@runtime_checkable
class MemoryProtocol(Protocol):
    """
    Abstract interface for memory/persistence

    Implement this protocol to add long-term memory capabilities
    """

    async def add(
        self,
        content: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a memory entry"""
        ...

    async def search(
        self,
        query: str,
        user_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search memories"""
        ...

    async def get_all(
        self,
        user_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get all memories for a user"""
        ...


@runtime_checkable
class ConfigLoaderProtocol(Protocol):
    """
    Abstract interface for loading agent configurations

    Implement this to load agent configs from YAML, database, etc.
    """

    def load_agent_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all agent configurations

        Returns:
            Dict mapping agent_type to config dict
        """
        ...

    def load_group_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Load agent group configurations

        Returns:
            Dict mapping group_name to config dict
        """
        ...
