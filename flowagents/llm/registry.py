"""
LLM Registry - Singleton registry for managing LLM clients

Provides centralized management of LLM clients that can be injected into agents.

Example:
    # Define providers in llm_providers.yaml
    llm_providers:
      openai_main:
        provider: openai
        model: gpt-4o
        api_key_env: OPENAI_API_KEY

      openai_fast:
        provider: openai
        model: gpt-4o-mini
        api_key_env: OPENAI_API_KEY

      deepseek:
        provider: dashscope
        model: deepseek-chat
        api_key_env: DEEPSEEK_API_KEY

    # Reference in agents.yaml
    agents:
      WeatherAgent:
        llm_provider: openai_fast

      SendEmailAgent:
        llm_provider: openai_main

    # Programmatic usage
    from flowagents.llm import LLMRegistry

    registry = LLMRegistry.get_instance()
    client = registry.get("openai_fast")
"""

import os
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass, field

from .base import BaseLLMClient, LLMConfig

logger = logging.getLogger(__name__)


@dataclass
class LLMProviderConfig:
    """Configuration for an LLM provider"""
    name: str
    provider: str  # openai, anthropic, dashscope, gemini, ollama
    model: str
    api_key_env: Optional[str] = None  # Environment variable name for API key
    api_key: Optional[str] = None  # Direct API key (not recommended)
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout: float = 60.0
    extra: Dict[str, Any] = field(default_factory=dict)


class LLMRegistry:
    """
    Singleton registry for managing LLM clients.

    Similar to ToolRegistry, provides centralized management of LLM clients
    that can be injected into agents based on configuration.

    Example:
        # Get singleton instance
        registry = LLMRegistry.get_instance()

        # Register a client
        registry.register("openai_main", OpenAIClient(model="gpt-4o"))

        # Or register from config
        registry.register_from_config(LLMProviderConfig(
            name="openai_fast",
            provider="openai",
            model="gpt-4o-mini",
            api_key_env="OPENAI_API_KEY"
        ))

        # Get a client
        client = registry.get("openai_fast")

        # Use in agent creation
        agent = MyAgent(user_id="123", llm_client=registry.get("openai_main"))
    """

    _instance: Optional["LLMRegistry"] = None

    @classmethod
    def get_instance(cls) -> "LLMRegistry":
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (mainly for testing)"""
        if cls._instance is not None:
            cls._instance.clear()
        cls._instance = None

    def __init__(self):
        self._clients: Dict[str, BaseLLMClient] = {}
        self._configs: Dict[str, LLMProviderConfig] = {}
        self._default_provider: Optional[str] = None
        self._routing_provider: Optional[str] = None

    def register(self, name: str, client: BaseLLMClient) -> None:
        """
        Register an LLM client instance.

        Args:
            name: Provider name (e.g., "openai_main", "deepseek")
            client: LLM client instance
        """
        if name in self._clients:
            logger.warning(f"Overwriting LLM client: {name}")

        self._clients[name] = client
        logger.info(f"Registered LLM client: {name}")

        # Set first registered as default
        if self._default_provider is None:
            self._default_provider = name

    def register_from_config(self, config: LLMProviderConfig) -> None:
        """
        Register an LLM client from configuration.

        Creates the appropriate client based on provider type.

        Args:
            config: LLM provider configuration
        """
        self._configs[config.name] = config

        # Resolve API key from environment variable
        api_key = config.api_key
        if config.api_key_env:
            api_key = os.environ.get(config.api_key_env)
            if not api_key:
                logger.warning(f"API key env var not set: {config.api_key_env}")

        # Create appropriate client based on provider
        client = self._create_client(config, api_key)

        if client:
            self._clients[config.name] = client
            logger.info(f"Registered LLM client from config: {config.name} ({config.provider}/{config.model})")

            if self._default_provider is None:
                self._default_provider = config.name

    def _create_client(self, config: LLMProviderConfig, api_key: Optional[str]) -> Optional[BaseLLMClient]:
        """Create LLM client based on provider type"""
        try:
            llm_config = LLMConfig(
                model=config.model,
                api_key=api_key,
                base_url=config.base_url,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout,
                extra=config.extra or {},
            )

            provider = config.provider.lower()

            if provider == "openai":
                from .openai_client import OpenAIClient
                return OpenAIClient(config=llm_config)

            elif provider == "anthropic":
                from .anthropic_client import AnthropicClient
                return AnthropicClient(config=llm_config)

            elif provider == "dashscope":
                from .dashscope_client import DashScopeClient
                return DashScopeClient(config=llm_config)

            elif provider == "gemini":
                from .gemini_client import GeminiClient
                return GeminiClient(config=llm_config)

            elif provider == "ollama":
                from .ollama_client import OllamaClient
                return OllamaClient(config=llm_config)

            elif provider == "azure":
                from .azure_client import AzureOpenAIClient
                return AzureOpenAIClient(config=llm_config)

            else:
                logger.error(f"Unknown LLM provider: {provider}")
                return None

        except ImportError as e:
            logger.error(f"Failed to import {config.provider} client: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to create {config.provider} client: {e}")
            return None

    def get(self, name: str) -> Optional[BaseLLMClient]:
        """
        Get an LLM client by name.

        Args:
            name: Provider name

        Returns:
            LLM client or None if not found
        """
        return self._clients.get(name)

    def get_default(self) -> Optional[BaseLLMClient]:
        """Get the default LLM client"""
        if self._default_provider:
            return self._clients.get(self._default_provider)
        return None

    def set_default(self, name: str) -> bool:
        """
        Set the default LLM provider for agents.

        Args:
            name: Provider name

        Returns:
            True if successful, False if provider not found
        """
        if name in self._clients:
            self._default_provider = name
            logger.info(f"Set default LLM provider: {name}")
            return True
        return False

    def get_routing(self) -> Optional[BaseLLMClient]:
        """Get the LLM client for orchestrator routing"""
        if self._routing_provider:
            return self._clients.get(self._routing_provider)
        # Fall back to default if routing not set
        return self.get_default()

    def set_routing(self, name: str) -> bool:
        """
        Set the LLM provider for orchestrator routing.

        Args:
            name: Provider name

        Returns:
            True if successful, False if provider not found
        """
        if name in self._clients:
            self._routing_provider = name
            logger.info(f"Set routing LLM provider: {name}")
            return True
        return False

    def get_or_default(self, name: Optional[str]) -> Optional[BaseLLMClient]:
        """
        Get an LLM client by name, or return default if name is None.

        Args:
            name: Provider name or None

        Returns:
            LLM client or None
        """
        if name:
            return self.get(name)
        return self.get_default()

    def list_providers(self) -> list:
        """List all registered provider names"""
        return list(self._clients.keys())

    def get_config(self, name: str) -> Optional[LLMProviderConfig]:
        """Get the config for a provider"""
        return self._configs.get(name)

    def clear(self) -> None:
        """Clear all registered clients"""
        # Close any clients that support it
        for name, client in self._clients.items():
            if hasattr(client, 'close'):
                try:
                    import asyncio
                    if asyncio.iscoroutinefunction(client.close):
                        # Try to close async
                        try:
                            loop = asyncio.get_running_loop()
                            loop.create_task(client.close())
                        except RuntimeError:
                            # No running loop, skip async close
                            pass
                    else:
                        client.close()
                except Exception as e:
                    logger.warning(f"Error closing LLM client {name}: {e}")

        self._clients.clear()
        self._configs.clear()
        self._default_provider = None
        self._routing_provider = None
        logger.info("LLM Registry cleared")

    def __contains__(self, name: str) -> bool:
        return name in self._clients

    def __len__(self) -> int:
        return len(self._clients)
