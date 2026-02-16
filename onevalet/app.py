"""
OneValet Application - Single entry point for the AI agent system.

Usage:
    from onevalet import OneValet

    app = OneValet("config.yaml")

    # Personal deployment
    result = await app.chat("What's the weather in Tokyo?")

    # Multi-tenant
    result = await app.chat("user1", "What's the weather in Tokyo?")
"""

import logging
import os
import re
from typing import Any, AsyncIterator, Dict, List, Optional

from .result import AgentResult
from .streaming.models import AgentEvent

logger = logging.getLogger(__name__)

# Provider -> default env var name for API key
_PROVIDER_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
    "dashscope": "DASHSCOPE_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "ollama": None,  # No API key needed
}

# Provider -> LLM client class import path
_PROVIDER_CLIENTS = {
    "openai": ("onevalet.llm.openai_client", "OpenAIClient"),
    "anthropic": ("onevalet.llm.anthropic_client", "AnthropicClient"),
    "azure": ("onevalet.llm.azure_client", "AzureOpenAIClient"),
    "dashscope": ("onevalet.llm.dashscope_client", "DashScopeClient"),
    "gemini": ("onevalet.llm.gemini_client", "GeminiClient"),
    "ollama": ("onevalet.llm.ollama_client", "OllamaClient"),
}


def _load_config(path: str) -> dict:
    """Read YAML config file with ${VAR} environment variable substitution."""
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "pyyaml is required for config file loading. "
            "Install with: pip install pyyaml"
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Replace ${VAR} with environment variable values
    def _replace_env(match):
        var_name = match.group(1)
        value = os.environ.get(var_name)
        if value is None:
            raise ValueError(
                f"Environment variable '{var_name}' not set "
                f"(referenced in config file '{path}')"
            )
        return value

    resolved = re.sub(r"\$\{(\w+)\}", _replace_env, raw)
    return yaml.safe_load(resolved)


class OneValet:
    """
    OneValet Application entry point.

    Wraps the entire AI agent system behind a simple interface.
    Sync constructor reads config; async initialization is deferred
    to the first chat() or stream() call.

    Args:
        config: Path to YAML configuration file.

    Example:
        app = OneValet("config.yaml")
        result = await app.chat("What's the weather in Tokyo?")
    """

    def __init__(self, config: str):
        self._config = _load_config(config)
        self._initialized = False

        # Validate required fields
        if "database" not in self._config:
            raise ValueError("Missing required config field: 'database'")
        llm_cfg = self._config.get("llm", {})
        if not llm_cfg.get("provider") or not llm_cfg.get("model"):
            raise ValueError("Missing required config fields: 'llm.provider' and 'llm.model'")
        if not self._config.get("embedding"):
            raise ValueError("Missing required config field: 'embedding'")

        provider = llm_cfg["provider"]
        if provider not in _PROVIDER_CLIENTS:
            raise ValueError(
                f"Unsupported provider: '{provider}'. "
                f"Supported: {', '.join(_PROVIDER_CLIENTS.keys())}"
            )

        # Will be set during lazy initialization
        self._llm_client = None
        self._database = None
        self._credential_store = None
        self._momex = None
        self._agent_registry = None
        self._orchestrator = None
        self._event_bus = None
        self._trigger_engine = None
        self._email_handler = None

    async def _ensure_initialized(self) -> None:
        """Lazy initialization — runs once on first chat()/stream() call."""
        if self._initialized:
            return

        cfg = self._config
        llm_cfg = cfg["llm"]
        provider = llm_cfg["provider"]
        model = llm_cfg["model"]

        # 1. LLM client
        api_key = llm_cfg.get("api_key")
        if api_key is None:
            env_var = _PROVIDER_ENV_VARS.get(provider)
            if env_var:
                api_key = os.environ.get(env_var)

        client_kwargs = {"model": model}
        if api_key:
            client_kwargs["api_key"] = api_key
        if llm_cfg.get("base_url"):
            client_kwargs["base_url"] = llm_cfg["base_url"]

        module_path, class_name = _PROVIDER_CLIENTS[provider]
        import importlib
        mod = importlib.import_module(module_path)
        ClientClass = getattr(mod, class_name)
        self._llm_client = ClientClass(**client_kwargs)
        logger.info(f"LLM client: provider={provider}, model={model}, client={class_name}")

        # 2. Database + all tables
        from .db import Database, ensure_schema
        self._database = Database(dsn=cfg["database"])
        await self._database.initialize()
        await ensure_schema(self._database)

        # 3. CredentialStore
        from .credentials import CredentialStore
        self._credential_store = CredentialStore(db=self._database)

        # Set default store for AccountResolver (agents call it as classmethod)
        from .providers.email.resolver import AccountResolver
        AccountResolver.set_default_store(self._credential_store)

        # 4. MomexMemory
        from .memory.momex import MomexMemory
        momex_provider = provider
        # Map OneValet provider names to momex provider names
        if momex_provider in ("gemini", "ollama"):
            momex_provider = "openai"  # fallback: momex only supports openai/azure/anthropic/deepseek/qwen

        # Embedding config
        embedding_cfg = cfg["embedding"]
        emb_provider = embedding_cfg.get("provider", "openai")
        emb_model = embedding_cfg.get("model", "text-embedding-3-small")
        emb_api_key = embedding_cfg.get("api_key", "")
        emb_api_base = embedding_cfg.get("base_url", "")
        emb_api_version = embedding_cfg.get("api_version", "")

        self._momex = MomexMemory(
            llm_provider=momex_provider,
            llm_model=model,
            llm_api_key=api_key or "",
            llm_api_base=llm_cfg.get("base_url", ""),
            database_url=cfg["database"],
            embedding_provider=emb_provider,
            embedding_model=emb_model,
            embedding_api_key=emb_api_key,
            embedding_api_base=emb_api_base,
            embedding_api_version=emb_api_version,
        )

        # 5. Agent discovery — scan builtin_agents
        from .agents.discovery import AgentDiscovery
        discovery = AgentDiscovery()
        discovery.scan_package("onevalet.builtin_agents")
        discovery.sync_from_global_registry()
        logger.info(
            f"Discovered {len(discovery.get_discovered_agents())} builtin agents"
        )

        # 6. Register builtin tools
        from .builtin_agents.tools import register_all_builtin_tools
        register_all_builtin_tools()

        # 7. AgentRegistry
        from .config import AgentRegistry
        self._agent_registry = AgentRegistry()
        await self._agent_registry.initialize()

        # Register LLM as default in LLMRegistry
        from .llm.registry import LLMRegistry
        llm_registry = LLMRegistry.get_instance()
        llm_registry.register("default", self._llm_client)
        llm_registry.set_default("default")

        # 7. TriggerEngine + EventBus + Notifications
        from .triggers import (
            TriggerEngine, EventBus, OrchestratorExecutor,
            PipelineExecutor, CallbackNotification, EmailEventHandler,
        )

        # EventBus (Redis Streams) — optional, requires redis config
        redis_url = cfg.get("redis", {}).get("url") if isinstance(cfg.get("redis"), dict) else cfg.get("redis")
        if redis_url:
            self._event_bus = EventBus(redis_url=redis_url)
            await self._event_bus.initialize()
            logger.info(f"EventBus initialized (redis: {redis_url})")

        # TriggerEngine
        self._trigger_engine = TriggerEngine()

        # CallbackNotification — if callbacks.notify_url configured
        callback_url = cfg.get("callbacks", {}).get("notify_url") if isinstance(cfg.get("callbacks"), dict) else None
        callback_notification = None
        if callback_url:
            callback_notification = CallbackNotification(callback_url=callback_url)
            self._trigger_engine._notifications.append(callback_notification)
            logger.info(f"CallbackNotification configured: {callback_url}")

        # 8. Orchestrator
        from .orchestrator import Orchestrator
        self._orchestrator = Orchestrator(
            momex=self._momex,
            llm_client=self._llm_client,
            agent_registry=self._agent_registry,
            credential_store=self._credential_store,
            database=self._database,
            system_prompt=cfg.get("system_prompt", ""),
            trigger_engine=self._trigger_engine,
        )
        await self._orchestrator.initialize()

        # Register executors with TriggerEngine
        orchestrator_executor = OrchestratorExecutor(self._orchestrator)
        self._trigger_engine.register_executor("orchestrator", orchestrator_executor)

        pipeline_executor = PipelineExecutor(
            orchestrator=self._orchestrator,
            llm_client=self._llm_client,
            notification=callback_notification,
        )
        self._trigger_engine.register_executor("pipeline", pipeline_executor)

        # EmailEventHandler — if EventBus and callback_url are configured
        if self._event_bus and callback_url:
            self._email_handler = EmailEventHandler(
                llm_client=self._llm_client,
                event_bus=self._event_bus,
                callback_url=callback_url,
            )
            await self._email_handler.start()
            logger.info("EmailEventHandler started")

        # 9. Load API key credentials into env vars for agent access
        await self._load_api_keys_to_env()

        self._initialized = True
        logger.info("OneValet initialized")

    _API_KEY_ENV_MAP = {
        "amadeus": {"api_key": "AMADEUS_API_KEY", "api_secret": "AMADEUS_API_SECRET"},
        "weather_api": {"api_key": "WEATHER_API_KEY"},
        "google_api": {
            "api_key": ["GOOGLE_MAPS_API_KEY", "GOOGLE_SEARCH_API_KEY"],
            "search_engine_id": "GOOGLE_SEARCH_ENGINE_ID",
        },
        "google_oauth_app": {"client_id": "GOOGLE_CLIENT_ID", "client_secret": "GOOGLE_CLIENT_SECRET"},
        "microsoft_oauth_app": {
            "client_id": "MICROSOFT_CLIENT_ID",
            "client_secret": "MICROSOFT_CLIENT_SECRET",
            "tenant_id": "MICROSOFT_TENANT_ID",
        },
        "notion": {"api_key": "NOTION_API_KEY"},
        "composio": {"api_key": "COMPOSIO_API_KEY"},
    }

    async def _load_api_keys_to_env(self) -> None:
        """Load static API key credentials from config file into env vars.

        Reads the ``credentials`` section of config.yaml. These are static
        secrets (API keys, client IDs) that don't change at runtime.
        Dynamic OAuth tokens are managed separately by the OAuth handlers.
        """
        file_creds = self._config.get("credentials", {})
        for service, mapping in self._API_KEY_ENV_MAP.items():
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
            logger.debug(f"Loaded {service} credentials from config")

    @property
    def config(self) -> dict:
        """Return a copy of the raw configuration dict."""
        return dict(self._config)

    async def shutdown(self) -> None:
        """Shut down the application, closing all connections."""
        if not self._initialized:
            return
        try:
            if self._orchestrator:
                await self._orchestrator.shutdown()
            if self._event_bus:
                await self._event_bus.close()
            if self._database:
                await self._database.close()
        except Exception as e:
            logger.warning(f"Error during shutdown: {e}")
        finally:
            self._initialized = False
            self._llm_client = None
            self._database = None
            self._credential_store = None
            self._momex = None
            self._agent_registry = None
            self._orchestrator = None
            self._event_bus = None
            self._trigger_engine = None
            self._email_handler = None
            logger.info("OneValet shut down")

    async def chat(
        self,
        message_or_tenant_id: str,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """
        Send a message and get a response.

        Can be called two ways:
            app.chat("Hello!")                    # personal (tenant_id="default")
            app.chat("user1", "Hello!")           # multi-tenant

        Args:
            message_or_tenant_id: The message (single-arg) or tenant_id (two-arg).
            message: The message when using multi-tenant mode.
            metadata: Optional metadata dict passed to the orchestrator.

        Returns:
            AgentResult with the response.
        """
        if message is None:
            tenant_id = "default"
            actual_message = message_or_tenant_id
        else:
            tenant_id = message_or_tenant_id
            actual_message = message

        await self._ensure_initialized()
        return await self._orchestrator.handle_message(
            tenant_id=tenant_id,
            message=actual_message,
            metadata=metadata,
        )

    async def stream(
        self,
        message_or_tenant_id: str,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[AgentEvent]:
        """
        Send a message and stream the response.

        Can be called two ways:
            async for event in app.stream("Hello!"): ...
            async for event in app.stream("user1", "Hello!"): ...

        Args:
            message_or_tenant_id: The message (single-arg) or tenant_id (two-arg).
            message: The message when using multi-tenant mode.
            metadata: Optional metadata dict passed to the orchestrator.

        Returns:
            AsyncIterator of AgentEvent.
        """
        if message is None:
            tenant_id = "default"
            actual_message = message_or_tenant_id
        else:
            tenant_id = message_or_tenant_id
            actual_message = message

        await self._ensure_initialized()
        async for event in self._orchestrator.stream_message(
            tenant_id=tenant_id,
            message=actual_message,
            metadata=metadata,
        ):
            yield event
