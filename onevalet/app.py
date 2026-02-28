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
        self._model_router = None
        self._cron_service = None

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

        from .llm.litellm_client import LiteLLMClient
        from .llm.base import LLMConfig
        llm_config = LLMConfig(model=model, api_key=api_key, base_url=llm_cfg.get("base_url"))
        self._llm_client = LiteLLMClient(config=llm_config, provider_name=provider)
        logger.info(f"LLM client: provider={provider}, model={model}")

        # 2. Database
        from .db import Database
        self._database = Database(dsn=cfg["database"])
        await self._database.initialize()

        # 3. CredentialStore (ensure_table creates credentials + oauth_states tables)
        from .credentials import CredentialStore
        self._credential_store = CredentialStore(db=self._database)
        await self._credential_store.initialize()

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

        # 6. AgentRegistry
        from .config import AgentRegistry
        self._agent_registry = AgentRegistry()
        await self._agent_registry.initialize()

        # Register LLM as default in LLMRegistry
        from .llm.registry import LLMRegistry
        llm_registry = LLMRegistry.get_instance()
        llm_registry.register("default", self._llm_client)
        llm_registry.set_default("default")

        # 6b. Additional LLM providers (for model routing)
        from .llm.base import LLMConfig as _LLMConfig
        from .llm.litellm_client import LiteLLMClient as _LiteLLMClient
        for name, prov_cfg in cfg.get("llm_providers", {}).items():
            if name == "default":
                continue  # already registered above
            try:
                _prov_llm_config = _LLMConfig(
                    model=prov_cfg["model"],
                    api_key=prov_cfg.get("api_key"),
                    base_url=prov_cfg.get("base_url"),
                )
                _prov_client = _LiteLLMClient(
                    config=_prov_llm_config,
                    provider_name=prov_cfg.get("provider", "openai"),
                )
                llm_registry.register(name, _prov_client)
                logger.info(f"Registered LLM provider: {name} ({prov_cfg.get('provider')}/{prov_cfg['model']})")
            except Exception as e:
                logger.warning(f"Failed to register LLM provider '{name}': {e}")

        # 6c. Model Router (complexity-based routing)
        self._model_router = None
        routing_cfg = cfg.get("model_routing", {})
        if routing_cfg.get("enabled"):
            from .llm.router import ModelRouter, RoutingRule
            rules = []
            for rule_cfg in routing_cfg.get("rules", []):
                score_range = rule_cfg.get("score_range", [0, 100])
                rules.append(RoutingRule(
                    min_score=score_range[0],
                    max_score=score_range[1],
                    provider=rule_cfg["provider"],
                ))
            self._model_router = ModelRouter(
                registry=llm_registry,
                classifier_provider=routing_cfg.get("classifier_provider", "default"),
                rules=rules or None,
                default_provider=routing_cfg.get("default_provider", "default"),
            )
            logger.info(
                f"ModelRouter enabled: classifier={routing_cfg.get('classifier_provider', 'default')}, "
                f"{len(rules or [])} rules"
            )

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

        # 8. Checkpoint storage (PostgreSQL)
        from .checkpoint import CheckpointManager, PostgreSQLStorage
        checkpoint_storage = PostgreSQLStorage(db=self._database)
        await checkpoint_storage.initialize()
        checkpoint_manager = CheckpointManager(storage=checkpoint_storage)

        # 9. Orchestrator
        from .orchestrator import Orchestrator
        self._orchestrator = Orchestrator(
            momex=self._momex,
            llm_client=self._llm_client,
            agent_registry=self._agent_registry,
            credential_store=self._credential_store,
            database=self._database,
            system_prompt=cfg.get("system_prompt", ""),
            trigger_engine=self._trigger_engine,
            model_router=self._model_router,
            checkpoint_manager=checkpoint_manager,
        )
        await self._orchestrator.initialize()

        # CronService setup
        from .triggers.cron.store import CronJobStore
        from .triggers.cron.run_log import CronRunLog
        from .triggers.cron.executor import CronExecutor as CronJobExecutor
        from .triggers.cron.delivery import CronDeliveryHandler
        from .triggers.cron.service import CronService

        cron_store_path = cfg.get("cron", {}).get("store", "~/.onevalet/cron/jobs.json") if isinstance(cfg.get("cron"), dict) else "~/.onevalet/cron/jobs.json"
        cron_store = CronJobStore(store_path=cron_store_path)
        await cron_store.load()

        cron_data_dir = str(cron_store.store_path.parent)
        cron_run_log = CronRunLog(data_dir=cron_data_dir)
        cron_delivery = CronDeliveryHandler(
            notifications=self._trigger_engine._notifications,
        )
        cron_executor = CronJobExecutor(
            orchestrator=self._orchestrator,
            store=cron_store,
            run_log=cron_run_log,
            delivery=cron_delivery,
        )
        self._cron_service = CronService(
            store=cron_store,
            executor=cron_executor,
            run_log=cron_run_log,
        )
        self._trigger_engine.set_cron_service(self._cron_service)
        logger.info(f"CronService initialized (store: {cron_store_path})")

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

        # 10. Ensure expense/budget/receipt tables exist
        try:
            from .builtin_agents.expense.repository import ExpenseRepository
            from .builtin_agents.expense.budget_repository import BudgetRepository
            from .builtin_agents.expense.receipt_repository import ReceiptRepository
            for RepoClass in (ExpenseRepository, BudgetRepository, ReceiptRepository):
                repo = RepoClass(self._database)
                await repo.ensure_table()
            logger.info("Expense/budget/receipt tables ensured")
        except Exception as e:
            logger.warning(f"Could not initialize expense tables: {e}")

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
            if self._cron_service:
                await self._cron_service.stop()
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
            self._model_router = None
            self._cron_service = None
            logger.info("OneValet shut down")

    # ── Public API methods (issue #12) ──

    async def list_credentials(self, tenant_id: str = "default", service: Optional[str] = None) -> List[dict]:
        """List stored credentials for a tenant."""
        await self._ensure_initialized()
        return await self._credential_store.list(tenant_id, service=service)

    async def save_credential(self, tenant_id: str, service: str, credentials: dict, account_name: str = "primary") -> None:
        """Save a credential entry and reload API keys into env."""
        await self._ensure_initialized()
        await self._credential_store.save(tenant_id=tenant_id, service=service, credentials=credentials, account_name=account_name)
        await self._load_api_keys_to_env()

    async def delete_credential(self, tenant_id: str, service: str, account_name: str) -> bool:
        """Delete a credential entry. Returns True if deleted."""
        await self._ensure_initialized()
        return await self._credential_store.delete(tenant_id=tenant_id, service=service, account_name=account_name)

    async def find_credential_by_email(self, email: str, service: Optional[str] = None):
        """Find credentials by email address."""
        await self._ensure_initialized()
        return await self._credential_store.find_by_email(email, service)

    async def get_credential(self, tenant_id: str, service: str, account_name: str = "primary"):
        """Get full credentials for a specific service/account."""
        await self._ensure_initialized()
        return await self._credential_store.get(tenant_id, service, account_name)

    async def save_credential_raw(self, tenant_id: str, service: str, credentials: dict, account_name: str = "primary") -> None:
        """Save credentials without reloading API keys (for internal/OAuth use)."""
        await self._ensure_initialized()
        await self._credential_store.save(tenant_id=tenant_id, service=service, credentials=credentials, account_name=account_name)

    async def save_oauth_state(self, tenant_id: str, service: str, redirect_after: Optional[str] = None, account_name: str = "primary") -> str:
        """Save OAuth state and return the state token."""
        await self._ensure_initialized()
        return await self._credential_store.save_oauth_state(
            tenant_id=tenant_id, service=service,
            redirect_after=redirect_after, account_name=account_name,
        )

    async def consume_oauth_state(self, state: str) -> Optional[dict]:
        """Consume and return OAuth state data."""
        await self._ensure_initialized()
        return await self._credential_store.consume_oauth_state(state)

    async def clear_session(self, tenant_id: str = "default") -> None:
        """Clear conversation history for a tenant."""
        await self._ensure_initialized()
        self._momex.clear_history(tenant_id=tenant_id, session_id=tenant_id)

    async def handle_message(
        self,
        tenant_id: str,
        message: str,
        images: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "AgentResult":
        """Send a message and get a response via the orchestrator."""
        await self._ensure_initialized()
        return await self._orchestrator.handle_message(
            tenant_id=tenant_id, message=message, images=images, metadata=metadata,
        )

    async def stream_message(
        self,
        tenant_id: str,
        message: str,
        images: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator["AgentEvent"]:
        """Stream a message response via the orchestrator."""
        await self._ensure_initialized()
        async for event in self._orchestrator.stream_message(
            tenant_id=tenant_id, message=message, images=images, metadata=metadata,
        ):
            yield event

    async def list_tasks(self, tenant_id: str = "default") -> list:
        """List trigger tasks for a tenant."""
        await self._ensure_initialized()
        if not self._trigger_engine:
            return []
        return await self._trigger_engine.list_tasks(user_id=tenant_id)

    async def create_task(self, **kwargs):
        """Create a new trigger task. Passes kwargs to TriggerEngine.create_task."""
        await self._ensure_initialized()
        if not self._trigger_engine:
            raise RuntimeError("TriggerEngine not available")
        return await self._trigger_engine.create_task(**kwargs)

    async def update_task(self, task_id: str, **kwargs):
        """Update a trigger task. Passes kwargs to TriggerEngine.update_task_status."""
        await self._ensure_initialized()
        if not self._trigger_engine:
            raise RuntimeError("TriggerEngine not available")
        return await self._trigger_engine.update_task_status(task_id, **kwargs)

    async def delete_task(self, task_id: str) -> bool:
        """Delete a trigger task."""
        await self._ensure_initialized()
        if not self._trigger_engine:
            raise RuntimeError("TriggerEngine not available")
        return await self._trigger_engine.delete_task(task_id)

    # ── Cron Job API ──

    @property
    def cron_service(self):
        """Access the cron service (may be None)."""
        return self._cron_service

    async def list_cron_jobs(self, tenant_id: str = "default", include_disabled: bool = False) -> list:
        """List cron jobs for a tenant."""
        await self._ensure_initialized()
        if not self._cron_service:
            return []
        return self._cron_service.list_jobs(user_id=tenant_id, include_disabled=include_disabled)

    async def get_cron_job(self, job_id: str):
        """Get a cron job by ID."""
        await self._ensure_initialized()
        if not self._cron_service:
            return None
        return self._cron_service.get_job(job_id)

    async def add_cron_job(self, **kwargs):
        """Create a new cron job."""
        await self._ensure_initialized()
        if not self._cron_service:
            raise RuntimeError("CronService not available")
        from .triggers.cron.models import CronJobCreate
        input_data = CronJobCreate(**kwargs)
        return await self._cron_service.add(input_data)

    async def remove_cron_job(self, job_id: str) -> bool:
        """Delete a cron job."""
        await self._ensure_initialized()
        if not self._cron_service:
            raise RuntimeError("CronService not available")
        return await self._cron_service.remove(job_id)

    async def cron_status(self) -> dict:
        """Get cron scheduler status."""
        await self._ensure_initialized()
        if not self._cron_service:
            return {"running": False, "total_jobs": 0}
        return await self._cron_service.status()

    async def get_config(self) -> dict:
        """Return a copy of the raw configuration dict."""
        return dict(self._config)

    async def save_config(self, config: dict) -> None:
        """Update the in-memory configuration."""
        self._config = config

    async def ingest_event(self, event) -> None:
        """Publish an event to the EventBus."""
        await self._ensure_initialized()
        if self._event_bus is None:
            raise RuntimeError("EventBus not available")
        await self._event_bus.publish(event)

    @property
    def trigger_engine(self):
        """Access the trigger engine (may be None)."""
        return self._trigger_engine

    @property
    def event_bus(self):
        """Access the event bus (may be None)."""
        return self._event_bus

    async def chat(
        self,
        message_or_tenant_id: str,
        message: Optional[str] = None,
        images: Optional[list] = None,
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
            images: Optional list of image dicts for multimodal input.
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
            images=images,
            metadata=metadata,
        )

    async def stream(
        self,
        message_or_tenant_id: str,
        message: Optional[str] = None,
        images: Optional[list] = None,
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
            images: Optional list of image dicts for multimodal input.
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
            images=images,
            metadata=metadata,
        ):
            yield event
