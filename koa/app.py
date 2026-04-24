"""
Koa Application - Single entry point for the AI agent system.

Usage:
    from koa import Koa

    app = Koa("config.yaml")

    # Personal deployment
    result = await app.chat("What's the weather in Tokyo?")

    # Multi-tenant
    result = await app.chat("user1", "What's the weather in Tokyo?")
"""

import logging
import os
import re
from typing import Any, AsyncIterator, Dict, List, Optional

from .config.schema import validate_config
from .result import AgentResult
from .streaming.models import AgentEvent

logger = logging.getLogger(__name__)


def _load_config(path: str) -> dict:
    """Read YAML config file with ${VAR} environment variable substitution."""
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "pyyaml is required for config file loading. Install with: pip install pyyaml"
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Replace ${VAR} or ${VAR:-default} with environment variable values
    # (skip comment lines)
    def _resolve_line(line: str) -> str:
        stripped = line.lstrip()
        if stripped.startswith("#"):
            return line  # skip comment lines — they may reference unused vars

        def _replace_env(match):
            var_name = match.group(1)
            default = match.group(2)  # None if no :- syntax
            value = os.environ.get(var_name)
            if value is None:
                if default is not None:
                    return default
                raise ValueError(
                    f"Environment variable '{var_name}' not set "
                    f"(referenced in config file '{path}')"
                )
            return value

        return re.sub(r"\$\{(\w+)(?::-([^}]*))?\}", _replace_env, line)

    resolved = "\n".join(_resolve_line(line) for line in raw.split("\n"))
    return yaml.safe_load(resolved)


class Koa:
    """
    Koa Application entry point.

    Wraps the entire AI agent system behind a simple interface.
    Sync constructor reads config; async initialization is deferred
    to the first chat() or stream() call.

    Args:
        config: Path to YAML configuration file.

    Example:
        app = Koa("config.yaml")
        result = await app.chat("What's the weather in Tokyo?")
    """

    def __init__(self, config: str):
        self._config = _load_config(config)
        self._initialized = False

        # Validate config schema
        errors = validate_config(self._config)
        if errors:
            for e in errors:
                logger.error(f"Config error: {e}")
            raise ValueError(f"Invalid config: {'; '.join(errors)}")

        # Keep the embedding check since it's not covered by our validation
        if not self._config.get("embedding"):
            raise ValueError("Missing required config field: 'embedding'")

        # Will be set during lazy initialization
        self._llm_client = None
        self._database = None
        self._credential_store = None
        self._momex = None
        self._agent_registry = None
        self._orchestrator = None
        self._trigger_engine = None
        self._email_handler = None
        self._model_router = None
        self._cron_service = None
        self._shipment_poller = None
        self._mcp_manager = None

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

        from .llm.base import LLMConfig
        from .llm.litellm_client import LiteLLMClient

        llm_config = LLMConfig(model=model, api_key=api_key, base_url=llm_cfg.get("base_url"))
        self._llm_client = LiteLLMClient(
            config=llm_config,
            provider_name=provider,
            api_version=llm_cfg.get("api_version"),
        )
        logger.info(f"LLM client: provider={provider}, model={model}")

        # 2. Database
        from .db import Database

        self._database = Database(dsn=cfg["database"])
        await self._database.initialize()

        # 3. CredentialStore
        from .credentials import CredentialStore

        self._credential_store = CredentialStore(db=self._database)
        await self._credential_store.initialize()

        # Set default store for AccountResolver (agents call it as classmethod)
        from .providers.email.resolver import AccountResolver

        AccountResolver.set_default_store(self._credential_store)

        # Register operator-provided image provider config (if any). Called
        # unconditionally so that reloading with `image:` removed clears stale state.
        from .providers.image.resolver import ImageProviderResolver

        ImageProviderResolver.set_global_config(cfg.get("image"))

        # 4. MomexMemory
        from .memory.momex import MomexMemory

        momex_provider = provider
        # Map Koa provider names to momex provider names
        if momex_provider in ("gemini", "ollama"):
            momex_provider = (
                "openai"  # fallback: momex only supports openai/azure/anthropic/deepseek/qwen
            )

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
        discovery.scan_package("koa.builtin_agents")
        discovery.sync_from_global_registry()
        logger.info(f"Discovered {len(discovery.get_discovered_agents())} builtin agents")

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
                logger.info(
                    f"Registered LLM provider: {name} ({prov_cfg.get('provider')}/{prov_cfg['model']})"
                )
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
                rules.append(
                    RoutingRule(
                        min_score=score_range[0],
                        max_score=score_range[1],
                        provider=rule_cfg["provider"],
                    )
                )
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

        # 7. TriggerEngine + Notifications
        from .triggers import (
            CallbackNotification,
            EmailEventHandler,
            OrchestratorExecutor,
            PipelineExecutor,
            TriggerEngine,
        )

        # TriggerEngine
        self._trigger_engine = TriggerEngine()

        # CallbackNotification — if callbacks.notify_url configured
        callback_url = (
            cfg.get("callbacks", {}).get("notify_url")
            if isinstance(cfg.get("callbacks"), dict)
            else None
        )
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
        from .orchestrator.reminder_guard import reminder_guard_hook

        self._orchestrator = Orchestrator(
            momex=self._momex,
            llm_client=self._llm_client,
            agent_registry=self._agent_registry,
            credential_store=self._credential_store,
            database=self._database,
            system_prompt=cfg.get("system_prompt", ""),
            system_prompt_mode=cfg.get("system_prompt_mode", "append"),
            trigger_engine=self._trigger_engine,
            model_router=self._model_router,
            checkpoint_manager=checkpoint_manager,
            post_process_hooks=[reminder_guard_hook],
        )
        await self._orchestrator.initialize()

        # 9b. MCP Servers (optional)
        mcp_servers_cfg = cfg.get("mcp_servers", {})
        if mcp_servers_cfg:
            try:
                from .mcp.models import MCPServerConfig, MCPTransportType
                from .mcp.provider import MCPManager
                from .mcp.sdk_client import MCPSDKClient

                # Verify mcp SDK is available before connecting servers
                try:
                    import mcp  # noqa: F401
                except ImportError:
                    raise ImportError("mcp")

                self._mcp_manager = MCPManager()
                for server_name, server_cfg in mcp_servers_cfg.items():
                    try:
                        transport = MCPTransportType(server_cfg.get("transport", "stdio"))
                        mcp_config = MCPServerConfig(
                            name=server_name,
                            transport=transport,
                            command=server_cfg.get("command"),
                            args=server_cfg.get("args", []),
                            url=server_cfg.get("url"),
                            env=server_cfg.get("env", {}),
                            headers=server_cfg.get("headers", {}),
                            timeout=float(server_cfg.get("timeout", 30.0)),
                        )
                        client = MCPSDKClient(mcp_config)
                        provider = await self._mcp_manager.add_server(client)
                        logger.info(
                            f"MCP server '{server_name}': "
                            f"{len(provider.get_tools())} tools registered"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to connect MCP server '{server_name}': {e}")

                # Inject MCP tools into orchestrator
                mcp_tools = self._mcp_manager.get_all_tools()
                if mcp_tools:
                    self._orchestrator.builtin_tools.extend(mcp_tools)
                    logger.info(f"Injected {len(mcp_tools)} MCP tools into orchestrator")
            except ImportError:
                logger.warning(
                    "mcp_servers configured but 'mcp' package not installed. "
                    "Install with: pip install mcp"
                )

        # CronService setup
        from .triggers.cron.delivery import CronDeliveryHandler
        from .triggers.cron.executor import CronExecutor as CronJobExecutor
        from .triggers.cron.pg_run_log import PostgresCronRunLog
        from .triggers.cron.pg_store import PostgresCronJobStore
        from .triggers.cron.service import CronService

        cron_store = PostgresCronJobStore(db=self._database)
        await cron_store.load()

        cron_run_log = PostgresCronRunLog(db=self._database)
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
        await self._cron_service.start()
        logger.info("CronService initialized and started (store: PostgreSQL)")

        # ShipmentPoller — DISABLED: 17TRACK webhooks handle status updates in real-time.
        # Keeping the code but not starting it to avoid unnecessary API calls.
        # from .services.shipment_poller import ShipmentPoller
        # self._shipment_poller = ShipmentPoller(db=self._database, notification=callback_notification)
        # await self._shipment_poller.start()

        # Register executors with TriggerEngine
        orchestrator_executor = OrchestratorExecutor(self._orchestrator)
        self._trigger_engine.register_executor("orchestrator", orchestrator_executor)

        pipeline_executor = PipelineExecutor(
            orchestrator=self._orchestrator,
            llm_client=self._llm_client,
            notification=callback_notification,
        )
        self._trigger_engine.register_executor("pipeline", pipeline_executor)

        # EmailEventHandler — if callback_url is configured
        if callback_url:
            self._email_handler = EmailEventHandler(
                llm_client=self._llm_client,
                callback_url=callback_url,
                database=self._database,
            )
            logger.info("EmailEventHandler initialized")

        # 9. Load API key credentials into env vars for agent access
        await self._load_credentials_to_env()

        # 10. Supabase Storage (optional — if supabase config is present)
        supabase_cfg = cfg.get("supabase")
        if isinstance(supabase_cfg, dict) and supabase_cfg.get("url"):
            from .providers.cloud_storage.supabase_storage import SupabaseStorageProvider

            self._supabase_storage = SupabaseStorageProvider(
                credentials={
                    "provider": "supabase",
                    "supabase_url": supabase_cfg["url"],
                    "supabase_key": supabase_cfg.get("service_role_key", ""),
                    "bucket": supabase_cfg.get("storage_bucket", "koa-files"),
                }
            )
            self._orchestrator._supabase_storage = self._supabase_storage
            logger.info("Supabase Storage configured")

        self._initialized = True
        logger.info("Koa initialized")

    async def _load_credentials_to_env(self) -> None:
        """Load credentials from config.yaml into environment variables.

        The ``credentials`` section is a flat dict of env-var-name → value.
        No hardcoded mapping — adding a new credential only requires a new
        line in config.yaml.
        """
        creds = self._config.get("credentials", {})
        if not isinstance(creds, dict):
            return
        for env_var, val in creds.items():
            if val and isinstance(val, str):
                os.environ.setdefault(env_var, val)
                logger.debug(f"Loaded credential {env_var} from config")

    async def _ensure_proactive_jobs(self, tenant_id: str) -> None:
        """Auto-create system proactive cron jobs for a tenant if they don't exist.

        Called when a user connects a new service (OAuth). Creates:
        - Calendar scan daily at 7am (schedules precise one-shot reminders)
        - Task check daily at 9am
        - Subscription check daily at 10am
        - Evening Summary at 9pm (F3)
        - Weekly Planning Sunday 7pm (F4)
        - Task Rollover at 8pm (F20)
        - Weather Alert at 7am/12pm/5pm (F9)

        Departure reminders, meeting prep, and other event-specific alerts are
        now created as one-shot "at" jobs when calendar events or tasks are
        created/updated (event-driven, not poll-driven).

        Each job is only created if not already present (idempotent).
        """
        if not self._cron_service:
            return

        try:
            existing = self._cron_service.list_jobs(user_id=tenant_id, include_disabled=True)
            existing_names = {j.name for j in existing}

            from .triggers.cron.models import (
                AgentTurnPayload,
                CronJobCreate,
                CronScheduleSpec,
                DeliveryConfig,
                DeliveryMode,
                SessionTarget,
                WakeMode,
            )

            jobs_to_create = []

            if "Proactive: Calendar Check" not in existing_names:
                jobs_to_create.append(
                    CronJobCreate(
                        name="Proactive: Calendar Check",
                        description="Morning scan of today's calendar; schedules precise one-shot reminders.",
                        user_id=tenant_id,
                        schedule=CronScheduleSpec(expr="0 7 * * *"),
                        session_target=SessionTarget.ISOLATED,
                        wake_mode=WakeMode.NEXT_HEARTBEAT,
                        payload=AgentTurnPayload(
                            message="Scan today's calendar events. For each event with a start time: "
                            "1) Create a reminder 30 minutes before the event with the event name, time, and meeting link. "
                            "2) If the event has a location, create a reminder 45 minutes before suggesting when to leave. "
                            "3) If the event has attendees, create a reminder 10 minutes before with attendee context. "
                            "Use the cron system to schedule each reminder as a one-shot 'at' job. "
                            "If no events today, respond with nothing_to_report."
                        ),
                        delivery=DeliveryConfig(mode=DeliveryMode.ANNOUNCE, channel="callback"),
                    )
                )

            if "Proactive: Task Check" not in existing_names:
                jobs_to_create.append(
                    CronJobCreate(
                        name="Proactive: Task Check",
                        description="Check for overdue and today-due tasks.",
                        user_id=tenant_id,
                        schedule=CronScheduleSpec(expr="0 9 * * *"),
                        session_target=SessionTarget.ISOLATED,
                        wake_mode=WakeMode.NEXT_HEARTBEAT,
                        payload=AgentTurnPayload(
                            message="Check for overdue and today-due tasks. "
                            "If any, notify the user with a brief summary. "
                            "If nothing, respond with nothing_to_report."
                        ),
                        delivery=DeliveryConfig(mode=DeliveryMode.ANNOUNCE, channel="callback"),
                    )
                )

            if "Proactive: Subscription Check" not in existing_names:
                jobs_to_create.append(
                    CronJobCreate(
                        name="Proactive: Subscription Check",
                        description="Check for subscriptions renewing in the next 3 days.",
                        user_id=tenant_id,
                        schedule=CronScheduleSpec(expr="0 10 * * *"),
                        session_target=SessionTarget.ISOLATED,
                        wake_mode=WakeMode.NEXT_HEARTBEAT,
                        payload=AgentTurnPayload(
                            message="Check for subscriptions renewing in the next 3 days. "
                            "If any, notify the user with name, price, and renewal date. "
                            "If nothing, respond with nothing_to_report."
                        ),
                        delivery=DeliveryConfig(mode=DeliveryMode.ANNOUNCE, channel="callback"),
                    )
                )

            # --- Evening Summary (F3) ---
            if "Proactive: Evening Summary" not in existing_names:
                jobs_to_create.append(
                    CronJobCreate(
                        name="Proactive: Evening Summary",
                        description="End-of-day recap and tomorrow preview.",
                        user_id=tenant_id,
                        schedule=CronScheduleSpec(expr="0 21 * * *"),
                        session_target=SessionTarget.ISOLATED,
                        wake_mode=WakeMode.NEXT_HEARTBEAT,
                        payload=AgentTurnPayload(
                            message="Generate a brief evening summary for the user. Include: "
                            "1) What they accomplished today (tasks completed, emails handled). "
                            "2) Tomorrow's first event and how many meetings they have. "
                            "3) Any overdue tasks they should tackle tomorrow. "
                            "Keep it warm and concise (3-4 lines). "
                            "If nothing notable happened and tomorrow is clear, respond with nothing_to_report."
                        ),
                        delivery=DeliveryConfig(mode=DeliveryMode.ANNOUNCE, channel="callback"),
                    )
                )

            # --- Weekly Planning (F4) ---
            if "Proactive: Weekly Planning" not in existing_names:
                jobs_to_create.append(
                    CronJobCreate(
                        name="Proactive: Weekly Planning",
                        description="Sunday evening next-week overview.",
                        user_id=tenant_id,
                        schedule=CronScheduleSpec(expr="0 19 * * 0"),
                        session_target=SessionTarget.ISOLATED,
                        wake_mode=WakeMode.NEXT_HEARTBEAT,
                        payload=AgentTurnPayload(
                            message="Generate a brief weekly preview for next week. Include: "
                            "1) Total meetings and busiest day. "
                            "2) Any deadlines or important dates. "
                            "3) Tasks carrying over from this week. "
                            "Keep it to 4-5 lines max. Be encouraging. "
                            "If next week is completely empty, say so briefly — don't skip."
                        ),
                        delivery=DeliveryConfig(mode=DeliveryMode.ANNOUNCE, channel="callback"),
                    )
                )

            # --- Task Rollover (F20) ---
            if "Proactive: Task Rollover" not in existing_names:
                jobs_to_create.append(
                    CronJobCreate(
                        name="Proactive: Task Rollover",
                        description="Evening check for incomplete today-due tasks.",
                        user_id=tenant_id,
                        schedule=CronScheduleSpec(expr="0 20 * * *"),
                        session_target=SessionTarget.ISOLATED,
                        wake_mode=WakeMode.NEXT_HEARTBEAT,
                        payload=AgentTurnPayload(
                            message="Check for tasks that were due today but not completed. "
                            "If any, list them briefly and suggest moving to tomorrow. "
                            "If all tasks are done or none were due, respond with nothing_to_report."
                        ),
                        delivery=DeliveryConfig(mode=DeliveryMode.ANNOUNCE, channel="callback"),
                    )
                )

            # --- Departure Weather Alert (F9) ---
            if "Proactive: Weather Alert" not in existing_names:
                jobs_to_create.append(
                    CronJobCreate(
                        name="Proactive: Weather Alert",
                        description="Intelligent weather alerts based on forecast changes.",
                        user_id=tenant_id,
                        schedule=CronScheduleSpec(expr="0 7,17 * * *"),
                        session_target=SessionTarget.ISOLATED,
                        wake_mode=WakeMode.NEXT_HEARTBEAT,
                        payload=AgentTurnPayload(
                            message="Check the weather forecast at the user's location for the next 24 hours. "
                            "Look for WEATHER CHANGES — specifically:\n"
                            "1. If rain/snow starts later today, tell the user WHEN it starts (e.g. 'Rain starting around 3 PM, bring an umbrella')\n"
                            "2. If temperature drops >10°C in next 12 hours, warn them\n"
                            "3. Extreme heat >35°C or cold <0°C expected\n"
                            "4. Severe weather warnings (storms, high winds)\n"
                            "If weather is stable and pleasant all day, respond with nothing_to_report.\n"
                            "Be specific about TIMING — 'rain at 3 PM' is useful, 'it might rain' is not."
                        ),
                        delivery=DeliveryConfig(mode=DeliveryMode.ANNOUNCE, channel="callback"),
                    )
                )

            # --- Habit Discovery (weekly) ---
            if "Proactive: Habit Discovery" not in existing_names:
                jobs_to_create.append(
                    CronJobCreate(
                        name="Proactive: Habit Discovery",
                        description="Weekly analysis of user behavior patterns.",
                        user_id=tenant_id,
                        schedule=CronScheduleSpec(expr="0 3 * * 0"),  # Sunday 3am
                        session_target=SessionTarget.ISOLATED,
                        wake_mode=WakeMode.NEXT_HEARTBEAT,
                        payload=AgentTurnPayload(
                            message="Analyze the user's behavior patterns from the past week. "
                            "Discover habits like active hours, most-used features, and weekly rhythm. "
                            "Store discoveries and adjust notification schedules accordingly. "
                            "Respond with nothing_to_report (this is a background analysis, don't notify the user)."
                        ),
                        delivery=DeliveryConfig(mode=DeliveryMode.NONE),
                    )
                )

            for job_input in jobs_to_create:
                await self._cron_service.add(job_input)
                logger.info(f"Auto-created proactive job '{job_input.name}' for tenant {tenant_id}")

        except Exception as e:
            logger.warning(f"Failed to ensure proactive jobs for {tenant_id}: {e}")

    @property
    def database(self):
        """Access the database instance (may be None before initialization)."""
        return self._database

    @property
    def config(self) -> dict:
        """Return a copy of the raw configuration dict."""
        return dict(self._config)

    async def shutdown(self) -> None:
        """Shut down the application, closing all connections."""
        if not self._initialized:
            return
        try:
            if self._shipment_poller:
                await self._shipment_poller.stop()
            if self._cron_service:
                await self._cron_service.stop()
            if self._mcp_manager:
                await self._mcp_manager.disconnect_all()
            if self._orchestrator:
                await self._orchestrator.shutdown()
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
            self._trigger_engine = None
            self._email_handler = None
            self._model_router = None
            self._cron_service = None
            self._shipment_poller = None
            self._mcp_manager = None
            logger.info("Koa shut down")

    # ── Public API methods (issue #12) ──

    async def list_credentials(
        self, tenant_id: str = "default", service: Optional[str] = None
    ) -> List[dict]:
        """List stored credentials for a tenant."""
        await self._ensure_initialized()
        return await self._credential_store.list(tenant_id, service=service)

    async def save_credential(
        self, tenant_id: str, service: str, credentials: dict, account_name: str = "primary"
    ) -> None:
        """Save a credential entry and reload API keys into env."""
        await self._ensure_initialized()
        await self._credential_store.save(
            tenant_id=tenant_id, service=service, credentials=credentials, account_name=account_name
        )
        await self._load_credentials_to_env()
        # Auto-register proactive cron jobs when a service is first connected
        await self._ensure_proactive_jobs(tenant_id)

    async def delete_credential(self, tenant_id: str, service: str, account_name: str) -> bool:
        """Delete a credential entry. Returns True if deleted."""
        await self._ensure_initialized()
        return await self._credential_store.delete(
            tenant_id=tenant_id, service=service, account_name=account_name
        )

    async def find_credential_by_email(
        self, email: str, service: Optional[str] = None, tenant_id: Optional[str] = None
    ):
        """Find credentials by email address, optionally scoped to a tenant."""
        await self._ensure_initialized()
        return await self._credential_store.find_by_email(email, service, tenant_id=tenant_id)

    async def list_credentials_by_service(self, service: str):
        """List all credentials for a given service across all tenants."""
        await self._ensure_initialized()
        return await self._credential_store.list_by_service(service)

    async def get_credential(self, tenant_id: str, service: str, account_name: str = "primary"):
        """Get full credentials for a specific service/account."""
        await self._ensure_initialized()
        return await self._credential_store.get(tenant_id, service, account_name)

    async def save_credential_raw(
        self, tenant_id: str, service: str, credentials: dict, account_name: str = "primary"
    ) -> None:
        """Save credentials without reloading API keys (for internal/OAuth use)."""
        await self._ensure_initialized()
        await self._credential_store.save(
            tenant_id=tenant_id, service=service, credentials=credentials, account_name=account_name
        )

    async def save_oauth_state(
        self,
        tenant_id: str,
        service: str,
        redirect_after: Optional[str] = None,
        account_name: str = "primary",
    ) -> str:
        """Save OAuth state and return the state token."""
        await self._ensure_initialized()
        return await self._credential_store.save_oauth_state(
            tenant_id=tenant_id,
            service=service,
            redirect_after=redirect_after,
            account_name=account_name,
        )

    async def consume_oauth_state(self, state: str) -> Optional[dict]:
        """Consume and return OAuth state data."""
        await self._ensure_initialized()
        return await self._credential_store.consume_oauth_state(state)

    async def clear_session(self, tenant_id: str = "default") -> None:
        """Clear conversation history for a tenant.

        Conversation history is managed by the app layer (KoiAI).
        This method is retained for API compatibility.
        """
        logger.info(f"clear_session called for {tenant_id} (history managed by app layer)")

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
            tenant_id=tenant_id,
            message=message,
            images=images,
            metadata=metadata,
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
            tenant_id=tenant_id,
            message=message,
            images=images,
            metadata=metadata,
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

    async def list_cron_jobs(
        self, tenant_id: str = "default", include_disabled: bool = False
    ) -> list:
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

    @property
    def trigger_engine(self):
        """Access the trigger engine (may be None)."""
        return self._trigger_engine

    @property
    def orchestrator(self):
        """Access the orchestrator (may be None before initialization)."""
        return self._orchestrator

    @property
    def email_handler(self):
        """Access the email handler (may be None)."""
        return self._email_handler

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
