"""Pydantic request/response models for the OneValet API."""

from typing import Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    tenant_id: str = "default"
    metadata: Optional[dict] = None


class ChatResponse(BaseModel):
    response: str
    status: str


class CredentialSaveRequest(BaseModel):
    account_name: str = "primary"
    credentials: dict


class LLMConfigRequest(BaseModel):
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class EmailEventRequest(BaseModel):
    tenant_id: str
    message_id: str
    sender: str
    subject: str
    snippet: str
    date: str
    unread: bool
    labels: Optional[list] = None
    account: Optional[dict] = None  # {"provider": "...", "account_name": "...", "email": "..."}


class EmbeddingConfigRequest(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_version: Optional[str] = None


class ConfigRequest(BaseModel):
    llm: LLMConfigRequest
    database: str
    embedding: EmbeddingConfigRequest
    system_prompt: Optional[str] = None


class TaskCreateRequest(BaseModel):
    tenant_id: str = "default"
    name: str = ""
    description: str = ""
    trigger_type: str  # "schedule", "event", "condition"
    trigger_params: dict  # e.g. {"cron": "0 8 * * *"} or {"source": "email", ...}
    executor: str = "orchestrator"
    instruction: str = ""
    action_config: Optional[dict] = None
    max_runs: Optional[int] = None
    metadata: Optional[dict] = None


class TaskUpdateRequest(BaseModel):
    status: Optional[str] = None  # "active", "paused", "disabled"


# ── Cron Job Models ──

class CronJobCreateRequest(BaseModel):
    tenant_id: str = "default"
    name: str
    instruction: str = ""
    schedule_type: str  # "at", "every", "cron"
    schedule_value: str  # ISO datetime, seconds, or cron expression
    timezone: str = ""
    session_target: str = "isolated"  # "main" or "isolated"
    wake_mode: str = "next-heartbeat"
    delivery_mode: str = "none"  # "none", "announce", "webhook"
    delivery_channel: Optional[str] = None
    webhook_url: Optional[str] = None
    delete_after_run: bool = False
    metadata: Optional[dict] = None


class CronJobUpdateRequest(BaseModel):
    enabled: Optional[bool] = None
    name: Optional[str] = None
    instruction: Optional[str] = None
    schedule_type: Optional[str] = None
    schedule_value: Optional[str] = None
    timezone: Optional[str] = None
