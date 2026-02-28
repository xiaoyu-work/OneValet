"""Cron job data models — mirrors OpenClaw's cron/types.ts in Python dataclasses."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union


# ---------------------------------------------------------------------------
# Schedule types (discriminated union via "kind")
# ---------------------------------------------------------------------------

@dataclass
class AtSchedule:
    """One-shot schedule at a specific ISO timestamp."""
    kind: Literal["at"] = "at"
    at: str = ""  # ISO 8601 datetime string

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "at": self.at}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AtSchedule":
        return cls(at=d.get("at", ""))


@dataclass
class EverySchedule:
    """Recurring interval schedule."""
    kind: Literal["every"] = "every"
    every_ms: int = 0  # interval in milliseconds
    anchor_ms: Optional[int] = None  # optional anchor point

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"kind": self.kind, "everyMs": self.every_ms}
        if self.anchor_ms is not None:
            d["anchorMs"] = self.anchor_ms
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EverySchedule":
        return cls(
            every_ms=d.get("everyMs", d.get("every_ms", 0)),
            anchor_ms=d.get("anchorMs", d.get("anchor_ms")),
        )


@dataclass
class CronScheduleSpec:
    """Cron expression schedule with timezone and stagger."""
    kind: Literal["cron"] = "cron"
    expr: str = ""  # cron expression (5 or 6 fields)
    tz: Optional[str] = None  # IANA timezone (e.g. "America/Los_Angeles")
    stagger_ms: Optional[int] = None  # max stagger window in ms

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"kind": self.kind, "expr": self.expr}
        if self.tz:
            d["tz"] = self.tz
        if self.stagger_ms is not None:
            d["staggerMs"] = self.stagger_ms
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CronScheduleSpec":
        return cls(
            expr=d.get("expr", ""),
            tz=d.get("tz"),
            stagger_ms=d.get("staggerMs", d.get("stagger_ms")),
        )


Schedule = Union[AtSchedule, EverySchedule, CronScheduleSpec]


def schedule_from_dict(d: Dict[str, Any]) -> Schedule:
    """Deserialize a schedule dict by its 'kind' discriminator."""
    kind = d.get("kind", "")
    if kind == "at":
        return AtSchedule.from_dict(d)
    elif kind == "every":
        return EverySchedule.from_dict(d)
    elif kind == "cron":
        return CronScheduleSpec.from_dict(d)
    raise ValueError(f"Unknown schedule kind: {kind!r}")


def schedule_to_dict(schedule: Schedule) -> Dict[str, Any]:
    return schedule.to_dict()


# ---------------------------------------------------------------------------
# Execution modes
# ---------------------------------------------------------------------------

class SessionTarget(str, Enum):
    MAIN = "main"
    ISOLATED = "isolated"


class WakeMode(str, Enum):
    NOW = "now"
    NEXT_HEARTBEAT = "next-heartbeat"


# ---------------------------------------------------------------------------
# Payload types (discriminated union via "kind")
# ---------------------------------------------------------------------------

@dataclass
class SystemEventPayload:
    """Payload for main-session system events."""
    kind: Literal["systemEvent"] = "systemEvent"
    text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "text": self.text}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SystemEventPayload":
        return cls(text=d.get("text", ""))


@dataclass
class AgentTurnPayload:
    """Payload for isolated agent turns."""
    kind: Literal["agentTurn"] = "agentTurn"
    message: str = ""
    model: Optional[str] = None
    thinking: Optional[str] = None
    timeout_seconds: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"kind": self.kind, "message": self.message}
        if self.model:
            d["model"] = self.model
        if self.thinking:
            d["thinking"] = self.thinking
        if self.timeout_seconds is not None:
            d["timeoutSeconds"] = self.timeout_seconds
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentTurnPayload":
        return cls(
            message=d.get("message", ""),
            model=d.get("model"),
            thinking=d.get("thinking"),
            timeout_seconds=d.get("timeoutSeconds", d.get("timeout_seconds")),
        )


CronPayload = Union[SystemEventPayload, AgentTurnPayload]


def payload_from_dict(d: Dict[str, Any]) -> CronPayload:
    kind = d.get("kind", "")
    if kind == "systemEvent":
        return SystemEventPayload.from_dict(d)
    elif kind == "agentTurn":
        return AgentTurnPayload.from_dict(d)
    raise ValueError(f"Unknown payload kind: {kind!r}")


def payload_to_dict(payload: CronPayload) -> Dict[str, Any]:
    return payload.to_dict()


# ---------------------------------------------------------------------------
# Delivery
# ---------------------------------------------------------------------------

class DeliveryMode(str, Enum):
    NONE = "none"
    ANNOUNCE = "announce"
    WEBHOOK = "webhook"


class DeliveryStatus(str, Enum):
    DELIVERED = "delivered"
    NOT_DELIVERED = "not-delivered"
    UNKNOWN = "unknown"
    NOT_REQUESTED = "not-requested"


@dataclass
class DeliveryConfig:
    """Delivery configuration for cron job results."""
    mode: DeliveryMode = DeliveryMode.NONE
    channel: Optional[str] = None  # "sms", "push", "callback", or channel ID
    to: Optional[str] = None  # destination (phone, user_id, URL, etc.)
    account_id: Optional[str] = None
    best_effort: bool = False  # don't fail job if delivery fails
    webhook_url: Optional[str] = None  # URL for webhook mode
    webhook_token: Optional[str] = None  # Bearer token for webhook
    conditional: bool = False  # only deliver when agent explicitly calls notify_user

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"mode": self.mode.value}
        if self.channel:
            d["channel"] = self.channel
        if self.to:
            d["to"] = self.to
        if self.account_id:
            d["accountId"] = self.account_id
        if self.best_effort:
            d["bestEffort"] = self.best_effort
        if self.webhook_url:
            d["webhookUrl"] = self.webhook_url
        if self.webhook_token:
            d["webhookToken"] = self.webhook_token
        if self.conditional:
            d["conditional"] = self.conditional
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DeliveryConfig":
        return cls(
            mode=DeliveryMode(d.get("mode", "none")),
            channel=d.get("channel"),
            to=d.get("to"),
            account_id=d.get("accountId", d.get("account_id")),
            best_effort=d.get("bestEffort", d.get("best_effort", False)),
            webhook_url=d.get("webhookUrl", d.get("webhook_url")),
            webhook_token=d.get("webhookToken", d.get("webhook_token")),
            conditional=d.get("conditional", False),
        )


# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------

@dataclass
class CronJobState:
    """Mutable runtime state for a cron job."""
    next_run_at_ms: Optional[int] = None
    running_at_ms: Optional[int] = None
    last_run_at_ms: Optional[int] = None
    last_run_status: Optional[str] = None  # "ok" | "error" | "skipped"
    last_error: Optional[str] = None
    last_duration_ms: Optional[int] = None
    consecutive_errors: int = 0
    schedule_error_count: int = 0
    last_delivery_status: Optional[str] = None
    last_delivery_error: Optional[str] = None
    last_delivered: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.next_run_at_ms is not None:
            d["nextRunAtMs"] = self.next_run_at_ms
        if self.running_at_ms is not None:
            d["runningAtMs"] = self.running_at_ms
        if self.last_run_at_ms is not None:
            d["lastRunAtMs"] = self.last_run_at_ms
        if self.last_run_status is not None:
            d["lastRunStatus"] = self.last_run_status
        if self.last_error is not None:
            d["lastError"] = self.last_error
        if self.last_duration_ms is not None:
            d["lastDurationMs"] = self.last_duration_ms
        if self.consecutive_errors:
            d["consecutiveErrors"] = self.consecutive_errors
        if self.schedule_error_count:
            d["scheduleErrorCount"] = self.schedule_error_count
        if self.last_delivery_status is not None:
            d["lastDeliveryStatus"] = self.last_delivery_status
        if self.last_delivery_error is not None:
            d["lastDeliveryError"] = self.last_delivery_error
        if self.last_delivered is not None:
            d["lastDelivered"] = self.last_delivered
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CronJobState":
        return cls(
            next_run_at_ms=d.get("nextRunAtMs", d.get("next_run_at_ms")),
            running_at_ms=d.get("runningAtMs", d.get("running_at_ms")),
            last_run_at_ms=d.get("lastRunAtMs", d.get("last_run_at_ms")),
            last_run_status=d.get("lastRunStatus", d.get("last_run_status")),
            last_error=d.get("lastError", d.get("last_error")),
            last_duration_ms=d.get("lastDurationMs", d.get("last_duration_ms")),
            consecutive_errors=d.get("consecutiveErrors", d.get("consecutive_errors", 0)),
            schedule_error_count=d.get("scheduleErrorCount", d.get("schedule_error_count", 0)),
            last_delivery_status=d.get("lastDeliveryStatus", d.get("last_delivery_status")),
            last_delivery_error=d.get("lastDeliveryError", d.get("last_delivery_error")),
            last_delivered=d.get("lastDelivered", d.get("last_delivered")),
        )


# ---------------------------------------------------------------------------
# CronJob — the main model
# ---------------------------------------------------------------------------

def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class CronJob:
    """A cron job — the primary model for scheduled execution."""
    id: str = ""
    agent_id: Optional[str] = None
    session_key: Optional[str] = None
    user_id: str = ""  # tenant_id
    name: str = ""
    description: Optional[str] = None
    enabled: bool = True
    delete_after_run: bool = False
    created_at_ms: int = 0
    updated_at_ms: int = 0
    schedule: Schedule = field(default_factory=lambda: CronScheduleSpec())
    session_target: SessionTarget = SessionTarget.ISOLATED
    wake_mode: WakeMode = WakeMode.NEXT_HEARTBEAT
    payload: CronPayload = field(default_factory=lambda: AgentTurnPayload())
    delivery: Optional[DeliveryConfig] = None
    state: CronJobState = field(default_factory=CronJobState)
    max_concurrent_runs: int = 1

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "enabled": self.enabled,
            "deleteAfterRun": self.delete_after_run,
            "createdAtMs": self.created_at_ms,
            "updatedAtMs": self.updated_at_ms,
            "schedule": schedule_to_dict(self.schedule),
            "sessionTarget": self.session_target.value,
            "wakeMode": self.wake_mode.value,
            "payload": payload_to_dict(self.payload),
            "state": self.state.to_dict(),
        }
        if self.agent_id:
            d["agentId"] = self.agent_id
        if self.session_key:
            d["sessionKey"] = self.session_key
        if self.user_id:
            d["userId"] = self.user_id
        if self.description:
            d["description"] = self.description
        if self.delivery:
            d["delivery"] = self.delivery.to_dict()
        if self.max_concurrent_runs != 1:
            d["maxConcurrentRuns"] = self.max_concurrent_runs
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CronJob":
        delivery_data = d.get("delivery")
        return cls(
            id=d.get("id", ""),
            agent_id=d.get("agentId", d.get("agent_id")),
            session_key=d.get("sessionKey", d.get("session_key")),
            user_id=d.get("userId", d.get("user_id", "")),
            name=d.get("name", ""),
            description=d.get("description"),
            enabled=d.get("enabled", True),
            delete_after_run=d.get("deleteAfterRun", d.get("delete_after_run", False)),
            created_at_ms=d.get("createdAtMs", d.get("created_at_ms", 0)),
            updated_at_ms=d.get("updatedAtMs", d.get("updated_at_ms", 0)),
            schedule=schedule_from_dict(d.get("schedule", {"kind": "cron"})),
            session_target=SessionTarget(d.get("sessionTarget", d.get("session_target", "isolated"))),
            wake_mode=WakeMode(d.get("wakeMode", d.get("wake_mode", "next-heartbeat"))),
            payload=payload_from_dict(d.get("payload", {"kind": "agentTurn"})),
            delivery=DeliveryConfig.from_dict(delivery_data) if delivery_data else None,
            state=CronJobState.from_dict(d.get("state", {})),
            max_concurrent_runs=d.get("maxConcurrentRuns", d.get("max_concurrent_runs", 1)),
        )


# ---------------------------------------------------------------------------
# Run log entry
# ---------------------------------------------------------------------------

@dataclass
class CronRunEntry:
    """A single run log entry."""
    ts: int = 0  # timestamp ms
    job_id: str = ""
    action: str = "finished"  # "finished"
    status: Optional[str] = None  # "ok" | "error" | "skipped"
    error: Optional[str] = None
    summary: Optional[str] = None
    delivered: Optional[bool] = None
    delivery_status: Optional[str] = None
    delivery_error: Optional[str] = None
    session_id: Optional[str] = None
    session_key: Optional[str] = None
    run_at_ms: Optional[int] = None
    duration_ms: Optional[int] = None
    next_run_at_ms: Optional[int] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    usage: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"ts": self.ts, "jobId": self.job_id, "action": self.action}
        if self.status is not None:
            d["status"] = self.status
        if self.error is not None:
            d["error"] = self.error
        if self.summary is not None:
            d["summary"] = self.summary
        if self.delivered is not None:
            d["delivered"] = self.delivered
        if self.delivery_status is not None:
            d["deliveryStatus"] = self.delivery_status
        if self.delivery_error is not None:
            d["deliveryError"] = self.delivery_error
        if self.session_id is not None:
            d["sessionId"] = self.session_id
        if self.session_key is not None:
            d["sessionKey"] = self.session_key
        if self.run_at_ms is not None:
            d["runAtMs"] = self.run_at_ms
        if self.duration_ms is not None:
            d["durationMs"] = self.duration_ms
        if self.next_run_at_ms is not None:
            d["nextRunAtMs"] = self.next_run_at_ms
        if self.model is not None:
            d["model"] = self.model
        if self.provider is not None:
            d["provider"] = self.provider
        if self.usage is not None:
            d["usage"] = self.usage
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CronRunEntry":
        return cls(
            ts=d.get("ts", 0),
            job_id=d.get("jobId", d.get("job_id", "")),
            action=d.get("action", "finished"),
            status=d.get("status"),
            error=d.get("error"),
            summary=d.get("summary"),
            delivered=d.get("delivered"),
            delivery_status=d.get("deliveryStatus", d.get("delivery_status")),
            delivery_error=d.get("deliveryError", d.get("delivery_error")),
            session_id=d.get("sessionId", d.get("session_id")),
            session_key=d.get("sessionKey", d.get("session_key")),
            run_at_ms=d.get("runAtMs", d.get("run_at_ms")),
            duration_ms=d.get("durationMs", d.get("duration_ms")),
            next_run_at_ms=d.get("nextRunAtMs", d.get("next_run_at_ms")),
            model=d.get("model"),
            provider=d.get("provider"),
            usage=d.get("usage"),
        )


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@dataclass
class CronEvent:
    """Event emitted on cron job state changes."""
    job_id: str = ""
    action: str = ""  # "added" | "updated" | "removed" | "started" | "finished"
    run_at_ms: Optional[int] = None
    duration_ms: Optional[int] = None
    status: Optional[str] = None
    error: Optional[str] = None
    summary: Optional[str] = None
    delivered: Optional[bool] = None
    delivery_status: Optional[str] = None
    delivery_error: Optional[str] = None
    session_id: Optional[str] = None
    session_key: Optional[str] = None
    next_run_at_ms: Optional[int] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    usage: Optional[Dict[str, int]] = None


# ---------------------------------------------------------------------------
# CRUD input types
# ---------------------------------------------------------------------------

@dataclass
class CronJobCreate:
    """Input for creating a new cron job."""
    name: str = ""
    description: Optional[str] = None
    user_id: str = ""
    agent_id: Optional[str] = None
    session_key: Optional[str] = None
    schedule: Schedule = field(default_factory=lambda: CronScheduleSpec())
    session_target: SessionTarget = SessionTarget.ISOLATED
    wake_mode: WakeMode = WakeMode.NEXT_HEARTBEAT
    payload: CronPayload = field(default_factory=lambda: AgentTurnPayload())
    delivery: Optional[DeliveryConfig] = None
    enabled: bool = True
    delete_after_run: Optional[bool] = None  # None = auto (True for "at")
    max_concurrent_runs: int = 1

    def to_job(self) -> CronJob:
        """Convert to a CronJob with generated ID and timestamps."""
        now = _now_ms()
        delete = self.delete_after_run
        if delete is None:
            delete = isinstance(self.schedule, AtSchedule)
        return CronJob(
            id=str(uuid.uuid4()),
            agent_id=self.agent_id,
            session_key=self.session_key,
            user_id=self.user_id,
            name=self.name,
            description=self.description,
            enabled=self.enabled,
            delete_after_run=delete,
            created_at_ms=now,
            updated_at_ms=now,
            schedule=self.schedule,
            session_target=self.session_target,
            wake_mode=self.wake_mode,
            payload=self.payload,
            delivery=self.delivery,
            state=CronJobState(),
            max_concurrent_runs=self.max_concurrent_runs,
        )


@dataclass
class CronJobPatch:
    """Partial update for a cron job."""
    name: Optional[str] = None
    description: Optional[str] = None
    enabled: Optional[bool] = None
    schedule: Optional[Schedule] = None
    session_target: Optional[SessionTarget] = None
    wake_mode: Optional[WakeMode] = None
    payload: Optional[CronPayload] = None
    delivery: Optional[DeliveryConfig] = None
    delete_after_run: Optional[bool] = None
    max_concurrent_runs: Optional[int] = None

    def apply(self, job: CronJob) -> None:
        """Apply patch fields to a CronJob in-place."""
        if self.name is not None:
            job.name = self.name
        if self.description is not None:
            job.description = self.description
        if self.enabled is not None:
            job.enabled = self.enabled
        if self.schedule is not None:
            job.schedule = self.schedule
        if self.session_target is not None:
            job.session_target = self.session_target
        if self.wake_mode is not None:
            job.wake_mode = self.wake_mode
        if self.payload is not None:
            job.payload = self.payload
        if self.delivery is not None:
            job.delivery = self.delivery
        if self.delete_after_run is not None:
            job.delete_after_run = self.delete_after_run
        if self.max_concurrent_runs is not None:
            job.max_concurrent_runs = self.max_concurrent_runs
        job.updated_at_ms = _now_ms()
