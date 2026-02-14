"""
OneValet Triggers — Proactive trigger system.

Supports schedule (cron/interval/one-time), event (Redis Streams),
and condition (periodic polling) triggers with dual execution paths:
OrchestratorExecutor (LLM-driven) and custom executors (deterministic).
"""

from .models import (
    Task,
    TaskStatus,
    TriggerConfig,
    TriggerType,
    ActionConfig,
    TriggerContext,
    ActionResult,
)
from .engine import TriggerEngine
from .event_bus import EventBus, Event
from .executor import OrchestratorExecutor
from .pipeline import PipelineExecutor
from .notification import SMSNotification, PushNotification
from .callback import CallbackNotification
from .email_handler import EmailEventHandler

__all__ = [
    # Models
    "Task",
    "TaskStatus",
    "TriggerConfig",
    "TriggerType",
    "ActionConfig",
    "TriggerContext",
    "ActionResult",
    # Engine
    "TriggerEngine",
    # EventBus
    "EventBus",
    "Event",
    # Executors
    "OrchestratorExecutor",
    "PipelineExecutor",
    # Email
    "EmailEventHandler",
    # Notifications
    "SMSNotification",
    "PushNotification",
    "CallbackNotification",
]
