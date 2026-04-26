"""Route registration for the Koa API."""

from fastapi import FastAPI

from .chat import router as chat_router
from .config import router as config_router
from .credentials import router as credentials_router
from .cron import router as cron_router
from .events import router as events_router
from .expenses import router as expenses_router
from .internal_events import router as internal_events_router
from .internal_routing_preferences import router as internal_routing_prefs_router
from .memory import router as memory_router
from .oauth import router as oauth_router
from .profile import router as profile_router
from .sensing import router as sensing_router
from .shipments import router as shipments_router
from .subscriptions import router as subscriptions_router
from .tasks import router as tasks_router


def register_routes(app: FastAPI):
    app.include_router(chat_router)
    app.include_router(config_router)
    app.include_router(credentials_router)
    app.include_router(events_router)
    app.include_router(internal_events_router)
    app.include_router(internal_routing_prefs_router)
    app.include_router(oauth_router)
    app.include_router(tasks_router)
    app.include_router(cron_router)
    app.include_router(memory_router)
    app.include_router(profile_router)
    app.include_router(sensing_router)
    app.include_router(shipments_router)
    app.include_router(expenses_router)
    app.include_router(subscriptions_router)
