"""Route registration for the OneValet API."""

from fastapi import FastAPI

from .chat import router as chat_router
from .config import router as config_router
from .credentials import router as credentials_router
from .events import router as events_router
from .oauth import router as oauth_router
from .tasks import router as tasks_router


def register_routes(app: FastAPI):
    app.include_router(chat_router)
    app.include_router(config_router)
    app.include_router(credentials_router)
    app.include_router(events_router)
    app.include_router(oauth_router)
    app.include_router(tasks_router)
