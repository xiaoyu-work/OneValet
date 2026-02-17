"""Demo UI routes (registered only with --ui flag)."""

from fastapi import FastAPI
from fastapi.responses import FileResponse

from .app import _STATIC_DIR


def register_ui_routes(app: FastAPI):
    """Register demo frontend routes. Only called when --ui flag is set."""

    @app.get("/")
    async def index():
        return FileResponse(_STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/settings")
    async def settings_page():
        return FileResponse(_STATIC_DIR / "settings.html", media_type="text/html")
