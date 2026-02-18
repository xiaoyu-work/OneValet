"""Status and configuration routes."""

import logging
import os

import yaml
from fastapi import APIRouter, Depends, HTTPException

from ...app import OneValet
from ..app import (
    _config_path,
    _SUPPORTED_PROVIDERS,
    get_app_instance,
    mask_config,
    require_app,
    set_app,
    verify_api_key,
)
from ..models import ConfigRequest

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/status")
async def get_status():
    return {"configured": get_app_instance() is not None}


@router.get("/api/config", dependencies=[Depends(verify_api_key)])
async def get_config():
    """Return current configuration (API key masked)."""
    _app = get_app_instance()
    if _app is not None:
        return mask_config(_app.config)
    if os.path.exists(_config_path):
        with open(_config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f.read()) or {}
        return mask_config(raw)
    return {}


@router.post("/api/config", dependencies=[Depends(verify_api_key)])
async def save_config(req: ConfigRequest):
    """Save configuration to config.yaml and reinitialize the app."""
    _app = get_app_instance()

    if req.llm.provider not in _SUPPORTED_PROVIDERS:
        raise HTTPException(400, f"Unsupported provider: {req.llm.provider}")

    llm_config = {
        "provider": req.llm.provider,
        "model": req.llm.model,
    }
    if req.llm.api_key:
        llm_config["api_key"] = req.llm.api_key
    elif _app is not None:
        old_llm = _app.config.get("llm", {})
        if old_llm.get("api_key"):
            llm_config["api_key"] = old_llm["api_key"]

    if req.llm.base_url:
        llm_config["base_url"] = req.llm.base_url

    config = {
        "llm": llm_config,
        "database": req.database,
    }
    emb = {k: v for k, v in req.embedding.model_dump().items() if v}
    if emb:
        config["embedding"] = emb
    if req.system_prompt:
        config["system_prompt"] = req.system_prompt

    # Shut down existing app
    if _app is not None:
        try:
            await _app.shutdown()
        except Exception as e:
            logger.warning(f"Error during shutdown: {e}")
        set_app(None)

    # Write config.yaml
    with open(_config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    # Reload and initialize
    try:
        new_app = OneValet(_config_path)
        await new_app._ensure_initialized()
        set_app(new_app)
        return {"success": True, "message": "Configuration saved and initialized."}
    except Exception as e:
        set_app(None)
        logger.error(f"Config saved but initialization failed: {e}")
        raise HTTPException(
            422,
            f"Config saved but initialization failed: {e}. "
            f"Check your database URL and API key.",
        )
