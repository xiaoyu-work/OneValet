"""
OneValet REST API Server

Usage:
    python -m onevalet
    # → POST /chat, POST /stream, GET /health
"""

import json
import logging
import os
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .app import OneValet

logger = logging.getLogger(__name__)

# Load config from ONEVALET_CONFIG env var, default to config.yaml
_config_path = os.getenv("ONEVALET_CONFIG", "config.yaml")
_app = OneValet(_config_path)

api = FastAPI(title="OneValet", version="0.1.1")


class ChatRequest(BaseModel):
    message: str
    tenant_id: str = "default"
    metadata: Optional[dict] = None


class ChatResponse(BaseModel):
    response: str
    status: str


@api.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    result = await _app.chat(
        message_or_tenant_id=req.tenant_id,
        message=req.message,
        metadata=req.metadata,
    )
    return ChatResponse(
        response=result.raw_message or "",
        status=result.status.value if result.status else "completed",
    )


@api.post("/stream")
async def stream(req: ChatRequest):
    async def event_generator():
        async for event in _app.stream(
            message_or_tenant_id=req.tenant_id,
            message=req.message,
            metadata=req.metadata,
        ):
            data = json.dumps({
                "type": event.type.value if event.type else "unknown",
                "data": event.data,
            }, ensure_ascii=False)
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


@api.get("/health")
async def health():
    return {"status": "ok"}


def main():
    import uvicorn

    host = os.getenv("ONEVALET_HOST", "0.0.0.0")
    port = int(os.getenv("ONEVALET_PORT", "8000"))
    uvicorn.run(api, host=host, port=port)
