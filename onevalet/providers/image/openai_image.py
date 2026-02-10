"""
OpenAI Image Provider - gpt-image-1 implementation via openai Python SDK

Uses OpenAI's image generation and editing APIs with async client.
"""

import logging
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from .base import BaseImageProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-image-1"
DEFAULT_SIZE = "1024x1024"
DEFAULT_QUALITY = "auto"
SUPPORTED_SIZES = ["1024x1024", "1024x1536", "1536x1024"]


class OpenAIImageProvider(BaseImageProvider):
    """OpenAI image provider using gpt-image-1 via the openai Python SDK."""

    def __init__(self, credentials: dict):
        super().__init__(credentials)
        self.model = credentials.get("model", DEFAULT_MODEL)
        self.client = AsyncOpenAI(api_key=self.api_key)
        logger.info(f"Initialized OpenAI image provider (model={self.model})")

    async def generate_image(
        self,
        prompt: str,
        size: Optional[str] = None,
        quality: Optional[str] = None,
        n: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate image(s) from text prompt using OpenAI gpt-image-1."""
        try:
            size = size or DEFAULT_SIZE
            quality = quality or DEFAULT_QUALITY

            logger.info(
                f"OpenAI generating image: size={size}, quality={quality}, n={n}"
            )

            response = await self.client.images.generate(
                model=self.model,
                prompt=prompt,
                size=size,
                quality=quality,
                n=n,
                response_format="b64_json",
            )

            images = []
            for item in response.data:
                images.append({
                    "base64": item.b64_json,
                    "url": None,
                    "revised_prompt": getattr(item, "revised_prompt", None),
                })

            logger.info(f"OpenAI generated {len(images)} image(s)")
            return {"success": True, "data": {"images": images}}

        except Exception as e:
            logger.error(f"OpenAI generate image error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def edit_image(
        self,
        image_data: bytes,
        prompt: str,
        size: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Edit an existing image using OpenAI gpt-image-1."""
        try:
            logger.info(f"OpenAI editing image: prompt={prompt[:80]}")

            edit_kwargs: Dict[str, Any] = {
                "model": self.model,
                "image": image_data,
                "prompt": prompt,
            }
            if size:
                edit_kwargs["size"] = size

            response = await self.client.images.edit(**edit_kwargs)

            images = []
            for item in response.data:
                images.append({
                    "base64": getattr(item, "b64_json", None),
                    "url": getattr(item, "url", None),
                    "revised_prompt": getattr(item, "revised_prompt", None),
                })

            logger.info(f"OpenAI edited image, returned {len(images)} result(s)")
            return {"success": True, "data": {"images": images}}

        except Exception as e:
            logger.error(f"OpenAI edit image error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def supports_editing(self) -> bool:
        """OpenAI gpt-image-1 supports image editing."""
        return True

    def get_supported_sizes(self) -> List[str]:
        """Return supported image sizes for OpenAI gpt-image-1."""
        return SUPPORTED_SIZES
