"""
Gemini Image Provider - Google Gemini image generation via google-genai SDK

Uses Gemini's multimodal generate_content API with response_modalities=['TEXT', 'IMAGE']
for both image generation and editing (native image editing support).
"""

import base64
import io
import logging
from typing import Any, Dict, List, Optional

try:
    from google import genai
    from google.genai import types

    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from .base import BaseImageProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash-image"
SUPPORTED_SIZES = ["1024x1024", "1024x1536", "1536x1024"]


class GeminiImageProvider(BaseImageProvider):
    """Google Gemini image provider using the google-genai SDK."""

    def __init__(self, credentials: dict):
        super().__init__(credentials)

        if not HAS_GENAI:
            raise ImportError(
                "google-genai package is required for Gemini image provider. "
                "Install with: pip install google-genai"
            )

        self.model = credentials.get("model", DEFAULT_MODEL)
        self.client = genai.Client(api_key=self.api_key)
        logger.info(f"Initialized Gemini image provider (model={self.model})")

    def _parse_response(self, response) -> Dict[str, Any]:
        """Extract image data and text from a Gemini generate_content response."""
        images = []
        revised_prompt = None

        if not response.candidates or not response.candidates[0].content.parts:
            return {"success": False, "error": "Gemini returned no content"}

        for part in response.candidates[0].content.parts:
            if part.text is not None:
                revised_prompt = part.text
            elif part.inline_data is not None:
                image_bytes = part.inline_data.data
                if isinstance(image_bytes, str):
                    b64_data = image_bytes
                else:
                    b64_data = base64.b64encode(image_bytes).decode("utf-8")
                images.append({
                    "base64": b64_data,
                    "url": None,
                    "revised_prompt": None,
                })

        if revised_prompt and images:
            images[0]["revised_prompt"] = revised_prompt

        if not images:
            return {"success": False, "error": "Gemini response contained no image data"}

        return {"success": True, "data": {"images": images}}

    async def generate_image(
        self,
        prompt: str,
        size: Optional[str] = None,
        quality: Optional[str] = None,
        n: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate image(s) from text prompt using Gemini."""
        try:
            logger.info(
                f"Gemini generating image: model={self.model}, n={n}"
            )

            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                ),
            )

            result = self._parse_response(response)
            if result["success"]:
                logger.info(f"Gemini generated {len(result['data']['images'])} image(s)")
            else:
                logger.warning(f"Gemini generate returned no images: {result.get('error')}")

            return result

        except Exception as e:
            logger.error(f"Gemini generate image error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def edit_image(
        self,
        image_data: bytes,
        prompt: str,
        size: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Edit an existing image using Gemini's native image editing."""
        try:
            if not HAS_PIL:
                return {
                    "success": False,
                    "error": "Pillow package is required for image editing. "
                    "Install with: pip install Pillow",
                }

            logger.info(f"Gemini editing image: prompt={prompt[:80]}")

            image = Image.open(io.BytesIO(image_data))

            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt, image],
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                ),
            )

            result = self._parse_response(response)
            if result["success"]:
                logger.info(f"Gemini edited image, returned {len(result['data']['images'])} result(s)")
            else:
                logger.warning(f"Gemini edit returned no images: {result.get('error')}")

            return result

        except Exception as e:
            logger.error(f"Gemini edit image error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def supports_editing(self) -> bool:
        """Gemini supports native image editing."""
        return True

    def get_supported_sizes(self) -> List[str]:
        """Return supported image sizes for Gemini."""
        return SUPPORTED_SIZES
