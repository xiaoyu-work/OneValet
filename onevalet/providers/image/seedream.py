"""
Seedream Image Provider - ByteDance Seedream implementation via BytePlus ModelArk REST API

Uses the OpenAI-compatible image generation endpoint on BytePlus ModelArk.
"""

import logging
from typing import Any, Dict, List, Optional

import httpx

from .base import BaseImageProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "seedream-4.5"
DEFAULT_ENDPOINT = "https://ark.ap-southeast.bytepluses.com/api/v3"
DEFAULT_SIZE = "1024x1024"
SUPPORTED_SIZES = ["1024x1024", "1024x1536", "1536x1024", "2048x2048"]
REQUEST_TIMEOUT = 120.0


class SeedreamProvider(BaseImageProvider):
    """ByteDance Seedream image provider via BytePlus ModelArk REST API."""

    def __init__(self, credentials: dict):
        super().__init__(credentials)
        self.endpoint = credentials.get("endpoint", DEFAULT_ENDPOINT).rstrip("/")
        self.model = credentials.get("model", DEFAULT_MODEL)
        logger.info(
            f"Initialized Seedream image provider (model={self.model}, endpoint={self.endpoint})"
        )

    def _get_headers(self) -> dict:
        """Get authorization headers for API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def generate_image(
        self,
        prompt: str,
        size: Optional[str] = None,
        quality: Optional[str] = None,
        n: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate image(s) from text prompt using Seedream via BytePlus ModelArk."""
        try:
            size = size or DEFAULT_SIZE
            url = f"{self.endpoint}/images/generations"

            body: Dict[str, Any] = {
                "model": self.model,
                "prompt": prompt,
                "size": size,
                "n": n,
                "response_format": "b64_json",
            }

            logger.info(f"Seedream generating image: size={size}, n={n}")

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers=self._get_headers(),
                    json=body,
                    timeout=REQUEST_TIMEOUT,
                )

                if response.status_code != 200:
                    error_detail = response.text
                    logger.error(
                        f"Seedream API error: {response.status_code} - {error_detail}"
                    )
                    return {
                        "success": False,
                        "error": f"Seedream API error: {response.status_code} - {error_detail}",
                    }

                data = response.json()
                images = []
                for item in data.get("data", []):
                    images.append({
                        "base64": item.get("b64_json"),
                        "url": item.get("url"),
                        "revised_prompt": item.get("revised_prompt"),
                    })

                logger.info(f"Seedream generated {len(images)} image(s)")
                return {"success": True, "data": {"images": images}}

        except httpx.TimeoutException:
            logger.error("Seedream generate image timed out")
            return {"success": False, "error": "Request timed out"}
        except httpx.ConnectError as e:
            logger.error(f"Seedream connection error: {e}")
            return {"success": False, "error": f"Connection error: {e}"}
        except Exception as e:
            logger.error(f"Seedream generate image error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def edit_image(
        self,
        image_data: bytes,
        prompt: str,
        size: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Image editing is not supported by Seedream."""
        return {
            "success": False,
            "error": "Image editing not supported by Seedream",
        }

    def supports_editing(self) -> bool:
        """Seedream does not support image editing."""
        return False

    def get_supported_sizes(self) -> List[str]:
        """Return supported image sizes for Seedream."""
        return SUPPORTED_SIZES
