"""
Azure OpenAI Image Provider - Azure OpenAI gpt-image-1 implementation

Uses the openai Python SDK configured for Azure endpoints.
Credentials dict expects:
    - api_key: Azure OpenAI API key
    - endpoint: Azure endpoint URL (e.g., https://xxx.openai.azure.com/)
    - api_version: API version (default: 2025-04-01-preview)
    - deployment: Deployment name (default: gpt-image-1)
    - provider: "azure"
"""

import logging
from typing import Any, Dict, List, Optional

from openai import AsyncAzureOpenAI

from .base import BaseImageProvider

logger = logging.getLogger(__name__)


class AzureImageProvider(BaseImageProvider):
    """Azure OpenAI image provider using gpt-image-1 deployment."""

    def __init__(self, credentials: dict):
        super().__init__(credentials)
        self.deployment = credentials.get("deployment", "gpt-image-1")
        self.api_version = credentials.get("api_version", "2025-04-01-preview")

        self.client = AsyncAzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )
        logger.info(
            f"Azure OpenAI image provider initialized: "
            f"endpoint={self.endpoint}, deployment={self.deployment}, "
            f"api_version={self.api_version}"
        )

    async def generate_image(
        self,
        prompt: str,
        size: Optional[str] = None,
        quality: Optional[str] = None,
        n: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate image(s) using Azure OpenAI gpt-image-1."""
        try:
            params: Dict[str, Any] = {
                "model": self.deployment,
                "prompt": prompt,
                "n": n,
                "response_format": "b64_json",
            }
            if size:
                params["size"] = size
            if quality:
                params["quality"] = quality

            logger.info(
                f"Azure image generate: prompt='{prompt[:80]}...', "
                f"size={size}, quality={quality}, n={n}"
            )

            response = await self.client.images.generate(**params)

            images = []
            for item in response.data:
                images.append({
                    "base64": item.b64_json,
                    "url": None,
                    "revised_prompt": getattr(item, "revised_prompt", None),
                })

            logger.info(f"Azure image generate succeeded: {len(images)} image(s)")
            return {"success": True, "data": {"images": images}}

        except Exception as e:
            logger.error(f"Azure image generate error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def edit_image(
        self,
        image_data: bytes,
        prompt: str,
        size: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Edit an image using Azure OpenAI gpt-image-1."""
        try:
            params: Dict[str, Any] = {
                "model": self.deployment,
                "image": image_data,
                "prompt": prompt,
                "response_format": "b64_json",
            }
            if size:
                params["size"] = size

            logger.info(f"Azure image edit: prompt='{prompt[:80]}...'")

            response = await self.client.images.edit(**params)

            images = []
            for item in response.data:
                images.append({
                    "base64": item.b64_json,
                    "url": None,
                    "revised_prompt": getattr(item, "revised_prompt", None),
                })

            logger.info(f"Azure image edit succeeded: {len(images)} image(s)")
            return {"success": True, "data": {"images": images}}

        except Exception as e:
            logger.error(f"Azure image edit error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def supports_editing(self) -> bool:
        """Azure OpenAI gpt-image-1 supports image editing."""
        return True

    def get_supported_sizes(self) -> List[str]:
        """Return supported sizes for Azure OpenAI gpt-image-1."""
        return ["1024x1024", "1024x1536", "1536x1024"]
