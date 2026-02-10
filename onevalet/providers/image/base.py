"""
Base Image Provider - Abstract interface for all image generation providers

Each image provider (OpenAI, Azure, Gemini, Seedream, etc.) must implement this interface.
Providers receive a credentials dict with API key and configuration.
Unlike OAuth-based providers, image providers use static API keys (no token refresh needed).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class BaseImageProvider(ABC):
    """
    Abstract base class for image generation providers.

    All image providers must implement:
    - generate_image()
    - edit_image()
    - supports_editing()
    - get_supported_sizes()
    """

    def __init__(self, credentials: dict):
        """
        Initialize provider with credentials dict.

        Args:
            credentials: Dict containing:
                - provider: str (openai, azure, gemini, seedream, etc.)
                - api_key: str
                - model: str (optional, provider-specific default)
                - endpoint: str (optional, for Azure/custom endpoints)
                - account_name: str (user-defined name)
        """
        self.credentials = credentials
        self.provider_name = credentials.get("provider", "")
        self.api_key = credentials.get("api_key", "")
        self.model = credentials.get("model", "")
        self.endpoint = credentials.get("endpoint", "")
        self.account_name = credentials.get("account_name", "primary")

    # ===== Abstract methods (must be implemented by subclasses) =====

    @abstractmethod
    async def generate_image(
        self,
        prompt: str,
        size: Optional[str] = None,
        quality: Optional[str] = None,
        n: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate image(s) from text prompt.

        Args:
            prompt: Text description of the image to generate
            size: Image size (e.g., "1024x1024", "1024x1536")
            quality: Quality level (e.g., "low", "medium", "high")
            n: Number of images to generate
            **kwargs: Provider-specific options

        Returns:
            {
                "success": bool,
                "data": {
                    "images": [
                        {
                            "base64": str or None,
                            "url": str or None,
                            "revised_prompt": str or None
                        }
                    ]
                },
                "error": str or None
            }
        """
        pass

    @abstractmethod
    async def edit_image(
        self,
        image_data: bytes,
        prompt: str,
        size: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Edit an existing image based on text prompt.

        Args:
            image_data: Raw image bytes (PNG/JPEG)
            prompt: Text description of the edit to apply
            size: Output image size
            **kwargs: Provider-specific options (e.g., mask for inpainting)

        Returns:
            Same format as generate_image()
        """
        pass

    @abstractmethod
    def supports_editing(self) -> bool:
        """Whether this provider supports image editing."""
        pass

    @abstractmethod
    def get_supported_sizes(self) -> List[str]:
        """Return list of supported image sizes for this provider."""
        pass

    def get_default_size(self) -> str:
        """Return default image size. Subclasses may override."""
        sizes = self.get_supported_sizes()
        return sizes[0] if sizes else "1024x1024"

    def get_provider_display_name(self) -> str:
        """Human-readable provider name for UI display."""
        return self.provider_name.replace("_", " ").title()

    def __repr__(self):
        return f"<{self.__class__.__name__} provider={self.provider_name} model={self.model}>"
