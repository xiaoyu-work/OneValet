"""
Image Agent - Unified image generation and editing

This agent handles both image generation and editing:
- Text only input -> generate a new image
- Text + attached image -> edit the existing image

Supports multiple providers (OpenAI, Azure, Gemini, Seedream) via the
image provider layer. Requires user approval before generating (costs money).
"""

import logging

from koa import StandardAgent, valet
from koa.constants import IMAGE_SERVICES

from .tools import generate_image

logger = logging.getLogger(__name__)


@valet(domain="lifestyle", requires_service=list(IMAGE_SERVICES))
class ImageAgent(StandardAgent):
    """Generate or edit images from a text description. Use when the user wants to create, modify, or design an image."""

    max_turns = 15

    _SYSTEM_PROMPT_TEMPLATE = """\
You create images for the user.

Available tools:
- generate_image: Generate a new image, or edit one the user attached.

Instructions:
1. Call generate_image with the user's description as `prompt`. Pass it through in
   their own words -- do not embellish it with details they did not ask for.
2. If the user attached an image, generate_image edits it instead of creating a new
   one. You do not need to do anything differently; just describe the change they
   asked for in `prompt`.
3. Only set `provider`, `size`, or `quality` when the user actually named one.
   Leave them empty otherwise so the user's configured default applies.
4. Generating costs money, so the tool asks the user to confirm first. Do not ask
   for confirmation yourself -- call the tool and let it prompt.
5. If the user only says something vague like "make me an image", ask what they
   want to see before calling the tool.
6. Respond in the same language the user used."""

    def get_system_prompt(self) -> str:
        prompt = self._SYSTEM_PROMPT_TEMPLATE
        if self.context_hints and self.context_hints.get("user_images"):
            prompt += (
                "\n\nIMPORTANT: The user has attached an image. Calling generate_image "
                "will edit that image rather than create a new one."
            )
        return prompt

    tools = (generate_image,)
