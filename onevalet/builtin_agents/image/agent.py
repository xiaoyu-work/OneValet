"""
Image Agent - Unified image generation and editing

This agent handles both image generation and editing:
- Text only input -> generate a new image
- Text + attached image -> edit the existing image

Supports multiple providers (OpenAI, Azure, Gemini, Seedream) via the
image provider layer. Requires user approval before generating (costs money).
"""
import logging
import json
import base64
from typing import Dict, Any

from onevalet import valet, StandardAgent, InputField, AgentStatus, AgentResult, Message, ApprovalResult, ImageBlock

logger = logging.getLogger(__name__)


@valet()
class ImageAgent(StandardAgent):
    """Unified image generation and editing agent with approval"""

    prompt = InputField(
        prompt="What image would you like?",
        description="Image description or edit instructions",
    )
    provider = InputField(
        prompt="Which provider?",
        description="Image provider (openai, azure, gemini, seedream)",
        required=False,
    )
    size = InputField(
        prompt="What size?",
        description="Image size like 1024x1024",
        required=False,
    )
    quality = InputField(
        prompt="What quality?",
        description="Image quality (low, medium, high, auto)",
        required=False,
    )

    def __init__(self, tenant_id: str = "", llm_client=None, **kwargs):
        super().__init__(
            tenant_id=tenant_id,
            llm_client=llm_client,
            **kwargs
        )
        self._resolved_credentials = None
        self._edit_mode = False
        self._image_data = None

    def needs_approval(self) -> bool:
        return True

    async def parse_approval_async(self, user_input: str):
        """Parse user's approval response using LLM."""
        prompt = f"""The user was asked to approve generating an image. Their response was:
"{user_input}"

What is the user's intent?
- APPROVED: User wants to proceed (yes, ok, go ahead, generate it, etc.)
- REJECTED: User wants to cancel (no, cancel, nevermind, etc.)
- MODIFY: User wants to change something (different size, another provider, change prompt, etc.)

Return ONLY one word: APPROVED, REJECTED, or MODIFY"""

        try:
            result = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                enable_thinking=False
            )
            response = result.content.strip().upper()

            if "APPROVED" in response:
                return ApprovalResult.APPROVED
            elif "REJECTED" in response:
                return ApprovalResult.REJECTED
            else:
                return ApprovalResult.MODIFY
        except Exception as e:
            logger.error(f"Failed to parse approval: {e}")
            return ApprovalResult.MODIFY

    # ===== State Handlers =====

    async def on_initializing(self, msg: Message) -> AgentResult:
        """Extract fields, detect attached images, resolve provider."""
        if msg:
            await self._extract_and_collect_fields(msg.get_text())

            # Detect if an image is attached (edit mode)
            if msg.has_blocks("image"):
                self._edit_mode = True
                self._image_data = self._extract_image_data(msg)
                logger.info("Edit mode: image attached to message")

        # Resolve provider
        await self._resolve_provider()

        # Check missing fields
        missing = self._get_missing_fields()
        if missing:
            return self.make_result(
                status=AgentStatus.WAITING_FOR_INPUT,
                raw_message=self._get_next_prompt(),
                missing_fields=missing
            )

        # All fields collected - go to approval
        return self.make_result(
            status=AgentStatus.WAITING_FOR_APPROVAL,
            raw_message=self.get_approval_prompt()
        )

    async def on_waiting_for_input(self, msg: Message) -> AgentResult:
        """Continue collecting fields from user."""
        if msg:
            await self._extract_and_collect_fields(msg.get_text())

            # User might attach an image in a follow-up message
            if not self._edit_mode and msg.has_blocks("image"):
                self._edit_mode = True
                self._image_data = self._extract_image_data(msg)

        missing = self._get_missing_fields()
        if missing:
            return self.make_result(
                status=AgentStatus.WAITING_FOR_INPUT,
                raw_message=self._get_next_prompt(),
                missing_fields=missing
            )

        # All fields collected - go to approval
        await self._resolve_provider()
        return self.make_result(
            status=AgentStatus.WAITING_FOR_APPROVAL,
            raw_message=self.get_approval_prompt()
        )

    async def on_waiting_for_approval(self, msg: Message) -> AgentResult:
        """Handle yes/no/modify responses."""
        user_input = msg.get_text() if msg else ""
        approval = await self.parse_approval_async(user_input)

        if approval == ApprovalResult.APPROVED:
            self.transition_to(AgentStatus.RUNNING)
            return await self.on_running(msg)

        elif approval == ApprovalResult.REJECTED:
            return self.make_result(
                status=AgentStatus.CANCELLED,
                raw_message="OK, cancelled."
            )

        else:  # MODIFY
            await self._extract_and_collect_fields(user_input)

            # Re-resolve provider in case it changed
            await self._resolve_provider()

            missing = self._get_missing_fields()
            if missing:
                return self.make_result(
                    status=AgentStatus.WAITING_FOR_INPUT,
                    raw_message=self._get_next_prompt(),
                    missing_fields=missing
                )

            return self.make_result(
                status=AgentStatus.WAITING_FOR_APPROVAL,
                raw_message=self.get_approval_prompt()
            )

    async def on_running(self, msg: Message) -> AgentResult:
        """Execute image generation or editing."""
        from onevalet.providers.image.factory import ImageProviderFactory

        fields = self.collected_fields
        prompt_text = fields.get("prompt", "")
        size = fields.get("size")
        quality = fields.get("quality")

        mode_label = "edit" if self._edit_mode else "generate"
        logger.info(f"Image {mode_label}: {prompt_text}")

        try:
            if not self._resolved_credentials:
                await self._resolve_provider()

            if not self._resolved_credentials:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message="No image providers configured. Please add one in settings."
                )

            provider = ImageProviderFactory.create_provider(self._resolved_credentials)
            if not provider:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message="Sorry, I can't use that image provider yet."
                )

            if self._edit_mode:
                if not provider.supports_editing():
                    provider_name = provider.get_provider_display_name()
                    return self.make_result(
                        status=AgentStatus.COMPLETED,
                        raw_message=f"{provider_name} doesn't support image editing. Try a different provider."
                    )

                if not self._image_data:
                    return self.make_result(
                        status=AgentStatus.COMPLETED,
                        raw_message="I couldn't read the attached image. Please try again."
                    )

                result = await provider.edit_image(
                    image_data=self._image_data,
                    prompt=prompt_text,
                    size=size,
                )
            else:
                result = await provider.generate_image(
                    prompt=prompt_text,
                    size=size,
                    quality=quality,
                )

            if result.get("success"):
                images = result.get("data", {}).get("images", [])
                if not images:
                    return self.make_result(
                        status=AgentStatus.COMPLETED,
                        raw_message="The provider returned no images. Please try again."
                    )

                image_info = images[0]
                revised_prompt = image_info.get("revised_prompt")
                image_base64 = image_info.get("base64")
                image_url = image_info.get("url")

                # Build response message
                provider_name = provider.get_provider_display_name()
                parts = []
                if self._edit_mode:
                    parts.append(f"Edited image via {provider_name}.")
                else:
                    parts.append(f"Generated image via {provider_name}.")
                if revised_prompt and revised_prompt != prompt_text:
                    parts.append(f"Revised prompt: {revised_prompt}")

                response_text = "\n".join(parts)

                # Build result with image content
                if image_base64:
                    image_block = ImageBlock(source={"type": "base64", "data": image_base64})
                    result_msg = Message(
                        name="assistant",
                        content=[
                            {"type": "text", "text": response_text},
                            image_block,
                        ],
                        role="assistant",
                    )
                    return self.make_result(
                        status=AgentStatus.COMPLETED,
                        raw_message=response_text,
                        metadata={"image_base64": image_base64, "message": result_msg.to_dict()},
                    )
                elif image_url:
                    image_block = ImageBlock(source={"type": "url", "url": image_url})
                    result_msg = Message(
                        name="assistant",
                        content=[
                            {"type": "text", "text": response_text},
                            image_block,
                        ],
                        role="assistant",
                    )
                    return self.make_result(
                        status=AgentStatus.COMPLETED,
                        raw_message=response_text,
                        metadata={"image_url": image_url, "message": result_msg.to_dict()},
                    )
                else:
                    return self.make_result(
                        status=AgentStatus.COMPLETED,
                        raw_message=response_text,
                    )
            else:
                error_msg = result.get("error", "Unknown error")
                logger.error(f"Image {mode_label} failed: {error_msg}")
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Image {mode_label} failed: {error_msg}"
                )

        except Exception as e:
            logger.error(f"Image {mode_label} failed: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=f"Something went wrong during image {mode_label}. Want to try again?"
            )

    # ===== Helper Methods =====

    async def _resolve_provider(self):
        """Resolve the image provider credentials."""
        from onevalet.providers.image.resolver import ImageProviderResolver

        provider_spec = self.collected_fields.get("provider")
        credentials = await ImageProviderResolver.resolve(self.tenant_id, provider_spec)

        if credentials:
            self._resolved_credentials = credentials
            logger.info(f"Resolved image provider: {credentials.get('provider')}")
        else:
            logger.warning(f"No image provider found for tenant {self.tenant_id}")
            self._resolved_credentials = None

    def _extract_image_data(self, msg: Message) -> bytes | None:
        """Extract raw image bytes from the first ImageBlock in the message."""
        image_blocks = msg.get_blocks("image")
        if not image_blocks:
            return None

        block = image_blocks[0]
        source = getattr(block, "source", None) or (block if isinstance(block, dict) else {})
        if isinstance(source, dict):
            if source.get("type") == "base64":
                data = source.get("data", "")
                if data:
                    try:
                        return base64.b64decode(data)
                    except Exception as e:
                        logger.error(f"Failed to decode base64 image: {e}")
                        return None
        return None

    def get_approval_prompt(self) -> str:
        """Generate image action summary for user approval."""
        prompt_text = self.collected_fields.get("prompt", "")
        size = self.collected_fields.get("size", "")
        quality = self.collected_fields.get("quality", "")

        provider_name = ""
        if self._resolved_credentials:
            provider_name = self._resolved_credentials.get("provider", "").replace("_", " ").title()

        if self._edit_mode:
            parts = ["Edit image:"]
        else:
            parts = ["Generate image:"]

        parts.append(f"Prompt: {prompt_text}")
        if provider_name:
            parts.append(f"Provider: {provider_name}")
        if size:
            parts.append(f"Size: {size}")
        if quality and not self._edit_mode:
            parts.append(f"Quality: {quality}")

        parts.append("\nProceed? (yes/no)")
        return "\n".join(parts)

    async def extract_fields(self, user_input: str) -> Dict[str, Any]:
        """Extract image generation parameters from user input using LLM."""
        extraction_prompt = f"""Extract image generation information from the user's message. The user may speak in any language.

User message: "{user_input}"

Return JSON with these fields (leave empty string if not mentioned):
{{
  "prompt": "",
  "provider": "",
  "size": "",
  "quality": ""
}}

Rules:
- prompt: The image description or edit instruction. Extract what the user wants to see/create.
- provider: Image provider if mentioned (openai, azure, gemini, seedream)
- size: Image size if mentioned (e.g., "1024x1024", "1024x1536", "wide", "square")
- quality: Quality level if mentioned (low, medium, high, auto)
- Do NOT add extra interpretation. Just extract what the user said."""

        try:
            result = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "Extract image parameters. Return valid JSON only."},
                    {"role": "user", "content": extraction_prompt}
                ],
                response_format="json_object",
                enable_thinking=False
            )

            response_text = result.content.strip()
            if not response_text:
                return {}

            extracted = json.loads(response_text)
            result_dict = {}

            for field_name in ["prompt", "provider", "size", "quality"]:
                value = extracted.get(field_name, "").strip()
                if value:
                    result_dict[field_name] = value

            logger.info(f"Extracted fields: {list(result_dict.keys())}")
            return result_dict

        except Exception as e:
            logger.error(f"Field extraction failed: {e}")
            return {}
