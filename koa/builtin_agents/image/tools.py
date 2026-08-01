"""Image tools — generate a new image, or edit one the user attached."""

import base64
import logging
from typing import Annotated, Optional

from koa.models import AgentToolContext, ToolOutput

from ...tool_decorator import tool

logger = logging.getLogger(__name__)


def _attached_image(context: AgentToolContext) -> Optional[bytes]:
    """Raw bytes of the first image the user attached, if any.

    Presence of an attachment is what distinguishes an edit from a generation,
    so this doubles as the mode check.
    """
    images = (context.context_hints or {}).get("user_images")
    if not images:
        return None

    data = images[0]
    if isinstance(data, bytes):
        return data
    if not isinstance(data, str):
        return None

    if data.startswith("data:"):  # data:image/png;base64,<payload>
        try:
            _, data = data.split(",", 1)
        except ValueError:
            return None
    try:
        return base64.b64decode(data)
    except Exception as e:
        logger.error(f"Failed to decode attached image: {e}")
        return None


async def _resolve_provider(context: AgentToolContext, provider_spec: str = ""):
    from koa.providers.image.resolver import ImageProviderResolver

    credentials = await ImageProviderResolver.resolve(context.tenant_id, provider_spec or None)
    if credentials:
        logger.info(f"Resolved image provider: {credentials.get('provider')}")
    else:
        logger.warning(f"No image provider found for tenant {context.tenant_id}")
    return credentials


def _display_provider(credentials) -> str:
    return (credentials or {}).get("provider", "").replace("_", " ").title()


def _image_media(image_base64: str, image_url: str) -> Optional[list]:
    """The generated image as a media attachment the client can render and store."""
    if image_base64:
        data = image_base64
        if not data.startswith("data:"):
            data = f"data:image/png;base64,{data}"
        return [
            {
                "type": "image",
                "data": data,
                "media_type": "image/png",
                "metadata": {"for_storage": True},
            }
        ]
    if image_url:
        return [
            {
                "type": "image",
                "data": image_url,
                "media_type": "image/png",
                "metadata": {"source_url": image_url, "for_storage": True},
            }
        ]
    return None


async def _preview_generate_image(args: dict, context: AgentToolContext) -> str:
    editing = _attached_image(context) is not None
    credentials = await _resolve_provider(context, args.get("provider", ""))

    parts = ["Edit image:" if editing else "Generate image:"]
    parts.append(f"Prompt: {args.get('prompt', '')}")
    provider_name = _display_provider(credentials)
    if provider_name:
        parts.append(f"Provider: {provider_name}")
    if args.get("size"):
        parts.append(f"Size: {args['size']}")
    if args.get("quality") and not editing:
        parts.append(f"Quality: {args['quality']}")
    parts.append("\nProceed? (yes/no)")
    return "\n".join(parts)


@tool(
    needs_approval=True,
    risk_level="write",
    category="image",
    get_preview=_preview_generate_image,
    renderer="image",
)
async def generate_image(
    prompt: Annotated[
        str, "What the image should show, or how to change an attached one."
    ],
    provider: Annotated[
        str, "Image provider if the user named one: openai, azure, gemini, seedream."
    ] = "",
    size: Annotated[str, "Image size if requested, e.g. '1024x1024' or '1024x1536'."] = "",
    quality: Annotated[str, "Quality if requested: low, medium, high, auto."] = "",
    *,
    context: AgentToolContext,
) -> str:
    """Generate an image, or edit the image the user attached. Costs money, so it needs approval."""
    from koa.providers.image.factory import ImageProviderFactory

    image_data = _attached_image(context)
    editing = image_data is not None
    mode_label = "edit" if editing else "generate"
    logger.info(f"Image {mode_label}: {prompt}")

    try:
        credentials = await _resolve_provider(context, provider)
        if not credentials:
            return "No image providers configured. Please add one in settings."

        image_provider = ImageProviderFactory.create_provider(credentials)
        if not image_provider:
            return "Sorry, I can't use that image provider yet."

        if editing:
            if not image_provider.supports_editing():
                name = image_provider.get_provider_display_name()
                return f"{name} doesn't support image editing. Try a different provider."
            result = await image_provider.edit_image(
                image_data=image_data, prompt=prompt, size=size or None
            )
        else:
            result = await image_provider.generate_image(
                prompt=prompt, size=size or None, quality=quality or None
            )

        if not result.get("success"):
            error_msg = result.get("error", "Unknown error")
            logger.error(f"Image {mode_label} failed: {error_msg}")
            return f"Image {mode_label} failed: {error_msg}"

        images = result.get("data", {}).get("images", [])
        if not images:
            return "The provider returned no images. Please try again."

        info = images[0]
        revised_prompt = info.get("revised_prompt")
        provider_name = image_provider.get_provider_display_name()

        parts = [f"{'Edited' if editing else 'Generated'} image via {provider_name}."]
        if revised_prompt and revised_prompt != prompt:
            parts.append(f"Revised prompt: {revised_prompt}")
        response_text = "\n".join(parts)

        media = _image_media(info.get("base64", ""), info.get("url", ""))
        if media:
            return ToolOutput(text=response_text, media=media)
        return response_text

    except Exception as e:
        logger.error(f"Image {mode_label} failed: {e}", exc_info=True)
        return f"Something went wrong during image {mode_label}. Want to try again?"
