"""Email preference tools — manage the user's custom email importance rules.

The rules are natural-language text stored on the tenant profile under
``email_importance_rules``. EmailEventHandler appends them to its classifier
prompt, where they take priority over the system defaults in both directions:
they can promote an email the defaults ignore, and suppress one the defaults
would have flagged.
"""

import logging
from typing import Annotated, Optional

from koa.models import AgentToolContext

from ...tool_decorator import tool
from .importance import EmailImportanceAgent

logger = logging.getLogger(__name__)

_NO_STORE = "Email preferences are unavailable right now. Please try again later."


def _credential_store(context: AgentToolContext):
    hints = context.context_hints or {}
    return hints.get("credential_store") or context.credentials


def _current_rules(context: AgentToolContext) -> str:
    """The user's rules as loaded into this request's context."""
    profile = context.user_profile or {}
    return (profile.get("email_importance_rules") or "").strip()


async def _save_rules(context: AgentToolContext, rules: Optional[str]) -> bool:
    store = _credential_store(context)
    if not store:
        return False
    await store.update_user_profile(context.tenant_id, {"email_importance_rules": rules})
    return True


async def _rewrite_rules(context: AgentToolContext, instruction: str) -> str:
    """Ask the model to produce the new rule text for a merge or removal.

    Rules are free-form prose, so combining or subtracting them is a language
    task rather than a string operation.
    """
    llm = context.llm_client
    if llm is None:
        raise RuntimeError("llm_client unavailable")
    result = await llm.chat_completion(
        messages=[{"role": "user", "content": instruction}],
        enable_thinking=False,
    )
    return (result.content or "").strip()


@tool(risk_level="read", category="email")
async def show_email_rules(*, context: AgentToolContext) -> str:
    """Show the user's current email importance rules and the system defaults."""
    current = _current_rules(context)
    if current:
        return (
            f"Your custom email importance rules:\n{current}\n\n"
            f"System default rules also apply:\n{EmailImportanceAgent.SYSTEM_RULES}"
        )
    return (
        "You haven't set custom rules yet. Using system default rules:\n"
        f"{EmailImportanceAgent.SYSTEM_RULES}"
    )


@tool(risk_level="write", category="email")
async def set_email_rules(
    rules: Annotated[str, "The importance rules to save, in the user's own words."],
    *,
    context: AgentToolContext,
) -> str:
    """Set email importance rules for the first time. Fails if rules already exist."""
    if not rules.strip():
        return "No rules provided. Tell me which emails you want flagged as important."

    current = _current_rules(context)
    if current:
        return (
            f"You already have custom rules:\n{current}\n\n"
            "Use add_email_rules to append, or replace_email_rules to overwrite."
        )

    if not await _save_rules(context, rules.strip()):
        return _NO_STORE
    logger.info(f"Email rules set for user {context.tenant_id}")
    return f"Email importance rules set successfully!\n\nYour rules:\n{rules.strip()}"


@tool(risk_level="write", category="email")
async def add_email_rules(
    rules: Annotated[str, "The rule to add, e.g. 'emails from my wife are important'."],
    *,
    context: AgentToolContext,
) -> str:
    """Add a rule to the user's existing email importance rules."""
    if not rules.strip():
        return "No rule provided. Tell me what you'd like to add."

    current = _current_rules(context)
    try:
        merged = await _rewrite_rules(
            context,
            "Merge the new rules with existing rules, avoiding duplicates and conflicts.\n\n"
            f"Existing rules: {current if current else 'None'}\n"
            f"New rules to add: {rules.strip()}\n\n"
            "Return the merged rules as a natural language description.\n"
            "Keep it concise and clear. Use bullet points or commas to separate different rules.",
        )
    except Exception as e:
        logger.error(f"Rule merge failed: {e}")
        return "I couldn't merge that rule. Please try again."

    if not await _save_rules(context, merged):
        return _NO_STORE
    logger.info(f"Email rules updated for user {context.tenant_id}")
    return f"Added new rule: {rules.strip()}\n\nUpdated rules:\n{merged}"


@tool(risk_level="write", category="email")
async def remove_email_rules(
    rules: Annotated[str, "The rule to remove, e.g. 'newsletters'."],
    *,
    context: AgentToolContext,
) -> str:
    """Remove a specific rule from the user's email importance rules."""
    current = _current_rules(context)
    if not current:
        return "No custom rules to remove. You're using system defaults only."
    if not rules.strip():
        return "Tell me which rule you'd like to remove."

    try:
        updated = await _rewrite_rules(
            context,
            "Remove the specified rules from the existing rules.\n\n"
            f"Existing rules: {current}\n"
            f"Rules to remove: {rules.strip()}\n\n"
            "Return the updated rules after removal. If all rules are removed or the result "
            "is empty, return an empty string.\n"
            "Keep the format consistent with the existing rules.",
        )
    except Exception as e:
        logger.error(f"Rule removal failed: {e}")
        return "I couldn't update that rule. Please try again."

    if not await _save_rules(context, updated or None):
        return _NO_STORE
    if updated:
        return f"Removed rule: {rules.strip()}\n\nRemaining rules:\n{updated}"
    return "All custom rules removed. Using system defaults only."


@tool(needs_approval=True, risk_level="write", category="email")
async def replace_email_rules(
    rules: Annotated[str, "The new rules that will replace all existing ones."],
    *,
    context: AgentToolContext,
) -> str:
    """Replace ALL of the user's email importance rules with new ones."""
    if not rules.strip():
        return "No rules provided. Use clear_email_rules to remove all rules instead."

    if not await _save_rules(context, rules.strip()):
        return _NO_STORE
    logger.info(f"Email rules replaced for user {context.tenant_id}")
    return f"Email importance rules replaced successfully!\n\nNew rules:\n{rules.strip()}"


@tool(needs_approval=True, risk_level="write", category="email")
async def clear_email_rules(*, context: AgentToolContext) -> str:
    """Clear all custom email rules, falling back to system defaults only."""
    if not await _save_rules(context, None):
        return _NO_STORE
    logger.info(f"Email rules cleared for user {context.tenant_id}")
    return (
        "All custom rules cleared. You'll receive notifications based on "
        "system default rules only."
    )


@tool(risk_level="write", category="email")
async def set_email_notifications(
    enabled: Annotated[bool, "True to turn email notifications on, False to turn them off."],
    *,
    context: AgentToolContext,
) -> str:
    """Turn email notifications on or off."""
    store = _credential_store(context)
    if not store:
        return _NO_STORE
    await store.update_notification_preferences(
        context.tenant_id, {"email_hook_enabled": enabled}
    )
    logger.info(
        f"Email notifications {'enabled' if enabled else 'disabled'} for user {context.tenant_id}"
    )
    if enabled:
        return "Email notifications turned on. I'll text you when important emails arrive."
    return "Email notifications turned off. I won't text you about emails anymore."
