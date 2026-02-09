"""
User Tools - User profile and connected account lookup

These tools use CredentialStore via ToolExecutionContext.credentials
to list connected accounts and user profile information.
"""

import logging

from onevalet.tools import ToolRegistry, ToolDefinition, ToolCategory, ToolExecutionContext

logger = logging.getLogger(__name__)


async def get_user_accounts_executor(args: dict, context: ToolExecutionContext = None) -> str:
    """Get user's connected accounts from CredentialStore."""
    if not context or not context.user_id:
        return "Error: User ID not available"

    cred_store = context.credentials
    if not cred_store:
        return "Error: Credential store not configured"

    try:
        accounts = await cred_store.list(context.user_id)

        if not accounts:
            return "You don't have any connected accounts yet."

        output = []
        for account in accounts:
            service = account.get("service", "unknown")
            account_name = account.get("account_name", "primary")
            creds = account.get("credentials", {})
            email = creds.get("account_identifier") or creds.get("email", "")

            label = f"- {service}"
            if account_name != "primary":
                label += f" ({account_name})"
            if email:
                label += f": {email}"
            output.append(label)

        return f"Connected accounts ({len(accounts)}):\n" + "\n".join(output)

    except Exception as e:
        logger.error(f"Error getting user accounts: {e}", exc_info=True)
        return f"Error retrieving account information: {e}"


async def get_user_profile_executor(args: dict, context: ToolExecutionContext = None) -> str:
    """Get user's profile information from metadata."""
    if not context or not context.user_id:
        return "Error: User ID not available"

    # Profile can be provided via context metadata by the application
    profile = context.metadata.get("user_profile")

    if not profile:
        return "No profile information found."

    output = []
    if profile.get("first_name"):
        name = f"{profile.get('first_name', '')} {profile.get('last_name', '')}".strip()
        output.append(f"Name: {name}")
    if profile.get("email"):
        output.append(f"Email: {profile.get('email')}")
    if profile.get("phone"):
        output.append(f"Phone: {profile.get('phone')}")
    if profile.get("timezone"):
        output.append(f"Timezone: {profile.get('timezone')}")

    if not output:
        return "Profile exists but no details available."

    return "User profile:\n" + "\n".join(output)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_user_tools() -> None:
    """Register user tools with the global ToolRegistry."""
    registry = ToolRegistry.get_instance()

    registry.register(ToolDefinition(
        name="get_user_accounts",
        description=(
            "Get the user's connected accounts (email, calendar). "
            "Use this when user asks about their connected accounts."
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        executor=get_user_accounts_executor,
        category=ToolCategory.USER,
    ))

    registry.register(ToolDefinition(
        name="get_user_profile",
        description=(
            "Get the user's profile information (name, email, phone, timezone). "
            "Use this when you need to know about the user."
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        executor=get_user_profile_executor,
        category=ToolCategory.USER,
    ))

    logger.info("User tools registered")
