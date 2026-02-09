"""
EmailPreferenceAgent - Manage user's email importance rules and notification preferences

Allows users to set, update, view, and manage custom rules for email importance
"""

import logging
import json
from typing import Dict, Any, List

from onevalet import valet, StandardAgent, InputField, AgentStatus, AgentResult, Message

logger = logging.getLogger(__name__)


@valet(triggers=["email preferences", "email rules", "email notification settings"])
class EmailPreferenceAgent(StandardAgent):
    """Manage user's email importance rules and notification preferences"""

    action = InputField(
        prompt="What would you like to do with your email rules?",
        description="Action to perform on email rules",
    )
    rules = InputField(
        prompt="Please describe your email importance rules",
        description="The email importance rules description",
        required=False,
    )

    def get_purpose_description(self) -> str:
        return "Set and manage email importance rules and notification preferences"

    def needs_approval(self) -> bool:
        action = self.collected_fields.get("action", "")
        return action == "replace"

    def get_approval_prompt(self) -> str:
        rules = self.collected_fields.get("rules", "")
        return f"This will replace all your existing email rules with:\n\n{rules}\n\nProceed?"

    def should_send_initial_response(self) -> bool:
        return False

    async def extract_fields(self, user_input: str) -> Dict[str, Any]:
        """Extract action and rules from user input using LLM"""

        prompt = f"""
Extract email preference action and rules from user input.

User input: "{user_input}"

Available actions:
- "show": Show current rules and notification status
- "set": Set new rules (first time or when no existing rules)
- "add": Add to existing rules
- "remove": Remove specific rules
- "replace": Completely replace existing rules
- "clear": Clear all custom rules (use system defaults only)
- "enable_notifications": Turn ON email notifications
- "disable_notifications": Turn OFF email notifications

Return JSON with this exact structure:
{{
    "action": "show" or "set" or "add" or "remove" or "replace" or "clear" or "enable_notifications" or "disable_notifications",
    "rules": "extracted rules description (only if applicable, otherwise null)"
}}

Examples:
"Show my email rules" -> {{"action": "show", "rules": null}}
"Add my wife's emails to important list" -> {{"action": "add", "rules": "emails from my wife"}}
"Don't notify me about newsletters" -> {{"action": "remove", "rules": "newsletters"}}
"My important emails are from boss and clients" -> {{"action": "set", "rules": "emails from boss and clients"}}
"Replace my rules with: only urgent emails" -> {{"action": "replace", "rules": "only urgent emails"}}
"Clear all my email rules" -> {{"action": "clear", "rules": null}}
"Turn off email notifications" -> {{"action": "disable_notifications", "rules": null}}
"Turn on email notifications" -> {{"action": "enable_notifications", "rules": null}}
"""

        try:
            result = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "Extract email preference actions. Return JSON only."},
                    {"role": "user", "content": prompt}
                ],
                response_format="json_object",
                enable_thinking=False
            )

            parsed = json.loads(result.content.strip())

            extracted = {}
            if "action" in parsed:
                extracted["action"] = parsed["action"]
            if parsed.get("rules"):
                extracted["rules"] = parsed["rules"]

            return extracted

        except Exception as e:
            logger.error(f"Field extraction failed: {e}")
            return {}

    async def on_running(self, msg: Message) -> AgentResult:
        """Execute email preference action"""
        fields = self.collected_fields
        action = fields.get("action")
        rules = fields.get("rules", "")

        try:
            # credential_store provides user profile access
            credential_store = self.context_hints.get("credential_store")

            if action == "show":
                result = await self._show_rules(credential_store)
            elif action == "set":
                result = await self._set_rules(credential_store, rules)
            elif action == "add":
                result = await self._add_rules(credential_store, rules)
            elif action == "remove":
                result = await self._remove_rules(credential_store, rules)
            elif action == "replace":
                result = await self._replace_rules(credential_store, rules)
            elif action == "clear":
                result = await self._clear_rules(credential_store)
            elif action == "enable_notifications":
                result = await self._toggle_notifications(credential_store, True)
            elif action == "disable_notifications":
                result = await self._toggle_notifications(credential_store, False)
            else:
                result = {"success": False, "message": f"Unknown action: {action}"}

            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=result.get("message", "Done.")
            )

        except Exception as e:
            logger.error(f"Email preference error: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=f"Something went wrong: {str(e)}"
            )

    async def _get_current_rules(self, credential_store) -> str:
        """Get user's current rules"""
        user_profile = self.context_hints.get("user_profile", {})
        if user_profile:
            return user_profile.get("email_importance_rules", "") or ""
        return ""

    async def _show_rules(self, credential_store) -> Dict[str, Any]:
        """Show current rules"""
        from onevalet.builtin_agents.email.importance import EmailImportanceAgent

        current_rules = await self._get_current_rules(credential_store)

        if current_rules:
            return {
                "success": True,
                "message": f"Your custom email importance rules:\n{current_rules}\n\nSystem default rules also apply:\n{EmailImportanceAgent.SYSTEM_RULES}",
            }
        else:
            return {
                "success": True,
                "message": f"You haven't set custom rules yet. Using system default rules:\n{EmailImportanceAgent.SYSTEM_RULES}",
            }

    async def _set_rules(self, credential_store, new_rules: str) -> Dict[str, Any]:
        """Set rules (first time or when no existing rules)"""
        current_rules = await self._get_current_rules(credential_store)

        if current_rules:
            return {
                "success": False,
                "message": f"You already have custom rules:\n{current_rules}\n\nUse 'add' to append or 'replace' to overwrite.",
            }

        if credential_store:
            await credential_store.update_user_profile(
                self.tenant_id,
                {"email_importance_rules": new_rules}
            )

        logger.info(f"Email rules set for user {self.tenant_id}")
        return {
            "success": True,
            "message": f"Email importance rules set successfully!\n\nYour rules:\n{new_rules}",
        }

    async def _add_rules(self, credential_store, additional_rules: str) -> Dict[str, Any]:
        """Add to existing rules"""
        current_rules = await self._get_current_rules(credential_store)

        prompt = f"""
Merge the new rules with existing rules, avoiding duplicates and conflicts.

Existing rules: {current_rules if current_rules else "None"}
New rules to add: {additional_rules}

Return the merged rules as a natural language description.
Keep it concise and clear. Use bullet points or commas to separate different rules.
"""

        result = await self.llm_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            enable_thinking=False
        )
        merged_rules = result.content.strip()

        if credential_store:
            await credential_store.update_user_profile(
                self.tenant_id,
                {"email_importance_rules": merged_rules}
            )

        logger.info(f"Email rules updated for user {self.tenant_id}")
        return {
            "success": True,
            "message": f"Added new rule: {additional_rules}\n\nUpdated rules:\n{merged_rules}",
        }

    async def _remove_rules(self, credential_store, rules_to_remove: str) -> Dict[str, Any]:
        """Remove specific rules"""
        current_rules = await self._get_current_rules(credential_store)

        if not current_rules:
            return {
                "success": False,
                "message": "No custom rules to remove. You're using system defaults only.",
            }

        prompt = f"""
Remove the specified rules from the existing rules.

Existing rules: {current_rules}
Rules to remove: {rules_to_remove}

Return the updated rules after removal. If all rules are removed or result is empty, return an empty string.
Keep the format consistent with the existing rules.
"""

        result = await self.llm_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            enable_thinking=False
        )
        updated_rules = result.content.strip()

        if credential_store:
            await credential_store.update_user_profile(
                self.tenant_id,
                {"email_importance_rules": updated_rules if updated_rules else None}
            )

        if updated_rules:
            return {
                "success": True,
                "message": f"Removed rule: {rules_to_remove}\n\nRemaining rules:\n{updated_rules}",
            }
        else:
            return {
                "success": True,
                "message": "All custom rules removed. Using system defaults only.",
            }

    async def _replace_rules(self, credential_store, new_rules: str) -> Dict[str, Any]:
        """Completely replace existing rules"""
        if credential_store:
            await credential_store.update_user_profile(
                self.tenant_id,
                {"email_importance_rules": new_rules}
            )

        logger.info(f"Email rules replaced for user {self.tenant_id}")
        return {
            "success": True,
            "message": f"Email importance rules replaced successfully!\n\nNew rules:\n{new_rules}",
        }

    async def _clear_rules(self, credential_store) -> Dict[str, Any]:
        """Clear all custom rules"""
        if credential_store:
            await credential_store.update_user_profile(
                self.tenant_id,
                {"email_importance_rules": None}
            )

        logger.info(f"Email rules cleared for user {self.tenant_id}")
        return {
            "success": True,
            "message": "All custom rules cleared. You'll receive notifications based on system default rules only.",
        }

    async def _toggle_notifications(self, credential_store, enabled: bool) -> Dict[str, Any]:
        """Enable or disable email notifications"""
        if credential_store:
            await credential_store.update_notification_preferences(
                self.tenant_id,
                {"email_hook_enabled": enabled}
            )

        action = "on" if enabled else "off"
        msg = (
            f"Email notifications turned {action}. "
            + ("I'll text you when important emails arrive." if enabled
               else "I won't text you about emails anymore.")
        )

        logger.info(f"Email notifications {'enabled' if enabled else 'disabled'} for user {self.tenant_id}")
        return {"success": True, "message": msg}
