"""
EmailImportanceAgent - Determine if emails are important and require notification

Combines system default rules and user custom rules to evaluate email importance
"""

import logging
from typing import Dict, Any, List

from onevalet import valet, StandardAgent, InputField

logger = logging.getLogger(__name__)


@valet(expose_as_tool=False)
class EmailImportanceAgent(StandardAgent):
    """Classify email importance level. Use internally to determine if an email is urgent or important."""

    # System default rules (LLM's own judgment criteria)
    SYSTEM_RULES = """
    1. Urgent matters: contains urgent, ASAP, immediate, critical, time-sensitive
    2. Security-related: verification codes, password reset, login alerts, suspicious activity, 2FA codes
    3. Financial: payment confirmation, transfer notification, bill due, invoice, payment failed
    4. Travel changes: flight changes, hotel cancellation, meeting reschedule, booking confirmation
    5. Important notices: interview invitation, job offer, contract signing, delivery confirmation
    6. Action required: approval needed, response required, deadline approaching
    """

    email = InputField(
        prompt="",  # This field comes from workflow trigger_data
        description="Email data to evaluate",
    )

    def get_purpose_description(self) -> str:
        return "Evaluate if an email is important enough to notify the user immediately"

    def needs_approval(self) -> bool:
        return False

    def should_send_initial_response(self) -> bool:
        return False

    async def _extract_fields(self, user_input: str) -> Dict[str, Any]:
        """
        This agent is typically called from workflows with trigger_data,
        not from direct user input
        """
        return {"email": None}

    async def _execute(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate email importance

        Args:
            fields: {
                "email": {
                    "sender": str,
                    "subject": str,
                    "snippet": str,
                    "date": str (optional),
                    "unread": bool (optional)
                }
            }

        Returns:
            {
                "is_important": bool,
                "rule_type": str ("system", "user", "both", "none"),
                "matched_rule": str,
                "reason": str (brief explanation for SMS)
            }
        """
        try:
            email = fields.get("email", {})

            if not email:
                return {
                    "is_important": False,
                    "rule_type": "none",
                    "matched_rule": "",
                    "reason": "No email data provided"
                }

            # Get user's custom importance rules from context_hints
            user_profile = self.context_hints.get("user_profile", {})
            user_rules = user_profile.get("email_importance_rules", "") if user_profile else ""

            # Build evaluation prompt
            prompt = f"""
Determine if this email is important and requires immediate notification to the user.

[SYSTEM DEFAULT RULES]
{self.SYSTEM_RULES}

{"[USER CUSTOM RULES]" if user_rules else "[USER CUSTOM RULES]"}
{user_rules if user_rules else "User has not set custom rules. Use system rules only."}

[EMAIL INFORMATION]
From: {email.get('sender', 'Unknown')}
Subject: {email.get('subject', '(No subject)')}
Preview: {email.get('snippet', '')[:300]}

EVALUATION CRITERIA:
1. If the email matches ANY user custom rule, it is IMPORTANT (user rules have highest priority)
2. If no user rules match, check against system default rules
3. If user explicitly excludes certain types (e.g., "don't notify about newsletters"), respect that
4. Be conservative - only mark as important if there's a clear match

Return JSON with this exact structure:
{{
    "is_important": true or false,
    "rule_type": "system" or "user" or "both" or "none",
    "matched_rule": "brief description of which rule matched",
    "reason": "very brief explanation (max 50 chars, suitable for SMS notification)"
}}

Examples:
- Email from boss about urgent meeting -> is_important: true, rule_type: "user", reason: "Email from boss"
- Verification code email -> is_important: true, rule_type: "system", reason: "Security alert"
- Newsletter from shopping site -> is_important: false, rule_type: "none", reason: "Promotional email"
"""

            # Call LLM for evaluation
            result = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You evaluate email importance. Return JSON only."},
                    {"role": "user", "content": prompt}
                ],
                response_format="json_object",
                enable_thinking=False
            )

            import json
            evaluated = json.loads(result.content.strip())

            logger.info(
                f"Email importance check: {email.get('subject', 'N/A')[:50]} "
                f"-> {evaluated.get('is_important', False)}"
            )

            return {
                "is_important": evaluated.get("is_important", False),
                "rule_type": evaluated.get("rule_type", "none"),
                "matched_rule": evaluated.get("matched_rule", ""),
                "reason": evaluated.get("reason", "")
            }

        except Exception as e:
            logger.error(f"Email importance check error: {e}", exc_info=True)
            # Default to not important on error to avoid spamming user
            return {
                "is_important": False,
                "rule_type": "error",
                "matched_rule": "",
                "reason": f"Error: {str(e)}"
            }
