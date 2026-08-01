"""
EmailPreferenceAgent - Manage user's email importance rules and notification preferences

Allows users to set, update, view, and manage custom rules for email importance.
The rules feed EmailEventHandler, which appends them to its classifier prompt.
"""

import logging

from koa import StandardAgent, valet

from .preference_tools import (
    add_email_rules,
    clear_email_rules,
    remove_email_rules,
    replace_email_rules,
    set_email_notifications,
    set_email_rules,
    show_email_rules,
)

logger = logging.getLogger(__name__)


@valet()
class EmailPreferenceAgent(StandardAgent):
    """Manage email notification rules. Use when the user wants to change which emails are flagged as important."""

    max_turns = 15

    _SYSTEM_PROMPT_TEMPLATE = """\
You manage the user's email importance rules — the natural-language criteria that
decide which incoming emails are worth interrupting them for.

Available tools:
- show_email_rules: Show current rules and the system defaults.
- set_email_rules: Save rules for the first time (fails if rules already exist).
- add_email_rules: Append a rule to the existing ones.
- remove_email_rules: Drop a specific rule.
- replace_email_rules: Overwrite every existing rule.
- clear_email_rules: Remove all custom rules, leaving only the system defaults.
- set_email_notifications: Turn email notifications on or off.

Instructions:
1. Pick the tool that matches what the user asked for. Adding one rule is not the
   same as replacing all of them — prefer add_email_rules unless the user clearly
   wants to start over.
2. Rules are free-form prose ("emails from my boss", "never notify me about
   newsletters"). Pass them through in the user's own words; do not invent
   criteria they did not state.
3. Rules can both promote and suppress: "don't tell me about receipts" is a valid
   rule, not a request to remove one.
4. If it is ambiguous whether the user wants to add or replace, ask before acting.
5. After the tool returns, report what changed in one short sentence.
6. Respond in the same language the user used."""

    def get_system_prompt(self) -> str:
        return self._SYSTEM_PROMPT_TEMPLATE

    tools = (
        show_email_rules,
        set_email_rules,
        add_email_rules,
        remove_email_rules,
        replace_email_rules,
        clear_email_rules,
        set_email_notifications,
    )

    def get_purpose_description(self) -> str:
        return "Set and manage email importance rules and notification preferences"

    def should_send_initial_response(self) -> bool:
        return False
