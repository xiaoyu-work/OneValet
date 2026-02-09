"""
Important Date Digest Agent - Get today's important date reminders

For morning digest: checks which dates need reminding today based on
remind_days_before settings (e.g., remind 7 days before, 1 day before, on the day).
"""
import logging
from typing import Dict, Any, List

from onevalet import valet, StandardAgent, AgentStatus, AgentResult, Message

logger = logging.getLogger(__name__)


@valet
class ImportantDateDigestAgent(StandardAgent):
    """Gets important dates that need reminding today"""

    def __init__(self, tenant_id: str = "", llm_client=None, **kwargs):
        super().__init__(
            tenant_id=tenant_id,
            llm_client=llm_client,
            **kwargs
        )

    def needs_approval(self) -> bool:
        return False

    def _get_db_client(self):
        """Get database client from context_hints"""
        return self.context_hints.get("db_client")

    async def on_running(self, msg: Message) -> AgentResult:
        """Get important dates that need reminding today"""
        try:
            db_client = self._get_db_client()
            if not db_client:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=""
                )

            dates = db_client.get_today_important_dates(self.tenant_id)

            if not dates:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=""
                )

            lines = []
            for d in dates:
                title = d.get("title", "Event")
                days_until = d.get("days_until", 0)
                date_type = d.get("date_type", "custom")

                icon = self._get_icon(date_type)

                if days_until == 0:
                    lines.append(f"{icon} {title} is TODAY!")
                elif days_until == 1:
                    lines.append(f"{icon} {title} is tomorrow")
                else:
                    lines.append(f"{icon} {title} in {days_until} days")

            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="\n".join(lines)
            )

        except Exception as e:
            logger.error(f"ImportantDateDigestAgent failed: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=""
            )

    def _get_icon(self, date_type: str) -> str:
        return {
            "birthday": "[birthday]",
            "anniversary": "[anniversary]",
            "holiday": "[holiday]",
            "custom": "[date]"
        }.get(date_type, "[date]")
