"""
03_leave_request.py - Approval workflow example

An agent that handles leave/vacation requests with:
- Field collection (dates, reason)
- Approval flow before submission
- Manager notification

This demonstrates:
- requires_approval=True
- get_approval_prompt() override
- Human-in-the-loop confirmation
"""

import asyncio
from datetime import datetime
from flowagents import StandardAgent, InputField, OutputField, flowagent, AgentStatus


def validate_date(value: str) -> bool:
    """Validate date format YYYY-MM-DD"""
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except ValueError:
        raise ValueError("Please use format: YYYY-MM-DD (e.g., 2024-12-25)")


@flowagent(
    triggers=["leave request", "vacation", "time off", "take leave"],
    requires_approval=True
)
class LeaveRequestAgent(StandardAgent):
    """Submit a leave/vacation request"""

    leave_type = InputField(
        "What type of leave? (annual/sick/personal)",
        description="Type of leave"
    )
    start_date = InputField(
        "Start date? (YYYY-MM-DD)",
        validator=validate_date
    )
    end_date = InputField(
        "End date? (YYYY-MM-DD)",
        validator=validate_date
    )
    reason = InputField(
        "Reason for leave?",
        description="Brief reason"
    )

    # Output field - set after successful submission
    request_id = OutputField(str, "The leave request ID")

    def get_approval_prompt(self) -> str:
        """Custom approval message shown to user"""
        return f"""
Please review your leave request:

  Type: {self.leave_type}
  From: {self.start_date}
  To: {self.end_date}
  Reason: {self.reason}

Submit this request? (yes/no)
        """.strip()

    async def on_running(self, msg):
        # Generate request ID
        import random
        self.request_id = f"LR-{random.randint(10000, 99999)}"

        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message=f"Leave request submitted! Request ID: {self.request_id}\n\nYour manager will be notified."
        )


async def main():
    from flowagents import Orchestrator, OpenAIClient

    llm = OpenAIClient(api_key="sk-xxx", model="gpt-4o-mini")
    orchestrator = Orchestrator(llm_client=llm)
    await orchestrator.initialize()

    conversations = [
        "I need to request time off",
        "annual",                    # leave type
        "2024-12-23",               # start date
        "2024-12-27",               # end date
        "Family vacation",          # reason
        "yes",                      # approval
    ]

    for user_input in conversations:
        print(f"User: {user_input}")
        result = await orchestrator.handle_message("user_1", user_input)
        print(f"Agent: {result.raw_message}\n")


if __name__ == "__main__":
    asyncio.run(main())
