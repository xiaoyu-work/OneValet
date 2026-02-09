"""
04_expense_workflow.py - Expense reimbursement with approval

Demonstrates:
- Field collection with validation
- Approval workflow (requires_approval=True)
- Custom approval prompt
"""

import asyncio
from flowagents import (
    StandardAgent, InputField, OutputField, flowagent, AgentStatus,
    Orchestrator, OpenAIClient
)


def validate_amount(value: str) -> bool:
    """Validate expense amount"""
    try:
        amount = float(value.replace("$", "").replace(",", ""))
        if amount <= 0:
            raise ValueError("Amount must be positive")
        return True
    except ValueError:
        raise ValueError("Please enter a valid amount (e.g., 75.50)")


@flowagent(
    triggers=["expense", "reimbursement", "submit expense"],
    requires_approval=True
)
class ExpenseAgent(StandardAgent):
    """Submit expense for reimbursement"""

    amount = InputField("How much was the expense?", validator=validate_amount)
    category = InputField("Category? (travel/meals/supplies/other)")
    description = InputField("Brief description?")

    def get_approval_prompt(self) -> str:
        return f"""
Please review your expense:

  Amount: ${float(self.amount.replace('$', '').replace(',', '')):.2f}
  Category: {self.category}
  Description: {self.description}

Submit this expense? (yes/no)
        """.strip()

    async def on_running(self, msg):
        import random
        expense_id = f"EXP-{random.randint(10000, 99999)}"

        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message=f"Expense submitted! ID: {expense_id}\nYou'll receive confirmation once approved."
        )


async def main():
    llm = OpenAIClient(api_key="sk-xxx", model="gpt-4o-mini")
    orchestrator = Orchestrator(llm_client=llm)
    await orchestrator.initialize()

    print("=== Expense Reimbursement ===\n")

    conversations = [
        "I need to submit an expense",
        "75.50",
        "meals",
        "Team lunch meeting",
        "yes",
    ]

    for user_input in conversations:
        print(f"User: {user_input}")
        result = await orchestrator.handle_message("user_1", user_input)
        print(f"Agent: {result.raw_message}\n")


if __name__ == "__main__":
    asyncio.run(main())
