"""
ExpenseAgent - Agent for all expense tracking, budget, and receipt management requests.

Provides a single agent with its own mini ReAct loop for:
- Logging expenses with smart category detection
- Querying and summarizing spending
- Deleting incorrect expense entries
- Setting and monitoring budgets
- Uploading and searching receipt images

The orchestrator sees only one "ExpenseAgent" tool. The internal LLM decides
which tools to call (log_expense, query_expenses, delete_expense,
spending_summary, set_budget, budget_status, upload_receipt, search_receipts)
based on the user's request.
"""

from datetime import datetime

from onevalet import valet
from onevalet.standard_agent import StandardAgent

from .tools import (
    log_expense,
    query_expenses,
    delete_expense,
    spending_summary,
    set_budget,
    budget_status,
    upload_receipt,
    search_receipts,
)


@valet(capabilities=["expense", "spending", "budget", "money", "cost", "payment", "receipt"])
class ExpenseAgent(StandardAgent):
    """Track expenses, scan receipts, analyze spending, and manage budgets. Use when
    the user mentions expenses, spending, costs, payments, budgets, or receipts."""

    max_turns = 5

    _SYSTEM_PROMPT_TEMPLATE = """\
You are an expense tracking and budget management assistant with access to expense, \
budget, and receipt tools.

Today's date: {today} ({weekday})

Available tools:
- log_expense: Log a new expense with amount, category, and optional details (description, \
merchant, date, currency).
- query_expenses: List expenses for a time period, optionally filtered by category or merchant.
- delete_expense: Delete an expense by searching for it with keywords.
- spending_summary: Show a spending breakdown by category for a given period with budget comparison.
- set_budget: Set a monthly spending limit for a category or overall total.
- budget_status: Show current budget utilization across all configured budgets.
- upload_receipt: Save a receipt image attached by the user to storage.
- search_receipts: Search previously saved receipts by keywords and time period.

Suggested categories: food, transport, shopping, entertainment, housing, health, education, other.

Instructions:
1. When the user mentions an expense (e.g. "lunch $15", "paid $50 for groceries", \
"Uber ride $12"), call log_expense with smart category detection:
   - Food-related words (lunch, dinner, coffee, groceries, restaurant) -> category=food
   - Transport words (uber, lyft, taxi, gas, parking, metro) -> category=transport
   - Shopping words (amazon, clothes, shoes, electronics) -> category=shopping
   - Entertainment words (movie, netflix, spotify, concert, game) -> category=entertainment
   - Housing words (rent, mortgage, utilities, electric, water) -> category=housing
   - Health words (doctor, pharmacy, medicine, gym, dental) -> category=health
   - Education words (course, book, tuition, school) -> category=education
   - When uncertain, use "other" and mention the detected category to the user.
2. For spending queries (how much, total, show expenses), call query_expenses or spending_summary.
3. For deleting entries, call delete_expense directly with identifying keywords. Do not ask for confirmation.
4. For budget management (set limit, how much left), call set_budget or budget_status.
5. After logging an expense, always mention the budget status if a budget is set for that category.
6. When user_images are present in context (the user attached an image), proactively call \
upload_receipt after log_expense to save the receipt image.
7. For receipt lookups ("find my receipt", "show receipt from"), call search_receipts.
8. If the user's request is ambiguous or missing critical information (like amount), \
ask for clarification in your text response WITHOUT calling any tools.
9. After getting tool results, provide a clear, concise summary to the user.
10. When showing amounts, always use the appropriate currency symbol.
11. Parse relative dates naturally: "yesterday", "last friday", "this morning" -> appropriate date.
12. If the user provides multiple expenses in one message, log each one separately."""

    def get_system_prompt(self) -> str:
        now = datetime.now()
        prompt = self._SYSTEM_PROMPT_TEMPLATE.format(
            today=now.strftime('%Y-%m-%d'),
            weekday=now.strftime('%A'),
        )
        # Tell the agent LLM that receipt images are available in context
        if self.context_hints and self.context_hints.get("user_images"):
            prompt += (
                "\n\nIMPORTANT: The user has attached image(s) in this conversation. "
                "After logging the expense, call upload_receipt to save the receipt image."
            )
        return prompt

    tools = (
        log_expense,
        query_expenses,
        delete_expense,
        spending_summary,
        set_budget,
        budget_status,
        upload_receipt,
        search_receipts,
    )
