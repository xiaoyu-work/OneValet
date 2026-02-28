"""
Expense Tools - Standalone API functions for ExpenseAgent's mini ReAct loop.

Provides tools for logging expenses, querying spending, managing budgets,
and handling receipt images.
"""

import base64
import logging
from datetime import date, datetime, timedelta
from typing import Annotated, Optional

from onevalet.models import AgentToolContext
from onevalet.tool_decorator import tool

logger = logging.getLogger(__name__)


# =============================================================================
# Shared Helpers
# =============================================================================

def _get_repos(context: AgentToolContext):
    """Get expense, budget, and receipt repositories from context hints.

    Returns:
        Tuple of (ExpenseRepository, BudgetRepository, ReceiptRepository).
        Any or all may be None if the database is not available.
    """
    db = context.context_hints.get("db") if context.context_hints else None
    if not db:
        return None, None, None
    from .repository import ExpenseRepository
    from .budget_repository import BudgetRepository
    from .receipt_repository import ReceiptRepository
    return ExpenseRepository(db), BudgetRepository(db), ReceiptRepository(db)


def _parse_date(date_str: str) -> date:
    """Parse a user-friendly date string into a date object.

    Supports:
        - "today" -> today's date
        - "yesterday" -> yesterday's date
        - ISO format string (YYYY-MM-DD)
    """
    lower = date_str.strip().lower()
    if lower == "today" or lower == "":
        return date.today()
    if lower == "yesterday":
        return date.today() - timedelta(days=1)
    try:
        return date.fromisoformat(date_str.strip())
    except ValueError:
        # Try dateutil as fallback for more flexible parsing
        try:
            from dateutil import parser as date_parser
            return date_parser.parse(date_str.strip()).date()
        except Exception:
            return date.today()


def _parse_period(period: str) -> tuple[date, date]:
    """Parse a period string into (start_date, end_date) inclusive.

    Supports:
        - "today"       -> today only
        - "this_week"   -> Monday through Sunday of the current week
        - "this_month"  -> first through last day of the current month
        - "last_month"  -> first through last day of last month
        - "YYYY-MM"     -> first through last day of that month
    """
    today = date.today()
    lower = period.strip().lower()

    if lower == "today":
        return today, today

    if lower == "this_week":
        start = today - timedelta(days=today.weekday())  # Monday
        end = start + timedelta(days=6)  # Sunday
        return start, end

    if lower in ("this_month", ""):
        start = today.replace(day=1)
        # Last day of this month
        if today.month == 12:
            end = today.replace(year=today.year + 1, month=1, day=1) - timedelta(days=1)
        else:
            end = today.replace(month=today.month + 1, day=1) - timedelta(days=1)
        return start, end

    if lower == "last_month":
        first_this_month = today.replace(day=1)
        end = first_this_month - timedelta(days=1)
        start = end.replace(day=1)
        return start, end

    # Try YYYY-MM format
    try:
        parts = lower.split("-")
        if len(parts) == 2:
            year, month = int(parts[0]), int(parts[1])
            start = date(year, month, 1)
            if month == 12:
                end = date(year + 1, 1, 1) - timedelta(days=1)
            else:
                end = date(year, month + 1, 1) - timedelta(days=1)
            return start, end
    except (ValueError, IndexError):
        pass

    # Default to this month
    start = today.replace(day=1)
    if today.month == 12:
        end = today.replace(year=today.year + 1, month=1, day=1) - timedelta(days=1)
    else:
        end = today.replace(month=today.month + 1, day=1) - timedelta(days=1)
    return start, end


def _format_amount(amount: float, currency: str = "USD") -> str:
    """Format an amount with currency symbol."""
    symbols = {
        "USD": "$", "EUR": "\u20ac", "GBP": "\u00a3", "JPY": "\u00a5",
        "CAD": "CA$", "AUD": "A$", "CHF": "CHF ",
    }
    symbol = symbols.get(currency.upper(), f"{currency.upper()} ")
    if currency.upper() == "JPY":
        return f"{symbol}{amount:,.0f}"
    return f"{symbol}{amount:,.2f}"


def _format_expense_table(expenses: list, currency: str = "USD") -> str:
    """Format a list of expense dicts as a readable text table."""
    if not expenses:
        return "No expenses found."

    lines = []
    lines.append(f"{'Date':<12} {'Amount':>10}  {'Category':<14} {'Merchant':<16} {'Description'}")
    lines.append("-" * 78)

    for exp in expenses:
        d = exp.get("date", "")
        if isinstance(d, date):
            d = d.isoformat()
        amount = _format_amount(exp.get("amount", 0), exp.get("currency", currency))
        cat = exp.get("category", "")[:14]
        merchant = exp.get("merchant", "")[:16]
        desc = exp.get("description", "")[:30]
        lines.append(f"{d:<12} {amount:>10}  {cat:<14} {merchant:<16} {desc}")

    return "\n".join(lines)


def _budget_warning(budget_repo, expense_repo, tenant_id: str,
                    category: str, currency: str) -> str:
    """Check if spending exceeds or approaches budget limits.

    Returns a warning string or empty string.
    """
    warnings = []
    today = date.today()
    start = today.replace(day=1)
    if today.month == 12:
        end = today.replace(year=today.year + 1, month=1, day=1) - timedelta(days=1)
    else:
        end = today.replace(month=today.month + 1, day=1) - timedelta(days=1)

    # Check category-specific budget
    cat_budget = budget_repo.get_budget(tenant_id, category)
    if cat_budget:
        cat_total = expense_repo.monthly_total(tenant_id, start, end, category=category)
        limit = cat_budget.get("monthly_limit", 0)
        if limit > 0:
            pct = (cat_total / limit) * 100
            if cat_total > limit:
                over = _format_amount(cat_total - limit, currency)
                warnings.append(
                    f"Warning: {category} spending is {over} OVER budget "
                    f"({_format_amount(cat_total, currency)} / {_format_amount(limit, currency)})."
                )
            elif pct >= 80:
                warnings.append(
                    f"Note: {category} spending at {pct:.0f}% of budget "
                    f"({_format_amount(cat_total, currency)} / {_format_amount(limit, currency)})."
                )

    # Check total budget
    total_budget = budget_repo.get_budget(tenant_id, "_total")
    if total_budget:
        grand_total = expense_repo.monthly_total(tenant_id, start, end)
        limit = total_budget.get("monthly_limit", 0)
        if limit > 0:
            pct = (grand_total / limit) * 100
            if grand_total > limit:
                over = _format_amount(grand_total - limit, currency)
                warnings.append(
                    f"Warning: Total spending is {over} OVER budget "
                    f"({_format_amount(grand_total, currency)} / {_format_amount(limit, currency)})."
                )
            elif pct >= 80:
                warnings.append(
                    f"Note: Total spending at {pct:.0f}% of budget "
                    f"({_format_amount(grand_total, currency)} / {_format_amount(limit, currency)})."
                )

    return " ".join(warnings)


# =============================================================================
# log_expense
# =============================================================================

@tool
async def log_expense(
    amount: Annotated[float, "Amount spent (e.g. 15.50)."],
    category: Annotated[str, "Expense category: food, transport, shopping, entertainment, housing, health, education, or other."],
    description: Annotated[str, "Brief description of the expense (e.g. 'lunch at cafe')."] = "",
    merchant: Annotated[str, "Merchant or vendor name (e.g. 'Starbucks')."] = "",
    date: Annotated[str, "Date of expense: 'today', 'yesterday', or YYYY-MM-DD. Defaults to today."] = "today",
    currency: Annotated[str, "Currency code (e.g. USD, EUR). Defaults to USD."] = "USD",
    *,
    context: AgentToolContext,
) -> str:
    """Log a new expense with amount, category, and optional details."""
    expense_repo, budget_repo, _ = _get_repos(context)
    if not expense_repo:
        return "Expense tracking is not available. Database not configured."

    try:
        expense_date = _parse_date(date)
    except Exception:
        expense_date = _parse_date("today")

    category_lower = category.strip().lower()

    try:
        expense_repo.add(
            tenant_id=context.tenant_id,
            amount=amount,
            category=category_lower,
            description=description,
            merchant=merchant,
            expense_date=expense_date,
            currency=currency.upper(),
        )
    except Exception as e:
        logger.error(f"Failed to log expense: {e}", exc_info=True)
        return "Sorry, I couldn't log that expense. Please try again."

    # Get monthly total for this category
    today = _parse_date("today")
    month_start = today.replace(day=1)
    if today.month == 12:
        month_end = today.replace(year=today.year + 1, month=1, day=1) - timedelta(days=1)
    else:
        month_end = today.replace(month=today.month + 1, day=1) - timedelta(days=1)

    cat_total = expense_repo.monthly_total(
        context.tenant_id, month_start, month_end, category=category_lower,
    )

    desc_part = f" ({description})" if description else ""
    result = (
        f"Logged: {category_lower} {_format_amount(amount, currency)}{desc_part}. "
        f"This month's {category_lower}: {_format_amount(cat_total, currency)}."
    )

    # Check budget warnings
    if budget_repo:
        warning = _budget_warning(
            budget_repo, expense_repo, context.tenant_id, category_lower, currency,
        )
        if warning:
            result += f" {warning}"

    return result


# =============================================================================
# query_expenses
# =============================================================================

@tool
async def query_expenses(
    period: Annotated[str, "Time period: 'today', 'this_week', 'this_month', 'last_month', or 'YYYY-MM'."] = "this_month",
    category: Annotated[str, "Filter by category (e.g. 'food'). Leave empty for all categories."] = "",
    merchant: Annotated[str, "Filter by merchant name. Leave empty for all merchants."] = "",
    limit: Annotated[int, "Maximum number of results to return."] = 20,
    *,
    context: AgentToolContext,
) -> str:
    """Query and list expenses for a given period, optionally filtered by category or merchant."""
    expense_repo, _, _ = _get_repos(context)
    if not expense_repo:
        return "Expense tracking is not available. Database not configured."

    start_date, end_date = _parse_period(period)

    try:
        expenses = expense_repo.query(
            tenant_id=context.tenant_id,
            start_date=start_date,
            end_date=end_date,
            category=category.strip().lower() if category else "",
            merchant=merchant.strip() if merchant else "",
            limit=limit,
        )
    except Exception as e:
        logger.error(f"Failed to query expenses: {e}", exc_info=True)
        return "Sorry, I couldn't retrieve your expenses. Please try again."

    if not expenses:
        period_label = period if period else "this month"
        filters = []
        if category:
            filters.append(f"category={category}")
        if merchant:
            filters.append(f"merchant={merchant}")
        filter_str = f" ({', '.join(filters)})" if filters else ""
        return f"No expenses found for {period_label}{filter_str}."

    # Calculate total
    total = sum(exp.get("amount", 0) for exp in expenses)
    currency = expenses[0].get("currency", "USD") if expenses else "USD"
    table = _format_expense_table(expenses, currency)
    period_label = period if period else "this month"

    return (
        f"Expenses for {period_label} ({len(expenses)} entries, "
        f"total: {_format_amount(total, currency)}):\n\n{table}"
    )


# =============================================================================
# delete_expense
# =============================================================================

async def _preview_delete_expense(args: dict, context) -> str:
    hint = args.get("hint", "")
    return f"Search for and delete expense matching: \"{hint}\"?"


@tool(needs_approval=True, get_preview=_preview_delete_expense)
async def delete_expense(
    hint: Annotated[str, "Keywords to find the expense to delete (e.g. 'Starbucks yesterday', '$15 lunch')."],
    *,
    context: AgentToolContext,
) -> str:
    """Delete an expense by searching for it with keywords. Returns list if multiple matches found."""
    expense_repo, _, _ = _get_repos(context)
    if not expense_repo:
        return "Expense tracking is not available. Database not configured."

    if not hint.strip():
        return "Please provide a description, amount, or merchant to identify the expense."

    try:
        matches = expense_repo.search(tenant_id=context.tenant_id, hint=hint.strip())
    except Exception as e:
        logger.error(f"Failed to search expenses: {e}", exc_info=True)
        return "Sorry, I couldn't search for that expense. Please try again."

    if not matches:
        return f"No matching expense found for \"{hint}\"."

    if len(matches) == 1:
        exp = matches[0]
        try:
            expense_repo.delete(tenant_id=context.tenant_id, expense_id=exp.get("id"))
            d = exp.get("date", "")
            if isinstance(d, date):
                d = d.isoformat()
            return (
                f"Deleted: {exp.get('category', '')} "
                f"{_format_amount(exp.get('amount', 0), exp.get('currency', 'USD'))} "
                f"on {d}"
                f"{' - ' + exp.get('description', '') if exp.get('description') else ''}."
            )
        except Exception as e:
            logger.error(f"Failed to delete expense: {e}", exc_info=True)
            return "Sorry, I couldn't delete that expense. Please try again."

    # Multiple matches - ask user to be more specific
    lines = [f"Found {len(matches)} expenses matching \"{hint}\":\n"]
    for i, exp in enumerate(matches[:10], 1):
        d = exp.get("date", "")
        if isinstance(d, date):
            d = d.isoformat()
        amount = _format_amount(exp.get("amount", 0), exp.get("currency", "USD"))
        cat = exp.get("category", "")
        desc = exp.get("description", "")
        merchant_name = exp.get("merchant", "")
        detail = desc or merchant_name
        lines.append(f"{i}. {d} | {amount} | {cat}" + (f" | {detail}" if detail else ""))
    lines.append("\nPlease be more specific so I can identify the right expense.")
    return "\n".join(lines)


# =============================================================================
# spending_summary
# =============================================================================

@tool
async def spending_summary(
    period: Annotated[str, "Time period: 'today', 'this_week', 'this_month', 'last_month', or 'YYYY-MM'."] = "this_month",
    *,
    context: AgentToolContext,
) -> str:
    """Show a spending summary broken down by category for the given period."""
    expense_repo, budget_repo, _ = _get_repos(context)
    if not expense_repo:
        return "Expense tracking is not available. Database not configured."

    start_date, end_date = _parse_period(period)

    try:
        summary = expense_repo.summary_by_category(
            tenant_id=context.tenant_id,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as e:
        logger.error(f"Failed to get spending summary: {e}", exc_info=True)
        return "Sorry, I couldn't generate a spending summary. Please try again."

    if not summary:
        period_label = period if period else "this month"
        return f"No expenses found for {period_label}."

    # Determine primary currency from first entry
    currency = summary[0].get("currency", "USD") if summary else "USD"

    lines = []
    period_label = period if period else "this month"
    lines.append(f"Spending summary for {period_label}:\n")
    lines.append(f"{'Category':<14} {'Total':>10}  {'Count':>5}  {'vs Budget'}")
    lines.append("-" * 52)

    grand_total = 0.0
    for row in summary:
        cat = row.get("category", "other")[:14]
        total = row.get("total", 0.0)
        count = row.get("count", 0)
        grand_total += total

        budget_str = ""
        if budget_repo:
            budget = budget_repo.get_budget(context.tenant_id, cat.strip())
            if budget:
                limit = budget.get("monthly_limit", 0)
                if limit > 0:
                    pct = (total / limit) * 100
                    budget_str = f"{pct:.0f}% of {_format_amount(limit, currency)}"
                    if total > limit:
                        budget_str += " OVER"

        lines.append(
            f"{cat:<14} {_format_amount(total, currency):>10}  {count:>5}  {budget_str}"
        )

    lines.append("-" * 52)

    # Grand total with optional total budget comparison
    total_budget_str = ""
    if budget_repo:
        total_budget = budget_repo.get_budget(context.tenant_id, "_total")
        if total_budget:
            limit = total_budget.get("monthly_limit", 0)
            if limit > 0:
                pct = (grand_total / limit) * 100
                total_budget_str = f"{pct:.0f}% of {_format_amount(limit, currency)}"
                if grand_total > limit:
                    total_budget_str += " OVER"

    lines.append(
        f"{'TOTAL':<14} {_format_amount(grand_total, currency):>10}  {'':>5}  {total_budget_str}"
    )

    return "\n".join(lines)


# =============================================================================
# set_budget
# =============================================================================

@tool
async def set_budget(
    category: Annotated[str, "Category to set budget for (e.g. 'food'), or '_total' for overall monthly budget."] = "_total",
    monthly_limit: Annotated[float, "Monthly spending limit amount."] = 0,
    currency: Annotated[str, "Currency code (e.g. USD, EUR). Defaults to USD."] = "USD",
    *,
    context: AgentToolContext,
) -> str:
    """Set a monthly spending budget for a specific category or overall total."""
    _, budget_repo, _ = _get_repos(context)
    if not budget_repo:
        return "Budget tracking is not available. Database not configured."

    if monthly_limit <= 0:
        return "Please provide a positive monthly limit amount."

    category_lower = category.strip().lower()

    try:
        budget_repo.set_budget(
            tenant_id=context.tenant_id,
            category=category_lower,
            monthly_limit=monthly_limit,
            currency=currency.upper(),
        )
    except Exception as e:
        logger.error(f"Failed to set budget: {e}", exc_info=True)
        return "Sorry, I couldn't set that budget. Please try again."

    if category_lower == "_total":
        return f"Budget set: total {_format_amount(monthly_limit, currency)}/month."
    return f"Budget set: {category_lower} {_format_amount(monthly_limit, currency)}/month."


# =============================================================================
# budget_status
# =============================================================================

@tool
async def budget_status(
    *,
    context: AgentToolContext,
) -> str:
    """Show current budget status with spending progress for all configured budgets."""
    expense_repo, budget_repo, _ = _get_repos(context)
    if not budget_repo:
        return "Budget tracking is not available. Database not configured."
    if not expense_repo:
        return "Expense tracking is not available. Database not configured."

    try:
        budgets = budget_repo.get_all_budgets(tenant_id=context.tenant_id)
    except Exception as e:
        logger.error(f"Failed to get budgets: {e}", exc_info=True)
        return "Sorry, I couldn't retrieve your budgets. Please try again."

    if not budgets:
        return "No budgets configured yet. Use set_budget to create one."

    today = date.today()
    month_start = today.replace(day=1)
    if today.month == 12:
        month_end = today.replace(year=today.year + 1, month=1, day=1) - timedelta(days=1)
    else:
        month_end = today.replace(month=today.month + 1, day=1) - timedelta(days=1)

    lines = []
    lines.append(f"Budget status for {today.strftime('%B %Y')}:\n")
    lines.append(f"{'Category':<14} {'Budget':>10}  {'Spent':>10}  {'Remaining':>10}  {'% Used':>7}")
    lines.append("-" * 62)

    for budget in budgets:
        cat = budget.get("category", "")
        limit = budget.get("monthly_limit", 0)
        currency = budget.get("currency", "USD")

        if cat == "_total":
            spent = expense_repo.monthly_total(context.tenant_id, month_start, month_end)
        else:
            spent = expense_repo.monthly_total(
                context.tenant_id, month_start, month_end, category=cat,
            )

        remaining = limit - spent
        pct = (spent / limit * 100) if limit > 0 else 0

        display_cat = "TOTAL" if cat == "_total" else cat[:14]

        status_marker = ""
        if pct >= 100:
            status_marker = " OVER"
        elif pct >= 80:
            status_marker = " !"

        lines.append(
            f"{display_cat:<14} "
            f"{_format_amount(limit, currency):>10}  "
            f"{_format_amount(spent, currency):>10}  "
            f"{_format_amount(remaining, currency):>10}  "
            f"{pct:>5.0f}%{status_marker}"
        )

    return "\n".join(lines)


# =============================================================================
# upload_receipt
# =============================================================================

@tool
async def upload_receipt(
    description: Annotated[str, "Description or notes about the receipt (e.g. 'dinner receipt from Friday')."] = "",
    *,
    context: AgentToolContext,
) -> str:
    """Save a receipt image from the conversation. The user must attach an image before calling this tool."""
    _, _, receipt_repo = _get_repos(context)
    if not receipt_repo:
        return "Receipt storage is not available. Database not configured."

    # Get images from context hints
    user_images = context.context_hints.get("user_images") if context.context_hints else None
    if not user_images or len(user_images) == 0:
        return "No image found. Please attach a receipt photo and try again."

    image_data = user_images[0]

    # Decode base64 if the image is a base64 string
    if isinstance(image_data, str):
        # Could be base64-encoded or a data URI
        if image_data.startswith("data:"):
            # data:image/png;base64,<data>
            try:
                _, encoded = image_data.split(",", 1)
                raw_bytes = base64.b64decode(encoded)
            except Exception:
                raw_bytes = image_data.encode()
        else:
            try:
                raw_bytes = base64.b64decode(image_data)
            except Exception:
                raw_bytes = image_data.encode()
    elif isinstance(image_data, bytes):
        raw_bytes = image_data
    elif isinstance(image_data, dict):
        # Image might be passed as a dict with 'data' key
        raw_data = image_data.get("data", "")
        if isinstance(raw_data, str):
            try:
                raw_bytes = base64.b64decode(raw_data)
            except Exception:
                raw_bytes = raw_data.encode()
        else:
            raw_bytes = raw_data
    else:
        return "Unsupported image format. Please try again with a different image."

    # Prepare filename
    today = date.today()
    filename = f"receipt_{today.isoformat()}_{datetime.now().strftime('%H%M%S')}.jpg"
    mime = "image/jpeg"
    if isinstance(image_data, dict):
        mime = image_data.get("media_type", mime)

    # Try to upload to cloud storage if available
    storage_url = ""
    storage_file_id = ""
    storage_provider = context.context_hints.get("cloud_storage_provider") if context.context_hints else None
    if storage_provider:
        try:
            folder_path = f"OneValet/Receipts/{today.strftime('%Y-%m')}"
            upload_result = await storage_provider.upload_file(
                file_name=filename,
                file_data=raw_bytes,
                mime_type=mime,
                folder_path=folder_path,
            )
            if upload_result.get("success") and upload_result.get("data"):
                storage_url = upload_result["data"].get("url", "")
                storage_file_id = upload_result["data"].get("id", "")
        except Exception as e:
            logger.warning(f"Failed to upload receipt to cloud storage: {e}")

    # Store in receipt repository
    try:
        record = await receipt_repo.add(
            tenant_id=context.tenant_id,
            file_name=filename,
            storage_file_id=storage_file_id or None,
            storage_url=storage_url or None,
            ocr_text=description,
        )
    except Exception as e:
        logger.error(f"Failed to store receipt record: {e}", exc_info=True)
        return "Sorry, I couldn't save the receipt record. Please try again."

    receipt_id = record.get("id", "") if record else ""
    result = f"Receipt saved (ID: {receipt_id})."
    if storage_url:
        result += f" Uploaded to: {storage_url}"
    if description:
        result += f" Notes: {description}"
    return result


# =============================================================================
# search_receipts
# =============================================================================

@tool
async def search_receipts(
    query: Annotated[str, "Search keywords to find receipts (e.g. 'restaurant', 'March', 'uber')."],
    period: Annotated[str, "Optional time period filter: 'this_month', 'last_month', or 'YYYY-MM'."] = "",
    *,
    context: AgentToolContext,
) -> str:
    """Search saved receipts by description or keywords, optionally filtered by time period."""
    expense_repo, _, receipt_repo = _get_repos(context)
    if not receipt_repo:
        return "Receipt storage is not available. Database not configured."

    start_date = None
    end_date = None
    if period:
        start_date, end_date = _parse_period(period)

    try:
        receipts = receipt_repo.search_by_text(
            tenant_id=context.tenant_id,
            query=query.strip(),
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as e:
        logger.error(f"Failed to search receipts: {e}", exc_info=True)
        return "Sorry, I couldn't search receipts. Please try again."

    # Cross-reference with expenses if available
    expense_matches = []
    if expense_repo:
        try:
            expense_matches = expense_repo.search(
                tenant_id=context.tenant_id,
                hint=query.strip(),
            )
        except Exception:
            pass

    if not receipts and not expense_matches:
        period_str = f" in {period}" if period else ""
        return f"No receipts found matching \"{query}\"{period_str}."

    lines = []

    if receipts:
        lines.append(f"Found {len(receipts)} receipt(s) matching \"{query}\":\n")
        for i, receipt in enumerate(receipts, 1):
            d = receipt.get("receipt_date", receipt.get("created_at", ""))
            if isinstance(d, (date, datetime)):
                d = d.isoformat()
            desc = receipt.get("description", "")
            url = receipt.get("storage_url", "")
            line = f"{i}. {d}"
            if desc:
                line += f" - {desc}"
            if url:
                line += f" [{url}]"
            lines.append(line)

    if expense_matches:
        if receipts:
            lines.append("")
        lines.append(f"Related expenses ({len(expense_matches)}):\n")
        for i, exp in enumerate(expense_matches[:5], 1):
            d = exp.get("date", "")
            if isinstance(d, date):
                d = d.isoformat()
            amount = _format_amount(exp.get("amount", 0), exp.get("currency", "USD"))
            cat = exp.get("category", "")
            desc = exp.get("description", "")
            lines.append(f"  {i}. {d} | {amount} | {cat}" + (f" | {desc}" if desc else ""))

    return "\n".join(lines)
