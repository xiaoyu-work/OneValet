"""Condition trigger — periodic polling + condition expression evaluation."""

import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Coroutine, Dict, Optional

logger = logging.getLogger(__name__)


async def evaluate_condition(
    trigger_params: Dict[str, Any],
    evaluator: Optional[Callable[..., Coroutine[Any, Any, bool]]] = None,
    last_run: Optional[datetime] = None,
) -> bool:
    """Check if a condition trigger should fire.

    Args:
        trigger_params: Trigger params with "expression", "poll_interval_minutes"
        evaluator: Optional async callable that evaluates the condition expression.
            Signature: async (expression: str, context: dict) -> bool
        last_run: When the task last fired

    Returns:
        True if the poll interval has elapsed AND the condition evaluates to True
    """
    # Poll interval gate
    poll_minutes = trigger_params.get("poll_interval_minutes", 30)
    if last_run:
        elapsed = (datetime.now() - last_run).total_seconds() / 60
        if elapsed < poll_minutes:
            return False

    # Evaluate condition
    expression = trigger_params.get("expression", "")
    if not expression:
        return False

    if evaluator is None:
        logger.debug(f"No condition evaluator configured, skipping: {expression}")
        return False

    try:
        context = trigger_params.get("context", {})
        return await evaluator(expression, context)
    except Exception as e:
        logger.warning(f"Condition evaluation failed for '{expression}': {e}")
        return False
