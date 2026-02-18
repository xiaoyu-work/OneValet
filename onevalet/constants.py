"""
Shared constants for the OneValet framework.

Centralizes values that are needed by both the orchestrator and
standard_agent modules to avoid circular imports and duplication.
"""

from typing import Any, Dict

COMPLETE_TASK_TOOL_NAME = "complete_task"

COMPLETE_TASK_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": COMPLETE_TASK_TOOL_NAME,
        "description": (
            "Call this tool to signal that you have completed the user's request "
            "and provide your final response. Use this when you have finished all "
            "necessary tool calls and are ready to deliver the final answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "result": {
                    "type": "string",
                    "description": (
                        "Your final response to the user. This should be comprehensive "
                        "and include all relevant information gathered from tool calls."
                    ),
                },
            },
            "required": ["result"],
        },
    },
}
