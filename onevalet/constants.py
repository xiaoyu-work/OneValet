"""
Shared constants for the OneValet framework.

Centralizes values that are needed by both the orchestrator and
standard_agent modules to avoid circular imports and duplication.
"""

from typing import Any, Dict, Tuple

# ── Credential service names ──
# These must match the ``service`` column in the ``credentials`` table.

# Email
SERVICE_GMAIL = "gmail"
SERVICE_OUTLOOK = "outlook"
EMAIL_SERVICES: Tuple[str, ...] = (SERVICE_GMAIL, SERVICE_OUTLOOK)

# Calendar
SERVICE_GOOGLE_CALENDAR = "google_calendar"
SERVICE_OUTLOOK_CALENDAR = "outlook_calendar"
CALENDAR_SERVICES: Tuple[str, ...] = (SERVICE_GOOGLE_CALENDAR, SERVICE_OUTLOOK_CALENDAR)

# Smart home
SERVICE_PHILIPS_HUE = "philips_hue"
SERVICE_SONOS = "sonos"
SMARTHOME_SERVICES: Tuple[str, ...] = (SERVICE_PHILIPS_HUE, SERVICE_SONOS)

# Cloud storage
SERVICE_GOOGLE_DRIVE = "google_drive"
SERVICE_ONEDRIVE = "onedrive"
SERVICE_DROPBOX = "dropbox"
STORAGE_SERVICES: Tuple[str, ...] = (SERVICE_GOOGLE_DRIVE, SERVICE_ONEDRIVE, SERVICE_DROPBOX)

# Image generation
SERVICE_IMAGE_OPENAI = "image_openai"
SERVICE_IMAGE_AZURE = "image_azure"
SERVICE_IMAGE_GEMINI = "image_gemini"
SERVICE_IMAGE_SEEDREAM = "image_seedream"
IMAGE_SERVICES: Tuple[str, ...] = (
    SERVICE_IMAGE_OPENAI, SERVICE_IMAGE_AZURE,
    SERVICE_IMAGE_GEMINI, SERVICE_IMAGE_SEEDREAM,
)

# Todo / tasks
SERVICE_TODOIST = "todoist"
SERVICE_GOOGLE_TASKS = "google_tasks"
SERVICE_MICROSOFT_TODO = "microsoft_todo"
TODO_SERVICES: Tuple[str, ...] = (SERVICE_TODOIST, SERVICE_GOOGLE_TASKS, SERVICE_MICROSOFT_TODO)

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
