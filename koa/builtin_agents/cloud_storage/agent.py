"""
CloudStorageAgent - Search and manage files across cloud storage providers.

Fans out across Google Drive, OneDrive, and Dropbox. Sharing is
approval-gated; everything else is read-only.
"""

import logging

from koa import StandardAgent, valet
from koa.constants import STORAGE_SERVICES

from .tools import (
    get_download_link,
    get_file_info,
    get_storage_usage,
    list_recent_files,
    search_files,
    share_file,
)

logger = logging.getLogger(__name__)


@valet(domain="productivity", requires_service=list(STORAGE_SERVICES))
class CloudStorageAgent(StandardAgent):
    """Search and manage files in cloud storage (Dropbox, Google Drive, OneDrive). Use when the user asks about their files or wants to share/upload."""

    max_turns = 15

    _SYSTEM_PROMPT_TEMPLATE = """\
You help the user find and manage files in their cloud storage
(Google Drive, OneDrive, Dropbox).

Available tools:
- search_files: Find files by name or keywords.
- list_recent_files: Show recently modified files.
- get_file_info: Details about one file (type, size, path, link).
- get_download_link: Get a download URL for a file.
- share_file: Share a file with someone by email.
- get_storage_usage: Show how much space is used.

Instructions:
1. Every tool searches all connected providers by default. Only set `provider`
   when the user names one ("on Dropbox", "in my Drive").
2. For "what files do I have" or "recent files", use list_recent_files rather
   than searching for an empty query.
3. To share, you need both the file and a recipient email. If the user has not
   given you an email address, ask for it before calling share_file.
4. share_file asks the user to confirm before it runs. Do not ask for
   confirmation yourself -- call the tool and let it prompt.
5. Results may note that an account could not be reached. Pass that along; do
   not present a partial result as if it were complete.
6. Respond in the same language the user used."""

    def get_system_prompt(self) -> str:
        return self._SYSTEM_PROMPT_TEMPLATE

    tools = (
        search_files,
        list_recent_files,
        get_file_info,
        get_download_link,
        share_file,
        get_storage_usage,
    )
