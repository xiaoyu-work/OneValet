"""
Google Workspace integration for OneValet

GoogleWorkspaceDomainAgent handles all Google Workspace operations
(Drive search, Docs read/create, Sheets read/write) via an internal
mini ReAct loop.
"""

from .agent import GoogleWorkspaceDomainAgent
from .create_doc import GoogleDocsCreateAgent
from .write_sheet import GoogleSheetsWriteAgent

__all__ = [
    "GoogleWorkspaceDomainAgent",
    "GoogleDocsCreateAgent",
    "GoogleSheetsWriteAgent",
]
