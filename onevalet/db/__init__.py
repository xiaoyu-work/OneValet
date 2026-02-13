"""
OneValet Database - Modular asyncpg-based data access.

- Database: shared connection pool manager (one per app)
- Repository: base class for domain-specific data access (one per table)
- ensure_all_tables: create all tables on first run
"""

from .database import Database
from .repository import Repository
from .initialize import ensure_schema

__all__ = ["Database", "Repository", "ensure_schema"]
