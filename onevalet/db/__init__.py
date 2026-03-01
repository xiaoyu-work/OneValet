"""
OneValet Database - Modular asyncpg-based data access.

- Database: shared connection pool manager (one per app)
- Repository: base class for domain-specific data access (one per table)

Schema creation is handled by Alembic migrations (see migrations/).
"""

from .database import Database
from .repository import Repository

__all__ = ["Database", "Repository"]
