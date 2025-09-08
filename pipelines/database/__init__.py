"""
Database module for content pipeline state management.

Provides SQLAlchemy models and connection management for PostgreSQL.
"""

from .connection import DatabaseManager, get_db_session
from .models import (
    Article,
    Base,
    ContentDuplicate,
    EnrichmentTask,
    PipelineRun,
    Source,
)

__all__ = [
    "DatabaseManager",
    "get_db_session",
    "Article",
    "ContentDuplicate",
    "EnrichmentTask",
    "PipelineRun",
    "Source",
    "Base",
]
