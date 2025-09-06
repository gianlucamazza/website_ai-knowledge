"""
Database connection management with connection pooling and transaction support.

Provides async SQLAlchemy engine and session management for PostgreSQL.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import asyncpg
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from ..config import config
from .models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and session lifecycle."""
    
    def __init__(self):
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker] = None
    
    @property
    def database_url(self) -> str:
        """Construct async PostgreSQL database URL."""
        db_config = config.database
        return (
            f"postgresql+asyncpg://{db_config.username}:{db_config.password}"
            f"@{db_config.host}:{db_config.port}/{db_config.database}"
        )
    
    @property
    def sync_database_url(self) -> str:
        """Construct sync PostgreSQL database URL for migrations."""
        db_config = config.database
        return (
            f"postgresql://{db_config.username}:{db_config.password}"
            f"@{db_config.host}:{db_config.port}/{db_config.database}"
        )
    
    async def initialize(self) -> None:
        """Initialize database connection and create tables if needed."""
        try:
            self._engine = create_async_engine(
                self.database_url,
                echo=config.database.echo,
                pool_size=config.database.pool_size,
                max_overflow=config.database.max_overflow,
                pool_pre_ping=True,
                pool_recycle=3600,  # Recycle connections every hour
            )
            
            self._session_factory = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
            
            # Test connection
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info("Database connection initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def close(self) -> None:
        """Close database connections."""
        if self._engine:
            await self._engine.dispose()
            logger.info("Database connections closed")
    
    def get_session(self) -> AsyncSession:
        """Get a new database session."""
        if not self._session_factory:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._session_factory()
    
    @asynccontextmanager
    async def session_scope(self) -> AsyncGenerator[AsyncSession, None]:
        """Provide a transactional scope around database operations."""
        session = self.get_session()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
    
    async def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            async with self.session_scope() as session:
                await session.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def create_sync_engine(self):
        """Create synchronous engine for migrations."""
        return create_engine(
            self.sync_database_url,
            echo=config.database.echo,
            poolclass=NullPool,
        )
    
    async def create_database_if_not_exists(self) -> None:
        """Create database if it doesn't exist."""
        db_config = config.database
        
        # Connect to postgres database to create our target database
        admin_url = (
            f"postgresql://{db_config.username}:{db_config.password}"
            f"@{db_config.host}:{db_config.port}/postgres"
        )
        
        try:
            conn = await asyncpg.connect(admin_url)
            
            # Check if database exists
            exists = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1",
                db_config.database
            )
            
            if not exists:
                await conn.execute(f'CREATE DATABASE "{db_config.database}"')
                logger.info(f"Created database: {db_config.database}")
            else:
                logger.info(f"Database already exists: {db_config.database}")
            
            await conn.close()
            
        except Exception as e:
            logger.error(f"Failed to create database: {e}")
            raise


# Global database manager instance
db_manager = DatabaseManager()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting database sessions."""
    async with db_manager.session_scope() as session:
        yield session


async def init_database() -> None:
    """Initialize database connection."""
    await db_manager.create_database_if_not_exists()
    await db_manager.initialize()


async def close_database() -> None:
    """Close database connections."""
    await db_manager.close()


# Migration utilities
def run_migrations() -> None:
    """Run database migrations using Alembic."""
    from alembic import command
    from alembic.config import Config
    
    # Configure Alembic
    alembic_cfg = Config()
    alembic_cfg.set_main_option("script_location", "alembic")
    alembic_cfg.set_main_option("sqlalchemy.url", db_manager.sync_database_url)
    
    try:
        command.upgrade(alembic_cfg, "head")
        logger.info("Database migrations completed successfully")
    except Exception as e:
        logger.error(f"Failed to run migrations: {e}")
        raise


def create_migration(message: str) -> None:
    """Create a new database migration."""
    from alembic import command
    from alembic.config import Config
    
    alembic_cfg = Config()
    alembic_cfg.set_main_option("script_location", "alembic")
    alembic_cfg.set_main_option("sqlalchemy.url", db_manager.sync_database_url)
    
    try:
        command.revision(alembic_cfg, message=message, autogenerate=True)
        logger.info(f"Created migration: {message}")
    except Exception as e:
        logger.error(f"Failed to create migration: {e}")
        raise