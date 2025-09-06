#!/usr/bin/env python3
"""
Initialize Supabase database with required tables for AI Knowledge Website.

This script creates all necessary tables using SQLAlchemy models.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine
from pipelines.database.models import Base

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def init_database():
    """Initialize database with all required tables."""
    
    # Get database URL from environment or use the provided Supabase pooler URL
    database_url = os.getenv(
        'DATABASE_URL', 
        'postgresql://postgres.dhidxgdfmknosqlthyzm:mdMTtu5CDtvWbkNj@aws-1-eu-central-1.pooler.supabase.com:6543/postgres'
    )
    
    # Convert to async URL for asyncpg
    async_database_url = database_url.replace('postgresql://', 'postgresql+asyncpg://')
    
    logger.info("Connecting to database...")
    
    try:
        # Create async engine
        engine = create_async_engine(
            async_database_url,
            echo=True,  # Show SQL statements
            pool_pre_ping=True,
        )
        
        # Create all tables
        async with engine.begin() as conn:
            logger.info("Creating database schema...")
            
            # Create schema if it doesn't exist
            await conn.execute(text("CREATE SCHEMA IF NOT EXISTS pipeline"))
            
            # Create all tables from SQLAlchemy models
            await conn.run_sync(Base.metadata.create_all)
            
            logger.info("Database schema created successfully!")
            
            # Verify tables were created
            result = await conn.execute(
                text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    ORDER BY table_name
                """)
            )
            tables = result.fetchall()
            
            logger.info(f"Created {len(tables)} tables:")
            for table in tables:
                logger.info(f"  - {table[0]}")
        
        # Create indexes for better performance
        async with engine.begin() as conn:
            logger.info("Creating performance indexes...")
            
            # Check if indexes already exist before creating
            indexes = [
                ("idx_articles_url", "CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(url)"),
                ("idx_articles_status", "CREATE INDEX IF NOT EXISTS idx_articles_status ON articles(status)"),
                ("idx_articles_source", "CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source_id)"),
                ("idx_articles_simhash", "CREATE INDEX IF NOT EXISTS idx_articles_simhash ON articles(simhash)"),
                ("idx_pipeline_runs_stage", "CREATE INDEX IF NOT EXISTS idx_pipeline_runs_stage ON pipeline_runs(current_stage)"),
                ("idx_pipeline_runs_status", "CREATE INDEX IF NOT EXISTS idx_pipeline_runs_status ON pipeline_runs(status)"),
                ("idx_sources_name", "CREATE INDEX IF NOT EXISTS idx_sources_name ON sources(name)"),
                ("idx_duplicates_article", "CREATE INDEX IF NOT EXISTS idx_duplicates_article ON content_duplicates(article_id)"),
            ]
            
            for index_name, index_sql in indexes:
                try:
                    await conn.execute(text(index_sql))
                    logger.info(f"  ✓ {index_name}")
                except Exception as e:
                    logger.warning(f"  ⚠ {index_name}: {e}")
        
        # Insert default sources
        async with engine.begin() as conn:
            logger.info("Inserting default content sources...")
            
            sources = [
                ("ArXiv CS.AI", "https://arxiv.org/rss/cs.AI", "rss"),
                ("Towards Data Science", "https://towardsdatascience.com/feed", "rss"),
                ("Google AI Blog", "https://ai.googleblog.com/feeds/posts/default", "rss"),
                ("OpenAI Blog", "https://openai.com/blog/rss.xml", "rss"),
                ("Anthropic Blog", "https://www.anthropic.com/rss.xml", "rss"),
            ]
            
            for name, url, source_type in sources:
                try:
                    await conn.execute(
                        text("""
                            INSERT INTO sources (name, base_url, source_type, is_active)
                            VALUES (:name, :url, :source_type, true)
                            ON CONFLICT (name) DO NOTHING
                        """),
                        {"name": name, "url": url, "source_type": source_type}
                    )
                    logger.info(f"  ✓ {name}")
                except Exception as e:
                    logger.warning(f"  ⚠ {name}: {e}")
        
        # Test the connection
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT COUNT(*) FROM sources"))
            count = result.scalar()
            logger.info(f"\n✅ Database initialized successfully with {count} sources!")
        
        await engine.dispose()
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def main():
    """Main entry point."""
    logger.info("AI Knowledge Website - Database Initialization")
    logger.info("=" * 50)
    
    try:
        asyncio.run(init_database())
        logger.info("\n🎉 Database initialization complete!")
        logger.info("\nNext steps:")
        logger.info("1. Run 'make dev' to start the development server")
        logger.info("2. Run 'python pipelines/run_graph.py sources --list' to verify sources")
        logger.info("3. Run 'python pipelines/run_graph.py ingest' to start content ingestion")
        
    except KeyboardInterrupt:
        logger.info("\n\nInitialization cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()