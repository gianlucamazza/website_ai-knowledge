"""
Pytest Configuration and Shared Fixtures

This module provides shared test configuration, fixtures, and utilities
for the AI Knowledge website testing framework.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Dict, Generator, List
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from pipelines.config import PipelineConfig, DatabaseConfig
from pipelines.database.connection import DatabaseManager
from pipelines.database.models import Base, Article, Source


# Test Configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_config() -> PipelineConfig:
    """Create test configuration with overrides."""
    config = PipelineConfig(
        environment="test",
        debug=True,
        database=DatabaseConfig(
            host="localhost",
            port=5432,
            database="ai_knowledge_test",
            username="postgres",
            password="test_password",
            pool_size=5,
            max_overflow=10,
            echo=False
        )
    )
    
    # Create test directories
    config.ensure_directories()
    return config


@pytest.fixture(scope="session")
def test_db_engine(test_config):
    """Create test database engine."""
    database_url = (
        f"postgresql+asyncpg://{test_config.database.username}:"
        f"{test_config.database.password}@{test_config.database.host}:"
        f"{test_config.database.port}/{test_config.database.database}"
    )
    
    engine = create_async_engine(
        database_url,
        echo=test_config.database.echo,
        pool_size=test_config.database.pool_size,
        max_overflow=test_config.database.max_overflow
    )
    
    yield engine
    engine.sync_engine.dispose()


@pytest.fixture(scope="function")
async def db_session(test_db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create isolated database session for each test."""
    async with test_db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
        async_session = sessionmaker(
            test_db_engine, class_=AsyncSession, expire_on_commit=False
        )
        
        async with async_session() as session:
            yield session
            await session.rollback()
        
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
def temp_directory() -> Generator[Path, None, None]:
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_article_data() -> Dict:
    """Sample article data for testing."""
    return {
        "title": "Test Article: Understanding Machine Learning",
        "url": "https://example.com/test-article",
        "content": "This is a comprehensive test article about machine learning concepts. "
                  "It covers various algorithms, techniques, and applications in the field. "
                  "Machine learning is a subset of artificial intelligence that enables "
                  "systems to learn and improve from experience without explicit programming.",
        "summary": "A test article covering machine learning fundamentals and applications.",
        "author": "Test Author",
        "published_date": "2024-01-15T10:00:00Z",
        "source_id": "test-source",
        "category": "machine-learning",
        "tags": ["ml", "ai", "algorithms", "data-science"],
        "metadata": {
            "word_count": 150,
            "reading_time": 2,
            "complexity_score": 0.6
        }
    }


@pytest.fixture
def sample_rss_feed() -> str:
    """Sample RSS feed XML for testing."""
    return """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
        <channel>
            <title>Test AI Blog</title>
            <description>Latest AI and ML articles</description>
            <link>https://example.com</link>
            <item>
                <title>Introduction to Neural Networks</title>
                <description>A beginner's guide to neural networks</description>
                <link>https://example.com/neural-networks</link>
                <pubDate>Mon, 15 Jan 2024 10:00:00 GMT</pubDate>
                <author>test@example.com (Test Author)</author>
                <category>machine-learning</category>
            </item>
            <item>
                <title>Deep Learning Fundamentals</title>
                <description>Understanding deep learning principles</description>
                <link>https://example.com/deep-learning</link>
                <pubDate>Tue, 16 Jan 2024 14:30:00 GMT</pubDate>
                <author>test@example.com (Test Author)</author>
                <category>deep-learning</category>
            </item>
        </channel>
    </rss>"""


@pytest.fixture
def sample_html_content() -> str:
    """Sample HTML content for testing."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Article: AI Ethics</title>
        <meta name="description" content="Exploring ethical considerations in AI">
        <meta name="author" content="Test Author">
    </head>
    <body>
        <header>
            <h1>AI Ethics: A Comprehensive Guide</h1>
            <p class="meta">Published on January 15, 2024 by Test Author</p>
        </header>
        <main>
            <section>
                <h2>Introduction</h2>
                <p>Artificial intelligence ethics is a critical field that examines the moral implications of AI systems. As AI becomes more prevalent in society, understanding ethical considerations becomes increasingly important.</p>
            </section>
            <section>
                <h2>Key Principles</h2>
                <ul>
                    <li>Transparency and Explainability</li>
                    <li>Fairness and Non-discrimination</li>
                    <li>Privacy and Data Protection</li>
                    <li>Accountability and Responsibility</li>
                </ul>
            </section>
        </main>
        <footer>
            <p>© 2024 Test AI Blog. All rights reserved.</p>
        </footer>
    </body>
    </html>
    """


# Mock Services
@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = AsyncMock(
        choices=[
            AsyncMock(
                message=AsyncMock(
                    content="This is a test AI-generated summary of the article content."
                )
            )
        ]
    )
    return mock_client


@pytest.fixture
def mock_http_session():
    """Mock HTTP session for web scraping tests."""
    mock_session = AsyncMock()
    
    async def mock_get(url, **kwargs):
        response = AsyncMock()
        response.status = 200
        response.text.return_value = "<html><body>Test content</body></html>"
        response.headers = {"content-type": "text/html"}
        return response
    
    mock_session.get = mock_get
    return mock_session


@pytest.fixture
def mock_database_session():
    """Mock database session for unit tests."""
    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()
    mock_session.close = AsyncMock()
    mock_session.execute = AsyncMock()
    mock_session.scalars = AsyncMock()
    return mock_session


# Test Data Helpers
@pytest.fixture
def create_test_articles():
    """Factory for creating test articles."""
    def _create_articles(count: int = 5) -> List[Dict]:
        articles = []
        for i in range(count):
            articles.append({
                "title": f"Test Article {i+1}",
                "url": f"https://example.com/article-{i+1}",
                "content": f"Content for test article {i+1}. " * 20,
                "source_id": f"source-{i % 3 + 1}",
                "category": ["ai", "ml", "data-science"][i % 3],
                "tags": [f"tag-{i}", "test"],
                "published_date": f"2024-01-{i+1:02d}T10:00:00Z"
            })
        return articles
    return _create_articles


@pytest.fixture
def performance_benchmarks() -> Dict[str, float]:
    """Performance benchmark thresholds for testing."""
    return {
        "content_ingestion_per_second": 10.0,
        "deduplication_time_per_1000_items": 5.0,
        "database_insert_time_per_item": 0.1,
        "content_processing_memory_mb": 500.0,
        "api_response_time_ms": 200.0,
        "page_load_time_ms": 1000.0
    }


# Test Markers and Configuration
pytest_plugins = ["pytest_asyncio"]

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interactions"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and load tests"
    )
    config.addinivalue_line(
        "markers", "security: Security and compliance tests"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end functional tests"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take more than 10 seconds"
    )
    config.addinivalue_line(
        "markers", "database: Tests that require database connection"
    )
    config.addinivalue_line(
        "markers", "external: Tests that make external API calls"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add slow marker to tests taking more than 10 seconds
        if hasattr(item.function, "_pytest_timeout"):
            if item.function._pytest_timeout > 10:
                item.add_marker(pytest.mark.slow)
        
        # Add database marker to tests using db fixtures
        if any(fixture in item.fixturenames for fixture in ["db_session", "test_db_engine"]):
            item.add_marker(pytest.mark.database)
        
        # Add external marker to tests making HTTP calls
        if "http" in item.name.lower() or "api" in item.name.lower():
            item.add_marker(pytest.mark.external)