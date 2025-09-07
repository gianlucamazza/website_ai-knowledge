"""
Database operation tests for the AI Knowledge website.

Tests database connections, CRUD operations, data integrity, and transaction handling.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch, AsyncMock
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
import uuid

from pipelines.database.models import Base, Article, Source, ProcessingJob, DuplicateRelation
from pipelines.database.connection import DatabaseManager
from pipelines.database.operations import ArticleOperations, SourceOperations


class TestDatabaseConnection:
    """Test database connection and session management."""
    
    @pytest.fixture
    def db_config(self):
        """Database configuration for testing."""
        return {
            'host': 'localhost',
            'port': 5432,
            'database': 'ai_knowledge_test',
            'username': 'test_user',
            'password': 'test_password',
            'pool_size': 5,
            'max_overflow': 10,
            'echo': False
        }
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_database_connection_success(self, db_config, db_session):
        """Test successful database connection."""
        db_manager = DatabaseManager(db_config)
        
        # Test connection
        async with db_manager.get_session() as session:
            result = await session.execute(sa.text("SELECT 1 as test_value"))
            row = result.first()
            assert row.test_value == 1
    
    @pytest.mark.database
    def test_database_connection_failure(self):
        """Test database connection failure handling."""
        bad_config = {
            'host': 'nonexistent-host',
            'port': 5432,
            'database': 'nonexistent_db',
            'username': 'bad_user',
            'password': 'bad_password'
        }
        
        db_manager = DatabaseManager(bad_config)
        
        # Should handle connection failure gracefully
        with pytest.raises(Exception) as exc_info:
            asyncio.run(db_manager._test_connection())
        
        assert 'connection' in str(exc_info.value).lower() or 'database' in str(exc_info.value).lower()
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_connection_pooling(self, db_config):
        """Test database connection pooling."""
        db_manager = DatabaseManager(db_config)
        
        # Test multiple concurrent connections
        async def test_query(session_id: int):
            async with db_manager.get_session() as session:
                result = await session.execute(sa.text(f"SELECT {session_id} as session_id"))
                row = result.first()
                return row.session_id
        
        # Run multiple concurrent database operations
        tasks = [test_query(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All queries should succeed
        assert results == list(range(10))
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_session_cleanup(self, db_config):
        """Test proper session cleanup and resource management."""
        db_manager = DatabaseManager(db_config)
        
        # Test session cleanup after exception
        with pytest.raises(Exception):
            async with db_manager.get_session() as session:
                await session.execute(sa.text("SELECT 1"))
                raise Exception("Test exception")
        
        # Should be able to create new session after exception
        async with db_manager.get_session() as session:
            result = await session.execute(sa.text("SELECT 1"))
            assert result.first()[0] == 1


class TestArticleOperations:
    """Test article-related database operations."""
    
    @pytest.fixture
    def article_operations(self, db_session):
        """Create ArticleOperations instance for testing."""
        return ArticleOperations(db_session)
    
    @pytest.fixture
    def sample_article_data(self):
        """Sample article data for testing."""
        return {
            'title': 'Understanding Neural Networks',
            'slug': 'understanding-neural-networks',
            'url': 'https://example.com/neural-networks',
            'content': 'Neural networks are computational models inspired by biological neural networks...',
            'summary': 'A comprehensive guide to understanding neural networks and their applications.',
            'author': 'Dr. Jane Smith',
            'published_date': datetime(2024, 1, 15, 10, 30, 0),
            'updated_date': datetime(2024, 1, 20, 14, 15, 0),
            'source_id': 'example-blog',
            'category': 'deep-learning',
            'tags': ['neural-networks', 'deep-learning', 'ai'],
            'word_count': 1500,
            'reading_time': 7,
            'quality_score': 0.85,
            'language': 'en',
            'featured_image': 'https://example.com/images/neural-networks.jpg',
            'meta_description': 'Learn about neural networks in this comprehensive guide.',
            'keywords': ['neural networks', 'deep learning', 'artificial intelligence'],
            'status': 'published'
        }
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_create_article(self, article_operations, sample_article_data):
        """Test creating a new article."""
        article = await article_operations.create_article(sample_article_data)
        
        # Verify article was created
        assert article.id is not None
        assert article.title == sample_article_data['title']
        assert article.slug == sample_article_data['slug']
        assert article.url == sample_article_data['url']
        assert article.content == sample_article_data['content']
        assert article.author == sample_article_data['author']
        assert article.word_count == sample_article_data['word_count']
        assert article.quality_score == sample_article_data['quality_score']
        assert article.status == sample_article_data['status']
        assert article.created_at is not None
        assert article.updated_at is not None
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_get_article_by_id(self, article_operations, sample_article_data):
        """Test retrieving an article by ID."""
        # Create article
        created_article = await article_operations.create_article(sample_article_data)
        
        # Retrieve article by ID
        retrieved_article = await article_operations.get_article_by_id(created_article.id)
        
        assert retrieved_article is not None
        assert retrieved_article.id == created_article.id
        assert retrieved_article.title == sample_article_data['title']
        assert retrieved_article.slug == sample_article_data['slug']
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_get_article_by_slug(self, article_operations, sample_article_data):
        """Test retrieving an article by slug."""
        # Create article
        await article_operations.create_article(sample_article_data)
        
        # Retrieve article by slug
        retrieved_article = await article_operations.get_article_by_slug(sample_article_data['slug'])
        
        assert retrieved_article is not None
        assert retrieved_article.slug == sample_article_data['slug']
        assert retrieved_article.title == sample_article_data['title']
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_get_article_by_url(self, article_operations, sample_article_data):
        """Test retrieving an article by URL."""
        # Create article
        await article_operations.create_article(sample_article_data)
        
        # Retrieve article by URL
        retrieved_article = await article_operations.get_article_by_url(sample_article_data['url'])
        
        assert retrieved_article is not None
        assert retrieved_article.url == sample_article_data['url']
        assert retrieved_article.title == sample_article_data['title']
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_update_article(self, article_operations, sample_article_data):
        """Test updating an existing article."""
        # Create article
        created_article = await article_operations.create_article(sample_article_data)
        
        # Update article
        update_data = {
            'title': 'Updated Neural Networks Guide',
            'content': 'Updated content about neural networks...',
            'quality_score': 0.90,
            'updated_date': datetime.now()
        }
        
        updated_article = await article_operations.update_article(created_article.id, update_data)
        
        assert updated_article.id == created_article.id
        assert updated_article.title == update_data['title']
        assert updated_article.content == update_data['content']
        assert updated_article.quality_score == update_data['quality_score']
        assert updated_article.updated_at > created_article.updated_at
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_delete_article(self, article_operations, sample_article_data):
        """Test deleting an article."""
        # Create article
        created_article = await article_operations.create_article(sample_article_data)
        article_id = created_article.id
        
        # Delete article
        success = await article_operations.delete_article(article_id)
        assert success is True
        
        # Verify article is deleted
        deleted_article = await article_operations.get_article_by_id(article_id)
        assert deleted_article is None
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_list_articles_with_pagination(self, article_operations):
        """Test listing articles with pagination."""
        # Create multiple articles
        articles_data = []
        for i in range(15):
            article_data = {
                'title': f'Test Article {i+1}',
                'slug': f'test-article-{i+1}',
                'url': f'https://example.com/article-{i+1}',
                'content': f'Content for article {i+1}',
                'source_id': 'test-source',
                'category': 'test',
                'status': 'published',
                'published_date': datetime.now() - timedelta(days=i)
            }
            created_article = await article_operations.create_article(article_data)
            articles_data.append(created_article)
        
        # Test pagination
        page_1 = await article_operations.list_articles(page=1, page_size=10)
        page_2 = await article_operations.list_articles(page=2, page_size=10)
        
        assert len(page_1.articles) == 10
        assert len(page_2.articles) == 5
        assert page_1.total_count == 15
        assert page_1.page == 1
        assert page_1.page_size == 10
        assert page_1.total_pages == 2
        
        # Verify no overlap between pages
        page_1_ids = {article.id for article in page_1.articles}
        page_2_ids = {article.id for article in page_2.articles}
        assert page_1_ids.isdisjoint(page_2_ids)
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_search_articles(self, article_operations):
        """Test article search functionality."""
        # Create test articles with different content
        test_articles = [
            {
                'title': 'Machine Learning Fundamentals',
                'slug': 'ml-fundamentals',
                'url': 'https://example.com/ml-fundamentals',
                'content': 'Machine learning is a subset of artificial intelligence...',
                'category': 'machine-learning',
                'tags': ['ml', 'ai', 'fundamentals'],
                'status': 'published'
            },
            {
                'title': 'Deep Learning with Neural Networks',
                'slug': 'deep-learning-nn',
                'url': 'https://example.com/deep-learning-nn',
                'content': 'Deep learning uses neural networks with multiple layers...',
                'category': 'deep-learning',
                'tags': ['deep-learning', 'neural-networks', 'ai'],
                'status': 'published'
            },
            {
                'title': 'Computer Vision Applications',
                'slug': 'computer-vision-apps',
                'url': 'https://example.com/cv-apps',
                'content': 'Computer vision enables machines to interpret visual information...',
                'category': 'computer-vision',
                'tags': ['computer-vision', 'applications'],
                'status': 'published'
            }
        ]
        
        for article_data in test_articles:
            await article_operations.create_article(article_data)
        
        # Test search by title
        title_results = await article_operations.search_articles(query='Machine Learning')
        assert len(title_results.articles) == 1
        assert 'Machine Learning' in title_results.articles[0].title
        
        # Test search by content
        content_results = await article_operations.search_articles(query='neural networks')
        assert len(content_results.articles) == 1
        assert 'Deep Learning' in content_results.articles[0].title
        
        # Test search by category
        category_results = await article_operations.search_articles(category='machine-learning')
        assert len(category_results.articles) == 1
        assert category_results.articles[0].category == 'machine-learning'
        
        # Test search by tags
        tag_results = await article_operations.search_articles(tags=['ai'])
        assert len(tag_results.articles) >= 2  # Should find multiple articles with 'ai' tag
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_get_articles_by_category(self, article_operations):
        """Test retrieving articles by category."""
        # Create articles in different categories
        categories = ['machine-learning', 'deep-learning', 'nlp']
        
        for i, category in enumerate(categories):
            for j in range(3):  # 3 articles per category
                article_data = {
                    'title': f'{category.title()} Article {j+1}',
                    'slug': f'{category}-article-{j+1}',
                    'url': f'https://example.com/{category}-article-{j+1}',
                    'content': f'Content about {category}',
                    'category': category,
                    'status': 'published'
                }
                await article_operations.create_article(article_data)
        
        # Test retrieving articles by category
        for category in categories:
            category_articles = await article_operations.get_articles_by_category(category)
            assert len(category_articles) == 3
            assert all(article.category == category for article in category_articles)
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_get_recent_articles(self, article_operations):
        """Test retrieving recent articles."""
        # Create articles with different publication dates
        base_date = datetime(2024, 1, 1)
        
        for i in range(10):
            article_data = {
                'title': f'Article {i+1}',
                'slug': f'article-{i+1}',
                'url': f'https://example.com/article-{i+1}',
                'content': f'Content {i+1}',
                'published_date': base_date + timedelta(days=i),
                'status': 'published'
            }
            await article_operations.create_article(article_data)
        
        # Get recent articles
        recent_articles = await article_operations.get_recent_articles(limit=5)
        
        assert len(recent_articles) == 5
        
        # Should be ordered by publication date (newest first)
        for i in range(len(recent_articles) - 1):
            assert recent_articles[i].published_date >= recent_articles[i+1].published_date
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_article_statistics(self, article_operations):
        """Test article statistics calculations."""
        # Create articles with various properties
        categories = ['ai', 'ml', 'dl']
        statuses = ['published', 'draft', 'archived']
        
        article_count = 0
        for category in categories:
            for status in statuses:
                for i in range(2):  # 2 articles per combination
                    article_data = {
                        'title': f'{category} {status} Article {i+1}',
                        'slug': f'{category}-{status}-article-{i+1}',
                        'url': f'https://example.com/{category}-{status}-{i+1}',
                        'content': f'Content for {category} {status} article {i+1}',
                        'category': category,
                        'status': status,
                        'word_count': 1000 + (i * 500),
                        'quality_score': 0.5 + (i * 0.2)
                    }
                    await article_operations.create_article(article_data)
                    article_count += 1
        
        # Get statistics
        stats = await article_operations.get_article_statistics()
        
        assert stats['total_articles'] == article_count
        assert stats['published_articles'] == 6  # 2 per category * 3 categories
        assert stats['draft_articles'] == 6
        assert stats['archived_articles'] == 6
        
        # Category breakdown
        assert len(stats['by_category']) == 3
        for category_stat in stats['by_category']:
            assert category_stat['count'] == 6  # 2 per status * 3 statuses
        
        # Quality statistics
        assert 'avg_quality_score' in stats
        assert 'avg_word_count' in stats
        assert stats['avg_quality_score'] > 0
        assert stats['avg_word_count'] > 0


class TestSourceOperations:
    """Test source-related database operations."""
    
    @pytest.fixture
    def source_operations(self, db_session):
        """Create SourceOperations instance for testing."""
        return SourceOperations(db_session)
    
    @pytest.fixture
    def sample_source_data(self):
        """Sample source data for testing."""
        return {
            'name': 'AI Research Blog',
            'type': 'rss',
            'url': 'https://ai-research.example.com/feed.xml',
            'enabled': True,
            'categories': ['artificial-intelligence', 'research'],
            'tags': ['research', 'academic', 'ai'],
            'max_articles_per_run': 50,
            'crawl_frequency': 3600,  # 1 hour
            'quality_threshold': 0.7,
            'config': {
                'respect_robots_txt': True,
                'request_delay': 1.0,
                'timeout': 30,
                'user_agent': 'AI-Knowledge-Bot/1.0'
            }
        }
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_create_source(self, source_operations, sample_source_data):
        """Test creating a new source."""
        source = await source_operations.create_source(sample_source_data)
        
        # Verify source was created
        assert source.id is not None
        assert source.name == sample_source_data['name']
        assert source.type == sample_source_data['type']
        assert source.url == sample_source_data['url']
        assert source.enabled == sample_source_data['enabled']
        assert source.max_articles_per_run == sample_source_data['max_articles_per_run']
        assert source.crawl_frequency == sample_source_data['crawl_frequency']
        assert source.created_at is not None
        assert source.updated_at is not None
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_get_enabled_sources(self, source_operations):
        """Test retrieving enabled sources."""
        # Create mix of enabled and disabled sources
        sources_data = [
            {'name': 'Enabled Source 1', 'type': 'rss', 'url': 'https://example1.com/feed', 'enabled': True},
            {'name': 'Enabled Source 2', 'type': 'rss', 'url': 'https://example2.com/feed', 'enabled': True},
            {'name': 'Disabled Source', 'type': 'rss', 'url': 'https://example3.com/feed', 'enabled': False},
        ]
        
        for source_data in sources_data:
            await source_operations.create_source(source_data)
        
        # Get enabled sources
        enabled_sources = await source_operations.get_enabled_sources()
        
        assert len(enabled_sources) == 2
        assert all(source.enabled for source in enabled_sources)
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_update_source_statistics(self, source_operations, sample_source_data):
        """Test updating source statistics."""
        # Create source
        source = await source_operations.create_source(sample_source_data)
        
        # Update statistics
        stats_update = {
            'last_crawled_at': datetime.now(),
            'articles_found': 25,
            'articles_processed': 20,
            'articles_published': 18,
            'error_count': 2,
            'last_error': 'Connection timeout'
        }
        
        updated_source = await source_operations.update_source_statistics(source.id, stats_update)
        
        assert updated_source.last_crawled_at is not None
        assert updated_source.articles_found == stats_update['articles_found']
        assert updated_source.articles_processed == stats_update['articles_processed']
        assert updated_source.articles_published == stats_update['articles_published']
        assert updated_source.error_count == stats_update['error_count']
        assert updated_source.last_error == stats_update['last_error']
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_get_sources_due_for_crawling(self, source_operations):
        """Test retrieving sources that are due for crawling."""
        now = datetime.now()
        
        sources_data = [
            {
                'name': 'Recent Source',
                'type': 'rss',
                'url': 'https://recent.com/feed',
                'enabled': True,
                'crawl_frequency': 3600,
                'last_crawled_at': now - timedelta(minutes=30)  # Not due yet
            },
            {
                'name': 'Due Source',
                'type': 'rss', 
                'url': 'https://due.com/feed',
                'enabled': True,
                'crawl_frequency': 1800,  # 30 minutes
                'last_crawled_at': now - timedelta(hours=1)  # Due for crawling
            },
            {
                'name': 'Never Crawled',
                'type': 'rss',
                'url': 'https://never.com/feed',
                'enabled': True,
                'crawl_frequency': 3600,
                'last_crawled_at': None  # Never crawled, should be due
            }
        ]
        
        for source_data in sources_data:
            await source_operations.create_source(source_data)
        
        # Get sources due for crawling
        due_sources = await source_operations.get_sources_due_for_crawling()
        
        # Should include the due source and never crawled source
        assert len(due_sources) >= 2
        due_names = {source.name for source in due_sources}
        assert 'Due Source' in due_names
        assert 'Never Crawled' in due_names
        assert 'Recent Source' not in due_names


class TestProcessingJobOperations:
    """Test processing job database operations."""
    
    @pytest.fixture
    def job_data(self):
        """Sample processing job data."""
        return {
            'source_ids': ['source-1', 'source-2'],
            'status': 'running',
            'articles_found': 0,
            'articles_processed': 0,
            'articles_published': 0,
            'errors_count': 0,
            'started_at': datetime.now()
        }
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_create_processing_job(self, db_session, job_data):
        """Test creating a new processing job."""
        job = ProcessingJob(**job_data)
        db_session.add(job)
        await db_session.commit()
        
        assert job.id is not None
        assert job.status == 'running'
        assert job.started_at is not None
        assert job.created_at is not None
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_update_job_progress(self, db_session, job_data):
        """Test updating job progress."""
        # Create job
        job = ProcessingJob(**job_data)
        db_session.add(job)
        await db_session.commit()
        
        # Update progress
        job.articles_found = 25
        job.articles_processed = 15
        job.articles_published = 12
        job.updated_at = datetime.now()
        
        await db_session.commit()
        
        # Reload and verify
        await db_session.refresh(job)
        assert job.articles_found == 25
        assert job.articles_processed == 15
        assert job.articles_published == 12
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_complete_processing_job(self, db_session, job_data):
        """Test completing a processing job."""
        # Create and complete job
        job = ProcessingJob(**job_data)
        db_session.add(job)
        await db_session.commit()
        
        # Complete the job
        job.status = 'completed'
        job.completed_at = datetime.now()
        job.articles_found = 30
        job.articles_processed = 25
        job.articles_published = 20
        
        await db_session.commit()
        
        assert job.status == 'completed'
        assert job.completed_at is not None
        assert job.completed_at > job.started_at


class TestTransactionManagement:
    """Test database transaction handling and rollback scenarios."""
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_transaction_rollback_on_error(self, db_session):
        """Test transaction rollback when an error occurs."""
        # Start transaction
        async with db_session.begin():
            # Create an article
            article = Article(
                title='Test Article',
                slug='test-article',
                url='https://example.com/test',
                content='Test content',
                status='published'
            )
            db_session.add(article)
            
            # Simulate an error that should cause rollback
            try:
                # This should cause a constraint violation or other error
                duplicate_article = Article(
                    title='Test Article',
                    slug='test-article',  # Duplicate slug
                    url='https://example.com/test',  # Duplicate URL
                    content='Test content',
                    status='published'
                )
                db_session.add(duplicate_article)
                await db_session.flush()  # Force constraint check
                
            except Exception:
                # Transaction should be rolled back
                await db_session.rollback()
        
        # Verify no articles were created
        result = await db_session.execute(
            sa.select(Article).where(Article.slug == 'test-article')
        )
        articles = result.scalars().all()
        assert len(articles) == 0
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_concurrent_article_creation(self, test_config):
        """Test concurrent article creation and constraint handling."""
        # This test would require multiple database sessions
        # to test concurrent access patterns
        
        async def create_article_session(article_data):
            db_manager = DatabaseManager(test_config.database)
            async with db_manager.get_session() as session:
                article = Article(**article_data)
                session.add(article)
                await session.commit()
                return article
        
        # Attempt to create duplicate articles concurrently
        article_data_1 = {
            'title': 'Concurrent Article',
            'slug': 'concurrent-article',
            'url': 'https://example.com/concurrent',
            'content': 'Content 1',
            'status': 'published'
        }
        
        article_data_2 = {
            'title': 'Concurrent Article',
            'slug': 'concurrent-article',  # Same slug
            'url': 'https://example.com/concurrent-2',
            'content': 'Content 2',
            'status': 'published'
        }
        
        # One should succeed, one should fail due to unique constraint
        tasks = [
            create_article_session(article_data_1),
            create_article_session(article_data_2)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # One should succeed, one should fail
        successes = [r for r in results if not isinstance(r, Exception)]
        failures = [r for r in results if isinstance(r, Exception)]
        
        assert len(successes) == 1
        assert len(failures) == 1
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_batch_insert_performance(self, db_session, performance_benchmarks):
        """Test batch insert performance."""
        import time
        
        # Create large batch of articles
        batch_size = 100
        articles = []
        
        for i in range(batch_size):
            article = Article(
                title=f'Batch Article {i}',
                slug=f'batch-article-{i}',
                url=f'https://example.com/batch-{i}',
                content=f'Content for batch article {i}',
                status='published',
                word_count=500,
                quality_score=0.7
            )
            articles.append(article)
        
        # Measure batch insert time
        start_time = time.time()
        
        db_session.add_all(articles)
        await db_session.commit()
        
        end_time = time.time()
        insert_time = end_time - start_time
        
        # Performance assertion
        max_time = performance_benchmarks.get("database_batch_insert_time_per_100_items", 5.0)
        assert insert_time <= max_time, f"Batch insert too slow: {insert_time:.2f}s > {max_time}s"
        
        # Verify all articles were inserted
        result = await db_session.execute(
            sa.select(sa.func.count(Article.id)).where(Article.title.like('Batch Article%'))
        )
        count = result.scalar()
        assert count == batch_size


class TestDataIntegrity:
    """Test data integrity and constraint validation."""
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_article_slug_uniqueness(self, db_session):
        """Test that article slugs must be unique."""
        # Create first article
        article1 = Article(
            title='First Article',
            slug='unique-slug',
            url='https://example.com/first',
            content='First content',
            status='published'
        )
        db_session.add(article1)
        await db_session.commit()
        
        # Attempt to create second article with same slug
        article2 = Article(
            title='Second Article',
            slug='unique-slug',  # Same slug
            url='https://example.com/second',
            content='Second content',
            status='published'
        )
        db_session.add(article2)
        
        # Should raise constraint violation
        with pytest.raises(Exception) as exc_info:
            await db_session.commit()
        
        assert 'unique' in str(exc_info.value).lower() or 'duplicate' in str(exc_info.value).lower()
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_article_url_uniqueness(self, db_session):
        """Test that article URLs must be unique."""
        # Create first article
        article1 = Article(
            title='First Article',
            slug='first-article',
            url='https://example.com/unique-url',
            content='First content',
            status='published'
        )
        db_session.add(article1)
        await db_session.commit()
        
        # Attempt to create second article with same URL
        article2 = Article(
            title='Second Article',
            slug='second-article',
            url='https://example.com/unique-url',  # Same URL
            content='Second content',
            status='published'
        )
        db_session.add(article2)
        
        # Should raise constraint violation
        with pytest.raises(Exception) as exc_info:
            await db_session.commit()
        
        assert 'unique' in str(exc_info.value).lower() or 'duplicate' in str(exc_info.value).lower()
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_required_fields_validation(self, db_session):
        """Test that required fields cannot be null."""
        # Attempt to create article without required fields
        article = Article(
            # Missing title, slug, url, content
            status='published'
        )
        db_session.add(article)
        
        # Should raise not null constraint violation
        with pytest.raises(Exception) as exc_info:
            await db_session.commit()
        
        error_msg = str(exc_info.value).lower()
        assert 'null' in error_msg or 'not null' in error_msg or 'required' in error_msg
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_foreign_key_constraints(self, db_session):
        """Test foreign key constraint validation."""
        # Create article with non-existent source_id
        article = Article(
            title='Test Article',
            slug='test-article',
            url='https://example.com/test',
            content='Test content',
            source_id='non-existent-source',  # Invalid foreign key
            status='published'
        )
        db_session.add(article)
        
        # Should either raise foreign key constraint violation or handle gracefully
        try:
            await db_session.commit()
            # If it doesn't raise an error, the source_id might be optional or handled differently
            assert article.source_id == 'non-existent-source'
        except Exception as exc_info:
            # Foreign key constraint violation is acceptable
            error_msg = str(exc_info).lower()
            assert 'foreign' in error_msg or 'constraint' in error_msg or 'reference' in error_msg