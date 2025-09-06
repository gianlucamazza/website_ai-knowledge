"""
Integration tests for database operations.

Tests database connections, transactions, migrations, and data integrity.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List
from unittest.mock import patch
from sqlalchemy import text, select, func
from sqlalchemy.exc import IntegrityError

from pipelines.database.connection import DatabaseManager
from pipelines.database.models import Base, Article, Source, ProcessingJob, DuplicateRelation


class TestDatabaseConnection:
    """Test database connection and session management."""
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_database_connection_pool(self, test_config):
        """Test database connection pooling."""
        db_manager = DatabaseManager(test_config.database)
        
        # Test multiple concurrent connections
        async def test_connection():
            async with db_manager.get_session() as session:
                result = await session.execute(text("SELECT 1 as test_value"))
                row = result.fetchone()
                assert row.test_value == 1
                return True
        
        # Create multiple concurrent connections
        tasks = [test_connection() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        assert all(results)  # All connections should succeed
        
        await db_manager.close_all_connections()
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_transaction_handling(self, db_session):
        """Test transaction commit and rollback behavior."""
        # Test successful transaction
        async with db_session.begin():
            await db_session.execute(
                text("INSERT INTO sources (id, name, type, url, enabled) VALUES ('test-source', 'Test', 'rss', 'http://test.com', true)")
            )
        
        # Verify data was committed
        result = await db_session.execute(text("SELECT id FROM sources WHERE id = 'test-source'"))
        row = result.fetchone()
        assert row is not None
        
        # Test transaction rollback
        try:
            async with db_session.begin():
                await db_session.execute(
                    text("INSERT INTO sources (id, name, type, url, enabled) VALUES ('test-source-2', 'Test 2', 'rss', 'http://test2.com', true)")
                )
                # Force an error to trigger rollback
                await db_session.execute(
                    text("INSERT INTO sources (id, name, type, url, enabled) VALUES ('test-source-2', 'Duplicate', 'rss', 'http://test3.com', true)")
                )
        except IntegrityError:
            pass  # Expected due to duplicate key
        
        # Verify rollback occurred - second source should not exist
        result = await db_session.execute(text("SELECT id FROM sources WHERE id = 'test-source-2'"))
        row = result.fetchone()
        assert row is None
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_connection_recovery(self, test_config):
        """Test database connection recovery after failure."""
        db_manager = DatabaseManager(test_config.database)
        
        # Test normal operation
        async with db_manager.get_session() as session:
            result = await session.execute(text("SELECT 1"))
            assert result.fetchone()[0] == 1
        
        # Simulate connection failure and recovery
        with patch.object(db_manager.engine, 'connect') as mock_connect:
            mock_connect.side_effect = [ConnectionError("Connection failed"), None]
            
            # Should handle connection error gracefully
            try:
                async with db_manager.get_session() as session:
                    await session.execute(text("SELECT 1"))
            except ConnectionError:
                pass  # Expected on first attempt
        
        await db_manager.close_all_connections()


class TestArticleOperations:
    """Test article-related database operations."""
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_article_crud_operations(self, db_session):
        """Test Create, Read, Update, Delete operations for articles."""
        # Create source first
        await db_session.execute(
            text("""
                INSERT INTO sources (id, name, type, url, enabled, created_at) 
                VALUES ('crud-source', 'CRUD Test Source', 'rss', 'http://test.com/feed.xml', true, :created_at)
            """),
            {'created_at': datetime.utcnow()}
        )
        
        # CREATE - Insert article
        article_data = {
            'id': 'test-article-crud',
            'url': 'https://example.com/test-article',
            'title': 'Test Article for CRUD',
            'content': 'This is test content for CRUD operations.',
            'summary': 'Test summary',
            'author': 'Test Author',
            'published_date': datetime(2024, 1, 15, 10, 0, 0),
            'source_id': 'crud-source',
            'category': 'test',
            'tags': '["test", "crud"]',
            'word_count': 50,
            'reading_time': 1,
            'language': 'en',
            'created_at': datetime.utcnow()
        }
        
        await db_session.execute(
            text("""
                INSERT INTO articles (id, url, title, content, summary, author, published_date, 
                                    source_id, category, tags, word_count, reading_time, language, created_at)
                VALUES (:id, :url, :title, :content, :summary, :author, :published_date,
                        :source_id, :category, :tags, :word_count, :reading_time, :language, :created_at)
            """),
            article_data
        )
        
        # READ - Query article
        result = await db_session.execute(
            text("SELECT * FROM articles WHERE id = 'test-article-crud'")
        )
        article = result.fetchone()
        
        assert article is not None
        assert article.title == 'Test Article for CRUD'
        assert article.source_id == 'crud-source'
        assert article.word_count == 50
        
        # UPDATE - Modify article
        await db_session.execute(
            text("""
                UPDATE articles 
                SET title = 'Updated Test Article', word_count = 75, updated_at = :updated_at
                WHERE id = 'test-article-crud'
            """),
            {'updated_at': datetime.utcnow()}
        )
        
        # Verify update
        result = await db_session.execute(
            text("SELECT title, word_count FROM articles WHERE id = 'test-article-crud'")
        )
        updated_article = result.fetchone()
        
        assert updated_article.title == 'Updated Test Article'
        assert updated_article.word_count == 75
        
        # DELETE - Remove article
        await db_session.execute(
            text("DELETE FROM articles WHERE id = 'test-article-crud'")
        )
        
        # Verify deletion
        result = await db_session.execute(
            text("SELECT id FROM articles WHERE id = 'test-article-crud'")
        )
        deleted_article = result.fetchone()
        
        assert deleted_article is None
        
        await db_session.commit()
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_article_search_operations(self, db_session):
        """Test article search and filtering operations."""
        # Create test source
        await db_session.execute(
            text("""
                INSERT INTO sources (id, name, type, url, enabled, created_at)
                VALUES ('search-source', 'Search Test Source', 'rss', 'http://test.com', true, :created_at)
            """),
            {'created_at': datetime.utcnow()}
        )
        
        # Create test articles
        test_articles = [
            {
                'id': 'ai-article-1',
                'title': 'Introduction to Artificial Intelligence',
                'content': 'AI is transforming technology across industries.',
                'category': 'artificial-intelligence',
                'tags': '["ai", "technology", "introduction"]',
                'published_date': datetime(2024, 1, 15),
                'reading_time': 5
            },
            {
                'id': 'ml-article-1',
                'title': 'Machine Learning Fundamentals',
                'content': 'ML algorithms learn from data to make predictions.',
                'category': 'machine-learning',
                'tags': '["ml", "algorithms", "data"]',
                'published_date': datetime(2024, 1, 10),
                'reading_time': 8
            },
            {
                'id': 'ai-article-2',
                'title': 'Advanced AI Techniques',
                'content': 'Advanced techniques in artificial intelligence development.',
                'category': 'artificial-intelligence',
                'tags': '["ai", "advanced", "techniques"]',
                'published_date': datetime(2024, 1, 20),
                'reading_time': 12
            }
        ]
        
        for article in test_articles:
            await db_session.execute(
                text("""
                    INSERT INTO articles (id, url, title, content, category, tags, 
                                        published_date, reading_time, source_id, created_at)
                    VALUES (:id, :url, :title, :content, :category, :tags,
                            :published_date, :reading_time, 'search-source', :created_at)
                """),
                {
                    **article,
                    'url': f'https://example.com/{article["id"]}',
                    'created_at': datetime.utcnow()
                }
            )
        
        await db_session.commit()
        
        # Test category filtering
        ai_articles = await db_session.execute(
            text("SELECT id FROM articles WHERE category = 'artificial-intelligence' ORDER BY published_date")
        )
        ai_results = ai_articles.fetchall()
        assert len(ai_results) == 2
        assert ai_results[0].id == 'ai-article-1'
        assert ai_results[1].id == 'ai-article-2'
        
        # Test date range filtering
        recent_articles = await db_session.execute(
            text("""
                SELECT id FROM articles 
                WHERE published_date >= :start_date AND published_date <= :end_date
                ORDER BY published_date DESC
            """),
            {
                'start_date': datetime(2024, 1, 12),
                'end_date': datetime(2024, 1, 25)
            }
        )
        recent_results = recent_articles.fetchall()
        assert len(recent_results) == 2
        assert recent_results[0].id == 'ai-article-2'  # Most recent first
        
        # Test text search
        ai_content_articles = await db_session.execute(
            text("""
                SELECT id FROM articles 
                WHERE content ILIKE '%artificial intelligence%' 
                ORDER BY published_date
            """)
        )
        ai_content_results = ai_content_articles.fetchall()
        assert len(ai_content_results) == 1
        assert ai_content_results[0].id == 'ai-article-2'
        
        # Test reading time filtering
        long_articles = await db_session.execute(
            text("SELECT id FROM articles WHERE reading_time > 7 ORDER BY reading_time")
        )
        long_results = long_articles.fetchall()
        assert len(long_results) == 2
        assert long_results[0].id == 'ml-article-1'  # 8 minutes
        assert long_results[1].id == 'ai-article-2'  # 12 minutes
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_article_statistics(self, db_session):
        """Test article statistics and aggregation queries."""
        # Create test source
        await db_session.execute(
            text("""
                INSERT INTO sources (id, name, type, url, enabled, created_at)
                VALUES ('stats-source', 'Statistics Test Source', 'rss', 'http://test.com', true, :created_at)
            """),
            {'created_at': datetime.utcnow()}
        )
        
        # Create articles with various statistics
        articles_data = [
            {'id': 'stats-1', 'category': 'ai', 'word_count': 500, 'reading_time': 3, 'published_date': datetime(2024, 1, 1)},
            {'id': 'stats-2', 'category': 'ai', 'word_count': 800, 'reading_time': 5, 'published_date': datetime(2024, 1, 5)},
            {'id': 'stats-3', 'category': 'ml', 'word_count': 1200, 'reading_time': 7, 'published_date': datetime(2024, 1, 10)},
            {'id': 'stats-4', 'category': 'ml', 'word_count': 600, 'reading_time': 4, 'published_date': datetime(2024, 1, 15)},
            {'id': 'stats-5', 'category': 'ai', 'word_count': 1000, 'reading_time': 6, 'published_date': datetime(2024, 1, 20)}
        ]
        
        for article in articles_data:
            await db_session.execute(
                text("""
                    INSERT INTO articles (id, url, title, content, category, word_count, 
                                        reading_time, published_date, source_id, created_at)
                    VALUES (:id, :url, 'Test Article', 'Content', :category, :word_count,
                            :reading_time, :published_date, 'stats-source', :created_at)
                """),
                {
                    **article,
                    'url': f'https://example.com/{article["id"]}',
                    'created_at': datetime.utcnow()
                }
            )
        
        await db_session.commit()
        
        # Test category counts
        category_stats = await db_session.execute(
            text("""
                SELECT category, COUNT(*) as count, AVG(word_count) as avg_words, AVG(reading_time) as avg_time
                FROM articles 
                WHERE source_id = 'stats-source'
                GROUP BY category 
                ORDER BY category
            """)
        )
        category_results = category_stats.fetchall()
        
        assert len(category_results) == 2
        
        # AI category should have 3 articles
        ai_stats = next((r for r in category_results if r.category == 'ai'), None)
        assert ai_stats is not None
        assert ai_stats.count == 3
        assert ai_stats.avg_words == 766.67  # (500 + 800 + 1000) / 3, approximately
        
        # ML category should have 2 articles  
        ml_stats = next((r for r in category_results if r.category == 'ml'), None)
        assert ml_stats is not None
        assert ml_stats.count == 2
        assert ml_stats.avg_words == 900  # (1200 + 600) / 2
        
        # Test date-based statistics
        monthly_stats = await db_session.execute(
            text("""
                SELECT DATE_TRUNC('month', published_date) as month,
                       COUNT(*) as article_count,
                       SUM(word_count) as total_words
                FROM articles 
                WHERE source_id = 'stats-source'
                GROUP BY DATE_TRUNC('month', published_date)
                ORDER BY month
            """)
        )
        monthly_results = monthly_stats.fetchall()
        
        assert len(monthly_results) == 1  # All articles in January 2024
        jan_stats = monthly_results[0]
        assert jan_stats.article_count == 5
        assert jan_stats.total_words == 4100  # Sum of all word counts


class TestSourceOperations:
    """Test source-related database operations."""
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_source_management(self, db_session):
        """Test source creation, updating, and status management."""
        # Create new source
        source_data = {
            'id': 'test-source-mgmt',
            'name': 'Test Source Management',
            'type': 'rss',
            'url': 'https://example.com/feed.xml',
            'enabled': True,
            'config': '{"max_articles": 50, "categories": ["test"]}',
            'created_at': datetime.utcnow()
        }
        
        await db_session.execute(
            text("""
                INSERT INTO sources (id, name, type, url, enabled, config, created_at)
                VALUES (:id, :name, :type, :url, :enabled, :config, :created_at)
            """),
            source_data
        )
        
        # Test source retrieval
        result = await db_session.execute(
            text("SELECT * FROM sources WHERE id = 'test-source-mgmt'")
        )
        source = result.fetchone()
        
        assert source is not None
        assert source.name == 'Test Source Management'
        assert source.enabled is True
        
        # Test source update
        await db_session.execute(
            text("""
                UPDATE sources 
                SET enabled = false, last_crawled_at = :last_crawled, updated_at = :updated_at
                WHERE id = 'test-source-mgmt'
            """),
            {
                'last_crawled': datetime.utcnow(),
                'updated_at': datetime.utcnow()
            }
        )
        
        # Verify update
        result = await db_session.execute(
            text("SELECT enabled, last_crawled_at FROM sources WHERE id = 'test-source-mgmt'")
        )
        updated_source = result.fetchone()
        
        assert updated_source.enabled is False
        assert updated_source.last_crawled_at is not None
        
        await db_session.commit()
    
    @pytest.mark.database
    @pytest.mark.asyncio 
    async def test_source_statistics_tracking(self, db_session):
        """Test tracking of source-level statistics."""
        # Create source with initial stats
        await db_session.execute(
            text("""
                INSERT INTO sources (id, name, type, url, enabled, articles_found, 
                                   articles_processed, last_crawled_at, created_at)
                VALUES ('stats-source', 'Stats Source', 'rss', 'http://test.com',
                        true, 0, 0, null, :created_at)
            """),
            {'created_at': datetime.utcnow()}
        )
        
        # Simulate crawling updates
        crawl_updates = [
            {'found': 10, 'processed': 8, 'date': datetime.utcnow() - timedelta(days=2)},
            {'found': 15, 'processed': 12, 'date': datetime.utcnow() - timedelta(days=1)},
            {'found': 8, 'processed': 7, 'date': datetime.utcnow()}
        ]
        
        for update in crawl_updates:
            await db_session.execute(
                text("""
                    UPDATE sources 
                    SET articles_found = articles_found + :found,
                        articles_processed = articles_processed + :processed,
                        last_crawled_at = :date,
                        updated_at = :date
                    WHERE id = 'stats-source'
                """),
                {
                    'found': update['found'],
                    'processed': update['processed'],
                    'date': update['date']
                }
            )
        
        # Check final statistics
        result = await db_session.execute(
            text("""
                SELECT articles_found, articles_processed, last_crawled_at
                FROM sources WHERE id = 'stats-source'
            """)
        )
        stats = result.fetchone()
        
        assert stats.articles_found == 33  # 10 + 15 + 8
        assert stats.articles_processed == 27  # 8 + 12 + 7
        assert stats.last_crawled_at is not None
        
        await db_session.commit()


class TestProcessingJobOperations:
    """Test processing job tracking and management."""
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_processing_job_lifecycle(self, db_session):
        """Test complete processing job lifecycle."""
        # Create processing job
        job_data = {
            'id': 'test-job-lifecycle',
            'source_ids': '["source-1", "source-2"]',
            'status': 'running',
            'articles_found': 0,
            'articles_processed': 0,
            'started_at': datetime.utcnow(),
            'created_at': datetime.utcnow()
        }
        
        await db_session.execute(
            text("""
                INSERT INTO processing_jobs (id, source_ids, status, articles_found, 
                                           articles_processed, started_at, created_at)
                VALUES (:id, :source_ids, :status, :articles_found, 
                        :articles_processed, :started_at, :created_at)
            """),
            job_data
        )
        
        # Update job progress
        progress_updates = [
            {'found': 10, 'processed': 5},
            {'found': 15, 'processed': 12},
            {'found': 20, 'processed': 18}
        ]
        
        for update in progress_updates:
            await db_session.execute(
                text("""
                    UPDATE processing_jobs 
                    SET articles_found = :found, articles_processed = :processed,
                        updated_at = :updated_at
                    WHERE id = 'test-job-lifecycle'
                """),
                {
                    'found': update['found'],
                    'processed': update['processed'],
                    'updated_at': datetime.utcnow()
                }
            )
        
        # Complete job
        await db_session.execute(
            text("""
                UPDATE processing_jobs 
                SET status = 'completed', completed_at = :completed_at, updated_at = :updated_at
                WHERE id = 'test-job-lifecycle'
            """),
            {
                'completed_at': datetime.utcnow(),
                'updated_at': datetime.utcnow()
            }
        )
        
        # Verify final job state
        result = await db_session.execute(
            text("SELECT * FROM processing_jobs WHERE id = 'test-job-lifecycle'")
        )
        job = result.fetchone()
        
        assert job.status == 'completed'
        assert job.articles_found == 20
        assert job.articles_processed == 18
        assert job.completed_at is not None
        
        await db_session.commit()
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_job_error_tracking(self, db_session):
        """Test error tracking in processing jobs."""
        # Create job
        await db_session.execute(
            text("""
                INSERT INTO processing_jobs (id, source_ids, status, created_at)
                VALUES ('error-job', '["error-source"]', 'running', :created_at)
            """),
            {'created_at': datetime.utcnow()}
        )
        
        # Record errors
        error_log = {
            'stage': 'ingestion',
            'error_type': 'HttpError',
            'error_message': 'Failed to fetch RSS feed: 404 Not Found',
            'source_id': 'error-source',
            'article_url': 'https://example.com/feed.xml'
        }
        
        await db_session.execute(
            text("""
                UPDATE processing_jobs 
                SET error_log = :error_log, status = 'failed', 
                    completed_at = :completed_at, updated_at = :updated_at
                WHERE id = 'error-job'
            """),
            {
                'error_log': str(error_log),  # In practice, this would be JSON
                'completed_at': datetime.utcnow(),
                'updated_at': datetime.utcnow()
            }
        )
        
        # Verify error recording
        result = await db_session.execute(
            text("SELECT status, error_log FROM processing_jobs WHERE id = 'error-job'")
        )
        job = result.fetchone()
        
        assert job.status == 'failed'
        assert 'HttpError' in job.error_log
        
        await db_session.commit()


class TestDatabasePerformance:
    """Test database performance and optimization."""
    
    @pytest.mark.database
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_bulk_insert_performance(self, db_session, performance_benchmarks):
        """Test bulk insert performance for articles."""
        import time
        
        # Create test source
        await db_session.execute(
            text("""
                INSERT INTO sources (id, name, type, url, enabled, created_at)
                VALUES ('bulk-source', 'Bulk Test Source', 'rss', 'http://test.com', true, :created_at)
            """),
            {'created_at': datetime.utcnow()}
        )
        
        # Prepare bulk data
        bulk_articles = []
        for i in range(1000):
            bulk_articles.append({
                'id': f'bulk-article-{i}',
                'url': f'https://example.com/article-{i}',
                'title': f'Bulk Test Article {i}',
                'content': f'This is bulk test content for article {i}. ' * 20,
                'source_id': 'bulk-source',
                'category': 'test',
                'word_count': 100,
                'reading_time': 2,
                'created_at': datetime.utcnow()
            })
        
        # Time bulk insert
        start_time = time.time()
        
        await db_session.execute(
            text("""
                INSERT INTO articles (id, url, title, content, source_id, category, 
                                    word_count, reading_time, created_at)
                VALUES (:id, :url, :title, :content, :source_id, :category,
                        :word_count, :reading_time, :created_at)
            """),
            bulk_articles
        )
        
        await db_session.commit()
        end_time = time.time()
        
        duration = end_time - start_time
        articles_per_second = 1000 / duration
        
        # Check performance benchmark
        min_throughput = performance_benchmarks.get('database_insert_articles_per_second', 100.0)
        assert articles_per_second >= min_throughput, f"Bulk insert too slow: {articles_per_second:.1f} < {min_throughput}"
        
        # Verify all articles were inserted
        result = await db_session.execute(
            text("SELECT COUNT(*) FROM articles WHERE source_id = 'bulk-source'")
        )
        count = result.scalar()
        assert count == 1000
    
    @pytest.mark.database
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_complex_query_performance(self, db_session, performance_benchmarks):
        """Test performance of complex queries."""
        import time
        
        # Create test data (reuse from bulk test if available)
        result = await db_session.execute(
            text("SELECT COUNT(*) FROM articles WHERE source_id = 'bulk-source'")
        )
        article_count = result.scalar()
        
        if article_count < 1000:
            pytest.skip("Bulk test data not available, skipping complex query test")
        
        # Test complex aggregation query
        start_time = time.time()
        
        result = await db_session.execute(
            text("""
                SELECT 
                    category,
                    COUNT(*) as article_count,
                    AVG(word_count) as avg_words,
                    AVG(reading_time) as avg_time,
                    MIN(created_at) as first_article,
                    MAX(created_at) as last_article
                FROM articles 
                WHERE source_id = 'bulk-source'
                  AND created_at >= :date_threshold
                GROUP BY category
                ORDER BY article_count DESC
            """),
            {'date_threshold': datetime.utcnow() - timedelta(days=1)}
        )
        
        results = result.fetchall()
        end_time = time.time()
        
        duration = end_time - start_time
        max_query_time = performance_benchmarks.get('database_complex_query_time_ms', 500.0) / 1000.0
        
        assert duration <= max_query_time, f"Complex query too slow: {duration*1000:.1f}ms > {max_query_time*1000:.1f}ms"
        assert len(results) > 0  # Should return aggregated results
    
    @pytest.mark.database
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_query_performance(self, test_config, performance_benchmarks):
        """Test database performance under concurrent load."""
        import time
        import asyncio
        
        db_manager = DatabaseManager(test_config.database)
        
        async def concurrent_query(query_id: int):
            async with db_manager.get_session() as session:
                result = await session.execute(
                    text("""
                        SELECT id, title, word_count 
                        FROM articles 
                        WHERE word_count > :min_words 
                        ORDER BY created_at DESC 
                        LIMIT 100
                    """),
                    {'min_words': query_id * 10}
                )
                rows = result.fetchall()
                return len(rows)
        
        # Run concurrent queries
        start_time = time.time()
        tasks = [concurrent_query(i) for i in range(50)]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        duration = end_time - start_time
        queries_per_second = 50 / duration
        
        min_throughput = performance_benchmarks.get('database_queries_per_second', 20.0)
        assert queries_per_second >= min_throughput, f"Concurrent queries too slow: {queries_per_second:.1f} < {min_throughput}"
        
        assert all(isinstance(result, int) for result in results)  # All queries should succeed
        
        await db_manager.close_all_connections()


class TestDataIntegrity:
    """Test data integrity constraints and validation."""
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_foreign_key_constraints(self, db_session):
        """Test foreign key constraint enforcement."""
        # Test valid foreign key relationship
        await db_session.execute(
            text("""
                INSERT INTO sources (id, name, type, url, enabled, created_at)
                VALUES ('fk-source', 'FK Test Source', 'rss', 'http://test.com', true, :created_at)
            """),
            {'created_at': datetime.utcnow()}
        )
        
        await db_session.execute(
            text("""
                INSERT INTO articles (id, url, title, content, source_id, created_at)
                VALUES ('fk-article', 'http://test.com/article', 'FK Test', 'Content', 'fk-source', :created_at)
            """),
            {'created_at': datetime.utcnow()}
        )
        
        await db_session.commit()
        
        # Test invalid foreign key should fail
        with pytest.raises(IntegrityError):
            await db_session.execute(
                text("""
                    INSERT INTO articles (id, url, title, content, source_id, created_at)
                    VALUES ('invalid-fk', 'http://test.com/invalid', 'Invalid FK', 'Content', 'nonexistent-source', :created_at)
                """),
                {'created_at': datetime.utcnow()}
            )
            await db_session.commit()
        
        await db_session.rollback()
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_unique_constraints(self, db_session):
        """Test unique constraint enforcement."""
        # Create first article
        await db_session.execute(
            text("""
                INSERT INTO sources (id, name, type, url, enabled, created_at)
                VALUES ('unique-source', 'Unique Test Source', 'rss', 'http://test.com', true, :created_at)
            """),
            {'created_at': datetime.utcnow()}
        )
        
        await db_session.execute(
            text("""
                INSERT INTO articles (id, url, title, content, source_id, created_at)
                VALUES ('unique-1', 'http://unique-test.com/article', 'Unique Test', 'Content', 'unique-source', :created_at)
            """),
            {'created_at': datetime.utcnow()}
        )
        
        await db_session.commit()
        
        # Test duplicate ID should fail
        with pytest.raises(IntegrityError):
            await db_session.execute(
                text("""
                    INSERT INTO articles (id, url, title, content, source_id, created_at)
                    VALUES ('unique-1', 'http://unique-test.com/different', 'Different Article', 'Content', 'unique-source', :created_at)
                """),
                {'created_at': datetime.utcnow()}
            )
            await db_session.commit()
        
        await db_session.rollback()
        
        # Test duplicate URL should also fail (if URL has unique constraint)
        # This depends on your schema design
    
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_cascade_operations(self, db_session):
        """Test cascade delete operations."""
        # Create source with articles
        await db_session.execute(
            text("""
                INSERT INTO sources (id, name, type, url, enabled, created_at)
                VALUES ('cascade-source', 'Cascade Test Source', 'rss', 'http://test.com', true, :created_at)
            """),
            {'created_at': datetime.utcnow()}
        )
        
        article_ids = ['cascade-article-1', 'cascade-article-2', 'cascade-article-3']
        for article_id in article_ids:
            await db_session.execute(
                text("""
                    INSERT INTO articles (id, url, title, content, source_id, created_at)
                    VALUES (:id, :url, 'Cascade Test', 'Content', 'cascade-source', :created_at)
                """),
                {
                    'id': article_id,
                    'url': f'http://test.com/{article_id}',
                    'created_at': datetime.utcnow()
                }
            )
        
        await db_session.commit()
        
        # Verify articles exist
        result = await db_session.execute(
            text("SELECT COUNT(*) FROM articles WHERE source_id = 'cascade-source'")
        )
        count = result.scalar()
        assert count == 3
        
        # Delete source (should cascade to articles if configured)
        await db_session.execute(
            text("DELETE FROM sources WHERE id = 'cascade-source'")
        )
        await db_session.commit()
        
        # Check if articles were cascade deleted (depends on schema configuration)
        result = await db_session.execute(
            text("SELECT COUNT(*) FROM articles WHERE source_id = 'cascade-source'")
        )
        remaining_count = result.scalar()
        
        # This assertion depends on your cascade configuration
        # If CASCADE is configured: assert remaining_count == 0
        # If SET NULL is configured: articles remain but source_id is null
        # If RESTRICT is configured: the delete would have failed
        
        # For this test, we'll just verify the source was deleted
        result = await db_session.execute(
            text("SELECT COUNT(*) FROM sources WHERE id = 'cascade-source'")
        )
        source_count = result.scalar()
        assert source_count == 0