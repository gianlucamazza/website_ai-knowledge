"""
Integration tests for end-to-end pipeline workflows.

Tests the complete content processing pipeline from ingestion to publication.
"""

import pytest
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from unittest.mock import AsyncMock, patch

from pipelines.orchestrators.langgraph.workflow import ContentPipelineWorkflow
from pipelines.ingest.rss_parser import RSSParser
from pipelines.normalize.content_extractor import ContentExtractor
from pipelines.dedup.simhash import SimHashDeduplicator
from pipelines.enrich.summarizer import ContentSummarizer
from pipelines.publish.markdown_generator import MarkdownGenerator
from pipelines.database.models import Article, Source, ProcessingJob


class TestEndToEndPipeline:
    """Test complete pipeline workflow integration."""
    
    @pytest.fixture
    async def pipeline_workflow(self, test_config, db_session):
        """Create pipeline workflow for integration testing."""
        workflow = ContentPipelineWorkflow(config=test_config)
        workflow.db_session = db_session
        return workflow
    
    @pytest.fixture
    def sample_rss_sources(self):
        """Sample RSS sources for testing."""
        return [
            {
                'id': 'ai_blog',
                'name': 'AI Research Blog',
                'type': 'rss',
                'url': 'https://ai-blog.example.com/feed.xml',
                'enabled': True,
                'categories': ['artificial-intelligence'],
                'tags': ['research', 'ai'],
                'max_articles_per_run': 10
            },
            {
                'id': 'ml_news',
                'name': 'ML News Feed',
                'type': 'rss', 
                'url': 'https://ml-news.example.com/rss',
                'enabled': True,
                'categories': ['machine-learning'],
                'tags': ['news', 'ml'],
                'max_articles_per_run': 20
            }
        ]
    
    @pytest.fixture
    def mock_rss_feeds(self):
        """Mock RSS feed content for testing."""
        return {
            'https://ai-blog.example.com/feed.xml': """<?xml version="1.0"?>
            <rss version="2.0">
                <channel>
                    <title>AI Research Blog</title>
                    <item>
                        <title>Advances in Neural Networks</title>
                        <link>https://ai-blog.example.com/neural-networks</link>
                        <description>Latest advances in neural network architectures</description>
                        <pubDate>Mon, 15 Jan 2024 10:00:00 GMT</pubDate>
                        <category>deep-learning</category>
                    </item>
                    <item>
                        <title>Machine Learning Ethics</title>
                        <link>https://ai-blog.example.com/ml-ethics</link>
                        <description>Exploring ethical considerations in ML systems</description>
                        <pubDate>Tue, 16 Jan 2024 14:30:00 GMT</pubDate>
                        <category>ethics</category>
                    </item>
                </channel>
            </rss>""",
            'https://ml-news.example.com/rss': """<?xml version="1.0"?>
            <rss version="2.0">
                <channel>
                    <title>ML News</title>
                    <item>
                        <title>New Deep Learning Framework Released</title>
                        <link>https://ml-news.example.com/framework-release</link>
                        <description>A new framework for deep learning applications</description>
                        <pubDate>Wed, 17 Jan 2024 09:15:00 GMT</pubDate>
                        <category>news</category>
                    </item>
                </channel>
            </rss>"""
        }
    
    @pytest.fixture
    def mock_article_content(self):
        """Mock full article content for testing."""
        return {
            'https://ai-blog.example.com/neural-networks': """
            <html>
                <head>
                    <title>Advances in Neural Networks</title>
                    <meta name="description" content="Latest advances in neural network architectures">
                    <meta name="author" content="Dr. Jane Smith">
                </head>
                <body>
                    <article>
                        <h1>Advances in Neural Networks</h1>
                        <p>Neural networks have seen remarkable advances in recent years. These advances have 
                        been driven by improvements in both algorithmic techniques and computational resources.</p>
                        
                        <h2>Key Innovations</h2>
                        <p>Several key innovations have contributed to these advances:</p>
                        <ul>
                            <li>Attention mechanisms for better sequence modeling</li>
                            <li>Residual connections for deeper networks</li>
                            <li>Batch normalization for stable training</li>
                            <li>Advanced optimization algorithms</li>
                        </ul>
                        
                        <h2>Applications</h2>
                        <p>These advances have enabled new applications in computer vision, natural language 
                        processing, and reinforcement learning. The impact on industry has been substantial.</p>
                        
                        <p>As we continue to push the boundaries of what's possible with neural networks, 
                        we must also consider the ethical implications and ensure responsible development.</p>
                    </article>
                </body>
            </html>
            """,
            'https://ai-blog.example.com/ml-ethics': """
            <html>
                <head>
                    <title>Machine Learning Ethics</title>
                    <meta name="description" content="Exploring ethical considerations in ML systems">
                    <meta name="author" content="Dr. Bob Johnson">
                </head>
                <body>
                    <article>
                        <h1>Machine Learning Ethics</h1>
                        <p>As machine learning systems become more prevalent in society, it's crucial to 
                        address the ethical implications of these technologies.</p>
                        
                        <h2>Core Ethical Principles</h2>
                        <p>Several core principles should guide ML development:</p>
                        <ul>
                            <li>Fairness and non-discrimination</li>
                            <li>Transparency and explainability</li>
                            <li>Privacy protection</li>
                            <li>Accountability and responsibility</li>
                        </ul>
                        
                        <h2>Challenges and Solutions</h2>
                        <p>Implementing ethical ML practices faces several challenges, but there are 
                        emerging solutions and frameworks to help address these issues.</p>
                    </article>
                </body>
            </html>
            """
        }
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_pipeline_workflow(
        self, 
        pipeline_workflow,
        sample_rss_sources,
        mock_rss_feeds,
        mock_article_content,
        temp_directory
    ):
        """Test complete pipeline from RSS ingestion to publication."""
        
        # Mock HTTP responses for RSS feeds and articles
        from aioresponses import aioresponses
        
        with aioresponses() as mock_http:
            # Mock RSS feeds
            for url, content in mock_rss_feeds.items():
                mock_http.get(url, status=200, body=content, headers={'content-type': 'application/rss+xml'})
            
            # Mock article content
            for url, content in mock_article_content.items():
                mock_http.get(url, status=200, body=content, headers={'content-type': 'text/html'})
            
            # Mock AI services
            with patch('openai.AsyncOpenAI') as mock_openai:
                mock_client = AsyncMock()
                mock_response = AsyncMock()
                mock_response.choices = [
                    AsyncMock(message=AsyncMock(content="AI-generated summary of the article content."))
                ]
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.return_value = mock_client
                
                # Set output directory
                output_dir = temp_directory / "output"
                pipeline_workflow.config.publishing.output_directory = str(output_dir)
                
                # Run complete pipeline
                result = await pipeline_workflow.process_sources(sample_rss_sources)
                
                # Verify results
                assert result['success'] is True
                assert result['processed_articles'] > 0
                assert result['published_articles'] > 0
                
                # Check that articles were processed through all stages
                assert result['ingestion']['articles_found'] >= 3  # Total from both feeds
                assert result['normalization']['articles_processed'] >= 3
                assert result['deduplication']['unique_articles'] >= 3
                assert result['enrichment']['articles_enriched'] >= 3
                assert result['publication']['files_created'] >= 3
                
                # Verify output files were created
                articles_dir = output_dir / "articles"
                assert articles_dir.exists()
                
                markdown_files = list(articles_dir.glob("*.md"))
                assert len(markdown_files) >= 3
                
                # Check content of generated files
                for md_file in markdown_files:
                    content = md_file.read_text()
                    assert content.startswith('---')  # Has frontmatter
                    assert 'title:' in content
                    assert 'category:' in content
                    assert '---' in content[3:]  # Frontmatter closes
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pipeline_with_duplicates(
        self, 
        pipeline_workflow,
        mock_rss_feeds,
        mock_article_content,
        temp_directory
    ):
        """Test pipeline handles duplicate content correctly."""
        
        # Create sources with overlapping content
        duplicate_sources = [
            {
                'id': 'source_1',
                'name': 'Source 1',
                'type': 'rss',
                'url': 'https://ai-blog.example.com/feed.xml',
                'enabled': True,
                'categories': ['ai'],
                'tags': ['test']
            },
            {
                'id': 'source_2', 
                'name': 'Source 2',
                'type': 'rss',
                'url': 'https://ai-blog.example.com/feed.xml',  # Same URL = duplicate content
                'enabled': True,
                'categories': ['ai'],
                'tags': ['test']
            }
        ]
        
        from aioresponses import aioresponses
        
        with aioresponses() as mock_http:
            # Mock the same feed for both sources
            for source in duplicate_sources:
                mock_http.get(
                    source['url'], 
                    status=200, 
                    body=mock_rss_feeds['https://ai-blog.example.com/feed.xml'],
                    headers={'content-type': 'application/rss+xml'}
                )
            
            # Mock article content
            for url, content in mock_article_content.items():
                mock_http.get(url, status=200, body=content, headers={'content-type': 'text/html'})
            
            # Run pipeline
            result = await pipeline_workflow.process_sources(duplicate_sources)
            
            # Should detect and handle duplicates
            assert result['deduplication']['duplicates_found'] > 0
            assert result['deduplication']['unique_articles'] < result['ingestion']['articles_found']
            
            # Should still publish unique content
            assert result['published_articles'] > 0
    
    @pytest.mark.integration
    @pytest.mark.asyncio  
    async def test_pipeline_error_recovery(
        self,
        pipeline_workflow,
        sample_rss_sources,
        mock_rss_feeds
    ):
        """Test pipeline error handling and recovery mechanisms."""
        
        from aioresponses import aioresponses
        
        with aioresponses() as mock_http:
            # Mock one successful feed and one failing feed
            mock_http.get(
                'https://ai-blog.example.com/feed.xml',
                status=200,
                body=mock_rss_feeds['https://ai-blog.example.com/feed.xml'],
                headers={'content-type': 'application/rss+xml'}
            )
            mock_http.get(
                'https://ml-news.example.com/rss',
                status=500  # Server error
            )
            
            # Mock article content (some successful, some failing)
            mock_http.get(
                'https://ai-blog.example.com/neural-networks',
                status=200,
                body="<html><body>Valid content</body></html>",
                headers={'content-type': 'text/html'}
            )
            mock_http.get(
                'https://ai-blog.example.com/ml-ethics',
                status=404  # Not found
            )
            
            # Run pipeline
            result = await pipeline_workflow.process_sources(sample_rss_sources)
            
            # Should handle errors gracefully
            assert 'errors' in result
            assert len(result['errors']) > 0  # Should record errors
            
            # Should still process what it can
            assert result.get('processed_articles', 0) > 0
            
            # Should have partial success
            assert result['success'] is True  # Pipeline completes despite some errors
    
    @pytest.mark.integration
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_pipeline_database_integration(
        self,
        pipeline_workflow,
        db_session,
        sample_rss_sources,
        mock_rss_feeds,
        mock_article_content
    ):
        """Test pipeline database operations and persistence."""
        
        from aioresponses import aioresponses
        
        with aioresponses() as mock_http:
            # Mock HTTP responses
            for url, content in mock_rss_feeds.items():
                mock_http.get(url, status=200, body=content, headers={'content-type': 'application/rss+xml'})
            
            for url, content in mock_article_content.items():
                mock_http.get(url, status=200, body=content, headers={'content-type': 'text/html'})
            
            # Mock AI services
            with patch('openai.AsyncOpenAI') as mock_openai:
                mock_client = AsyncMock()
                mock_response = AsyncMock()
                mock_response.choices = [AsyncMock(message=AsyncMock(content="Test summary"))]
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.return_value = mock_client
                
                # Run pipeline
                result = await pipeline_workflow.process_sources(sample_rss_sources)
                
                # Verify database entries were created
                articles = await db_session.execute(
                    "SELECT * FROM articles WHERE source_id IN ('ai_blog', 'ml_news')"
                )
                article_rows = articles.fetchall()
                
                assert len(article_rows) >= 3  # Should have stored articles
                
                # Check article data integrity
                for row in article_rows:
                    assert row.title is not None
                    assert row.url is not None
                    assert row.content is not None
                    assert row.source_id in ['ai_blog', 'ml_news']
                    assert row.created_at is not None
                
                # Check processing job was recorded
                jobs = await db_session.execute("SELECT * FROM processing_jobs")
                job_rows = jobs.fetchall()
                
                assert len(job_rows) > 0
                latest_job = max(job_rows, key=lambda x: x.created_at)
                assert latest_job.status in ['completed', 'completed_with_errors']
                assert latest_job.articles_processed > 0
    
    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_pipeline_performance_large_batch(
        self,
        pipeline_workflow,
        performance_benchmarks,
        temp_directory
    ):
        """Test pipeline performance with large batch of articles."""
        
        # Create large RSS feed
        large_feed = self._create_large_rss_feed(100)
        large_sources = [{
            'id': 'large_source',
            'name': 'Large Test Source',
            'type': 'rss',
            'url': 'https://example.com/large-feed.xml',
            'enabled': True,
            'categories': ['test'],
            'tags': ['performance'],
            'max_articles_per_run': 100
        }]
        
        from aioresponses import aioresponses
        
        with aioresponses() as mock_http:
            mock_http.get(
                'https://example.com/large-feed.xml',
                status=200,
                body=large_feed,
                headers={'content-type': 'application/rss+xml'}
            )
            
            # Mock article content for all URLs
            for i in range(100):
                article_url = f'https://example.com/article-{i+1}'
                mock_http.get(
                    article_url,
                    status=200,
                    body=f"<html><body><h1>Article {i+1}</h1><p>Content for article {i+1}</p></body></html>",
                    headers={'content-type': 'text/html'}
                )
            
            # Mock AI services with faster responses
            with patch('openai.AsyncOpenAI') as mock_openai:
                mock_client = AsyncMock()
                mock_response = AsyncMock()
                mock_response.choices = [AsyncMock(message=AsyncMock(content="Quick summary"))]
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.return_value = mock_client
                
                # Measure performance
                start_time = asyncio.get_event_loop().time()
                result = await pipeline_workflow.process_sources(large_sources)
                end_time = asyncio.get_event_loop().time()
                
                duration = end_time - start_time
                articles_per_second = result.get('processed_articles', 0) / duration
                
                # Check performance benchmarks
                min_throughput = performance_benchmarks.get('pipeline_articles_per_second', 5.0)
                assert articles_per_second >= min_throughput, f"Pipeline too slow: {articles_per_second:.2f} < {min_throughput}"
                
                # Check that all articles were processed
                assert result['processed_articles'] >= 50  # Should process most articles
                assert result['success'] is True
    
    def _create_large_rss_feed(self, item_count: int) -> str:
        """Create large RSS feed for performance testing."""
        items = []
        for i in range(item_count):
            items.append(f"""
            <item>
                <title>Performance Test Article {i+1}</title>
                <link>https://example.com/article-{i+1}</link>
                <description>This is performance test article {i+1}</description>
                <pubDate>Mon, {(i % 28) + 1:02d} Jan 2024 10:00:00 GMT</pubDate>
                <category>performance</category>
            </item>
            """)
        
        return f"""<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Large Performance Test Feed</title>
                <description>Large feed for performance testing</description>
                <link>https://example.com</link>
                {''.join(items)}
            </channel>
        </rss>"""


class TestPipelineComponentIntegration:
    """Test integration between pipeline components."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ingestion_to_normalization(self, sample_rss_feed, sample_html_content):
        """Test data flow from ingestion to normalization."""
        # Setup components
        rss_parser = RSSParser()
        content_extractor = ContentExtractor()
        
        from aioresponses import aioresponses
        
        with aioresponses() as mock_http:
            # Mock RSS feed
            mock_http.get(
                'https://example.com/feed.xml',
                status=200,
                body=sample_rss_feed,
                headers={'content-type': 'application/rss+xml'}
            )
            
            # Mock article content
            mock_http.get(
                'https://example.com/neural-networks',
                status=200,
                body=sample_html_content,
                headers={'content-type': 'text/html'}
            )
            mock_http.get(
                'https://example.com/deep-learning',
                status=200,
                body=sample_html_content,
                headers={'content-type': 'text/html'}
            )
            
            # Ingest articles
            source_config = {'max_articles_per_run': 10}
            articles = await rss_parser.parse_feed('https://example.com/feed.xml', source_config)
            
            assert len(articles) > 0
            
            # Normalize first article
            article = articles[0]
            assert 'raw_html' in article or 'content' in article
            
            raw_content = article.get('raw_html', article.get('content', ''))
            normalized_result = await content_extractor.extract_content(raw_content, article['url'])
            
            # Verify normalization results
            assert normalized_result['title'] is not None
            assert normalized_result['content'] is not None
            assert normalized_result['word_count'] > 0
            assert normalized_result['quality_score'] >= 0
            
            # Check that data flows correctly between stages
            assert len(normalized_result['content']) > len(article.get('summary', ''))
            assert normalized_result['language'] in ['en', 'unknown']
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_normalization_to_deduplication(self):
        """Test data flow from normalization to deduplication."""
        content_extractor = ContentExtractor()
        deduplicator = SimHashDeduplicator(k=3)
        
        # Create test articles with varying similarity
        test_articles = [
            {
                'url': 'https://example.com/article1',
                'content': """
                <html><body>
                <h1>Machine Learning Fundamentals</h1>
                <p>Machine learning is a subset of artificial intelligence that enables 
                systems to learn and improve from experience without explicit programming.</p>
                </body></html>
                """
            },
            {
                'url': 'https://example.com/article2',
                'content': """
                <html><body>
                <h1>ML Basics</h1>
                <p>Machine learning represents a subset of AI that allows systems to 
                learn and improve from experience without explicit programming.</p>
                </body></html>
                """
            },
            {
                'url': 'https://example.com/article3',
                'content': """
                <html><body>
                <h1>Deep Learning Guide</h1>
                <p>Deep learning uses neural networks with multiple layers to process 
                complex data patterns and extract meaningful features.</p>
                </body></html>
                """
            }
        ]
        
        # Normalize and deduplicate
        normalized_articles = []
        for i, article in enumerate(test_articles):
            normalized = await content_extractor.extract_content(article['content'], article['url'])
            normalized['id'] = f'article_{i+1}'
            normalized_articles.append(normalized)
        
        # Add to deduplicator
        for article in normalized_articles:
            deduplicator.add_content(article['id'], article['content'])
        
        # Test duplicate detection
        test_content = normalized_articles[0]['content']
        duplicates = deduplicator.find_duplicates(test_content, exclude_id='article_1')
        
        # Should find article_2 as similar (both about ML)
        duplicate_ids = [dup[0] for dup in duplicates]
        similarities = [dup[1] for dup in duplicates]
        
        assert 'article_2' in duplicate_ids
        article_2_similarity = similarities[duplicate_ids.index('article_2')]
        assert article_2_similarity > 0.5  # Should be similar
        
        # article_3 should be less similar or not found
        if 'article_3' in duplicate_ids:
            article_3_similarity = similarities[duplicate_ids.index('article_3')]
            assert article_3_similarity < article_2_similarity
    
    @pytest.mark.integration 
    @pytest.mark.asyncio
    async def test_enrichment_to_publication(self, mock_openai_client):
        """Test data flow from enrichment to publication."""
        summarizer = ContentSummarizer()
        markdown_generator = MarkdownGenerator()
        
        # Mock AI services
        with patch.object(summarizer, 'openai_client', mock_openai_client):
            # Test article data
            article_data = {
                'id': 'test-article',
                'title': 'Test Article',
                'url': 'https://example.com/test',
                'content': """
                Artificial intelligence has revolutionized many industries through advanced
                machine learning techniques. These systems can process vast amounts of data
                and identify patterns that would be difficult for humans to detect.
                
                The applications of AI span across healthcare, finance, transportation,
                and many other sectors. As we continue to develop these technologies,
                it's important to consider ethical implications and ensure responsible
                development practices.
                """,
                'author': 'Test Author',
                'published_date': datetime(2024, 1, 15, 10, 0, 0),
                'category': 'artificial-intelligence',
                'tags': ['ai', 'machine-learning'],
                'source_id': 'test-source'
            }
            
            # Enrich content
            summary = await summarizer.generate_summary(
                article_data['content'],
                max_length=200,
                provider='openai'
            )
            
            key_points = await summarizer.generate_key_points(
                article_data['content'],
                max_points=3
            )
            
            metadata_enhancements = await summarizer.enhance_metadata(
                article_data['content'],
                {'category': article_data['category'], 'tags': article_data['tags']}
            )
            
            # Add enriched data to article
            enriched_article = {
                **article_data,
                'summary': summary,
                'key_points': key_points,
                'ai_generated_summary': True,
                'enhanced_tags': metadata_enhancements.get('suggested_tags', []),
                'complexity_level': metadata_enhancements.get('complexity_level', 'intermediate')
            }
            
            # Generate publication markdown
            markdown_content = markdown_generator.generate_article_markdown(enriched_article)
            
            # Verify enriched data is included in publication
            assert summary in markdown_content
            assert 'keyPoints:' in markdown_content
            assert 'complexityLevel:' in markdown_content
            
            # Validate markdown structure
            validation_result = markdown_generator.validate_markdown_output(markdown_content)
            assert validation_result['valid'] is True
            assert validation_result['has_frontmatter'] is True
            assert validation_result['has_content'] is True


class TestPipelineStateManagement:
    """Test pipeline state management and recovery."""
    
    @pytest.mark.integration
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_pipeline_resume_after_failure(self, db_session, temp_directory):
        """Test pipeline can resume from previous state after failure."""
        from pipelines.orchestrators.langgraph.workflow import ContentPipelineWorkflow
        from pipelines.config import PipelineConfig
        
        # Create pipeline with test config
        test_config = PipelineConfig(
            environment='test',
            publishing=type('obj', (object,), {
                'output_directory': str(temp_directory / 'output')
            })()
        )
        
        workflow = ContentPipelineWorkflow(config=test_config)
        workflow.db_session = db_session
        
        # Simulate partial pipeline execution
        test_sources = [{
            'id': 'resume_test',
            'name': 'Resume Test Source',
            'type': 'rss',
            'url': 'https://example.com/resume-test.xml',
            'enabled': True
        }]
        
        # Create processing job record
        from sqlalchemy import text
        await db_session.execute(
            text("""
                INSERT INTO processing_jobs (id, source_ids, status, articles_found, articles_processed, created_at)
                VALUES ('test-job-1', '["resume_test"]', 'failed', 5, 2, :created_at)
            """),
            {'created_at': datetime.utcnow()}
        )
        
        # Create some processed articles
        await db_session.execute(
            text("""
                INSERT INTO articles (id, url, title, content, source_id, created_at, processing_status)
                VALUES 
                ('article-1', 'https://example.com/1', 'Article 1', 'Content 1', 'resume_test', :created_at, 'normalized'),
                ('article-2', 'https://example.com/2', 'Article 2', 'Content 2', 'resume_test', :created_at, 'enriched')
            """),
            {'created_at': datetime.utcnow()}
        )
        
        await db_session.commit()
        
        # Test resume functionality
        resume_result = await workflow.resume_failed_job('test-job-1')
        
        # Should identify articles that need further processing
        assert resume_result['resumable'] is True
        assert resume_result['articles_to_process'] > 0
        
        # Should be able to continue from where it left off
        assert 'next_stage' in resume_result
        assert resume_result['next_stage'] in ['deduplication', 'enrichment', 'publication']
    
    @pytest.mark.integration
    @pytest.mark.database
    @pytest.mark.asyncio
    async def test_pipeline_checkpoint_creation(self, db_session):
        """Test pipeline creates checkpoints during processing."""
        from pipelines.orchestrators.langgraph.workflow import ContentPipelineWorkflow
        
        workflow = ContentPipelineWorkflow()
        workflow.db_session = db_session
        
        # Test checkpoint creation
        checkpoint_data = {
            'job_id': 'checkpoint-test',
            'stage': 'normalization',
            'processed_count': 10,
            'total_count': 50,
            'last_processed_id': 'article-10',
            'metadata': {'batch_size': 10, 'errors': []}
        }
        
        checkpoint_id = await workflow.create_checkpoint(checkpoint_data)
        assert checkpoint_id is not None
        
        # Test checkpoint retrieval
        retrieved_checkpoint = await workflow.get_checkpoint(checkpoint_id)
        assert retrieved_checkpoint is not None
        assert retrieved_checkpoint['job_id'] == 'checkpoint-test'
        assert retrieved_checkpoint['stage'] == 'normalization'
        assert retrieved_checkpoint['processed_count'] == 10