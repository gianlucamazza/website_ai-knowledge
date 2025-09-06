"""
Unit tests for content ingestion components.

Tests RSS parsing, web scraping, and source management functionality.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urljoin

import feedparser
from aioresponses import aioresponses

from pipelines.ingest.rss_parser import RSSParser
from pipelines.ingest.scraper import EthicalScraper
from pipelines.ingest.source_manager import SourceManager
from pipelines.database.models import ContentType


class TestRSSParser:
    """Test RSS feed parsing functionality."""
    
    @pytest.fixture
    def rss_parser(self):
        """Create RSSParser instance for testing."""
        return RSSParser()
    
    @pytest.fixture
    def sample_source_config(self):
        """Sample source configuration."""
        return {
            'max_articles_per_run': 10,
            'categories': ['ai', 'machine-learning'],
            'tags': ['test', 'automation'],
            'language': 'en',
            'content_type': 'article'
        }
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_parse_feed_success(self, rss_parser, sample_rss_feed, sample_source_config):
        """Test successful RSS feed parsing."""
        with aioresponses() as m:
            feed_url = "https://example.com/feed.xml"
            m.get(feed_url, status=200, body=sample_rss_feed, headers={'content-type': 'application/rss+xml'})
            
            with patch.object(rss_parser, '_fetch_full_content', return_value="Full article content here"):
                articles = await rss_parser.parse_feed(feed_url, sample_source_config)
                
                assert len(articles) == 2
                
                # Check first article
                article = articles[0]
                assert article['title'] == "Introduction to Neural Networks"
                assert article['url'] == "https://example.com/neural-networks"
                assert 'ai' in article['categories']
                assert 'test' in article['tags']
                assert article['language'] == 'en'
                assert article['content_type'] == ContentType.ARTICLE
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_parse_feed_empty_response(self, rss_parser, sample_source_config):
        """Test handling of empty feed response."""
        with aioresponses() as m:
            feed_url = "https://example.com/empty-feed.xml"
            m.get(feed_url, status=404)
            
            articles = await rss_parser.parse_feed(feed_url, sample_source_config)
            assert articles == []
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_parse_feed_malformed_xml(self, rss_parser, sample_source_config):
        """Test handling of malformed XML feed."""
        malformed_xml = "<rss><channel><item><title>Test</title></item>"  # Missing closing tags
        
        with aioresponses() as m:
            feed_url = "https://example.com/malformed.xml"
            m.get(feed_url, status=200, body=malformed_xml)
            
            # Should still attempt to parse despite warnings
            articles = await rss_parser.parse_feed(feed_url, sample_source_config)
            # May return empty list or partial results depending on feedparser behavior
            assert isinstance(articles, list)
    
    @pytest.mark.unit
    def test_extract_content_from_entry(self, rss_parser):
        """Test content extraction from feed entry."""
        # Mock feed entry with content
        entry = MagicMock()
        entry.content = [{'value': '<p>This is the full article content.</p>'}]
        entry.summary = "This is a summary"
        
        content = rss_parser._extract_content(entry)
        assert content == '<p>This is the full article content.</p>'
        
        # Test fallback to summary
        del entry.content
        content = rss_parser._extract_content(entry)
        assert content == "This is a summary"
    
    @pytest.mark.unit
    def test_extract_author(self, rss_parser):
        """Test author extraction from feed entry."""
        entry = MagicMock()
        
        # Test with author_detail dict
        entry.author_detail = {'name': 'John Doe', 'email': 'john@example.com'}
        author = rss_parser._extract_author(entry)
        assert author == 'John Doe'
        
        # Test with plain author string
        del entry.author_detail
        entry.author = 'Jane Smith'
        author = rss_parser._extract_author(entry)
        assert author == 'Jane Smith'
        
        # Test with no author
        del entry.author
        author = rss_parser._extract_author(entry)
        assert author is None
    
    @pytest.mark.unit
    def test_extract_publish_date(self, rss_parser):
        """Test publish date extraction from feed entry."""
        import time
        
        entry = MagicMock()
        
        # Test with parsed time struct
        test_time = time.struct_time((2024, 1, 15, 10, 0, 0, 0, 15, 0))
        entry.published_parsed = test_time
        
        date = rss_parser._extract_publish_date(entry)
        assert date == datetime(2024, 1, 15, 10, 0, 0)
        
        # Test with string date
        del entry.published_parsed
        entry.published = "2024-01-15T10:00:00Z"
        
        date = rss_parser._extract_publish_date(entry)
        assert date.year == 2024
        assert date.month == 1
        assert date.day == 15
    
    @pytest.mark.unit
    def test_extract_tags(self, rss_parser, sample_source_config):
        """Test tag extraction from feed entry."""
        entry = MagicMock()
        
        # Mock tags
        tag1 = MagicMock()
        tag1.term = "Machine Learning"
        tag2 = MagicMock()
        tag2.term = "AI"
        
        entry.tags = [tag1, tag2]
        entry.categories = ["Technology", "Science"]
        
        tags = rss_parser._extract_tags(entry, sample_source_config)
        
        # Should include feed tags, categories, and source tags
        expected_tags = {'machine learning', 'ai', 'technology', 'science', 'test', 'automation'}
        assert set(tags) == expected_tags
        assert len(tags) <= 10  # Should be limited to 10 tags
    
    @pytest.mark.unit
    def test_determine_content_type(self, rss_parser, sample_source_config):
        """Test content type determination."""
        entry = MagicMock()
        
        # Test tutorial detection
        entry.title = "How to Build Neural Networks: A Complete Guide"
        content_type = rss_parser._determine_content_type(entry, sample_source_config)
        assert content_type == ContentType.TUTORIAL
        
        # Test news detection  
        entry.title = "OpenAI Announces GPT-5 Release"
        content_type = rss_parser._determine_content_type(entry, sample_source_config)
        assert content_type == ContentType.NEWS
        
        # Test default to article
        entry.title = "Understanding Machine Learning Concepts"
        content_type = rss_parser._determine_content_type(entry, sample_source_config)
        assert content_type == ContentType.ARTICLE
    
    @pytest.mark.unit
    def test_validate_article_data(self, rss_parser):
        """Test article data validation."""
        # Valid article data
        valid_data = {
            'url': 'https://example.com/article',
            'title': 'Test Article',
            'raw_html': '<p>This is a test article with sufficient content for validation.</p>',
            'summary': 'Test summary'
        }
        
        assert rss_parser.validate_article_data(valid_data) is True
        
        # Missing required field
        invalid_data = valid_data.copy()
        del invalid_data['title']
        assert rss_parser.validate_article_data(invalid_data) is False
        
        # Invalid URL
        invalid_data = valid_data.copy()
        invalid_data['url'] = 'not-a-valid-url'
        assert rss_parser.validate_article_data(invalid_data) is False
        
        # Content too short
        invalid_data = valid_data.copy()
        invalid_data['raw_html'] = 'Short'
        del invalid_data['summary']
        assert rss_parser.validate_article_data(invalid_data) is False


class TestEthicalScraper:
    """Test web scraping functionality with ethical considerations."""
    
    @pytest.fixture
    def scraper(self):
        """Create EthicalScraper instance for testing."""
        return EthicalScraper()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_fetch_url_success(self, scraper, sample_html_content):
        """Test successful URL fetching."""
        with aioresponses() as m:
            url = "https://example.com/article"
            m.get(url, status=200, body=sample_html_content, headers={'content-type': 'text/html'})
            
            result = await scraper.fetch_url(url)
            
            assert result is not None
            assert result['status_code'] == 200
            assert result['url'] == url
            assert 'AI Ethics' in result['content']
            assert result['headers']['content-type'] == 'text/html'
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_fetch_url_404(self, scraper):
        """Test handling of 404 responses."""
        with aioresponses() as m:
            url = "https://example.com/not-found"
            m.get(url, status=404)
            
            result = await scraper.fetch_url(url)
            assert result is None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_fetch_url_timeout(self, scraper):
        """Test handling of request timeouts."""
        with aioresponses() as m:
            url = "https://example.com/slow"
            m.get(url, exception=asyncio.TimeoutError())
            
            result = await scraper.fetch_url(url)
            assert result is None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_respect_robots_txt(self, scraper):
        """Test robots.txt compliance."""
        robots_content = """
        User-agent: *
        Disallow: /private/
        Allow: /public/
        """
        
        with aioresponses() as m:
            base_url = "https://example.com"
            m.get(f"{base_url}/robots.txt", status=200, body=robots_content)
            
            # Should allow public URLs
            allowed_url = f"{base_url}/public/article"
            m.get(allowed_url, status=200, body="<html>Content</html>")
            
            result = await scraper.fetch_url(allowed_url)
            assert result is not None
            
            # Should disallow private URLs
            disallowed_url = f"{base_url}/private/secret"
            result = await scraper.fetch_url(disallowed_url)
            assert result is None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_request_delay(self, scraper):
        """Test request delay implementation."""
        with aioresponses() as m:
            urls = [
                "https://example.com/article1",
                "https://example.com/article2"
            ]
            
            for url in urls:
                m.get(url, status=200, body="<html>Content</html>")
            
            start_time = datetime.now()
            
            # Fetch multiple URLs
            results = []
            for url in urls:
                result = await scraper.fetch_url(url)
                results.append(result)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Should have waited between requests (default delay is 1 second)
            assert duration >= 1.0
            assert all(result is not None for result in results)


class TestSourceManager:
    """Test source configuration and management."""
    
    @pytest.fixture
    def temp_sources_file(self, temp_directory):
        """Create temporary sources configuration file."""
        sources_content = """
        sources:
          ai_blog:
            name: "AI Research Blog"
            type: "rss"
            url: "https://ai-blog.com/feed.xml"
            enabled: true
            categories: ["ai", "research"]
            tags: ["academic", "ai"]
            max_articles_per_run: 50
            language: "en"
          
          tech_news:
            name: "Tech News"
            type: "rss"
            url: "https://tech-news.com/rss"
            enabled: false
            categories: ["technology", "news"]
            tags: ["news", "tech"]
            max_articles_per_run: 100
            language: "en"
        """
        
        sources_file = temp_directory / "sources.yaml"
        sources_file.write_text(sources_content)
        return sources_file
    
    @pytest.fixture
    def source_manager(self, temp_sources_file):
        """Create SourceManager with test configuration."""
        return SourceManager(str(temp_sources_file))
    
    @pytest.mark.unit
    async def test_load_sources(self, source_manager):
        """Test loading source configurations."""
        sources = await source_manager.get_enabled_sources()
        
        # Only enabled sources should be returned
        assert len(sources) == 1
        assert "ai_blog" in sources
        assert sources["ai_blog"]["name"] == "AI Research Blog"
        assert sources["ai_blog"]["enabled"] is True
    
    @pytest.mark.unit
    async def test_get_all_sources(self, source_manager):
        """Test getting all sources including disabled ones."""
        all_sources = await source_manager.get_all_sources()
        
        assert len(all_sources) == 2
        assert "ai_blog" in all_sources
        assert "tech_news" in all_sources
    
    @pytest.mark.unit
    async def test_validate_source_config(self, source_manager):
        """Test source configuration validation."""
        # Valid configuration
        valid_config = {
            "name": "Test Source",
            "type": "rss",
            "url": "https://example.com/feed.xml",
            "enabled": True
        }
        
        is_valid = source_manager.validate_source_config(valid_config)
        assert is_valid is True
        
        # Invalid configuration - missing required fields
        invalid_config = {
            "name": "Test Source"
            # Missing type and url
        }
        
        is_valid = source_manager.validate_source_config(invalid_config)
        assert is_valid is False
    
    @pytest.mark.unit
    async def test_add_source(self, source_manager, temp_directory):
        """Test adding new source configuration."""
        new_source = {
            "name": "New AI Blog",
            "type": "rss",
            "url": "https://new-ai-blog.com/feed.xml",
            "enabled": True,
            "categories": ["ai"],
            "tags": ["new"]
        }
        
        success = await source_manager.add_source("new_blog", new_source)
        assert success is True
        
        # Verify source was added
        all_sources = await source_manager.get_all_sources()
        assert "new_blog" in all_sources
        assert all_sources["new_blog"]["name"] == "New AI Blog"


# Performance tests for ingestion components
class TestIngestionPerformance:
    """Performance tests for ingestion components."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_rss_parsing_performance(self, rss_parser, performance_benchmarks):
        """Test RSS parsing performance with large feeds."""
        # Create large RSS feed for testing
        large_feed = self._create_large_rss_feed(1000)
        
        with aioresponses() as m:
            feed_url = "https://example.com/large-feed.xml"
            m.get(feed_url, status=200, body=large_feed)
            
            start_time = datetime.now()
            articles = await rss_parser.parse_feed(feed_url, {"max_articles_per_run": 1000})
            end_time = datetime.now()
            
            duration = (end_time - start_time).total_seconds()
            articles_per_second = len(articles) / duration if duration > 0 else 0
            
            # Should meet performance benchmark
            min_rate = performance_benchmarks["content_ingestion_per_second"]
            assert articles_per_second >= min_rate, f"RSS parsing too slow: {articles_per_second:.2f} < {min_rate}"
    
    def _create_large_rss_feed(self, item_count: int) -> str:
        """Create large RSS feed for performance testing."""
        items = []
        for i in range(item_count):
            items.append(f"""
            <item>
                <title>Article {i+1}: AI Topic</title>
                <link>https://example.com/article-{i+1}</link>
                <description>This is article {i+1} about AI topics</description>
                <pubDate>Mon, {i+1:02d} Jan 2024 10:00:00 GMT</pubDate>
                <category>ai</category>
            </item>
            """)
        
        return f"""<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Large Test Feed</title>
                <description>Large feed for performance testing</description>
                <link>https://example.com</link>
                {''.join(items)}
            </channel>
        </rss>"""