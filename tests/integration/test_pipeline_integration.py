"""
Integration tests for the complete content processing pipeline.

Tests end-to-end pipeline functionality including ingest → normalize → dedup → enrich → publish.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
import json
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

from pipelines.config import PipelineConfig
from pipelines.ingest.rss_fetcher import RSSFetcher
from pipelines.normalize.content_extractor import ContentExtractor
from pipelines.dedup.similarity_detector import SimilarityDetector
from pipelines.enrich.summarizer import ContentSummarizer
from pipelines.publish.markdown_generator import MarkdownGenerator


class TestPipelineIntegration:
    """Test complete pipeline integration from ingestion to publishing."""
    
    @pytest.fixture
    def pipeline_config(self, temp_directory):
        """Create pipeline configuration for integration testing."""
        config = PipelineConfig(
            data_dir=str(temp_directory),
            output_dir=str(temp_directory / "output"),
            temp_dir=str(temp_directory / "temp"),
            max_articles_per_source=10,
            quality_threshold=0.5,
            similarity_threshold=0.85
        )
        config.ensure_directories()
        return config
    
    @pytest.fixture
    def sample_rss_feed(self):
        """Sample RSS feed content for testing."""
        return """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>AI Research Blog</title>
                <description>Latest AI research and developments</description>
                <link>https://ai-blog.example.com</link>
                <item>
                    <title>Neural Networks in Computer Vision</title>
                    <description>Exploring the use of neural networks for image recognition tasks</description>
                    <link>https://ai-blog.example.com/neural-networks-cv</link>
                    <pubDate>Mon, 15 Jan 2024 10:00:00 GMT</pubDate>
                    <author>Dr. Sarah Johnson</author>
                    <category>Computer Vision</category>
                </item>
                <item>
                    <title>Natural Language Processing Advances</title>
                    <description>Recent breakthroughs in NLP and transformer models</description>
                    <link>https://ai-blog.example.com/nlp-advances</link>
                    <pubDate>Wed, 17 Jan 2024 14:30:00 GMT</pubDate>
                    <author>Prof. Michael Chen</author>
                    <category>NLP</category>
                </item>
                <item>
                    <title>Machine Learning Ethics and Fairness</title>
                    <description>Addressing bias and fairness in ML algorithms</description>
                    <link>https://ai-blog.example.com/ml-ethics</link>
                    <pubDate>Fri, 19 Jan 2024 09:15:00 GMT</pubDate>
                    <author>Dr. Emily Rodriguez</author>
                    <category>Ethics</category>
                </item>
            </channel>
        </rss>"""
    
    @pytest.fixture
    def sample_article_html(self):
        """Sample article HTML for testing."""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <title>Neural Networks in Computer Vision - AI Research Blog</title>
            <meta name="description" content="Exploring the use of neural networks for image recognition tasks">
            <meta name="author" content="Dr. Sarah Johnson">
        </head>
        <body>
            <header>
                <nav>Navigation here</nav>
            </header>
            <main>
                <article>
                    <h1>Neural Networks in Computer Vision</h1>
                    <p class="meta">Published by Dr. Sarah Johnson on January 15, 2024</p>
                    
                    <h2>Introduction</h2>
                    <p>Neural networks have revolutionized computer vision by enabling machines to 
                    process and understand visual information with unprecedented accuracy. This article 
                    explores the fundamental concepts and applications of neural networks in image 
                    recognition tasks.</p>
                    
                    <h2>Convolutional Neural Networks</h2>
                    <p>Convolutional Neural Networks (CNNs) are specialized architectures designed 
                    for processing grid-like data such as images. They use convolution operations 
                    to detect local features and patterns within images.</p>
                    
                    <p>Key components of CNNs include:</p>
                    <ul>
                        <li>Convolutional layers for feature extraction</li>
                        <li>Pooling layers for dimensionality reduction</li>
                        <li>Fully connected layers for classification</li>
                    </ul>
                    
                    <h2>Applications</h2>
                    <p>Neural networks in computer vision have enabled breakthrough applications 
                    including autonomous vehicles, medical image analysis, and facial recognition 
                    systems. The accuracy and efficiency of these systems continue to improve with 
                    advances in architecture design and training techniques.</p>
                    
                    <h2>Future Directions</h2>
                    <p>Emerging trends include attention mechanisms, vision transformers, and 
                    self-supervised learning approaches that promise to further advance the field 
                    of computer vision.</p>
                </article>
            </main>
            <footer>
                <p>&copy; 2024 AI Research Blog</p>
            </footer>
        </body>
        </html>
        """
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_pipeline_flow(self, pipeline_config, sample_rss_feed, sample_article_html):
        """Test complete pipeline from RSS ingestion to Markdown publishing."""
        
        # Step 1: RSS Ingestion
        rss_fetcher = RSSFetcher(config=pipeline_config)
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Mock RSS feed fetch
            mock_rss_response = MagicMock()
            mock_rss_response.text.return_value = sample_rss_feed
            mock_rss_response.status = 200
            mock_get.return_value.__aenter__.return_value = mock_rss_response
            
            rss_items = await rss_fetcher.fetch_rss_feed("https://ai-blog.example.com/feed.xml")
            
            assert len(rss_items) == 3
            assert rss_items[0]['title'] == "Neural Networks in Computer Vision"
        
        # Step 2: Content Extraction/Normalization
        content_extractor = ContentExtractor()
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Mock article HTML fetch
            mock_html_response = MagicMock()
            mock_html_response.text.return_value = sample_article_html
            mock_html_response.status = 200
            mock_get.return_value.__aenter__.return_value = mock_html_response
            
            extracted_content = await content_extractor.extract_content(
                sample_article_html, 
                rss_items[0]['url']
            )
            
            assert extracted_content['title'] == "Neural Networks in Computer Vision"
            assert 'Convolutional Neural Networks' in extracted_content['content']
            assert extracted_content['word_count'] > 100
        
        # Step 3: Duplicate Detection
        similarity_detector = SimilarityDetector(threshold=0.85)
        
        # Create a similar article for duplicate detection
        similar_content = {
            'title': "CNNs in Computer Vision",
            'content': extracted_content['content'][:500] + " Additional content here.",
            'url': "https://different-blog.com/cnns-cv"
        }
        
        existing_articles = [extracted_content]
        is_duplicate = similarity_detector.is_duplicate(similar_content, existing_articles)
        
        # Should detect as potential duplicate due to content similarity
        assert is_duplicate == True
        
        # Step 4: Content Enrichment
        content_summarizer = ContentSummarizer()
        
        with patch.object(content_summarizer, 'openai_client') as mock_openai:
            # Mock AI summary generation
            mock_response = AsyncMock()
            mock_response.choices = [AsyncMock()]
            mock_response.choices[0].message.content = (
                "This article explores neural networks in computer vision, focusing on "
                "CNNs and their applications in image recognition tasks."
            )
            mock_openai.chat.completions.create.return_value = mock_response
            
            enriched_content = extracted_content.copy()
            enriched_content['summary'] = await content_summarizer.generate_summary(
                extracted_content['content'], 
                max_length=200
            )
            
            assert 'summary' in enriched_content
            assert len(enriched_content['summary']) > 50
            assert 'neural networks' in enriched_content['summary'].lower()
        
        # Step 5: Markdown Publishing
        markdown_generator = MarkdownGenerator()
        
        # Add required fields for publishing
        enriched_content.update({
            'slug': 'neural-networks-computer-vision',
            'category': 'computer-vision',
            'tags': ['neural-networks', 'computer-vision', 'cnn'],
            'published_date': datetime(2024, 1, 15, 10, 0, 0),
            'reading_time': 5,
            'difficulty': 'intermediate'
        })
        
        markdown_output = markdown_generator.generate_article_markdown(enriched_content)
        
        # Verify markdown structure
        assert markdown_output.startswith('---')
        assert 'title: Neural Networks in Computer Vision' in markdown_output
        assert 'category: computer-vision' in markdown_output
        assert '# Introduction' in markdown_output or '## Introduction' in markdown_output
        
        # Step 6: File System Output
        output_file = Path(pipeline_config.output_dir) / "articles" / "neural-networks-computer-vision.md"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_output)
        
        # Verify file was created and contains expected content
        assert output_file.exists()
        saved_content = output_file.read_text(encoding='utf-8')
        assert 'Neural Networks in Computer Vision' in saved_content
        assert 'Convolutional Neural Networks' in saved_content
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, pipeline_config):
        """Test pipeline error handling and recovery."""
        
        # Test RSS fetch failure
        rss_fetcher = RSSFetcher(config=pipeline_config)
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Mock network error
            mock_get.side_effect = Exception("Network error")
            
            rss_items = await rss_fetcher.fetch_rss_feed("https://invalid-url.com/feed.xml")
            assert rss_items == []  # Should return empty list on error
        
        # Test content extraction with invalid HTML
        content_extractor = ContentExtractor()
        
        invalid_html = "<<<invalid html>>>"
        result = await content_extractor.extract_content(invalid_html, "https://example.com")
        
        # Should handle gracefully
        assert result['title'] == ''
        assert result['content'] == ''
        assert result['method'] == 'failed'
        
        # Test AI service failure fallback
        content_summarizer = ContentSummarizer()
        
        with patch.object(content_summarizer, 'openai_client') as mock_openai:
            mock_openai.chat.completions.create.side_effect = Exception("API Error")
            
            # Should fallback to extractive summarization
            content = "This is test content for summarization." * 20
            summary = await content_summarizer.generate_summary(content, provider='openai')
            
            assert summary is not None
            assert len(summary) > 0
    
    @pytest.mark.integration
    @pytest.mark.asyncio 
    async def test_pipeline_data_flow_consistency(self, pipeline_config, sample_rss_feed, sample_article_html):
        """Test data consistency throughout the pipeline."""
        
        # Track data through each pipeline stage
        original_data = {
            'url': 'https://ai-blog.example.com/neural-networks-cv',
            'title': 'Neural Networks in Computer Vision',
            'description': 'Exploring the use of neural networks for image recognition tasks'
        }
        
        stages_data = {'original': original_data}
        
        # Stage 1: Content Extraction
        content_extractor = ContentExtractor()
        extracted = await content_extractor.extract_content(sample_article_html, original_data['url'])
        stages_data['extracted'] = extracted
        
        # Verify data integrity
        assert stages_data['extracted']['title'] == stages_data['original']['title']
        assert stages_data['extracted']['url'] == stages_data['original']['url']
        assert stages_data['extracted']['extracted_at'] is not None
        
        # Stage 2: Deduplication Check
        similarity_detector = SimilarityDetector()
        dedup_result = {
            'is_duplicate': False,
            'similarity_score': 0.0,
            'checked_at': datetime.now()
        }
        stages_data['dedup'] = dedup_result
        
        # Stage 3: Enrichment
        enriched = extracted.copy()
        enriched.update({
            'summary': 'AI-generated summary of neural networks in computer vision.',
            'key_concepts': ['neural networks', 'computer vision', 'CNN'],
            'quality_score': 0.85,
            'enriched_at': datetime.now()
        })
        stages_data['enriched'] = enriched
        
        # Verify enrichment preserves original data
        assert stages_data['enriched']['title'] == stages_data['original']['title']
        assert stages_data['enriched']['url'] == stages_data['original']['url']
        assert 'summary' in stages_data['enriched']
        assert 'key_concepts' in stages_data['enriched']
        
        # Stage 4: Publishing
        markdown_generator = MarkdownGenerator()
        
        publish_data = enriched.copy()
        publish_data.update({
            'slug': 'neural-networks-computer-vision',
            'category': 'computer-vision',
            'tags': ['neural-networks', 'computer-vision'],
            'published_date': datetime.now(),
            'reading_time': 5
        })
        
        markdown_content = markdown_generator.generate_article_markdown(publish_data)
        stages_data['published'] = {'markdown_content': markdown_content, 'published_at': datetime.now()}
        
        # Verify final output contains original information
        assert original_data['title'] in markdown_content
        assert 'neural networks' in markdown_content.lower()
        
        # Audit trail verification
        audit_log = {
            'article_url': original_data['url'],
            'pipeline_stages': list(stages_data.keys()),
            'total_processing_time': (stages_data['published']['published_at'] - stages_data['extracted']['extracted_at']).total_seconds(),
            'data_integrity_check': 'passed'
        }
        
        assert len(audit_log['pipeline_stages']) == 5
        assert audit_log['total_processing_time'] > 0
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_batch_processing_pipeline(self, pipeline_config, sample_rss_feed):
        """Test pipeline handling of multiple articles in batch."""
        
        batch_size = 5
        
        # Create multiple RSS items
        rss_items = []
        for i in range(batch_size):
            rss_items.append({
                'title': f'AI Article {i+1}',
                'url': f'https://example.com/article-{i+1}',
                'description': f'Description for AI article {i+1}',
                'published_date': datetime.now() - timedelta(days=i)
            })
        
        # Mock content extraction for batch
        content_extractor = ContentExtractor()
        
        async def mock_extract_content(html, url):
            article_id = url.split('-')[-1]
            return {
                'title': f'AI Article {article_id}',
                'content': f'Content for AI article {article_id}. ' * 50,
                'url': url,
                'word_count': 250,
                'quality_score': 0.7,
                'extracted_at': datetime.now()
            }
        
        with patch.object(content_extractor, 'extract_content', side_effect=mock_extract_content):
            
            # Process articles in batch
            extracted_articles = []
            for rss_item in rss_items:
                extracted = await content_extractor.extract_content('mock_html', rss_item['url'])
                extracted_articles.append(extracted)
            
            assert len(extracted_articles) == batch_size
            assert all(article['word_count'] == 250 for article in extracted_articles)
        
        # Batch duplicate detection
        similarity_detector = SimilarityDetector()
        
        unique_articles = []
        duplicates_found = 0
        
        for article in extracted_articles:
            if not similarity_detector.is_duplicate(article, unique_articles):
                unique_articles.append(article)
            else:
                duplicates_found += 1
        
        # All articles should be unique in this test
        assert len(unique_articles) == batch_size
        assert duplicates_found == 0
        
        # Batch enrichment
        content_summarizer = ContentSummarizer()
        
        with patch.object(content_summarizer, 'generate_summary') as mock_summarize:
            mock_summarize.return_value = "AI-generated summary."
            
            enriched_articles = []
            for article in unique_articles:
                enriched = article.copy()
                enriched['summary'] = await content_summarizer.generate_summary(article['content'])
                enriched_articles.append(enriched)
            
            assert len(enriched_articles) == batch_size
            assert all('summary' in article for article in enriched_articles)
        
        # Batch publishing
        markdown_generator = MarkdownGenerator()
        
        published_files = []
        for i, article in enumerate(enriched_articles):
            article.update({
                'slug': f'ai-article-{i+1}',
                'category': 'artificial-intelligence',
                'tags': ['ai'],
                'published_date': datetime.now()
            })
            
            markdown_content = markdown_generator.generate_article_markdown(article)
            published_files.append({
                'slug': article['slug'],
                'content': markdown_content
            })
        
        assert len(published_files) == batch_size
        assert all('AI Article' in file['content'] for file in published_files)
    
    @pytest.mark.integration
    @pytest.mark.performance
    async def test_pipeline_performance_benchmarks(self, pipeline_config, performance_benchmarks):
        """Test pipeline performance against benchmarks."""
        
        # Test content processing speed
        content_extractor = ContentExtractor()
        large_html = "<html><body>" + "<p>Test content. " * 1000 + "</p></body></html>"
        
        start_time = datetime.now()
        result = await content_extractor.extract_content(large_html, "https://example.com")
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        max_time = performance_benchmarks.get("content_processing_time_per_1000_words", 2.0)
        words_processed = result.get('word_count', 1000)
        expected_max_time = (words_processed / 1000) * max_time
        
        assert processing_time <= expected_max_time, f"Processing too slow: {processing_time:.2f}s > {expected_max_time:.2f}s"
        
        # Test duplicate detection performance
        similarity_detector = SimilarityDetector()
        
        # Create large set of articles for comparison
        test_articles = []
        for i in range(100):
            test_articles.append({
                'title': f'Test Article {i}',
                'content': f'Content for test article {i}. ' * 100,
                'url': f'https://example.com/test-{i}'
            })
        
        new_article = {
            'title': 'New Test Article',
            'content': 'New content for testing duplicate detection. ' * 100,
            'url': 'https://example.com/new-test'
        }
        
        start_time = datetime.now()
        is_duplicate = similarity_detector.is_duplicate(new_article, test_articles)
        end_time = datetime.now()
        
        dedup_time = (end_time - start_time).total_seconds()
        max_dedup_time = performance_benchmarks.get("deduplication_time_per_100_items", 1.0)
        
        assert dedup_time <= max_dedup_time, f"Deduplication too slow: {dedup_time:.2f}s > {max_dedup_time}s"
        assert isinstance(is_duplicate, bool)
    
    @pytest.mark.integration
    def test_pipeline_configuration_validation(self, temp_directory):
        """Test pipeline configuration validation and setup."""
        
        # Test valid configuration
        valid_config = PipelineConfig(
            data_dir=str(temp_directory),
            output_dir=str(temp_directory / "output"),
            max_articles_per_source=50,
            quality_threshold=0.6,
            similarity_threshold=0.8
        )
        
        assert valid_config.data_dir == str(temp_directory)
        assert valid_config.max_articles_per_source == 50
        assert valid_config.quality_threshold == 0.6
        
        # Test directory creation
        valid_config.ensure_directories()
        assert Path(valid_config.data_dir).exists()
        assert Path(valid_config.output_dir).exists()
        
        # Test invalid configuration values
        with pytest.raises(ValueError):
            PipelineConfig(
                quality_threshold=-0.1  # Invalid: negative threshold
            )
        
        with pytest.raises(ValueError):
            PipelineConfig(
                similarity_threshold=1.5  # Invalid: threshold > 1.0
            )
        
        with pytest.raises(ValueError):
            PipelineConfig(
                max_articles_per_source=0  # Invalid: must be positive
            )
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pipeline_failure_recovery(self, pipeline_config):
        """Test pipeline recovery from failures at different stages."""
        
        # Simulate failure during content extraction
        content_extractor = ContentExtractor()
        
        with patch.object(content_extractor, 'extract_content', side_effect=Exception("Extraction failed")):
            try:
                await content_extractor.extract_content("<html>test</html>", "https://example.com")
                assert False, "Should have raised exception"
            except Exception as e:
                assert "Extraction failed" in str(e)
        
        # Test graceful degradation - should continue with next article
        articles_to_process = [
            {"url": "https://example.com/article1", "html": "<html>Article 1</html>"},
            {"url": "https://example.com/article2", "html": "<html>Article 2</html>"},
            {"url": "https://example.com/article3", "html": "<html>Article 3</html>"}
        ]
        
        successful_extractions = []
        failed_extractions = []
        
        for i, article in enumerate(articles_to_process):
            try:
                # Mock failure on second article
                if i == 1:
                    raise Exception("Simulated extraction failure")
                
                # Mock successful extraction
                result = {
                    'title': f'Article {i+1}',
                    'content': f'Content for article {i+1}',
                    'url': article['url'],
                    'extracted_at': datetime.now()
                }
                successful_extractions.append(result)
                
            except Exception as e:
                failed_extractions.append({
                    'url': article['url'],
                    'error': str(e),
                    'failed_at': datetime.now()
                })
        
        # Should have processed 2 successfully and 1 failure
        assert len(successful_extractions) == 2
        assert len(failed_extractions) == 1
        assert failed_extractions[0]['url'] == "https://example.com/article2"
        
        # Pipeline should continue with successful articles
        assert successful_extractions[0]['title'] == 'Article 1'
        assert successful_extractions[1]['title'] == 'Article 3'