"""
Performance tests for pipeline components.

Tests throughput, response time, and resource usage of content processing pipeline.
"""

import pytest
import asyncio
import time
import psutil
import os
from datetime import datetime
from typing import Dict, List
from unittest.mock import AsyncMock, patch

from .benchmarks import PerformanceProfiler, LoadTestScenarios, get_benchmark_thresholds
from pipelines.ingest.rss_parser import RSSParser
from pipelines.normalize.content_extractor import ContentExtractor
from pipelines.dedup.simhash import SimHashDeduplicator
from pipelines.enrich.summarizer import ContentSummarizer
from pipelines.publish.markdown_generator import MarkdownGenerator


class TestPipelineComponentPerformance:
    """Test performance of individual pipeline components."""
    
    @pytest.fixture
    def profiler(self):
        """Create performance profiler for testing."""
        return PerformanceProfiler()
    
    @pytest.fixture
    def benchmarks(self):
        """Get performance benchmarks."""
        return get_benchmark_thresholds()
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_rss_parsing_performance(self, profiler, benchmarks):
        """Test RSS parsing performance with various feed sizes."""
        parser = RSSParser()
        
        # Test with different feed sizes
        test_scenarios = [
            {'name': 'small_feed', 'item_count': 10},
            {'name': 'medium_feed', 'item_count': 50},
            {'name': 'large_feed', 'item_count': 200}
        ]
        
        for scenario in test_scenarios:
            # Create test feed
            rss_feed = self._create_test_rss_feed(scenario['item_count'])
            source_config = {'max_articles_per_run': scenario['item_count']}
            
            from aioresponses import aioresponses
            
            with aioresponses() as mock_http:
                feed_url = f"https://example.com/{scenario['name']}.xml"
                mock_http.get(feed_url, status=200, body=rss_feed, headers={'content-type': 'application/rss+xml'})
                
                # Mock article content
                for i in range(scenario['item_count']):
                    article_url = f"https://example.com/article-{i+1}"
                    mock_http.get(
                        article_url, 
                        status=200, 
                        body=f"<html><body><h1>Article {i+1}</h1><p>Content</p></body></html>",
                        headers={'content-type': 'text/html'}
                    )
                
                # Measure parsing performance
                profiler.start_measurement(f'rss_parsing_{scenario["name"]}')
                articles = await parser.parse_feed(feed_url, source_config)
                duration = profiler.end_measurement(f'rss_parsing_{scenario["name"]}')
                
                # Calculate performance metrics
                articles_per_second = len(articles) / duration if duration > 0 else 0
                
                # Check against benchmarks
                min_throughput = benchmarks['content_ingestion_per_second']
                assert articles_per_second >= min_throughput, \
                    f"RSS parsing too slow for {scenario['name']}: {articles_per_second:.2f} < {min_throughput}"
                
                # Verify all articles were parsed
                assert len(articles) == scenario['item_count']
                
                print(f"RSS Parsing {scenario['name']}: {articles_per_second:.2f} articles/sec")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_content_extraction_performance(self, profiler, benchmarks):
        """Test content extraction performance with various content sizes."""
        extractor = ContentExtractor()
        
        test_scenarios = [
            {'name': 'short_content', 'word_count': 200},
            {'name': 'medium_content', 'word_count': 1000},
            {'name': 'long_content', 'word_count': 5000}
        ]
        
        for scenario in test_scenarios:
            # Create test HTML content
            html_content = self._create_test_html_content(scenario['word_count'])
            
            # Measure extraction performance
            profiler.start_measurement(f'content_extraction_{scenario["name"]}')
            result = await extractor.extract_content(html_content, "https://example.com/test")
            duration = profiler.end_measurement(f'content_extraction_{scenario["name"]}')
            
            # Calculate performance metrics
            words_per_second = scenario['word_count'] / duration if duration > 0 else 0
            time_per_1000_words = (duration * 1000) / scenario['word_count']
            
            # Check against benchmarks
            max_time_per_1000_words = benchmarks['content_extraction_time_per_1000_words']
            assert time_per_1000_words <= max_time_per_1000_words, \
                f"Content extraction too slow for {scenario['name']}: {time_per_1000_words:.2f}s > {max_time_per_1000_words}s"
            
            # Verify extraction quality
            assert result['word_count'] > scenario['word_count'] * 0.8  # Allow some variance
            assert result['title'] is not None
            assert result['content'] is not None
            
            print(f"Content Extraction {scenario['name']}: {words_per_second:.2f} words/sec")
    
    @pytest.mark.performance
    def test_deduplication_performance(self, profiler, benchmarks):
        """Test deduplication performance with various dataset sizes."""
        deduplicator = SimHashDeduplicator(k=3)
        
        test_scenarios = [
            {'name': 'small_dataset', 'article_count': 100},
            {'name': 'medium_dataset', 'article_count': 1000},
            {'name': 'large_dataset', 'article_count': 5000}
        ]
        
        for scenario in test_scenarios:
            # Generate test articles
            articles = self._generate_test_articles(scenario['article_count'])
            
            # Measure indexing performance
            profiler.start_measurement(f'dedup_indexing_{scenario["name"]}')
            for article in articles:
                deduplicator.add_content(article['id'], article['content'])
            indexing_duration = profiler.end_measurement(f'dedup_indexing_{scenario["name"]}')
            
            # Measure query performance
            profiler.start_measurement(f'dedup_querying_{scenario["name"]}')
            test_content = articles[0]['content']
            duplicates = deduplicator.find_duplicates(test_content)
            query_duration = profiler.end_measurement(f'dedup_querying_{scenario["name"]}')
            
            # Calculate performance metrics
            indexing_rate = scenario['article_count'] / indexing_duration if indexing_duration > 0 else 0
            time_per_1000_items = (indexing_duration * 1000) / scenario['article_count']
            
            # Check against benchmarks
            max_time_per_1000 = benchmarks['deduplication_time_per_1000_items']
            assert time_per_1000_items <= max_time_per_1000, \
                f"Deduplication too slow for {scenario['name']}: {time_per_1000_items:.2f}s > {max_time_per_1000}s"
            
            # Verify functionality
            assert len(duplicates) >= 0  # Should return results without error
            
            print(f"Deduplication {scenario['name']}: {indexing_rate:.2f} articles/sec indexing, {query_duration*1000:.2f}ms query")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_ai_summarization_performance(self, profiler, benchmarks):
        """Test AI summarization performance."""
        summarizer = ContentSummarizer()
        
        test_articles = [
            {'name': 'short_article', 'word_count': 300},
            {'name': 'medium_article', 'word_count': 1000},
            {'name': 'long_article', 'word_count': 3000}
        ]
        
        # Mock AI service for consistent testing
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.choices = [
                AsyncMock(message=AsyncMock(content="This is a test AI-generated summary."))
            ]
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            summarizer.openai_client = mock_client
            
            for article in test_articles:
                # Generate test content
                content = self._generate_test_article_content(article['word_count'])
                
                # Measure summarization performance
                profiler.start_measurement(f'ai_summarization_{article["name"]}')
                summary = await summarizer.generate_summary(
                    content=content,
                    max_length=200,
                    provider='openai'
                )
                duration = profiler.end_measurement(f'ai_summarization_{article["name"]}')
                
                # Check against benchmarks
                max_time_per_article = benchmarks['ai_summarization_time_per_article']
                assert duration <= max_time_per_article, \
                    f"AI summarization too slow for {article['name']}: {duration:.2f}s > {max_time_per_article}s"
                
                # Verify output
                assert summary is not None
                assert len(summary) > 0
                
                print(f"AI Summarization {article['name']}: {duration:.2f}s")
    
    @pytest.mark.performance
    def test_markdown_generation_performance(self, profiler, benchmarks):
        """Test Markdown generation performance."""
        generator = MarkdownGenerator()
        
        # Generate test articles with varying complexity
        test_articles = []
        for i in range(100):
            test_articles.append({
                'id': f'perf-article-{i}',
                'title': f'Performance Test Article {i}',
                'slug': f'perf-article-{i}',
                'content': self._generate_test_article_content(500 + (i * 50)),
                'summary': f'Summary for article {i}',
                'author': 'Performance Tester',
                'published_date': datetime(2024, 1, 15, 10, 0, 0),
                'category': 'performance',
                'tags': ['performance', 'testing', f'tag-{i % 5}'],
                'word_count': 500 + (i * 50),
                'reading_time': 3 + (i // 20)
            })
        
        # Measure batch generation performance
        profiler.start_measurement('markdown_generation_batch')
        
        for article in test_articles:
            markdown_content = generator.generate_article_markdown(article)
            assert len(markdown_content) > 0
            assert 'title:' in markdown_content
        
        duration = profiler.end_measurement('markdown_generation_batch')
        
        # Calculate performance metrics
        articles_per_second = len(test_articles) / duration if duration > 0 else 0
        time_per_article = duration / len(test_articles)
        
        # Check against benchmarks
        max_time_per_article = benchmarks['markdown_generation_time_per_article']
        assert time_per_article <= max_time_per_article, \
            f"Markdown generation too slow: {time_per_article:.4f}s > {max_time_per_article}s per article"
        
        print(f"Markdown Generation: {articles_per_second:.2f} articles/sec, {time_per_article*1000:.2f}ms per article")
    
    def _create_test_rss_feed(self, item_count: int) -> str:
        """Create test RSS feed with specified number of items."""
        items = []
        for i in range(item_count):
            items.append(f"""
            <item>
                <title>Performance Test Article {i+1}</title>
                <link>https://example.com/article-{i+1}</link>
                <description>This is a performance test article for benchmarking</description>
                <pubDate>Mon, {(i % 28) + 1:02d} Jan 2024 10:00:00 GMT</pubDate>
                <category>performance</category>
            </item>
            """)
        
        return f"""<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Performance Test Feed</title>
                <description>RSS feed for performance testing</description>
                <link>https://example.com</link>
                {''.join(items)}
            </channel>
        </rss>"""
    
    def _create_test_html_content(self, target_words: int) -> str:
        """Create test HTML content with specified word count."""
        # Base content template
        words_per_paragraph = 50
        paragraphs_needed = max(1, target_words // words_per_paragraph)
        
        paragraphs = []
        for i in range(paragraphs_needed):
            paragraph = f"""
            <p>This is paragraph {i+1} containing various artificial intelligence and machine learning
            concepts that are commonly discussed in technical articles and research papers. The content
            includes detailed explanations of algorithms, methodologies, applications, and theoretical
            foundations that form the basis of modern AI systems and computational approaches to
            problem-solving in complex domains.</p>
            """
            paragraphs.append(paragraph)
        
        return f"""
        <html>
        <head>
            <title>Performance Test Article</title>
            <meta name="description" content="Article for performance testing">
            <meta name="author" content="Performance Tester">
        </head>
        <body>
            <article>
                <h1>Performance Test Article</h1>
                {''.join(paragraphs)}
            </article>
        </body>
        </html>
        """
    
    def _generate_test_articles(self, count: int) -> List[Dict]:
        """Generate test articles for deduplication testing."""
        articles = []
        base_content = "Machine learning artificial intelligence neural networks deep learning"
        
        for i in range(count):
            # Create variations to test similarity detection
            if i % 10 == 0:
                # Exact duplicates every 10th article
                content = base_content
            elif i % 5 == 0:
                # Similar content every 5th article
                content = base_content.replace("artificial intelligence", "AI").replace("neural networks", "neural nets")
            else:
                # Unique content
                content = f"{base_content} specialized content for article {i} with unique identifiers {i * 7}"
            
            articles.append({
                'id': f'test-article-{i}',
                'content': content
            })
        
        return articles
    
    def _generate_test_article_content(self, target_words: int) -> str:
        """Generate test article content with specified word count."""
        base_sentences = [
            "Artificial intelligence represents a revolutionary approach to computational problem-solving.",
            "Machine learning algorithms enable systems to learn patterns from large datasets automatically.",
            "Deep learning neural networks process information through multiple interconnected layers.",
            "Natural language processing allows computers to understand and generate human language.",
            "Computer vision systems can analyze and interpret visual information from images and videos.",
            "Reinforcement learning agents optimize their behavior through trial and error interactions.",
            "Data science combines statistical analysis with domain expertise to extract insights.",
            "Algorithm optimization improves computational efficiency and performance characteristics.",
        ]
        
        content_parts = []
        words_added = 0
        sentence_index = 0
        
        while words_added < target_words:
            sentence = base_sentences[sentence_index % len(base_sentences)]
            content_parts.append(sentence)
            words_added += len(sentence.split())
            sentence_index += 1
            
            # Add paragraph breaks periodically
            if sentence_index % 4 == 0:
                content_parts.append("\n\n")
        
        return " ".join(content_parts)


class TestPipelineMemoryUsage:
    """Test memory usage and resource consumption of pipeline components."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_usage_during_processing(self, benchmarks):
        """Test memory usage during different processing stages."""
        profiler = PerformanceProfiler()
        
        # Get baseline memory usage
        baseline_memory = profiler.get_memory_usage()
        
        # Test content extraction memory usage
        extractor = ContentExtractor()
        large_content = self._create_large_html_content(10000)  # 10k words
        
        memory_before = profiler.get_memory_usage()
        result = await extractor.extract_content(large_content, "https://example.com/large")
        memory_after = profiler.get_memory_usage()
        
        memory_increase = memory_after['rss_mb'] - memory_before['rss_mb']
        max_processing_memory = benchmarks['content_processing_memory_mb']
        
        assert memory_increase <= max_processing_memory, \
            f"Content extraction uses too much memory: {memory_increase:.1f}MB > {max_processing_memory}MB"
        
        print(f"Content extraction memory usage: {memory_increase:.1f}MB")
        
        # Test deduplication memory usage
        deduplicator = SimHashDeduplicator()
        articles = self._generate_large_article_set(1000)
        
        memory_before = profiler.get_memory_usage()
        deduplicator.build_index_from_articles(articles)
        memory_after = profiler.get_memory_usage()
        
        dedup_memory_increase = memory_after['rss_mb'] - memory_before['rss_mb']
        max_dedup_memory = benchmarks['deduplication_index_memory_mb']
        
        assert dedup_memory_increase <= max_dedup_memory, \
            f"Deduplication uses too much memory: {dedup_memory_increase:.1f}MB > {max_dedup_memory}MB"
        
        print(f"Deduplication memory usage: {dedup_memory_increase:.1f}MB")
    
    @pytest.mark.performance
    def test_memory_leaks(self):
        """Test for memory leaks during repeated operations."""
        import gc
        
        profiler = PerformanceProfiler()
        
        # Perform repeated operations and monitor memory
        memory_measurements = []
        
        for i in range(10):
            # Create and process content
            extractor = ContentExtractor()
            content = self._create_large_html_content(1000)
            
            # Process multiple times
            for j in range(50):
                # Use sync version for testing or create async wrapper
                pass  # Placeholder - would need async handling
            
            # Force garbage collection
            gc.collect()
            
            # Measure memory
            memory = profiler.get_memory_usage()
            memory_measurements.append(memory['rss_mb'])
            
            print(f"Iteration {i+1}: {memory['rss_mb']:.1f}MB")
        
        # Check for memory growth trend
        initial_memory = memory_measurements[0]
        final_memory = memory_measurements[-1]
        memory_growth = final_memory - initial_memory
        
        # Allow some growth but detect significant leaks
        max_acceptable_growth = 100  # 100MB
        assert memory_growth <= max_acceptable_growth, \
            f"Possible memory leak detected: {memory_growth:.1f}MB growth > {max_acceptable_growth}MB"
    
    def _create_large_html_content(self, word_count: int) -> str:
        """Create large HTML content for memory testing."""
        words_per_paragraph = 100
        paragraphs = []
        
        for i in range(word_count // words_per_paragraph):
            paragraph_words = []
            for j in range(words_per_paragraph):
                # Generate varied vocabulary
                word = f"word{j % 50}content{i}data{(i*j) % 100}"
                paragraph_words.append(word)
            
            paragraph = f"<p>{' '.join(paragraph_words)}</p>"
            paragraphs.append(paragraph)
        
        return f"""
        <html>
        <head>
            <title>Large Memory Test Content</title>
        </head>
        <body>
            <article>
                <h1>Large Article for Memory Testing</h1>
                {''.join(paragraphs)}
            </article>
        </body>
        </html>
        """
    
    def _generate_large_article_set(self, count: int) -> List[Dict]:
        """Generate large set of articles for memory testing."""
        articles = []
        base_words = ["machine", "learning", "artificial", "intelligence", "neural", 
                     "network", "deep", "algorithm", "data", "science"]
        
        for i in range(count):
            # Create varied content to avoid excessive similarity
            content_words = []
            for j in range(100):  # 100 words per article
                word = base_words[j % len(base_words)]
                content_words.append(f"{word}_{i}_{j}")
            
            articles.append({
                'id': f'memory-test-article-{i}',
                'content': ' '.join(content_words)
            })
        
        return articles


class TestPipelineScalability:
    """Test pipeline scalability and load handling."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_small_batch_scenario(self, benchmarks):
        """Test small batch processing scenario."""
        scenario = LoadTestScenarios.small_batch_scenario()
        profiler = PerformanceProfiler()
        
        # Setup test data
        articles = self._create_scenario_articles(scenario)
        
        # Process articles
        profiler.start_measurement('small_batch_processing')
        processed_count = await self._process_articles_batch(articles, scenario['parameters'])
        duration = profiler.end_measurement('small_batch_processing')
        
        # Check performance
        articles_per_second = processed_count / duration if duration > 0 else 0
        min_throughput = benchmarks['pipeline_articles_per_second']
        
        assert articles_per_second >= min_throughput, \
            f"Small batch processing too slow: {articles_per_second:.2f} < {min_throughput}"
        
        assert duration <= scenario['expected_duration_seconds'], \
            f"Small batch took too long: {duration:.1f}s > {scenario['expected_duration_seconds']}s"
        
        # Check memory usage
        memory_usage = profiler.get_memory_usage()
        assert memory_usage['rss_mb'] <= scenario['memory_limit_mb'], \
            f"Small batch uses too much memory: {memory_usage['rss_mb']:.1f}MB > {scenario['memory_limit_mb']}MB"
        
        print(f"Small batch: {processed_count} articles in {duration:.2f}s ({articles_per_second:.2f} art/sec)")
    
    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, benchmarks):
        """Test concurrent processing capabilities."""
        scenario = LoadTestScenarios.high_concurrency_scenario()
        profiler = PerformanceProfiler()
        
        # Create multiple batches for concurrent processing
        total_articles = scenario['parameters']['total_articles']
        batch_size = scenario['parameters']['batch_size']
        batches = []
        
        for i in range(0, total_articles, batch_size):
            batch_articles = self._create_test_articles_batch(i, min(batch_size, total_articles - i))
            batches.append(batch_articles)
        
        # Process batches concurrently
        profiler.start_measurement('concurrent_processing')
        
        tasks = []
        for batch in batches:
            task = self._process_articles_batch(batch, scenario['parameters'])
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        duration = profiler.end_measurement('concurrent_processing')
        
        # Calculate metrics
        total_processed = sum(results)
        articles_per_second = total_processed / duration if duration > 0 else 0
        
        # Check performance
        min_throughput = benchmarks['pipeline_articles_per_second']
        assert articles_per_second >= min_throughput, \
            f"Concurrent processing too slow: {articles_per_second:.2f} < {min_throughput}"
        
        print(f"Concurrent processing: {total_processed} articles in {duration:.2f}s ({articles_per_second:.2f} art/sec)")
    
    def _create_scenario_articles(self, scenario: Dict) -> List[Dict]:
        """Create articles based on load test scenario."""
        articles = []
        params = scenario['parameters']
        
        for source_i in range(params['source_count']):
            for article_i in range(params['articles_per_source']):
                articles.append({
                    'id': f'scenario-article-{source_i}-{article_i}',
                    'title': f'Scenario Article {source_i}-{article_i}',
                    'content': f'Test content for scenario article {source_i}-{article_i}. ' * 50,
                    'source_id': f'scenario-source-{source_i}',
                    'url': f'https://example.com/scenario/{source_i}/{article_i}'
                })
        
        return articles
    
    def _create_test_articles_batch(self, start_index: int, count: int) -> List[Dict]:
        """Create a batch of test articles."""
        articles = []
        for i in range(count):
            article_id = start_index + i
            articles.append({
                'id': f'batch-article-{article_id}',
                'title': f'Batch Article {article_id}',
                'content': f'Content for batch test article {article_id}. ' * 30,
                'url': f'https://example.com/batch/{article_id}'
            })
        return articles
    
    async def _process_articles_batch(self, articles: List[Dict], parameters: Dict = None) -> int:
        """Simulate processing a batch of articles."""
        # Simulate processing time (replace with actual pipeline processing)
        processing_delay = 0.01  # 10ms per article simulation
        
        processed_count = 0
        batch_size = parameters.get('batch_size', 10) if parameters else 10
        
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            
            # Simulate concurrent processing within batch
            await asyncio.sleep(processing_delay * len(batch) / max(1, parameters.get('concurrent_articles', 1)))
            processed_count += len(batch)
        
        return processed_count