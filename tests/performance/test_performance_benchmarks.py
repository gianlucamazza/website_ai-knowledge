"""
Performance benchmark tests for the AI Knowledge website pipeline.

Tests system performance under various load conditions and ensures acceptable response times.
"""

import pytest
import time
import asyncio
import psutil
import gc
from datetime import datetime, timedelta
from typing import Dict, List
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock, patch, AsyncMock

from pipelines.ingest.rss_fetcher import RSSFetcher
from pipelines.normalize.content_extractor import ContentExtractor
from pipelines.dedup.similarity_detector import SimilarityDetector
from pipelines.enrich.summarizer import ContentSummarizer
from pipelines.publish.markdown_generator import MarkdownGenerator


class TestPerformanceBenchmarks:
    """Performance benchmark tests for pipeline components."""
    
    @pytest.fixture
    def performance_config(self):
        """Performance testing configuration."""
        return {
            'max_processing_time_per_article_seconds': 10.0,
            'max_memory_usage_mb': 500,
            'max_concurrent_articles': 50,
            'target_throughput_articles_per_minute': 30,
            'content_extraction_time_per_kb_ms': 100,
            'duplicate_detection_time_per_comparison_ms': 50,
            'ai_summarization_timeout_seconds': 30
        }
    
    @pytest.fixture
    def large_content_samples(self):
        """Generate large content samples for performance testing."""
        samples = []
        
        # Small articles (1-2KB)
        for i in range(10):
            content = f"Article {i+1}. " + "This is sample content about AI and machine learning. " * 50
            samples.append({
                'size': 'small',
                'title': f'Small Article {i+1}',
                'content': content,
                'expected_words': len(content.split()),
                'url': f'https://example.com/small-{i+1}'
            })
        
        # Medium articles (5-10KB)
        for i in range(10):
            content = f"Medium Article {i+1}. " + "This is detailed content about artificial intelligence, machine learning, and deep learning technologies. " * 200
            samples.append({
                'size': 'medium',
                'title': f'Medium Article {i+1}',
                'content': content,
                'expected_words': len(content.split()),
                'url': f'https://example.com/medium-{i+1}'
            })
        
        # Large articles (20-50KB)
        for i in range(5):
            content = f"Large Article {i+1}. " + "This is comprehensive content about advanced artificial intelligence concepts, including neural networks, deep learning architectures, computer vision, natural language processing, and their real-world applications across various industries. " * 500
            samples.append({
                'size': 'large',
                'title': f'Large Article {i+1}',
                'content': content,
                'expected_words': len(content.split()),
                'url': f'https://example.com/large-{i+1}'
            })
        
        return samples
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_content_extraction_performance(self, large_content_samples, performance_config):
        """Test content extraction performance across different content sizes."""
        content_extractor = ContentExtractor()
        
        performance_results = {
            'small': [],
            'medium': [], 
            'large': []
        }
        
        for sample in large_content_samples:
            # Create mock HTML
            html_content = f"""
            <html>
            <head><title>{sample['title']}</title></head>
            <body>
                <article>
                    <h1>{sample['title']}</h1>
                    <p>{sample['content']}</p>
                </article>
            </body>
            </html>
            """
            
            # Measure extraction performance
            start_time = time.perf_counter()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            result = await content_extractor.extract_content(html_content, sample['url'])
            
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            processing_time = end_time - start_time
            memory_used = end_memory - start_memory
            
            performance_results[sample['size']].append({
                'processing_time_ms': processing_time * 1000,
                'memory_used_mb': memory_used,
                'words_extracted': result.get('word_count', 0),
                'content_size_kb': len(html_content) / 1024,
                'throughput_words_per_second': result.get('word_count', 0) / processing_time if processing_time > 0 else 0
            })
            
            # Verify extraction success
            assert result['title'] == sample['title']
            assert result['word_count'] > 0
            
            # Performance assertions
            content_size_kb = len(html_content) / 1024
            max_time_ms = performance_config['content_extraction_time_per_kb_ms'] * content_size_kb
            assert processing_time * 1000 <= max_time_ms, f"Extraction too slow: {processing_time*1000:.1f}ms > {max_time_ms:.1f}ms for {content_size_kb:.1f}KB"
        
        # Analyze performance by content size
        for size_category, results in performance_results.items():
            if results:
                avg_time = statistics.mean(r['processing_time_ms'] for r in results)
                avg_memory = statistics.mean(r['memory_used_mb'] for r in results)
                avg_throughput = statistics.mean(r['throughput_words_per_second'] for r in results)
                
                print(f"\n{size_category.title()} Content Performance:")
                print(f"  Average processing time: {avg_time:.1f}ms")
                print(f"  Average memory usage: {avg_memory:.1f}MB")
                print(f"  Average throughput: {avg_throughput:.0f} words/second")
                
                # Performance thresholds by content size
                if size_category == 'small':
                    assert avg_time <= 500, f"Small content extraction too slow: {avg_time:.1f}ms"
                elif size_category == 'medium':
                    assert avg_time <= 2000, f"Medium content extraction too slow: {avg_time:.1f}ms"
                elif size_category == 'large':
                    assert avg_time <= 5000, f"Large content extraction too slow: {avg_time:.1f}ms"
                
                assert avg_memory <= performance_config['max_memory_usage_mb'], f"Memory usage too high: {avg_memory:.1f}MB"
    
    @pytest.mark.performance
    def test_duplicate_detection_performance(self, large_content_samples, performance_config):
        """Test duplicate detection performance with large article sets."""
        similarity_detector = SimilarityDetector(threshold=0.85)
        
        # Create test article database
        test_articles = []
        for i, sample in enumerate(large_content_samples):
            test_articles.append({
                'id': f'article_{i}',
                'title': sample['title'],
                'content': sample['content'],
                'url': sample['url']
            })
        
        # Test duplicate detection against various database sizes
        database_sizes = [10, 25, 50, 100, len(test_articles)]
        performance_results = []
        
        for db_size in database_sizes:
            if db_size > len(test_articles):
                continue
                
            test_db = test_articles[:db_size]
            new_article = {
                'title': 'New Test Article',
                'content': 'This is new content that should not match existing articles. ' * 100,
                'url': 'https://example.com/new-test'
            }
            
            # Measure duplicate detection performance
            start_time = time.perf_counter()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            is_duplicate = similarity_detector.is_duplicate(new_article, test_db)
            
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            processing_time = end_time - start_time
            memory_used = end_memory - start_memory
            
            performance_results.append({
                'database_size': db_size,
                'processing_time_ms': processing_time * 1000,
                'memory_used_mb': memory_used,
                'comparisons_per_second': db_size / processing_time if processing_time > 0 else 0,
                'is_duplicate': is_duplicate
            })
            
            # Performance assertion
            max_time_ms = performance_config['duplicate_detection_time_per_comparison_ms'] * db_size
            assert processing_time * 1000 <= max_time_ms, f"Duplicate detection too slow: {processing_time*1000:.1f}ms > {max_time_ms:.1f}ms for {db_size} articles"
            
            assert is_duplicate == False  # Should not be duplicate in this test
        
        # Analyze performance scaling
        print(f"\nDuplicate Detection Performance Scaling:")
        for result in performance_results:
            print(f"  {result['database_size']:3d} articles: {result['processing_time_ms']:6.1f}ms ({result['comparisons_per_second']:6.0f} comp/sec)")
        
        # Check that performance scales reasonably (should be sub-quadratic)
        if len(performance_results) >= 3:
            small_db_time = performance_results[0]['processing_time_ms']
            large_db_time = performance_results[-1]['processing_time_ms']
            size_ratio = performance_results[-1]['database_size'] / performance_results[0]['database_size']
            time_ratio = large_db_time / small_db_time
            
            # Time should scale better than O(n²)
            assert time_ratio <= size_ratio ** 1.5, f"Duplicate detection scaling too poor: {time_ratio:.1f}x time for {size_ratio:.1f}x data"
    
    @pytest.mark.performance
    @pytest.mark.asyncio 
    async def test_concurrent_processing_performance(self, performance_config):
        """Test concurrent article processing performance."""
        
        # Create mock articles for concurrent processing
        test_articles = []
        for i in range(performance_config['max_concurrent_articles']):
            test_articles.append({
                'id': f'concurrent_article_{i}',
                'title': f'Concurrent Test Article {i}',
                'content': f'Test content for concurrent processing article {i}. ' * 100,
                'url': f'https://example.com/concurrent-{i}',
                'html': f'<html><body><h1>Article {i}</h1><p>{"Content " * 100}</p></body></html>'
            })
        
        content_extractor = ContentExtractor()
        
        # Test sequential processing baseline
        start_time = time.perf_counter()
        
        sequential_results = []
        for article in test_articles[:10]:  # Test with 10 articles for baseline
            result = await content_extractor.extract_content(article['html'], article['url'])
            sequential_results.append(result)
        
        sequential_time = time.perf_counter() - start_time
        sequential_throughput = len(sequential_results) / sequential_time
        
        print(f"\nSequential Processing Baseline:")
        print(f"  Processed {len(sequential_results)} articles in {sequential_time:.2f}s")
        print(f"  Throughput: {sequential_throughput:.1f} articles/second")
        
        # Test concurrent processing
        start_time = time.perf_counter()
        
        async def process_article(article):
            return await content_extractor.extract_content(article['html'], article['url'])
        
        # Process articles concurrently in batches
        batch_size = 10
        concurrent_results = []
        
        for i in range(0, min(len(test_articles), 30), batch_size):  # Test with 30 articles max
            batch = test_articles[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [process_article(article) for article in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out any exceptions
            successful_results = [r for r in batch_results if not isinstance(r, Exception)]
            concurrent_results.extend(successful_results)
        
        concurrent_time = time.perf_counter() - start_time
        concurrent_throughput = len(concurrent_results) / concurrent_time
        
        print(f"\nConcurrent Processing Performance:")
        print(f"  Processed {len(concurrent_results)} articles in {concurrent_time:.2f}s")
        print(f"  Throughput: {concurrent_throughput:.1f} articles/second")
        print(f"  Speedup: {concurrent_throughput/sequential_throughput:.1f}x")
        
        # Performance assertions
        assert len(concurrent_results) > 0, "No articles processed successfully"
        assert concurrent_throughput >= performance_config['target_throughput_articles_per_minute'] / 60, f"Throughput too low: {concurrent_throughput:.1f} < {performance_config['target_throughput_articles_per_minute']/60:.1f} articles/second"
        
        # Concurrent should be faster than sequential for the same workload
        if len(concurrent_results) >= len(sequential_results):
            normalized_concurrent_time = concurrent_time * (len(sequential_results) / len(concurrent_results))
            assert normalized_concurrent_time <= sequential_time * 1.2, "Concurrent processing not providing expected speedup"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_ai_summarization_performance(self, large_content_samples, performance_config):
        """Test AI summarization performance and timeout handling."""
        content_summarizer = ContentSummarizer()
        
        performance_results = []
        
        # Mock OpenAI client with variable response times
        with patch.object(content_summarizer, 'openai_client') as mock_openai:
            
            def create_mock_response(delay_ms=100):
                async def delayed_response(*args, **kwargs):
                    await asyncio.sleep(delay_ms / 1000)  # Simulate API delay
                    mock_response = AsyncMock()
                    mock_response.choices = [AsyncMock()]
                    mock_response.choices[0].message.content = "AI-generated summary for performance testing."
                    return mock_response
                return delayed_response
            
            # Test with different simulated API response times
            api_delays = [50, 100, 500, 1000, 2000]  # milliseconds
            
            for delay_ms in api_delays:
                mock_openai.chat.completions.create = create_mock_response(delay_ms)
                
                # Test with medium-sized content
                test_content = large_content_samples[15]['content']  # Medium article
                
                start_time = time.perf_counter()
                
                try:
                    summary = await asyncio.wait_for(
                        content_summarizer.generate_summary(test_content, max_length=200),
                        timeout=performance_config['ai_summarization_timeout_seconds']
                    )
                    
                    end_time = time.perf_counter()
                    processing_time = end_time - start_time
                    
                    performance_results.append({
                        'api_delay_ms': delay_ms,
                        'processing_time_ms': processing_time * 1000,
                        'success': True,
                        'summary_length': len(summary) if summary else 0
                    })
                    
                    assert summary is not None
                    assert len(summary) > 0
                    
                except asyncio.TimeoutError:
                    performance_results.append({
                        'api_delay_ms': delay_ms,
                        'processing_time_ms': performance_config['ai_summarization_timeout_seconds'] * 1000,
                        'success': False,
                        'summary_length': 0
                    })
        
        # Analyze performance results
        print(f"\nAI Summarization Performance:")
        for result in performance_results:
            status = "SUCCESS" if result['success'] else "TIMEOUT"
            print(f"  API delay {result['api_delay_ms']:4d}ms: {result['processing_time_ms']:6.1f}ms ({status})")
        
        # Performance assertions
        successful_results = [r for r in performance_results if r['success']]
        assert len(successful_results) > 0, "No successful AI summarizations"
        
        # Fast API calls should succeed
        fast_results = [r for r in performance_results if r['api_delay_ms'] <= 500]
        assert all(r['success'] for r in fast_results), "Fast API calls should not timeout"
        
        # Very slow API calls should timeout appropriately
        slow_results = [r for r in performance_results if r['api_delay_ms'] >= 2000]
        if slow_results:
            assert any(not r['success'] for r in slow_results), "Timeout mechanism should work for slow API calls"
    
    @pytest.mark.performance
    def test_memory_usage_under_load(self, large_content_samples, performance_config):
        """Test memory usage under high load conditions."""
        
        # Monitor memory usage during processing
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_snapshots = [initial_memory]
        
        content_extractor = ContentExtractor()
        similarity_detector = SimilarityDetector()
        markdown_generator = MarkdownGenerator()
        
        processed_articles = []
        
        # Process articles and monitor memory
        for i, sample in enumerate(large_content_samples):
            
            # Content extraction
            html_content = f"<html><body><h1>{sample['title']}</h1><p>{sample['content']}</p></body></html>"
            
            # Simulate extraction result
            extracted_content = {
                'title': sample['title'],
                'content': sample['content'],
                'url': sample['url'],
                'word_count': sample['expected_words'],
                'extracted_at': datetime.now()
            }
            
            processed_articles.append(extracted_content)
            
            # Duplicate detection
            is_duplicate = similarity_detector.is_duplicate(extracted_content, processed_articles[:-1])
            
            # Markdown generation
            if not is_duplicate:
                markdown_data = extracted_content.copy()
                markdown_data.update({
                    'slug': f'article-{i}',
                    'category': 'test',
                    'tags': ['performance-test'],
                    'published_date': datetime.now()
                })
                
                markdown_content = markdown_generator.generate_article_markdown(markdown_data)
            
            # Take memory snapshot
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_snapshots.append(current_memory)
            
            # Force garbage collection periodically
            if i % 10 == 9:
                gc.collect()
                gc_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_snapshots.append(gc_memory)
        
        # Analyze memory usage
        max_memory = max(memory_snapshots)
        final_memory = memory_snapshots[-1]
        memory_growth = final_memory - initial_memory
        peak_usage = max_memory - initial_memory
        
        print(f"\nMemory Usage Analysis:")
        print(f"  Initial memory: {initial_memory:.1f}MB")
        print(f"  Final memory: {final_memory:.1f}MB")
        print(f"  Memory growth: {memory_growth:.1f}MB")
        print(f"  Peak additional usage: {peak_usage:.1f}MB")
        print(f"  Articles processed: {len(processed_articles)}")
        print(f"  Memory per article: {memory_growth/len(processed_articles):.2f}MB")
        
        # Memory usage assertions
        assert peak_usage <= performance_config['max_memory_usage_mb'], f"Peak memory usage too high: {peak_usage:.1f}MB > {performance_config['max_memory_usage_mb']}MB"
        
        # Memory should not grow unboundedly
        assert memory_growth <= peak_usage * 1.2, "Memory not being released properly (potential leak)"
        
        # Per-article memory usage should be reasonable
        per_article_memory = memory_growth / len(processed_articles) if processed_articles else 0
        assert per_article_memory <= 5.0, f"Memory usage per article too high: {per_article_memory:.2f}MB"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_pipeline_throughput_benchmark(self, performance_config):
        """Test overall pipeline throughput benchmark."""
        
        # Create test workload
        num_articles = 50
        test_workload = []
        
        for i in range(num_articles):
            test_workload.append({
                'id': f'throughput_test_{i}',
                'title': f'Throughput Test Article {i}',
                'content': f'Content for throughput testing article {i}. ' * 200,  # ~400 words
                'url': f'https://example.com/throughput-{i}',
                'html': f'<html><body><h1>Article {i}</h1><p>{"Test content " * 200}</p></body></html>'
            })
        
        # Initialize pipeline components
        content_extractor = ContentExtractor()
        similarity_detector = SimilarityDetector(threshold=0.85)
        content_summarizer = ContentSummarizer()
        markdown_generator = MarkdownGenerator()
        
        # Mock AI summarization to avoid external API calls
        with patch.object(content_summarizer, 'generate_summary') as mock_summarize:
            mock_summarize.return_value = "Performance test summary."
            
            # Run throughput benchmark
            start_time = time.perf_counter()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            processed_count = 0
            duplicate_count = 0
            published_count = 0
            
            processed_articles = []
            
            # Process articles through complete pipeline
            for i, article in enumerate(test_workload):
                try:
                    # Stage 1: Content Extraction
                    extracted = await content_extractor.extract_content(article['html'], article['url'])
                    processed_count += 1
                    
                    # Stage 2: Duplicate Detection
                    is_duplicate = similarity_detector.is_duplicate(extracted, processed_articles)
                    if is_duplicate:
                        duplicate_count += 1
                        continue
                    
                    # Stage 3: Content Enrichment
                    summary = await content_summarizer.generate_summary(extracted['content'], max_length=150)
                    enriched = extracted.copy()
                    enriched['summary'] = summary
                    
                    # Stage 4: Publishing
                    publish_data = enriched.copy()
                    publish_data.update({
                        'slug': f'throughput-test-{i}',
                        'category': 'performance-test',
                        'tags': ['performance', 'benchmark'],
                        'published_date': datetime.now()
                    })
                    
                    markdown_content = markdown_generator.generate_article_markdown(publish_data)
                    published_count += 1
                    
                    processed_articles.append(extracted)
                    
                except Exception as e:
                    print(f"Error processing article {i}: {e}")
                    continue
            
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Calculate throughput metrics
            total_time = end_time - start_time
            throughput_per_second = processed_count / total_time
            throughput_per_minute = throughput_per_second * 60
            memory_used = end_memory - start_memory
            
            print(f"\nPipeline Throughput Benchmark Results:")
            print(f"  Total articles processed: {processed_count}/{num_articles}")
            print(f"  Duplicates detected: {duplicate_count}")
            print(f"  Articles published: {published_count}")
            print(f"  Total processing time: {total_time:.2f}s")
            print(f"  Throughput: {throughput_per_second:.2f} articles/second")
            print(f"  Throughput: {throughput_per_minute:.1f} articles/minute")
            print(f"  Memory used: {memory_used:.1f}MB")
            print(f"  Average time per article: {(total_time/processed_count)*1000:.1f}ms")
            
            # Performance assertions
            assert processed_count > num_articles * 0.9, f"Too many processing failures: {processed_count}/{num_articles}"
            assert throughput_per_minute >= performance_config['target_throughput_articles_per_minute'], f"Throughput too low: {throughput_per_minute:.1f} < {performance_config['target_throughput_articles_per_minute']} articles/minute"
            assert memory_used <= performance_config['max_memory_usage_mb'], f"Memory usage too high: {memory_used:.1f}MB"
            
            # Average processing time per article should be reasonable
            avg_time_per_article = total_time / processed_count
            assert avg_time_per_article <= performance_config['max_processing_time_per_article_seconds'], f"Processing time per article too high: {avg_time_per_article:.2f}s"
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_long_running_stability(self):
        """Test system stability under extended operation."""
        
        duration_minutes = 5  # Run for 5 minutes in CI, longer locally
        if not pytest.config.getoption("--run-slow-tests", default=False):
            pytest.skip("Long-running stability test skipped. Use --run-slow-tests to run.")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        content_extractor = ContentExtractor()
        similarity_detector = SimilarityDetector()
        
        iteration_count = 0
        error_count = 0
        memory_samples = []
        processing_times = []
        
        while time.time() < end_time:
            try:
                iteration_start = time.time()
                
                # Create test content
                test_html = f"""
                <html>
                <body>
                    <h1>Stability Test Article {iteration_count}</h1>
                    <p>{'Test content for stability testing. ' * 100}</p>
                </body>
                </html>
                """
                
                # Process content
                result = asyncio.run(content_extractor.extract_content(
                    test_html, 
                    f'https://example.com/stability-{iteration_count}'
                ))
                
                # Duplicate detection
                similarity_detector.is_duplicate(result, [])
                
                iteration_end = time.time()
                processing_times.append(iteration_end - iteration_start)
                
                # Memory monitoring
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
                
                iteration_count += 1
                
                # Brief pause to prevent overwhelming the system
                time.sleep(0.1)
                
            except Exception as e:
                error_count += 1
                print(f"Error in iteration {iteration_count}: {e}")
                
                # If too many errors, abort
                if error_count > iteration_count * 0.1:  # More than 10% error rate
                    break
        
        # Analyze stability results
        total_runtime = time.time() - start_time
        
        if processing_times:
            avg_processing_time = statistics.mean(processing_times)
            processing_time_std = statistics.stdev(processing_times) if len(processing_times) > 1 else 0
        else:
            avg_processing_time = 0
            processing_time_std = 0
        
        if memory_samples:
            initial_memory = memory_samples[0]
            final_memory = memory_samples[-1]
            max_memory = max(memory_samples)
            memory_growth = final_memory - initial_memory
        else:
            initial_memory = final_memory = max_memory = memory_growth = 0
        
        print(f"\nLong-Running Stability Test Results:")
        print(f"  Runtime: {total_runtime/60:.1f} minutes")
        print(f"  Iterations completed: {iteration_count}")
        print(f"  Errors encountered: {error_count}")
        print(f"  Error rate: {(error_count/iteration_count)*100:.1f}%")
        print(f"  Avg processing time: {avg_processing_time*1000:.1f}ms ±{processing_time_std*1000:.1f}ms")
        print(f"  Memory: {initial_memory:.1f}MB → {final_memory:.1f}MB (peak: {max_memory:.1f}MB)")
        print(f"  Memory growth: {memory_growth:.1f}MB")
        
        # Stability assertions
        assert error_count <= iteration_count * 0.05, f"Error rate too high: {(error_count/iteration_count)*100:.1f}%"
        assert memory_growth <= 100, f"Memory growth too high: {memory_growth:.1f}MB (potential leak)"
        assert processing_time_std <= avg_processing_time, f"Processing time too variable: {processing_time_std*1000:.1f}ms std dev"
        
        # Should have processed a reasonable number of items
        expected_iterations = duration_minutes * 30  # Rough estimate
        assert iteration_count >= expected_iterations * 0.5, f"Too few iterations: {iteration_count} < {expected_iterations * 0.5}"