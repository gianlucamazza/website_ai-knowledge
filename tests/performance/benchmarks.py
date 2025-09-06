"""
Performance benchmarks and baseline metrics for the AI knowledge website.

Defines performance thresholds and benchmarks for various system components.
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class PerformanceBenchmarks:
    """Performance benchmark thresholds for system components."""
    
    # Pipeline Processing Benchmarks
    content_ingestion_per_second: float = 10.0  # Articles per second
    content_extraction_time_per_1000_words: float = 2.0  # Seconds
    deduplication_time_per_1000_items: float = 5.0  # Seconds
    ai_summarization_time_per_article: float = 3.0  # Seconds
    markdown_generation_time_per_article: float = 0.1  # Seconds
    
    # Database Performance Benchmarks
    database_insert_articles_per_second: float = 100.0
    database_query_response_time_ms: float = 100.0
    database_complex_query_time_ms: float = 500.0
    database_queries_per_second: float = 50.0
    database_connection_pool_max: int = 20
    
    # Memory Usage Benchmarks (in MB)
    content_processing_memory_mb: float = 500.0
    deduplication_index_memory_mb: float = 200.0
    ai_processing_memory_mb: float = 1000.0
    total_pipeline_memory_mb: float = 2000.0
    
    # API and External Service Benchmarks
    rss_feed_fetch_time_ms: float = 2000.0
    web_scraping_time_per_page_ms: float = 1000.0
    openai_api_response_time_ms: float = 5000.0
    anthropic_api_response_time_ms: float = 3000.0
    
    # Scalability Benchmarks
    max_concurrent_articles: int = 100
    max_concurrent_sources: int = 20
    pipeline_articles_per_second: float = 5.0
    cross_linking_time_per_1000_articles: float = 2.0
    
    # Quality and Coverage Benchmarks
    test_coverage_minimum_percent: float = 95.0
    code_quality_score_minimum: float = 8.0
    duplicate_detection_accuracy_percent: float = 95.0
    content_extraction_accuracy_percent: float = 90.0
    
    # Frontend Performance Benchmarks (for website)
    page_load_time_ms: float = 2000.0
    first_contentful_paint_ms: float = 1500.0
    largest_contentful_paint_ms: float = 2500.0
    cumulative_layout_shift: float = 0.1
    first_input_delay_ms: float = 100.0


def get_benchmark_thresholds() -> Dict[str, Any]:
    """Get performance benchmark thresholds as dictionary."""
    benchmarks = PerformanceBenchmarks()
    return {
        # Pipeline Processing
        'content_ingestion_per_second': benchmarks.content_ingestion_per_second,
        'content_extraction_time_per_1000_words': benchmarks.content_extraction_time_per_1000_words,
        'deduplication_time_per_1000_items': benchmarks.deduplication_time_per_1000_items,
        'ai_summarization_time_per_article': benchmarks.ai_summarization_time_per_article,
        'markdown_generation_time_per_article': benchmarks.markdown_generation_time_per_article,
        
        # Database
        'database_insert_articles_per_second': benchmarks.database_insert_articles_per_second,
        'database_query_response_time_ms': benchmarks.database_query_response_time_ms,
        'database_complex_query_time_ms': benchmarks.database_complex_query_time_ms,
        'database_queries_per_second': benchmarks.database_queries_per_second,
        
        # Memory
        'content_processing_memory_mb': benchmarks.content_processing_memory_mb,
        'deduplication_index_memory_mb': benchmarks.deduplication_index_memory_mb,
        'ai_processing_memory_mb': benchmarks.ai_processing_memory_mb,
        'total_pipeline_memory_mb': benchmarks.total_pipeline_memory_mb,
        
        # API Services
        'rss_feed_fetch_time_ms': benchmarks.rss_feed_fetch_time_ms,
        'web_scraping_time_per_page_ms': benchmarks.web_scraping_time_per_page_ms,
        'openai_api_response_time_ms': benchmarks.openai_api_response_time_ms,
        'anthropic_api_response_time_ms': benchmarks.anthropic_api_response_time_ms,
        
        # Scalability
        'max_concurrent_articles': benchmarks.max_concurrent_articles,
        'max_concurrent_sources': benchmarks.max_concurrent_sources,
        'pipeline_articles_per_second': benchmarks.pipeline_articles_per_second,
        'cross_linking_time_per_1000_articles': benchmarks.cross_linking_time_per_1000_articles,
        
        # Quality
        'test_coverage_minimum_percent': benchmarks.test_coverage_minimum_percent,
        'duplicate_detection_accuracy_percent': benchmarks.duplicate_detection_accuracy_percent,
        'content_extraction_accuracy_percent': benchmarks.content_extraction_accuracy_percent,
        
        # Frontend
        'page_load_time_ms': benchmarks.page_load_time_ms,
        'first_contentful_paint_ms': benchmarks.first_contentful_paint_ms,
        'largest_contentful_paint_ms': benchmarks.largest_contentful_paint_ms,
        'cumulative_layout_shift': benchmarks.cumulative_layout_shift,
        'first_input_delay_ms': benchmarks.first_input_delay_ms,
    }


def get_performance_categories() -> Dict[str, list]:
    """Get performance benchmarks organized by category."""
    return {
        'pipeline': [
            'content_ingestion_per_second',
            'content_extraction_time_per_1000_words', 
            'deduplication_time_per_1000_items',
            'ai_summarization_time_per_article',
            'markdown_generation_time_per_article',
            'pipeline_articles_per_second'
        ],
        'database': [
            'database_insert_articles_per_second',
            'database_query_response_time_ms',
            'database_complex_query_time_ms',
            'database_queries_per_second'
        ],
        'memory': [
            'content_processing_memory_mb',
            'deduplication_index_memory_mb',
            'ai_processing_memory_mb',
            'total_pipeline_memory_mb'
        ],
        'external_apis': [
            'rss_feed_fetch_time_ms',
            'web_scraping_time_per_page_ms',
            'openai_api_response_time_ms',
            'anthropic_api_response_time_ms'
        ],
        'scalability': [
            'max_concurrent_articles',
            'max_concurrent_sources',
            'cross_linking_time_per_1000_articles'
        ],
        'quality': [
            'test_coverage_minimum_percent',
            'duplicate_detection_accuracy_percent',
            'content_extraction_accuracy_percent'
        ],
        'frontend': [
            'page_load_time_ms',
            'first_contentful_paint_ms',
            'largest_contentful_paint_ms',
            'cumulative_layout_shift',
            'first_input_delay_ms'
        ]
    }


class PerformanceProfiler:
    """Performance profiling utilities."""
    
    def __init__(self):
        self.measurements = {}
        self.benchmarks = PerformanceBenchmarks()
    
    def start_measurement(self, operation: str) -> None:
        """Start timing an operation."""
        import time
        self.measurements[operation] = {'start_time': time.perf_counter()}
    
    def end_measurement(self, operation: str) -> float:
        """End timing an operation and return duration."""
        import time
        if operation not in self.measurements:
            raise ValueError(f"No measurement started for operation: {operation}")
        
        duration = time.perf_counter() - self.measurements[operation]['start_time']
        self.measurements[operation]['duration'] = duration
        return duration
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    
    def check_benchmark(self, operation: str, measured_value: float, 
                       benchmark_key: str = None) -> Dict[str, Any]:
        """Check if measured value meets benchmark threshold."""
        if benchmark_key is None:
            benchmark_key = operation
        
        benchmark_value = getattr(self.benchmarks, benchmark_key, None)
        if benchmark_value is None:
            return {
                'operation': operation,
                'measured_value': measured_value,
                'benchmark_value': None,
                'meets_benchmark': None,
                'ratio': None
            }
        
        # For time-based metrics, lower is better
        if 'time' in benchmark_key or 'ms' in benchmark_key:
            meets_benchmark = measured_value <= benchmark_value
            ratio = measured_value / benchmark_value
        else:
            # For throughput metrics, higher is better
            meets_benchmark = measured_value >= benchmark_value
            ratio = measured_value / benchmark_value
        
        return {
            'operation': operation,
            'measured_value': measured_value,
            'benchmark_value': benchmark_value,
            'meets_benchmark': meets_benchmark,
            'ratio': ratio,
            'performance_delta': ((measured_value - benchmark_value) / benchmark_value) * 100
        }
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            'timestamp': self._get_timestamp(),
            'system_info': self._get_system_info(),
            'measurements': {},
            'memory_usage': self.get_memory_usage(),
            'benchmark_results': {},
            'summary': {
                'total_operations': 0,
                'operations_meeting_benchmarks': 0,
                'overall_performance_score': 0.0
            }
        }
        
        benchmark_scores = []
        
        for operation, data in self.measurements.items():
            if 'duration' in data:
                duration = data['duration']
                report['measurements'][operation] = {
                    'duration_seconds': duration,
                    'duration_ms': duration * 1000
                }
                
                # Check against benchmark if available
                benchmark_result = self.check_benchmark(operation, duration)
                report['benchmark_results'][operation] = benchmark_result
                
                if benchmark_result['meets_benchmark'] is not None:
                    report['summary']['total_operations'] += 1
                    if benchmark_result['meets_benchmark']:
                        report['summary']['operations_meeting_benchmarks'] += 1
                    
                    # Add to performance score calculation
                    benchmark_scores.append(min(1.0, 1.0 / benchmark_result['ratio']))
        
        # Calculate overall performance score
        if benchmark_scores:
            report['summary']['overall_performance_score'] = sum(benchmark_scores) / len(benchmark_scores)
        
        return report
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow().isoformat()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        import platform
        import psutil
        
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else None,
            'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'disk_usage_percent': psutil.disk_usage('/').percent
        }


class LoadTestScenarios:
    """Predefined load test scenarios."""
    
    @staticmethod
    def small_batch_scenario() -> Dict[str, Any]:
        """Small batch processing scenario (10-50 articles)."""
        return {
            'name': 'Small Batch Processing',
            'description': 'Process 10-50 articles from 2-5 RSS sources',
            'parameters': {
                'source_count': 3,
                'articles_per_source': 15,
                'total_articles': 45,
                'concurrent_sources': 2,
                'batch_size': 10
            },
            'expected_duration_seconds': 30,
            'memory_limit_mb': 500
        }
    
    @staticmethod
    def medium_batch_scenario() -> Dict[str, Any]:
        """Medium batch processing scenario (100-500 articles)."""
        return {
            'name': 'Medium Batch Processing',
            'description': 'Process 100-500 articles from 10-20 RSS sources',
            'parameters': {
                'source_count': 15,
                'articles_per_source': 25,
                'total_articles': 375,
                'concurrent_sources': 5,
                'batch_size': 25
            },
            'expected_duration_seconds': 180,
            'memory_limit_mb': 1000
        }
    
    @staticmethod
    def large_batch_scenario() -> Dict[str, Any]:
        """Large batch processing scenario (1000+ articles)."""
        return {
            'name': 'Large Batch Processing',
            'description': 'Process 1000+ articles from 50+ RSS sources',
            'parameters': {
                'source_count': 50,
                'articles_per_source': 25,
                'total_articles': 1250,
                'concurrent_sources': 10,
                'batch_size': 50
            },
            'expected_duration_seconds': 600,
            'memory_limit_mb': 2000
        }
    
    @staticmethod
    def high_concurrency_scenario() -> Dict[str, Any]:
        """High concurrency processing scenario."""
        return {
            'name': 'High Concurrency Processing',
            'description': 'Process articles with maximum concurrency settings',
            'parameters': {
                'source_count': 20,
                'articles_per_source': 20,
                'total_articles': 400,
                'concurrent_sources': 20,  # Maximum concurrency
                'concurrent_articles': 50,
                'batch_size': 10
            },
            'expected_duration_seconds': 120,
            'memory_limit_mb': 1500
        }
    
    @staticmethod
    def memory_stress_scenario() -> Dict[str, Any]:
        """Memory stress test scenario."""
        return {
            'name': 'Memory Stress Test',
            'description': 'Process large articles to test memory usage',
            'parameters': {
                'source_count': 5,
                'articles_per_source': 50,
                'total_articles': 250,
                'article_size_words': 5000,  # Large articles
                'concurrent_sources': 3,
                'batch_size': 20
            },
            'expected_duration_seconds': 300,
            'memory_limit_mb': 3000
        }
    
    @staticmethod
    def get_all_scenarios() -> Dict[str, Dict[str, Any]]:
        """Get all predefined load test scenarios."""
        return {
            'small_batch': LoadTestScenarios.small_batch_scenario(),
            'medium_batch': LoadTestScenarios.medium_batch_scenario(),
            'large_batch': LoadTestScenarios.large_batch_scenario(),
            'high_concurrency': LoadTestScenarios.high_concurrency_scenario(),
            'memory_stress': LoadTestScenarios.memory_stress_scenario()
        }