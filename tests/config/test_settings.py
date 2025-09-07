"""
Test configuration settings for the AI Knowledge Website project.

Provides centralized configuration for all test environments and scenarios.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import timedelta

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

class TestConfig:
    """Base test configuration class."""
    
    # Environment settings
    ENVIRONMENT = os.getenv('TEST_ENVIRONMENT', 'local')
    DEBUG = os.getenv('TEST_DEBUG', 'false').lower() == 'true'
    VERBOSE = os.getenv('TEST_VERBOSE', 'false').lower() == 'true'
    
    # Test execution settings
    PARALLEL_TESTS = os.getenv('TEST_PARALLEL', 'false').lower() == 'true'
    MAX_WORKERS = int(os.getenv('TEST_MAX_WORKERS', '4'))
    TEST_TIMEOUT = int(os.getenv('TEST_TIMEOUT', '300'))  # 5 minutes
    
    # Coverage settings
    COVERAGE_THRESHOLD = float(os.getenv('COVERAGE_THRESHOLD', '95.0'))
    COVERAGE_FAIL_UNDER = float(os.getenv('COVERAGE_FAIL_UNDER', '95.0'))
    
    # Database settings for testing
    DATABASE_URL = os.getenv('TEST_DATABASE_URL', 'sqlite:///test_ai_knowledge.db')
    DATABASE_RESET = os.getenv('TEST_DATABASE_RESET', 'true').lower() == 'true'
    
    # External API settings for testing
    OPENAI_API_KEY = os.getenv('TEST_OPENAI_API_KEY', 'test-key-placeholder')
    ANTHROPIC_API_KEY = os.getenv('TEST_ANTHROPIC_API_KEY', 'test-key-placeholder')
    MOCK_AI_APIS = os.getenv('TEST_MOCK_AI_APIS', 'true').lower() == 'true'
    
    # File system paths
    TEST_DATA_DIR = PROJECT_ROOT / 'tests' / 'data'
    TEST_OUTPUT_DIR = PROJECT_ROOT / 'tests' / 'output'
    TEST_FIXTURES_DIR = PROJECT_ROOT / 'tests' / 'fixtures'
    TEMP_DIR = PROJECT_ROOT / 'tests' / 'temp'
    
    # Performance benchmarks
    PERFORMANCE_BENCHMARKS = {
        'content_processing_time_per_1000_words': 2.0,  # seconds
        'duplicate_detection_time_per_100_items': 1.0,   # seconds
        'database_batch_insert_time_per_100_items': 5.0, # seconds
        'cross_linking_time_per_1000_articles': 2.0,     # seconds
        'ai_summarization_timeout_seconds': 30,
        'rss_fetch_timeout_seconds': 10,
        'content_extraction_time_per_kb_ms': 100,        # milliseconds
        'max_memory_usage_mb': 500,
        'target_throughput_articles_per_minute': 30
    }
    
    # Security test settings
    SECURITY_SCAN_ENABLED = os.getenv('TEST_SECURITY_SCAN', 'true').lower() == 'true'
    XSS_PAYLOADS_FILE = TEST_FIXTURES_DIR / 'security' / 'xss_payloads.json'
    SQL_INJECTION_PAYLOADS_FILE = TEST_FIXTURES_DIR / 'security' / 'sql_payloads.json'
    
    # Network and API settings
    NETWORK_TIMEOUT = int(os.getenv('TEST_NETWORK_TIMEOUT', '10'))
    RETRY_ATTEMPTS = int(os.getenv('TEST_RETRY_ATTEMPTS', '3'))
    RATE_LIMIT_DELAY = float(os.getenv('TEST_RATE_LIMIT_DELAY', '1.0'))
    
    # Content generation settings
    SAMPLE_CONTENT_SIZE = {
        'small': 1000,   # ~200 words
        'medium': 5000,  # ~1000 words  
        'large': 25000,  # ~5000 words
    }
    
    @classmethod
    def ensure_test_directories(cls) -> None:
        """Ensure all test directories exist."""
        directories = [
            cls.TEST_DATA_DIR,
            cls.TEST_OUTPUT_DIR,
            cls.TEST_FIXTURES_DIR,
            cls.TEMP_DIR,
            cls.TEST_FIXTURES_DIR / 'security',
            cls.TEST_FIXTURES_DIR / 'content',
            cls.TEST_FIXTURES_DIR / 'database'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_database_config(cls) -> Dict[str, Any]:
        """Get database configuration for testing."""
        return {
            'url': cls.DATABASE_URL,
            'reset_on_startup': cls.DATABASE_RESET,
            'echo': cls.DEBUG,
            'pool_pre_ping': True,
            'pool_recycle': 300
        }
    
    @classmethod
    def get_performance_config(cls) -> Dict[str, Any]:
        """Get performance testing configuration."""
        return {
            'benchmarks': cls.PERFORMANCE_BENCHMARKS.copy(),
            'timeout': cls.TEST_TIMEOUT,
            'parallel': cls.PARALLEL_TESTS,
            'max_workers': cls.MAX_WORKERS
        }


class LocalTestConfig(TestConfig):
    """Configuration for local development testing."""
    
    # More lenient timeouts for local development
    TEST_TIMEOUT = 600  # 10 minutes
    NETWORK_TIMEOUT = 30
    
    # Enable all test types by default
    RUN_SLOW_TESTS = True
    RUN_NETWORK_TESTS = True
    RUN_AI_API_TESTS = False  # Usually mocked locally
    
    # Local database settings
    DATABASE_URL = 'sqlite:///local_test_ai_knowledge.db'
    DATABASE_RESET = True
    
    # Enhanced logging for debugging
    DEBUG = True
    VERBOSE = True


class CITestConfig(TestConfig):
    """Configuration for CI/CD pipeline testing."""
    
    # Stricter timeouts for CI
    TEST_TIMEOUT = 300  # 5 minutes
    NETWORK_TIMEOUT = 10
    
    # Controlled test execution in CI
    RUN_SLOW_TESTS = False
    RUN_NETWORK_TESTS = False
    RUN_AI_API_TESTS = False  # Always mocked in CI
    
    # Parallel execution enabled
    PARALLEL_TESTS = True
    MAX_WORKERS = 2  # Conservative for CI resources
    
    # Stricter coverage requirements
    COVERAGE_THRESHOLD = 95.0
    COVERAGE_FAIL_UNDER = 95.0
    
    # In-memory database for speed
    DATABASE_URL = 'sqlite:///:memory:'
    DATABASE_RESET = True


class StagingTestConfig(TestConfig):
    """Configuration for staging environment testing."""
    
    # Real services but with test data
    RUN_SLOW_TESTS = True
    RUN_NETWORK_TESTS = True
    RUN_AI_API_TESTS = True
    
    # Staging database
    DATABASE_URL = os.getenv('STAGING_DATABASE_URL', 'sqlite:///staging_test.db')
    DATABASE_RESET = False  # Don't reset staging data
    
    # Real API keys for staging
    MOCK_AI_APIS = False


class ProductionTestConfig(TestConfig):
    """Configuration for production smoke tests (read-only)."""
    
    # Only safe, read-only tests
    RUN_SLOW_TESTS = False
    RUN_NETWORK_TESTS = True
    RUN_AI_API_TESTS = False
    
    # Production database (read-only)
    DATABASE_URL = os.getenv('PRODUCTION_DATABASE_URL')
    DATABASE_RESET = False
    
    # No mocking - test real services
    MOCK_AI_APIS = False
    
    # Very strict timeouts
    TEST_TIMEOUT = 60
    NETWORK_TIMEOUT = 5


def get_test_config() -> TestConfig:
    """Get the appropriate test configuration based on environment."""
    environment = os.getenv('TEST_ENVIRONMENT', 'local').lower()
    
    config_map = {
        'local': LocalTestConfig,
        'ci': CITestConfig,
        'staging': StagingTestConfig,
        'production': ProductionTestConfig
    }
    
    config_class = config_map.get(environment, LocalTestConfig)
    return config_class()


# Global test configuration instance
test_config = get_test_config()

# Performance benchmark utilities
def get_performance_threshold(metric_name: str, default: float = 1.0) -> float:
    """Get performance threshold for a specific metric."""
    return test_config.PERFORMANCE_BENCHMARKS.get(metric_name, default)


def is_performance_test_enabled() -> bool:
    """Check if performance tests should be run."""
    return os.getenv('TEST_PERFORMANCE', 'true').lower() == 'true'


def is_slow_test_enabled() -> bool:
    """Check if slow tests should be run."""
    return getattr(test_config, 'RUN_SLOW_TESTS', False) or \
           os.getenv('TEST_RUN_SLOW', 'false').lower() == 'true'


def is_network_test_enabled() -> bool:
    """Check if network tests should be run."""
    return getattr(test_config, 'RUN_NETWORK_TESTS', False) or \
           os.getenv('TEST_NETWORK', 'false').lower() == 'true'


def is_ai_api_test_enabled() -> bool:
    """Check if AI API tests should be run."""
    return getattr(test_config, 'RUN_AI_API_TESTS', False) or \
           os.getenv('TEST_AI_API', 'false').lower() == 'true'


# Test data utilities
def get_sample_content(size: str = 'medium') -> str:
    """Generate sample content of specified size."""
    sizes = test_config.SAMPLE_CONTENT_SIZE
    char_count = sizes.get(size, sizes['medium'])
    
    base_content = (
        "This is sample content for testing the AI Knowledge Website pipeline. "
        "It discusses artificial intelligence, machine learning, and deep learning concepts. "
        "Neural networks are computational models inspired by biological neural networks. "
        "They consist of interconnected nodes that process information through weighted connections. "
    )
    
    # Repeat content to reach desired size
    repeat_count = max(1, char_count // len(base_content))
    return (base_content * repeat_count)[:char_count]


# Initialize test environment
def setup_test_environment():
    """Setup the test environment with required directories and configuration."""
    test_config.ensure_test_directories()
    
    # Set environment variables for pytest
    os.environ['PYTEST_CURRENT_TEST'] = 'true'
    os.environ['TZ'] = 'UTC'
    
    # Ensure coverage directory exists
    coverage_dir = PROJECT_ROOT / 'coverage'
    coverage_dir.mkdir(exist_ok=True)


if __name__ == '__main__':
    # For testing the configuration
    setup_test_environment()
    config = get_test_config()
    print(f"Test environment: {config.ENVIRONMENT}")
    print(f"Database URL: {config.DATABASE_URL}")
    print(f"Coverage threshold: {config.COVERAGE_THRESHOLD}%")
    print(f"Performance benchmarks: {len(config.PERFORMANCE_BENCHMARKS)} defined")