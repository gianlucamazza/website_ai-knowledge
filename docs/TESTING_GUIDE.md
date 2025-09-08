# Testing Guide

This comprehensive guide covers testing strategies, frameworks, and best practices for the AI Knowledge Website system, including unit testing, integration testing, performance testing, and quality assurance procedures.

## Table of Contents

1. [Testing Philosophy](#testing-philosophy)
2. [Testing Strategy](#testing-strategy)
3. [Test Structure and Organization](#test-structure-and-organization)
4. [Unit Testing](#unit-testing)
5. [Integration Testing](#integration-testing)
6. [End-to-End Testing](#end-to-end-testing)
7. [Performance Testing](#performance-testing)
8. [Security Testing](#security-testing)
9. [Content Quality Testing](#content-quality-testing)
10. [Test Automation and CI/CD](#test-automation-and-cicd)
11. [Test Data Management](#test-data-management)
12. [Testing Tools and Frameworks](#testing-tools-and-frameworks)

## Testing Philosophy

### Quality Principles

1. **Quality Built-In**: Testing is integrated throughout development, not an afterthought
2. **Shift-Left Testing**: Catch issues early in the development cycle
3. **Risk-Based Testing**: Focus testing efforts on high-risk areas
4. **Continuous Testing**: Automated testing in CI/CD pipelines
5. **Test-Driven Development**: Write tests before implementing features

### Testing Pyramid

```
        ┌─────────────────┐
        │   E2E Tests     │ ← Few, slow, expensive
        │   (UI, API)     │
        └─────────────────┘
      ┌───────────────────────┐
      │  Integration Tests    │ ← Some, moderate speed
      │  (APIs, Database)     │
      └───────────────────────┘
    ┌─────────────────────────────┐
    │      Unit Tests             │ ← Many, fast, cheap
    │  (Functions, Classes)       │
    └─────────────────────────────┘
```

### Coverage Goals

| Test Type | Coverage Target | Purpose |
|-----------|-----------------|---------|
| **Unit Tests** | 90%+ | Individual component functionality |
| **Integration Tests** | 80%+ | Component interaction |
| **E2E Tests** | Critical paths | User journey validation |
| **Performance Tests** | Key scenarios | Performance regression detection |
| **Security Tests** | All attack vectors | Vulnerability prevention |

## Testing Strategy

### Test Classification

**Functional Testing:**
- Unit tests for individual components
- Integration tests for component interactions
- API tests for endpoint functionality
- E2E tests for complete user workflows

**Non-Functional Testing:**
- Performance tests for speed and scalability
- Security tests for vulnerability detection
- Reliability tests for system stability
- Usability tests for user experience

**Content Pipeline Testing:**
- Data ingestion validation
- Content processing accuracy
- Duplicate detection effectiveness
- Quality scoring validation
- Publication workflow testing

### Risk-Based Test Prioritization

**Critical Risk Areas:**

1. **Content Pipeline (High Risk)**
   - Data loss or corruption
   - Duplicate content publication
   - Quality degradation
   - Processing failures

2. **API Security (High Risk)**
   - Authentication bypass
   - Authorization vulnerabilities
   - Data exposure
   - Injection attacks

3. **Data Integrity (High Risk)**
   - Database consistency
   - Content accuracy
   - Link integrity
   - Source attribution

4. **Performance (Medium Risk)**
   - Response time degradation
   - Memory leaks
   - Scalability limits
   - Resource exhaustion

## Test Structure and Organization

### Directory Structure

```
tests/
├── unit/                       # Unit tests
│   ├── pipeline/              # Pipeline component tests
│   │   ├── test_ingest.py
│   │   ├── test_normalize.py
│   │   ├── test_dedup.py
│   │   ├── test_enrich.py
│   │   └── test_publish.py
│   ├── api/                   # API unit tests
│   │   ├── test_auth.py
│   │   ├── test_content.py
│   │   └── test_pipeline.py
│   ├── models/                # Data model tests
│   │   ├── test_content_model.py
│   │   └── test_user_model.py
│   └── utils/                 # Utility function tests
│       ├── test_validation.py
│       └── test_helpers.py
├── integration/               # Integration tests
│   ├── api/                  # API integration tests
│   │   ├── test_content_api.py
│   │   └── test_pipeline_api.py
│   ├── database/             # Database integration tests
│   │   ├── test_content_ops.py
│   │   └── test_migrations.py
│   ├── pipeline/             # Pipeline integration tests
│   │   ├── test_full_pipeline.py
│   │   └── test_stage_integration.py
│   └── external/             # External service tests
│       ├── test_openai_integration.py
│       └── test_source_scraping.py
├── e2e/                      # End-to-end tests
│   ├── frontend/             # Frontend E2E tests
│   │   ├── test_content_browsing.py
│   │   └── test_search_functionality.py
│   ├── api/                  # API E2E tests
│   │   ├── test_content_workflow.py
│   │   └── test_admin_workflow.py
│   └── pipeline/             # Pipeline E2E tests
│       └── test_content_lifecycle.py
├── performance/              # Performance tests
│   ├── load/                # Load testing
│   │   ├── test_api_load.py
│   │   └── test_pipeline_load.py
│   ├── stress/              # Stress testing
│   │   └── test_resource_limits.py
│   └── benchmarks/          # Benchmark tests
│       └── test_performance_benchmarks.py
├── security/                 # Security tests
│   ├── test_auth_security.py
│   ├── test_input_validation.py
│   ├── test_sql_injection.py
│   └── test_xss_prevention.py
├── content/                  # Content quality tests
│   ├── test_schema_validation.py
│   ├── test_link_checking.py
│   └── test_content_quality.py
├── fixtures/                 # Test data and fixtures
│   ├── sample_content.json
│   ├── test_sources.yaml
│   └── mock_responses.py
├── conftest.py              # Pytest configuration
├── pytest.ini              # Pytest settings
└── requirements-test.txt    # Test dependencies
```

### Test Naming Conventions

```python
# Test file naming: test_[module_name].py
# Test class naming: Test[ClassName]
# Test method naming: test_[what_is_being_tested]_[expected_outcome]

class TestContentDeduplication:
    def test_identical_content_detected_as_duplicate(self):
        """Test that identical content is correctly identified as duplicate."""
        pass
    
    def test_similar_content_above_threshold_detected_as_duplicate(self):
        """Test that similar content above threshold is detected as duplicate."""
        pass
    
    def test_different_content_not_detected_as_duplicate(self):
        """Test that clearly different content is not flagged as duplicate."""
        pass
    
    def test_edge_case_empty_content_handled_gracefully(self):
        """Test that empty content input is handled without errors."""
        pass
```

## Unit Testing

### Python Unit Testing with pytest

**Basic Test Structure:**
```python
import pytest
from unittest.mock import Mock, patch, MagicMock
from pipelines.dedup.simhash import SimHashDuplicateDetector
from pipelines.exceptions import ContentProcessingError

class TestSimHashDuplicateDetector:
    
    @pytest.fixture
    def detector(self):
        """Create a SimHashDuplicateDetector instance for testing."""
        return SimHashDuplicateDetector(threshold=3)
    
    @pytest.fixture
    def sample_content(self):
        """Provide sample content for testing."""
        return {
            'identical': "This is a test article about machine learning algorithms.",
            'similar': "This is a test article about machine learning methods.",
            'different': "This article discusses quantum computing principles."
        }
    
    def test_compute_hash_returns_integer(self, detector):
        """Test that compute_hash returns an integer hash value."""
        content = "Sample content for hashing"
        hash_value = detector.compute_hash(content)
        
        assert isinstance(hash_value, int)
        assert hash_value > 0
    
    def test_identical_content_produces_identical_hashes(self, detector, sample_content):
        """Test that identical content produces identical hash values."""
        content = sample_content['identical']
        
        hash1 = detector.compute_hash(content)
        hash2 = detector.compute_hash(content)
        
        assert hash1 == hash2
    
    def test_are_duplicates_with_identical_hashes(self, detector):
        """Test duplicate detection with identical hash values."""
        hash1 = 12345
        hash2 = 12345
        
        assert detector.are_duplicates(hash1, hash2) is True
    
    def test_are_duplicates_within_threshold(self, detector):
        """Test duplicate detection within configured threshold."""
        # Hashes that differ by 2 bits (within threshold of 3)
        hash1 = 0b1010101010101010
        hash2 = 0b1010101010101100  # 2 bit difference
        
        assert detector.are_duplicates(hash1, hash2) is True
    
    def test_are_duplicates_beyond_threshold(self, detector):
        """Test that content beyond threshold is not detected as duplicate."""
        # Hashes that differ by 4 bits (beyond threshold of 3)
        hash1 = 0b1010101010101010
        hash2 = 0b1010101010100000  # 4 bit difference
        
        assert detector.are_duplicates(hash1, hash2) is False
    
    @pytest.mark.parametrize("threshold,hash1,hash2,expected", [
        (1, 0b1010, 0b1010, True),   # Identical
        (1, 0b1010, 0b1011, True),   # 1 bit diff, threshold 1
        (1, 0b1010, 0b1000, False),  # 2 bit diff, threshold 1
        (3, 0b1010, 0b1000, True),   # 2 bit diff, threshold 3
        (3, 0b1010, 0b0000, False),  # 4 bit diff, threshold 3
    ])
    def test_threshold_parameterized(self, threshold, hash1, hash2, expected):
        """Test duplicate detection with various thresholds."""
        detector = SimHashDuplicateDetector(threshold=threshold)
        assert detector.are_duplicates(hash1, hash2) == expected
    
    def test_compute_hash_with_empty_content_raises_error(self, detector):
        """Test that empty content raises appropriate error."""
        with pytest.raises(ContentProcessingError) as exc_info:
            detector.compute_hash("")
        
        assert "empty content" in str(exc_info.value).lower()
    
    def test_compute_hash_with_none_content_raises_error(self, detector):
        """Test that None content raises appropriate error."""
        with pytest.raises(ContentProcessingError):
            detector.compute_hash(None)
    
    @patch('pipelines.dedup.simhash.xxhash.xxh64')
    def test_compute_hash_uses_xxhash(self, mock_xxhash, detector):
        """Test that compute_hash uses xxhash for consistent hashing."""
        mock_hash = Mock()
        mock_hash.intdigest.return_value = 12345
        mock_xxhash.return_value = mock_hash
        
        result = detector.compute_hash("test content")
        
        mock_xxhash.assert_called_once()
        assert result == 12345
```

**Testing Async Functions:**
```python
import pytest
import asyncio
from pipelines.enrich.summarizer import ContentSummarizer

class TestContentSummarizer:
    
    @pytest.fixture
    async def summarizer(self):
        """Create ContentSummarizer instance."""
        return ContentSummarizer(api_key="test_key")
    
    @pytest.mark.asyncio
    async def test_summarize_content_returns_summary(self, summarizer):
        """Test that summarize_content returns a proper summary."""
        content = "This is a long article about artificial intelligence..."
        
        with patch.object(summarizer, '_call_openai_api') as mock_api:
            mock_api.return_value = "AI summary"
            
            summary = await summarizer.summarize_content(content)
            
            assert isinstance(summary, str)
            assert len(summary) > 0
            mock_api.assert_called_once_with(content)
    
    @pytest.mark.asyncio
    async def test_summarize_content_handles_api_error(self, summarizer):
        """Test error handling when API call fails."""
        content = "Test content"
        
        with patch.object(summarizer, '_call_openai_api') as mock_api:
            mock_api.side_effect = Exception("API Error")
            
            with pytest.raises(ContentProcessingError):
                await summarizer.summarize_content(content)
```

### Frontend Unit Testing with Vitest

**Astro Component Testing:**
```typescript
// tests/unit/components/ArticleCard.test.tsx
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import ArticleCard from '../../../src/components/ArticleCard.astro';

describe('ArticleCard', () => {
  const mockArticle = {
    data: {
      title: 'Neural Network Fundamentals',
      description: 'Introduction to neural networks and deep learning',
      publishDate: new Date('2024-01-01'),
      author: 'AI Expert',
      category: 'machine-learning',
      tags: ['ai', 'ml']
    },
    slug: 'neural-network-fundamentals'
  };

  it('renders article title and description', () => {
    render(<ArticleCard article={mockArticle} />);
    
    expect(screen.getByText('Neural Network Fundamentals')).toBeInTheDocument();
    expect(screen.getByText('Introduction to neural networks and deep learning')).toBeInTheDocument();
  });

  it('displays formatted publish date', () => {
    render(<ArticleCard article={mockArticle} />);
    
    expect(screen.getByText(/January 1, 2024/)).toBeInTheDocument();
  });

  it('shows correct author information', () => {
    render(<ArticleCard article={mockArticle} />);
    
    expect(screen.getByText('By Test Author')).toBeInTheDocument();
  });

  it('includes correct navigation link', () => {
    render(<ArticleCard article={mockArticle} />);
    
    const link = screen.getByRole('link');
    expect(link).toHaveAttribute('href', '/articles/neural-network-fundamentals');
  });

  it('applies featured styling when featured prop is true', () => {
    render(<ArticleCard article={mockArticle} featured={true} />);
    
    const card = screen.getByRole('article');
    expect(card).toHaveClass('featured');
  });
});
```

**Utility Function Testing:**
```typescript
// tests/unit/utils/formatDate.test.ts
import { describe, it, expect } from 'vitest';
import { formatDate } from '../../../src/utils/date';

describe('formatDate', () => {
  it('formats date with default locale and options', () => {
    const date = new Date('2024-01-15T12:00:00Z');
    const formatted = formatDate(date);
    
    expect(formatted).toBe('January 15, 2024');
  });

  it('formats date with custom locale', () => {
    const date = new Date('2024-01-15T12:00:00Z');
    const formatted = formatDate(date, 'de-DE');
    
    expect(formatted).toBe('15. Januar 2024');
  });

  it('formats date with custom options', () => {
    const date = new Date('2024-01-15T12:00:00Z');
    const formatted = formatDate(date, 'en-US', {
      year: '2-digit',
      month: 'short',
      day: 'numeric'
    });
    
    expect(formatted).toBe('Jan 15, 24');
  });

  it('handles edge case dates correctly', () => {
    const date = new Date('2024-02-29T00:00:00Z'); // Leap year
    const formatted = formatDate(date);
    
    expect(formatted).toBe('February 29, 2024');
  });
});
```

## Integration Testing

### API Integration Testing

```python
# tests/integration/api/test_content_api.py
import pytest
import httpx
from fastapi.testclient import TestClient
from pipelines.api.main import app
from pipelines.database.models import ContentItem
from pipelines.database.connection import get_session

class TestContentAPI:
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Provide authentication headers."""
        return {"Authorization": "Bearer test_token"}
    
    @pytest.fixture
    def sample_content_item(self, db_session):
        """Create sample content item in database."""
        content_item = ContentItem(
            id="test_item_001",
            title="Test Article",
            content="This is test content for integration testing.",
            source_url="https://example.com/test",
            status="published"
        )
        db_session.add(content_item)
        db_session.commit()
        return content_item
    
    def test_get_content_item_success(self, client, auth_headers, sample_content_item):
        """Test successful retrieval of content item."""
        response = client.get(
            f"/api/v1/content/{sample_content_item.id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == sample_content_item.id
        assert data["title"] == "Test Article"
        assert data["status"] == "published"
    
    def test_get_content_item_not_found(self, client, auth_headers):
        """Test retrieval of non-existent content item."""
        response = client.get(
            "/api/v1/content/nonexistent",
            headers=auth_headers
        )
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_get_content_item_unauthorized(self, client, sample_content_item):
        """Test unauthorized access to content item."""
        response = client.get(f"/api/v1/content/{sample_content_item.id}")
        
        assert response.status_code == 401
    
    def test_create_content_item(self, client, auth_headers):
        """Test content item creation."""
        content_data = {
            "title": "New Test Article",
            "content": "New test content",
            "source_url": "https://example.com/new-test",
            "category": "machine-learning",
            "tags": ["ai", "testing"]
        }
        
        response = client.post(
            "/api/v1/content",
            json=content_data,
            headers=auth_headers
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "New Test Article"
        assert data["status"] == "draft"
        assert "id" in data
    
    def test_update_content_item_status(self, client, auth_headers, sample_content_item):
        """Test updating content item status."""
        update_data = {"status": "published"}
        
        response = client.patch(
            f"/api/v1/content/{sample_content_item.id}/status",
            json=update_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "published"
    
    @pytest.mark.parametrize("invalid_data", [
        {"title": ""},  # Empty title
        {"title": "A" * 300},  # Title too long
        {"content": ""},  # Empty content
        {"source_url": "not-a-url"},  # Invalid URL
        {"tags": []},  # Empty tags array
        {"category": "invalid-category"}  # Invalid category
    ])
    def test_create_content_item_validation_errors(self, client, auth_headers, invalid_data):
        """Test content creation with various invalid data."""
        base_data = {
            "title": "Valid Title",
            "content": "Valid content",
            "source_url": "https://example.com/valid",
            "category": "machine-learning",
            "tags": ["ai"]
        }
        base_data.update(invalid_data)
        
        response = client.post(
            "/api/v1/content",
            json=base_data,
            headers=auth_headers
        )
        
        assert response.status_code == 422
        assert "validation error" in response.json()["detail"][0]["type"]
```

### Database Integration Testing

```python
# tests/integration/database/test_content_operations.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pipelines.database.models import Base, ContentItem, Source
from pipelines.database.operations import ContentOperations

class TestContentOperations:
    
    @pytest.fixture(scope="class")
    def engine(self):
        """Create test database engine."""
        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(engine)
        return engine
    
    @pytest.fixture
    def db_session(self, engine):
        """Create database session for each test."""
        Session = sessionmaker(bind=engine)
        session = Session()
        yield session
        session.rollback()
        session.close()
    
    @pytest.fixture
    def content_ops(self, db_session):
        """Create ContentOperations instance."""
        return ContentOperations(db_session)
    
    @pytest.fixture
    def sample_source(self, db_session):
        """Create sample source."""
        source = Source(
            name="test_source",
            url="https://example.com/test",
            type="rss",
            active=True
        )
        db_session.add(source)
        db_session.commit()
        return source
    
    def test_create_content_item_success(self, content_ops, sample_source):
        """Test successful content item creation."""
        content_data = {
            "title": "Test Article",
            "content": "Test content",
            "source_id": sample_source.id,
            "source_url": "https://example.com/article",
            "content_hash": "test_hash_123"
        }
        
        content_item = content_ops.create_content_item(**content_data)
        
        assert content_item.id is not None
        assert content_item.title == "Test Article"
        assert content_item.source_id == sample_source.id
        assert content_item.status == "pending"
    
    def test_get_content_by_hash_existing(self, content_ops, sample_source):
        """Test retrieval of content by existing hash."""
        # Create content item
        content_item = content_ops.create_content_item(
            title="Test Article",
            content="Test content", 
            source_id=sample_source.id,
            source_url="https://example.com/article",
            content_hash="unique_hash_456"
        )
        
        # Retrieve by hash
        found_item = content_ops.get_content_by_hash("unique_hash_456")
        
        assert found_item is not None
        assert found_item.id == content_item.id
        assert found_item.content_hash == "unique_hash_456"
    
    def test_get_content_by_hash_nonexistent(self, content_ops):
        """Test retrieval of content by non-existent hash."""
        found_item = content_ops.get_content_by_hash("nonexistent_hash")
        assert found_item is None
    
    def test_update_content_status(self, content_ops, sample_source):
        """Test content status update."""
        # Create content item
        content_item = content_ops.create_content_item(
            title="Test Article",
            content="Test content",
            source_id=sample_source.id,
            source_url="https://example.com/article"
        )
        
        # Update status
        updated = content_ops.update_content_status(content_item.id, "published")
        
        assert updated is True
        assert content_item.status == "published"
        assert content_item.updated_at is not None
    
    def test_bulk_insert_performance(self, content_ops, sample_source):
        """Test bulk insert performance for large datasets."""
        import time
        
        # Prepare bulk data
        bulk_data = []
        for i in range(1000):
            bulk_data.append({
                "title": f"Bulk Article {i}",
                "content": f"Bulk content {i}",
                "source_id": sample_source.id,
                "source_url": f"https://example.com/bulk/{i}",
                "content_hash": f"bulk_hash_{i}"
            })
        
        # Time the bulk insert
        start_time = time.time()
        inserted_count = content_ops.bulk_create_content_items(bulk_data)
        end_time = time.time()
        
        assert inserted_count == 1000
        assert end_time - start_time < 5.0  # Should complete in under 5 seconds
        
        # Verify data integrity
        total_count = content_ops.get_content_count()
        assert total_count >= 1000
```

### Pipeline Integration Testing

```python
# tests/integration/pipeline/test_full_pipeline.py
import pytest
import asyncio
from unittest.mock import patch, Mock
from pipelines.orchestrators.langgraph.workflow import ContentPipelineWorkflow
from pipelines.models import PipelineState, ContentItem

class TestFullPipelineIntegration:
    
    @pytest.fixture
    def mock_external_apis(self):
        """Mock external API calls."""
        with patch('pipelines.enrich.summarizer.OpenAIClient') as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = Mock(
                choices=[Mock(message=Mock(content="Test summary"))]
            )
            yield mock_openai
    
    @pytest.fixture
    def sample_pipeline_input(self):
        """Provide sample input for pipeline testing."""
        return {
            'sources': ['test_source'],
            'raw_content': [],
            'normalized_content': [],
            'deduplicated_content': [],
            'enriched_content': [],
            'published_files': [],
            'errors': [],
            'metadata': {'run_id': 'test_run_001'}
        }
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_pipeline_execution(self, sample_pipeline_input, mock_external_apis):
        """Test complete pipeline execution from start to finish."""
        workflow = ContentPipelineWorkflow()
        
        # Add sample content to process
        sample_pipeline_input['raw_content'] = [
            {
                'title': 'Test Article 1',
                'content': 'This is test content for pipeline testing.',
                'source_url': 'https://example.com/test1',
                'source': 'test_source'
            },
            {
                'title': 'Test Article 2', 
                'content': 'This is different test content for pipeline testing.',
                'source_url': 'https://example.com/test2',
                'source': 'test_source'
            }
        ]
        
        # Execute complete pipeline
        final_state = await workflow.run(sample_pipeline_input)
        
        # Verify pipeline completion
        assert len(final_state['errors']) == 0, f"Pipeline errors: {final_state['errors']}"
        assert len(final_state['published_files']) > 0
        assert final_state['metadata']['status'] == 'completed'
        
        # Verify content processing stages
        assert len(final_state['normalized_content']) == 2
        assert len(final_state['deduplicated_content']) <= 2  # May be fewer due to dedup
        assert len(final_state['enriched_content']) > 0
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_pipeline_error_handling(self, sample_pipeline_input):
        """Test pipeline error handling and recovery."""
        workflow = ContentPipelineWorkflow()
        
        # Add problematic content
        sample_pipeline_input['raw_content'] = [
            {
                'title': '',  # Invalid: empty title
                'content': 'Valid content',
                'source_url': 'invalid-url',  # Invalid URL
                'source': 'test_source'
            }
        ]
        
        # Execute pipeline
        final_state = await workflow.run(sample_pipeline_input)
        
        # Verify error handling
        assert len(final_state['errors']) > 0
        assert final_state['metadata']['status'] == 'completed_with_errors'
        
        # Should still complete but with logged errors
        error_types = [error['type'] for error in final_state['errors']]
        assert 'validation_error' in error_types
    
    @pytest.mark.asyncio 
    @pytest.mark.integration
    async def test_pipeline_duplicate_detection(self, sample_pipeline_input):
        """Test duplicate detection in full pipeline."""
        workflow = ContentPipelineWorkflow()
        
        # Add duplicate content
        duplicate_content = {
            'title': 'Duplicate Article',
            'content': 'This content appears twice in the pipeline.',
            'source_url': 'https://example.com/duplicate1',
            'source': 'test_source'
        }
        
        sample_pipeline_input['raw_content'] = [
            duplicate_content,
            {
                **duplicate_content,
                'source_url': 'https://example.com/duplicate2'  # Different URL, same content
            }
        ]
        
        # Execute pipeline
        final_state = await workflow.run(sample_pipeline_input)
        
        # Verify duplicate detection
        assert len(final_state['normalized_content']) == 2
        assert len(final_state['deduplicated_content']) == 1  # Duplicates removed
        assert len(final_state['published_files']) == 1
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_pipeline_performance_with_large_dataset(self, sample_pipeline_input):
        """Test pipeline performance with large content dataset."""
        import time
        
        workflow = ContentPipelineWorkflow()
        
        # Generate large dataset
        large_dataset = []
        for i in range(100):
            large_dataset.append({
                'title': f'Performance Test Article {i}',
                'content': f'This is performance test content number {i}. ' * 20,
                'source_url': f'https://example.com/perf-test/{i}',
                'source': 'test_source'
            })
        
        sample_pipeline_input['raw_content'] = large_dataset
        
        # Execute pipeline with timing
        start_time = time.time()
        final_state = await workflow.run(sample_pipeline_input)
        end_time = time.time()
        
        # Verify performance
        processing_time = end_time - start_time
        items_per_second = len(large_dataset) / processing_time
        
        assert processing_time < 60  # Should complete in under 1 minute
        assert items_per_second > 1  # Should process at least 1 item per second
        assert len(final_state['published_files']) > 0
        
        print(f"Processed {len(large_dataset)} items in {processing_time:.2f}s "
              f"({items_per_second:.2f} items/sec)")
```

## End-to-End Testing

### Frontend E2E Testing with Playwright

```typescript
// tests/e2e/content-browsing.spec.ts
import { test, expect } from '@playwright/test';

test.describe('Content Browsing', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should display homepage with articles', async ({ page }) => {
    // Check page loads correctly
    await expect(page).toHaveTitle(/AI Knowledge/);
    
    // Check main navigation is present
    await expect(page.locator('nav')).toBeVisible();
    await expect(page.getByRole('link', { name: 'Articles' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Glossary' })).toBeVisible();
    
    // Check featured content is displayed
    await expect(page.locator('.featured-articles')).toBeVisible();
    await expect(page.locator('.article-card').first()).toBeVisible();
  });

  test('should navigate to articles page', async ({ page }) => {
    await page.click('text=Articles');
    await expect(page).toHaveURL(/.*\/articles/);
    
    // Check articles are displayed
    await expect(page.locator('h1')).toContainText('Articles');
    await expect(page.locator('.article-card')).toHaveCount.greaterThan(0);
  });

  test('should open and display article content', async ({ page }) => {
    await page.goto('/articles');
    
    // Click on first article
    const firstArticle = page.locator('.article-card').first();
    const articleTitle = await firstArticle.locator('h2').textContent();
    await firstArticle.click();
    
    // Verify article page loads
    await expect(page.locator('h1')).toContainText(articleTitle);
    await expect(page.locator('article .content')).toBeVisible();
    
    // Check metadata is displayed
    await expect(page.locator('.article-meta')).toBeVisible();
    await expect(page.locator('.article-meta .author')).toBeVisible();
    await expect(page.locator('.article-meta .date')).toBeVisible();
  });

  test('should filter articles by category', async ({ page }) => {
    await page.goto('/articles');
    
    // Get initial article count
    const initialCount = await page.locator('.article-card').count();
    
    // Apply category filter
    await page.selectOption('[data-testid="category-filter"]', 'machine-learning');
    
    // Wait for filter to apply
    await page.waitForTimeout(500);
    
    // Verify filtering
    const filteredCount = await page.locator('.article-card').count();
    expect(filteredCount).toBeLessThanOrEqual(initialCount);
    
    // Check all visible articles have correct category
    const categoryTags = page.locator('.article-card .category-tag');
    const count = await categoryTags.count();
    for (let i = 0; i < count; i++) {
      await expect(categoryTags.nth(i)).toContainText('machine-learning');
    }
  });

  test('should search for articles', async ({ page }) => {
    await page.goto('/articles');
    
    // Perform search
    const searchTerm = 'neural networks';
    await page.fill('[data-testid="search-input"]', searchTerm);
    await page.press('[data-testid="search-input"]', 'Enter');
    
    // Wait for search results
    await page.waitForSelector('.search-results');
    
    // Verify search results
    await expect(page.locator('.search-results')).toBeVisible();
    const results = page.locator('.article-card');
    const count = await results.count();
    
    if (count > 0) {
      // Check that search term appears in results
      for (let i = 0; i < count; i++) {
        const articleText = await results.nth(i).textContent();
        expect(articleText.toLowerCase()).toContain(searchTerm.toLowerCase());
      }
    } else {
      // If no results, should show appropriate message
      await expect(page.locator('.no-results')).toBeVisible();
    }
  });
});

test.describe('Glossary Functionality', () => {
  test('should display glossary with terms', async ({ page }) => {
    await page.goto('/glossary');
    
    await expect(page.locator('h1')).toContainText('Glossary');
    await expect(page.locator('.glossary-term').first()).toBeVisible();
  });

  test('should navigate to glossary term', async ({ page }) => {
    await page.goto('/glossary');
    
    const firstTerm = page.locator('.glossary-term').first();
    const termTitle = await firstTerm.locator('h3').textContent();
    await firstTerm.click();
    
    // Verify term page loads
    await expect(page.locator('h1')).toContainText(termTitle);
    await expect(page.locator('.definition')).toBeVisible();
  });

  test('should show related terms', async ({ page }) => {
    await page.goto('/glossary/neural-network');
    
    // Check for related terms section
    const relatedSection = page.locator('.related-terms');
    if (await relatedSection.isVisible()) {
      await expect(relatedSection.locator('.term-link').first()).toBeVisible();
    }
  });
});
```

### API E2E Testing

```python
# tests/e2e/api/test_content_workflow.py
import pytest
import httpx
import asyncio
from typing import Dict, Any

class TestContentWorkflowE2E:
    
    @pytest.fixture
    def api_client(self):
        """Create API client for E2E tests."""
        return httpx.AsyncClient(
            base_url="http://localhost:8000",
            timeout=30.0
        )
    
    @pytest.fixture
    def admin_auth_headers(self):
        """Get admin authentication headers."""
        # In real tests, this would authenticate and get a real token
        return {"Authorization": "Bearer admin_test_token"}
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_complete_content_lifecycle(self, api_client, admin_auth_headers):
        """Test complete content lifecycle from creation to publication."""
        
        # Step 1: Create new content
        content_data = {
            "title": "E2E Test Article",
            "content": "This is an end-to-end test article for workflow validation.",
            "source_url": "https://example.com/e2e-test",
            "category": "machine-learning",
            "tags": ["testing", "e2e", "automation"]
        }
        
        create_response = await api_client.post(
            "/api/v1/content",
            json=content_data,
            headers=admin_auth_headers
        )
        
        assert create_response.status_code == 201
        content = create_response.json()
        content_id = content["id"]
        assert content["status"] == "draft"
        
        # Step 2: Validate content through pipeline
        validate_response = await api_client.post(
            "/api/v1/pipeline/validate",
            json={"content_id": content_id},
            headers=admin_auth_headers
        )
        
        assert validate_response.status_code == 200
        validation_result = validate_response.json()
        assert validation_result["valid"] is True
        
        # Step 3: Process content through enrichment
        enrich_response = await api_client.post(
            "/api/v1/pipeline/enrich",
            json={"content_ids": [content_id]},
            headers=admin_auth_headers
        )
        
        assert enrich_response.status_code == 200
        job = enrich_response.json()
        job_id = job["job_id"]
        
        # Step 4: Monitor job completion
        max_attempts = 30
        for attempt in range(max_attempts):
            status_response = await api_client.get(
                f"/api/v1/pipeline/jobs/{job_id}",
                headers=admin_auth_headers
            )
            
            assert status_response.status_code == 200
            job_status = status_response.json()
            
            if job_status["status"] == "completed":
                break
            elif job_status["status"] == "failed":
                pytest.fail(f"Job failed: {job_status}")
            
            await asyncio.sleep(2)
        else:
            pytest.fail("Job did not complete within timeout")
        
        # Step 5: Verify content enrichment
        content_response = await api_client.get(
            f"/api/v1/content/{content_id}",
            headers=admin_auth_headers
        )
        
        assert content_response.status_code == 200
        enriched_content = content_response.json()
        assert "summary" in enriched_content["metadata"]
        assert "quality_score" in enriched_content["metadata"]
        
        # Step 6: Publish content
        publish_response = await api_client.patch(
            f"/api/v1/content/{content_id}/status",
            json={"status": "published"},
            headers=admin_auth_headers
        )
        
        assert publish_response.status_code == 200
        published_content = publish_response.json()
        assert published_content["status"] == "published"
        
        # Step 7: Verify content is publicly accessible
        public_response = await api_client.get(f"/api/v1/content/{content_id}")
        assert public_response.status_code == 200
        
        # Step 8: Clean up - delete test content
        delete_response = await api_client.delete(
            f"/api/v1/content/{content_id}",
            headers=admin_auth_headers
        )
        assert delete_response.status_code == 200
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_pipeline_error_handling_e2e(self, api_client, admin_auth_headers):
        """Test end-to-end error handling in pipeline."""
        
        # Create content with intentional issues
        problematic_content = {
            "title": "Error Test",
            "content": "<script>alert('xss')</script>Invalid content with scripts",
            "source_url": "not-a-valid-url",
            "category": "invalid-category",
            "tags": []  # Empty tags should trigger validation error
        }
        
        create_response = await api_client.post(
            "/api/v1/content",
            json=problematic_content,
            headers=admin_auth_headers
        )
        
        # Should still create but with validation warnings
        assert create_response.status_code == 201
        content = create_response.json()
        content_id = content["id"]
        
        # Try to process through pipeline
        process_response = await api_client.post(
            "/api/v1/pipeline/process",
            json={"content_ids": [content_id]},
            headers=admin_auth_headers
        )
        
        assert process_response.status_code == 200
        job = process_response.json()
        job_id = job["job_id"]
        
        # Monitor job - should complete but with errors
        max_attempts = 20
        for attempt in range(max_attempts):
            status_response = await api_client.get(
                f"/api/v1/pipeline/jobs/{job_id}",
                headers=admin_auth_headers
            )
            
            job_status = status_response.json()
            
            if job_status["status"] in ["completed", "failed"]:
                break
                
            await asyncio.sleep(1)
        
        # Verify error handling
        assert len(job_status.get("errors", [])) > 0
        error_types = [error["type"] for error in job_status["errors"]]
        assert "validation_error" in error_types or "processing_error" in error_types
        
        # Clean up
        await api_client.delete(f"/api/v1/content/{content_id}", headers=admin_auth_headers)
```

---

**Testing Resources:**
- **Test Documentation**: /docs/testing/
- **CI/CD Pipeline**: /.github/workflows/
- **Test Reports**: Available in CI/CD artifacts
- **Coverage Reports**: https://coverage.ai-knowledge.org

**Testing Contacts:**
- **QA Team**: qa@ai-knowledge.org  
- **Test Automation**: test-automation@ai-knowledge.org
- **Performance Testing**: performance@ai-knowledge.org

**Last Updated**: January 2024  
**Review Cycle**: Quarterly  
**Owner**: Quality Engineering Team