# Development Guide

This guide provides comprehensive instructions for setting up a local development environment, understanding the codebase structure, and contributing to the AI Knowledge Website project.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Project Structure](#project-structure)
4. [Local Development](#local-development)
5. [Development Workflow](#development-workflow)
6. [Testing](#testing)
7. [Debugging](#debugging)
8. [Code Quality](#code-quality)
9. [Performance Optimization](#performance-optimization)
10. [Advanced Topics](#advanced-topics)

## Prerequisites

### Required Software

- **Node.js**: Version 18+ with npm 8+
- **Python**: Version 3.9+ with pip
- **PostgreSQL**: Version 14+ for local database
- **Redis**: Version 7+ for caching and job queues
- **Git**: For version control
- **Docker**: Optional, for containerized development

### Recommended Tools

- **VS Code**: With recommended extensions (see `.vscode/extensions.json`)
- **pgAdmin**: For database management
- **Redis Desktop Manager**: For Redis debugging
- **Postman**: For API testing
- **ngrok**: For webhook testing

### System Requirements

- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: At least 10GB free space
- **OS**: macOS, Linux, or Windows with WSL2

## Environment Setup

### Initial Setup

1. **Clone the Repository**:

```bash
git clone https://github.com/your-org/ai-knowledge-website.git
cd ai-knowledge-website
```

2. **Install Dependencies**:

```bash
# Install Python dependencies
pip install -e .
pip install -r requirements-dev.txt

# Install Node.js dependencies
cd apps/site
npm install
cd ../..
```

3. **Environment Configuration**:

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your local configuration
nano .env
```

### Environment Variables

Create a `.env` file with the following configuration:

```bash
# Environment
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/ai_knowledge_dev

# Redis
REDIS_URL=redis://localhost:6379/0

# API Keys (optional for local development)
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here

# Security
SECRET_KEY=your-local-secret-key-here

# Pipeline Configuration
SCRAPING_REQUEST_DELAY=0.5
SCRAPING_CONCURRENT_REQUESTS=3
DEDUP_SIMHASH_THRESHOLD=3

# Local Testing
PYTEST_CURRENT_TEST=true
```

### Database Setup

1. **Create Local Database**:

```bash
# Using PostgreSQL locally
createdb ai_knowledge_dev

# Or using Docker
docker run -d \
  --name ai-knowledge-postgres \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=ai_knowledge_dev \
  -p 5432:5432 \
  postgres:14
```

2. **Initialize Schema**:

```bash
# Run database migrations
python -m pipelines.database.init_db
python -m pipelines.database.migrate
```

3. **Load Sample Data**:

```bash
# Load development fixtures
python -m pipelines.database.load_fixtures --env development
```

### Redis Setup

1. **Install and Start Redis**:

```bash
# macOS with Homebrew
brew install redis
brew services start redis

# Ubuntu/Debian
sudo apt-get install redis-server
sudo systemctl start redis-server

# Docker
docker run -d \
  --name ai-knowledge-redis \
  -p 6379:6379 \
  redis:7
```

2. **Verify Redis Connection**:

```bash
redis-cli ping
# Should return: PONG
```

## Project Structure

### High-Level Overview

```
ai-knowledge-website/
├── apps/                   # Application code
│   └── site/              # Astro frontend application
├── pipelines/             # Python content pipeline
├── security/              # Security modules
├── tests/                 # Test suites
├── scripts/               # Automation scripts
├── docs/                  # Documentation
├── infrastructure/        # Infrastructure as code
└── monitoring/           # Monitoring configurations
```

### Frontend Structure (apps/site/)

```
apps/site/
├── src/
│   ├── components/        # Reusable UI components
│   │   ├── articles/     # Article-specific components
│   │   ├── glossary/     # Glossary components
│   │   ├── search/       # Search functionality
│   │   └── shared/       # Common UI components
│   ├── content/          # Content collections
│   │   ├── config.ts     # Zod schemas for content validation
│   │   ├── articles/     # MDX article files
│   │   ├── glossary/     # Markdown glossary entries
│   │   └── taxonomies/   # Categories and tags
│   ├── layouts/          # Page layout components
│   │   ├── BaseLayout.astro
│   │   └── GlossaryLayout.astro
│   ├── pages/            # Route definitions
│   │   ├── index.astro   # Homepage
│   │   ├── articles/     # Article pages
│   │   └── glossary/     # Glossary pages
│   ├── styles/           # Global styles
│   └── utils/            # Utility functions
├── public/               # Static assets
├── tests/                # Frontend tests
│   ├── unit/            # Component unit tests
│   └── e2e/             # End-to-end tests
└── package.json         # Dependencies and scripts
```

### Backend Structure (pipelines/)

```
pipelines/
├── ingest/               # Content ingestion
│   ├── scrapers/        # Website scrapers
│   ├── apis/            # API integrations
│   ├── feeds/           # RSS/Atom processors
│   └── sources.yaml     # Source configurations
├── normalize/            # Data normalization
│   ├── extractors/      # Content extractors
│   ├── cleaners/        # HTML/text cleaning
│   └── validators/      # Schema validation
├── dedup/                # Duplicate detection
│   ├── simhash.py       # SimHash implementation
│   ├── lsh_index.py     # LSH for similarity
│   └── clustering.py    # Content clustering
├── enrich/               # Content enrichment
│   ├── summarizer.py    # AI-powered summaries
│   ├── cross_linker.py  # Cross-reference detection
│   └── tagger.py        # Automatic tagging
├── publish/              # Content publication
│   ├── markdown_generator.py # Markdown output
│   └── validators.py    # Final validation
├── orchestrators/        # Workflow management
│   └── langgraph/       # LangGraph implementation
├── database/             # Database operations
│   ├── models.py        # SQLAlchemy models
│   ├── connection.py    # Database connections
│   └── migrations/      # Schema migrations
├── monitoring.py         # Metrics and logging
├── config.py            # Configuration management
└── exceptions.py        # Custom exceptions
```

## Local Development

### Starting the Development Environment

1. **Start Backend Services**:

```bash
# Start database and Redis (if using Docker)
docker-compose up -d postgres redis

# Start the content pipeline
python -m pipelines.run_server --host localhost --port 8000

# In another terminal, start background workers
python -m pipelines.run_workers
```

2. **Start Frontend Development Server**:

```bash
cd apps/site
npm run dev
```

3. **Verify Setup**:

Visit the following URLs to ensure everything is working:
- Frontend: http://localhost:4321
- API Health: http://localhost:8000/health
- API Docs: http://localhost:8000/docs

### Development Commands

```bash
# Make commands for common tasks
make install          # Install all dependencies
make dev             # Start development servers
make test            # Run all tests
make lint            # Run code quality checks
make format          # Format code
make health-check    # Verify system health

# Python-specific commands
python -m pipelines.cli ingest --source arxiv  # Test ingestion
python -m pipelines.cli pipeline-status       # Check pipeline
python -m pipelines.database.shell           # Database shell

# Frontend-specific commands
cd apps/site
npm run dev          # Development server
npm run build        # Build for production
npm run preview      # Preview production build
npm run lint         # Lint and fix issues
npm run type-check   # TypeScript validation
```

### Hot Reloading

The development environment supports hot reloading:

- **Frontend**: Astro automatically reloads on file changes
- **Backend**: FastAPI with `--reload` flag restarts on changes
- **Content**: Content collections automatically rebuild on change

### Environment Switching

```bash
# Switch between environments
export ENVIRONMENT=development  # or staging, production
export DEBUG=true              # Enable debug mode

# Use different database
export DATABASE_URL=postgresql://localhost:5432/ai_knowledge_test

# Load environment-specific configuration
python -m pipelines.config.load --env development
```

## Development Workflow

### Feature Development Process

1. **Create Feature Branch**:

```bash
git checkout -b feature/content-quality-scoring
```

2. **Develop and Test Locally**:

```bash
# Write code
# Add tests
make test

# Ensure code quality
make lint
make type-check
```

3. **Test Integration**:

```bash
# Run full pipeline test
make test-integration

# Test with real data
python -m pipelines.cli test-pipeline --source test-data
```

4. **Submit Pull Request**:

```bash
git add .
git commit -m "feat: add content quality scoring algorithm"
git push origin feature/content-quality-scoring
# Create PR through GitHub interface
```

### Code Organization Principles

1. **Separation of Concerns**:
   - Keep data models separate from business logic
   - Separate API handlers from core processing
   - Isolate external service integrations

2. **Configuration Management**:
   - Use environment variables for configuration
   - Keep secrets separate from code
   - Support multiple environments

3. **Error Handling**:
   - Use custom exceptions for domain-specific errors
   - Implement proper logging throughout
   - Handle graceful degradation

### Database Development

1. **Creating Migrations**:

```python
# Create new migration file: migrations/004_add_quality_metrics.py
from alembic import op
import sqlalchemy as sa

def upgrade():
    op.add_column('content_items', 
        sa.Column('quality_score', sa.Float, default=0.0))
    op.create_index('idx_quality_score', 'content_items', ['quality_score'])

def downgrade():
    op.drop_index('idx_quality_score')
    op.drop_column('content_items', 'quality_score')
```

2. **Running Migrations**:

```bash
# Apply migrations
python -m pipelines.database.migrate

# Rollback if needed
python -m pipelines.database.migrate --downgrade -1
```

3. **Database Testing**:

```python
# Use test fixtures
@pytest.fixture
def sample_content_item(db_session):
    item = ContentItem(
        title="Sample AI Article",
        content="Sample content about artificial intelligence",
        source_url="https://example.com"
    )
    db_session.add(item)
    db_session.commit()
    return item

def test_content_item_creation(sample_content_item):
    assert sample_content_item.title == "Sample AI Article"
```

### Frontend Development

1. **Component Development**:

```typescript
// src/components/ArticleCard.astro
---
import type { ArticleData } from '../content/config';

interface Props {
  article: ArticleData;
}

const { article } = Astro.props;
---

<article class="bg-white rounded-lg shadow-md p-6">
  <h2 class="text-xl font-semibold mb-2">{article.title}</h2>
  <p class="text-gray-600 mb-4">{article.description}</p>
  <a href={`/articles/${article.slug}`} class="text-blue-600 hover:underline">
    Read more →
  </a>
</article>
```

2. **Content Schema Development**:

```typescript
// src/content/config.ts
import { z, defineCollection } from 'astro:content';

const articleCollection = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    description: z.string(),
    publishDate: z.date(),
    author: z.string(),
    category: z.string(),
    tags: z.array(z.string()),
    featured: z.boolean().default(false),
    draft: z.boolean().default(false)
  })
});

export const collections = {
  articles: articleCollection
};
```

3. **Testing Components**:

```javascript
// tests/unit/components/ArticleCard.test.js
import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import ArticleCard from '../../../src/components/ArticleCard.astro';

describe('ArticleCard', () => {
  it('renders article title and description', () => {
    const article = {
      title: 'Understanding Neural Networks',
      description: 'Introduction to neural network fundamentals',
      slug: 'understanding-neural-networks'
    };
    
    const { getByText } = render(<ArticleCard article={article} />);
    expect(getByText('Understanding Neural Networks')).toBeDefined();
    expect(getByText('Introduction to neural network fundamentals')).toBeDefined();
  });
});
```

## Testing

### Test Structure

```
tests/
├── unit/                 # Unit tests
│   ├── pipeline/        # Pipeline component tests
│   ├── models/          # Database model tests
│   └── utils/           # Utility function tests
├── integration/         # Integration tests
│   ├── api/            # API endpoint tests
│   ├── pipeline/       # Full pipeline tests
│   └── database/       # Database integration tests
├── performance/         # Performance tests
│   ├── load_tests.py   # Load testing
│   └── benchmarks.py   # Performance benchmarks
├── security/            # Security tests
│   └── test_auth.py    # Authentication tests
├── fixtures/            # Test data
│   └── sample_data.py  # Sample content
└── conftest.py         # Pytest configuration
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test types
make test-unit
make test-integration
make test-performance
make test-security

# Run tests with coverage
pytest --cov=pipelines --cov-report=html

# Run specific test file
pytest tests/unit/pipeline/test_ingest.py -v

# Run tests matching pattern
pytest -k "test_duplicate_detection" -v

# Run tests with debugging
pytest --pdb tests/unit/pipeline/test_dedup.py
```

### Writing Tests

1. **Unit Test Example**:

```python
# tests/unit/pipeline/test_dedup.py
import pytest
from pipelines.dedup.simhash import SimHashDuplicateDetector

class TestSimHashDuplicateDetector:
    
    def test_identical_content_detected(self):
        detector = SimHashDuplicateDetector()
        content1 = "This is a test article about machine learning"
        content2 = "This is a test article about machine learning"
        
        hash1 = detector.compute_hash(content1)
        hash2 = detector.compute_hash(content2)
        
        assert detector.are_duplicates(hash1, hash2)
    
    def test_different_content_not_detected(self):
        detector = SimHashDuplicateDetector()
        content1 = "Article about machine learning"
        content2 = "Article about quantum computing"
        
        hash1 = detector.compute_hash(content1)
        hash2 = detector.compute_hash(content2)
        
        assert not detector.are_duplicates(hash1, hash2)
```

2. **Integration Test Example**:

```python
# tests/integration/test_pipeline_flow.py
import pytest
from pipelines.orchestrators.langgraph.workflow import ContentPipelineWorkflow

@pytest.mark.integration
async def test_full_pipeline_flow(sample_source_data):
    workflow = ContentPipelineWorkflow()
    
    # Start with raw source data
    initial_state = {
        'sources': [sample_source_data],
        'raw_content': [],
        'errors': []
    }
    
    # Run complete pipeline
    final_state = await workflow.run(initial_state)
    
    # Verify results
    assert len(final_state['published_files']) > 0
    assert len(final_state['errors']) == 0
    assert final_state['metadata']['success_rate'] > 0.95
```

3. **Performance Test Example**:

```python
# tests/performance/test_dedup_performance.py
import pytest
import time
from pipelines.dedup.lsh_index import LSHIndex

@pytest.mark.performance
def test_lsh_query_performance():
    lsh = LSHIndex(num_perm=256, threshold=0.8)
    
    # Insert 10,000 items
    start_time = time.time()
    for i in range(10000):
        lsh.insert(f"item_{i}", set([f"word_{j}" for j in range(i, i+100)]))
    insert_time = time.time() - start_time
    
    # Query performance
    start_time = time.time()
    for i in range(100):
        results = lsh.query(set([f"word_{j}" for j in range(i, i+100)]))
    query_time = time.time() - start_time
    
    # Performance assertions
    assert insert_time < 60  # Less than 1 minute for 10k inserts
    assert query_time < 5    # Less than 5 seconds for 100 queries
```

### Test Data Management

```python
# tests/fixtures/sample_data.py
@pytest.fixture
def sample_article_content():
    return {
        'title': 'Introduction to Transformers',
        'content': '''
        Transformers are a type of neural network architecture that has 
        revolutionized natural language processing...
        ''',
        'author': 'Jane Doe',
        'source_url': 'https://example.com/transformers',
        'published_date': '2024-01-01T00:00:00Z',
        'tags': ['machine-learning', 'nlp', 'transformers']
    }

@pytest.fixture
def mock_openai_response():
    return {
        'choices': [{
            'text': 'This is a test summary of the article content.',
            'finish_reason': 'stop'
        }],
        'usage': {'total_tokens': 50}
    }
```

## Debugging

### Backend Debugging

1. **Using Python Debugger**:

```python
# Add breakpoint in code
import pdb; pdb.set_trace()

# Or use IPython debugger
import IPdb; ipdb.set_trace()

# Remote debugging with debugpy
import debugpy
debugpy.listen(5678)
debugpy.wait_for_client()
```

2. **Database Query Debugging**:

```python
# Enable SQL query logging
import logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# Use query explain
from pipelines.database.connection import get_session
with get_session() as session:
    result = session.execute(
        text("EXPLAIN ANALYZE SELECT * FROM content_items WHERE status = :status"),
        {"status": "published"}
    )
    print(result.fetchall())
```

3. **Pipeline State Debugging**:

```python
# Add debugging to LangGraph workflow
from langgraph import StateGraph

def debug_node(state):
    print(f"Current state: {state}")
    return state

workflow = StateGraph(ContentPipelineState)
workflow.add_node("debug", debug_node)
```

### Frontend Debugging

1. **Astro Development Tools**:

```bash
# Enable verbose logging
DEBUG=astro:* npm run dev

# Build with debugging info
npm run build -- --verbose
```

2. **Browser Debugging**:

```javascript
// Add debugging to components
console.log('Component props:', Astro.props);
console.error('Component error:', error);

// Use browser dev tools
debugger; // Breaks in browser
```

### Production Debugging

1. **Remote Debugging**:

```bash
# Port forward to debug pod
kubectl port-forward deployment/ai-knowledge-pipeline 5678:5678 -n ai-knowledge-prod

# Connect with VS Code or PyCharm debugger
```

2. **Log Analysis**:

```bash
# Follow logs in real-time
kubectl logs -f deployment/ai-knowledge-pipeline -n ai-knowledge-prod

# Search logs for errors
kubectl logs deployment/ai-knowledge-pipeline -n ai-knowledge-prod --since=1h | grep ERROR

# Export logs for analysis
kubectl logs deployment/ai-knowledge-pipeline -n ai-knowledge-prod > pipeline.log
```

## Code Quality

### Linting and Formatting

1. **Python Code Quality**:

```bash
# Format with Black
black pipelines/ tests/

# Sort imports
isort pipelines/ tests/

# Lint with flake8
flake8 pipelines/ tests/

# Type checking with mypy
mypy pipelines/

# Security checks with bandit
bandit -r pipelines/
```

2. **Frontend Code Quality**:

```bash
cd apps/site

# Lint and fix
npm run lint
npm run lint:fix

# Type checking
npm run type-check

# Format with Prettier (if configured)
npx prettier --write src/
```

### Pre-commit Hooks

Install pre-commit hooks to ensure code quality:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### Code Review Guidelines

1. **What to Review**:
   - Code correctness and logic
   - Test coverage and quality
   - Performance implications
   - Security considerations
   - Documentation updates

2. **Review Checklist**:
   - [ ] Tests added for new functionality
   - [ ] Error handling implemented
   - [ ] Performance impact considered
   - [ ] Security implications reviewed
   - [ ] Documentation updated
   - [ ] Breaking changes documented

## Performance Optimization

### Profiling

1. **Python Profiling**:

```python
# Profile with cProfile
python -m cProfile -o profile.stats -m pipelines.run_pipeline

# Analyze with snakeviz
pip install snakeviz
snakeviz profile.stats

# Memory profiling
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Function code here
    pass
```

2. **Database Performance**:

```sql
-- Enable query timing
\timing on

-- Explain query plans
EXPLAIN ANALYZE SELECT * FROM content_items WHERE status = 'published';

-- Check index usage
SELECT schemaname, tablename, attname, n_distinct, correlation 
FROM pg_stats 
WHERE tablename = 'content_items';
```

3. **Frontend Performance**:

```bash
# Lighthouse audit
npx lighthouse http://localhost:4321 --output=json --output-path=lighthouse.json

# Bundle analysis
npm run build
npx astro build --analyze
```

### Optimization Techniques

1. **Database Optimization**:

```python
# Use connection pooling
from sqlalchemy.pool import StaticPool
engine = create_engine(
    DATABASE_URL,
    poolclass=StaticPool,
    pool_size=20,
    max_overflow=30
)

# Batch operations
def batch_insert_content(items, batch_size=100):
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        session.bulk_insert_mappings(ContentItem, batch)
        session.commit()
```

2. **Caching Strategy**:

```python
from functools import lru_cache
from pipelines.cache import redis_client

# Memory caching
@lru_cache(maxsize=1000)
def get_category_mapping(category_id):
    # Expensive computation
    return category_mapping

# Redis caching
def cached_function(key, func, timeout=3600):
    result = redis_client.get(key)
    if result is None:
        result = func()
        redis_client.setex(key, timeout, json.dumps(result))
    else:
        result = json.loads(result)
    return result
```

3. **Async Processing**:

```python
import asyncio
import aiohttp

async def process_urls_concurrently(urls, max_concurrent=10):
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_url(url):
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return await response.text()
    
    tasks = [process_url(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return results
```

## Advanced Topics

### Custom Pipeline Stages

Create custom pipeline stages by implementing the required interface:

```python
# pipelines/custom/my_stage.py
from pipelines.base import PipelineStage
from pipelines.models import ProcessingState

class MyCustomStage(PipelineStage):
    
    def __init__(self, config):
        self.config = config
    
    async def process(self, state: ProcessingState) -> ProcessingState:
        """Process content items through custom logic."""
        for item in state.content_items:
            # Custom processing logic
            processed_item = self.custom_logic(item)
            item.metadata.update(processed_item.metadata)
        
        return state
    
    def custom_logic(self, content_item):
        # Implement your custom logic here
        pass
```

### Plugin Development

Extend functionality with plugins:

```python
# pipelines/plugins/quality_scorer.py
from pipelines.interfaces import ContentPlugin

class QualityScorerPlugin(ContentPlugin):
    
    def __init__(self, config):
        self.model = self.load_quality_model()
    
    def process_content(self, content_item):
        """Add quality score to content item."""
        score = self.model.predict(content_item.content)
        content_item.metadata['quality_score'] = score
        return content_item
    
    def load_quality_model(self):
        # Load your quality scoring model
        pass

# Register plugin
from pipelines.registry import plugin_registry
plugin_registry.register('quality_scorer', QualityScorerPlugin)
```

### Custom Content Sources

Add new content sources:

```python
# pipelines/ingest/sources/custom_api.py
from pipelines.ingest.base import ContentSource
from pipelines.models import RawContent

class CustomAPISource(ContentSource):
    
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
    
    async def fetch_content(self) -> List[RawContent]:
        """Fetch content from custom API."""
        headers = {'Authorization': f'Bearer {self.api_key}'}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f'{self.base_url}/articles', headers=headers) as response:
                data = await response.json()
                
                content_items = []
                for item in data['articles']:
                    content_items.append(RawContent(
                        title=item['title'],
                        content=item['content'],
                        source_url=item['url'],
                        metadata=item.get('metadata', {})
                    ))
                
                return content_items
```

### Monitoring and Observability

Add custom metrics and tracing:

```python
# pipelines/monitoring/custom_metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time
import functools

# Define metrics
PROCESSING_DURATION = Histogram(
    'pipeline_processing_duration_seconds',
    'Time spent processing content',
    ['stage', 'source']
)

CONTENT_ITEMS_PROCESSED = Counter(
    'pipeline_content_items_total',
    'Total number of content items processed',
    ['stage', 'status']
)

ACTIVE_JOBS = Gauge(
    'pipeline_active_jobs',
    'Number of currently active pipeline jobs'
)

# Monitoring decorator
def monitor_stage(stage_name):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            ACTIVE_JOBS.inc()
            
            try:
                result = await func(*args, **kwargs)
                CONTENT_ITEMS_PROCESSED.labels(
                    stage=stage_name, 
                    status='success'
                ).inc()
                return result
            except Exception as e:
                CONTENT_ITEMS_PROCESSED.labels(
                    stage=stage_name, 
                    status='error'
                ).inc()
                raise
            finally:
                duration = time.time() - start_time
                PROCESSING_DURATION.labels(stage=stage_name).observe(duration)
                ACTIVE_JOBS.dec()
        
        return wrapper
    return decorator

# Usage
@monitor_stage('normalize')
async def normalize_content(content_items):
    # Processing logic
    pass
```

---

**Next Steps**: 
- Review the [Contributing Guide](CONTRIBUTING.md) for code submission guidelines
- Check the [Testing Guide](TESTING_GUIDE.md) for comprehensive testing practices
- Explore the [API Documentation](API_DOCUMENTATION.md) for integration details

**Last Updated**: January 2024  
**Version**: 1.0.0