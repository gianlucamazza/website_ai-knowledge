# AI Knowledge Testing Framework

This comprehensive testing framework ensures enterprise-grade quality for the AI Knowledge content pipeline and website.

## Overview

The testing framework provides:
- **Unit Tests**: Component-level testing with >95% coverage requirement
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Load testing and benchmark validation
- **Security Tests**: XSS prevention, input validation, and compliance checks
- **Frontend Tests**: Browser testing with Vitest and Playwright
- **Quality Gates**: Automated quality assurance and compliance validation

## Quick Start

### Local Testing

```bash
# Run all tests
./scripts/run-tests.sh

# Run specific test suites
./scripts/run-tests.sh --unit
./scripts/run-tests.sh --integration
./scripts/run-tests.sh --performance
./scripts/run-tests.sh --security
./scripts/run-tests.sh --frontend

# Skip setup (for rapid iterations)
./scripts/run-tests.sh --unit --skip-setup
```

### Python Tests Only

```bash
# Install dependencies
pip install -r requirements.txt

# Run unit tests with coverage
pytest tests/unit/ --cov=pipelines --cov-report=html

# Run integration tests
pytest tests/integration/ --cov=pipelines --cov-append

# Run performance benchmarks
pytest tests/performance/ --benchmark-only

# Run security tests
pytest tests/security/ -m security
```

### Frontend Tests Only

```bash
cd apps/site

# Unit tests
npm run test:unit

# E2E tests  
npm run test:e2e

# Coverage report
npm run test:unit -- --coverage
```

## Test Structure

```
tests/
├── conftest.py                 # Shared test configuration
├── requirements.txt           # Testing dependencies
├── pytest.ini               # Pytest configuration
├── fixtures/                 # Test data and fixtures
│   └── test_data.py          # Sample data factory
├── unit/                     # Unit tests
│   └── pipeline/             # Pipeline component tests
│       ├── test_ingest.py    # RSS parsing, web scraping
│       ├── test_normalize.py # Content extraction, HTML cleaning  
│       ├── test_dedup.py     # Deduplication algorithms
│       ├── test_enrich.py    # AI summarization, cross-linking
│       └── test_publish.py   # Markdown generation, publishing
├── integration/              # Integration tests
│   ├── test_pipeline_flow.py # End-to-end pipeline workflows
│   └── test_database_ops.py  # Database operations
├── performance/              # Performance tests
│   ├── benchmarks.py         # Performance benchmarks
│   └── test_pipeline_performance.py # Load testing
├── security/                 # Security tests
│   └── test_input_validation.py # XSS, injection prevention
└── reports/                  # Generated test reports
```

## Frontend Tests

```
apps/site/tests/
├── vitest.config.js          # Vitest configuration
├── setup.js                  # Test setup and globals
├── unit/                     # Component unit tests
│   └── components.test.js    # Component testing
└── e2e/                      # End-to-end tests
    ├── playwright.config.js  # Playwright configuration
    └── site-functionality.spec.js # User journey tests
```

## Test Categories

### Unit Tests (`pytest tests/unit/`)
- **Pipeline Components**: RSS parsing, content extraction, deduplication, AI enrichment
- **Data Processing**: HTML cleaning, normalization, quality analysis
- **Content Publishing**: Markdown generation, frontmatter creation
- **Mock External Services**: OpenAI, Anthropic, HTTP requests

### Integration Tests (`pytest tests/integration/`)
- **End-to-End Workflows**: Complete pipeline execution
- **Database Operations**: CRUD, transactions, connection pooling
- **Service Integration**: External API integration testing

### Performance Tests (`pytest tests/performance/`)
- **Component Benchmarks**: Individual component performance
- **Load Testing**: High-volume data processing
- **Memory Usage**: Memory efficiency validation
- **Scalability**: Performance under various data sizes

### Security Tests (`pytest tests/security/`)
- **XSS Prevention**: Cross-site scripting protection
- **Input Validation**: SQL injection, command injection prevention
- **Content Sanitization**: Malicious content filtering
- **GDPR Compliance**: Data privacy validation

### Frontend Tests
- **Unit Tests** (`npm run test:unit`): Component testing with Vitest
- **E2E Tests** (`npm run test:e2e`): Browser automation with Playwright
- **Accessibility**: WCAG compliance validation
- **Performance**: Page load times, Core Web Vitals

## Coverage Requirements

- **Minimum Coverage**: 95% for all components
- **Branch Coverage**: Required for critical paths
- **Integration Coverage**: End-to-end workflow coverage
- **Security Coverage**: All input validation paths

## Performance Benchmarks

| Component | Max Processing Time | Memory Limit |
|-----------|-------------------|--------------|
| RSS Parsing | 2s per feed | 50MB |
| Content Extraction | 1s per 1000 words | 100MB |
| Deduplication | 5s per 1000 articles | 200MB |
| AI Enrichment | 10s per article | 150MB |
| Publishing | 0.5s per article | 25MB |

## Quality Gates

Automated quality gates enforce:
- ✅ Test coverage ≥95%
- ✅ All security tests pass
- ✅ Performance benchmarks meet thresholds
- ✅ Code quality standards (Black, flake8, mypy)
- ✅ Dependency security scanning
- ✅ License compliance

## CI/CD Integration

### GitHub Actions Workflows
- **`test-and-coverage.yml`**: Main test suite execution
- **`quality-gates.yml`**: Code quality and compliance checks

### Coverage Reporting
- **Codecov**: Automated coverage reporting and tracking
- **HTML Reports**: Local coverage visualization
- **Quality Trends**: Historical quality metrics

## Mock Services

The testing framework includes comprehensive mocking for:

### External APIs
```python
# OpenAI API mocking
@pytest.fixture
def mock_openai_client():
    with patch('openai.ChatCompletion.create') as mock:
        mock.return_value = {
            'choices': [{'message': {'content': 'Mocked response'}}]
        }
        yield mock
```

### Database Operations
```python
# Database session mocking
@pytest.fixture
def mock_db_session():
    session = MagicMock()
    session.query.return_value.filter.return_value.first.return_value = None
    return session
```

### HTTP Requests
```python
# HTTP client mocking
@pytest.fixture
def mock_http_client():
    with patch('httpx.AsyncClient') as mock:
        mock.return_value.__aenter__.return_value.get.return_value.text = "Mock content"
        yield mock
```

## Test Data Factory

Realistic test data generation:

```python
from tests.fixtures.test_data import TestDataFactory

# Generate sample articles
articles = TestDataFactory.create_sample_articles(count=10)

# Generate glossary entries  
glossary = TestDataFactory.create_sample_glossary_entries(count=20)

# Generate content sources
sources = TestDataFactory.create_sample_sources(count=5)
```

## Security Testing

Comprehensive security validation:

### XSS Prevention
```python
def test_xss_prevention():
    malicious_input = '<script>alert("XSS")</script>'
    sanitized = content_sanitizer.sanitize_html(malicious_input)
    assert '<script>' not in sanitized
```

### Input Validation
```python
def test_sql_injection_detection():
    malicious_input = "'; DROP TABLE users; --"
    assert input_validator.contains_sql_injection(malicious_input) is True
```

### GDPR Compliance
```python
def test_gdpr_compliance():
    user_data = {'email': 'user@example.com', 'ip': '192.168.1.1'}
    compliance = compliance_checker.check_gdpr_compliance(user_data)
    assert compliance['requires_consent'] is True
```

## Troubleshooting

### Common Issues

1. **Database Connection Failures**
   ```bash
   # Start PostgreSQL locally
   pg_ctl start -D /usr/local/var/postgres
   
   # Or use SQLite fallback
   export DATABASE_URL="sqlite:///tests/temp/test.db"
   ```

2. **Frontend Test Failures**
   ```bash
   # Install Playwright browsers
   cd apps/site
   npx playwright install --with-deps
   ```

3. **Coverage Below Threshold**
   ```bash
   # Generate detailed coverage report
   coverage html
   open htmlcov/index.html
   ```

4. **Performance Test Failures**
   ```bash
   # Run with verbose benchmarking
   pytest tests/performance/ --benchmark-verbose
   ```

### Environment Variables

Required for full test suite:
```bash
export DATABASE_URL="postgresql://localhost:5432/ai_knowledge_test"
export OPENAI_API_KEY="your-api-key"
export ANTHROPIC_API_KEY="your-api-key"
export PYTHONPATH="${PWD}:${PYTHONPATH}"
```

## Contributing

When adding new features:

1. **Write tests first** (TDD approach)
2. **Maintain >95% coverage** for all new code
3. **Include security tests** for input handling
4. **Add performance benchmarks** for processing components
5. **Update fixtures** with relevant test data
6. **Document test scenarios** in docstrings

### Test Naming Conventions

```python
# Unit tests
def test_component_function_expected_behavior():
    pass

# Integration tests  
def test_integration_workflow_success():
    pass

# Performance tests
def test_component_performance_benchmark():
    pass

# Security tests
def test_security_validation_attack_prevention():
    pass
```

## Reporting Issues

For test-related issues:
1. Include full error output and stack trace
2. Specify Python/Node.js versions
3. Include relevant environment variables
4. Attach generated test reports from `tests/reports/`

The testing framework ensures enterprise-grade quality and reliability for the AI Knowledge platform.