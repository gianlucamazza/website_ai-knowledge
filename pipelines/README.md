# AI Knowledge Content Pipeline

A robust, scalable content pipeline for ingesting, processing, and publishing AI/ML knowledge content with deduplication, enrichment, and quality assurance capabilities.

## Architecture Overview

The pipeline follows a 5-stage architecture with event-driven processing and PostgreSQL state persistence:

```
Ingest → Normalize → Dedup → Enrich → Publish
   ↓        ↓         ↓       ↓        ↓
 [State Management & Orchestration via LangGraph]
```

### Core Features

- **Ethical Web Scraping**: Respects robots.txt, implements rate limiting, and follows best practices
- **Content Normalization**: Advanced HTML cleaning, readability analysis, and content extraction
- **Duplicate Detection**: SimHash + MinHash LSH with <2% false positive rate
- **AI-Powered Enrichment**: Content summarization, cross-linking, and metadata enhancement
- **Quality Assurance**: Content validation, quality scoring, and human-in-the-loop review
- **Monitoring & Observability**: Comprehensive logging, metrics, and health checking

## Quick Start

### Prerequisites

- Python 3.9+
- PostgreSQL 12+
- (Optional) OpenAI/Anthropic API keys for enrichment

### Installation

1. **Install dependencies**:
```bash
cd pipelines/
pip install -r requirements.txt
```

2. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Initialize database**:
```bash
python run_graph.py sources --sync
```

4. **Run the pipeline**:
```bash
# Full pipeline
python run_graph.py run

# Individual stages
python run_graph.py ingest
python run_graph.py publish --all
```

## Configuration

### Environment Variables

Create a `.env` file in the pipelines directory:

```env
# Database
DATABASE__HOST=localhost
DATABASE__PORT=5432
DATABASE__DATABASE=ai_knowledge
DATABASE__USERNAME=postgres
DATABASE__PASSWORD=your_password

# AI Services (Optional)
ENRICHMENT__OPENAI_API_KEY=your_openai_key
ENRICHMENT__ANTHROPIC_API_KEY=your_anthropic_key

# Scraping
SCRAPING__USER_AGENT=AI-Knowledge-Bot/1.0
SCRAPING__REQUEST_DELAY=1.0
SCRAPING__MAX_RETRIES=3

# Logging
LOGGING__LEVEL=INFO
LOGGING__FORMAT=json
```

### Source Configuration

Edit `ingest/sources.yaml` to configure content sources:

```yaml
sources:
  - name: "arxiv_cs_ai"
    type: "rss"
    base_url: "http://export.arxiv.org/rss/cs.AI"
    config:
      categories: ["research", "ai"]
      crawl_frequency: 3600
      max_articles_per_run: 50
    active: true
```

## Pipeline Stages

### 1. Ingest Stage

**Purpose**: Discover and fetch content from configured sources

**Components**:
- `SourceManager`: Orchestrates ingestion from multiple sources
- `EthicalScraper`: Handles web scraping with robots.txt compliance
- `RSSParser`: Parses RSS/Atom feeds for content discovery

**Key Features**:
- Respects robots.txt and rate limits
- RSS/Atom feed parsing
- Sitemap crawling
- Concurrent processing with backoff
- Source scheduling and configuration

### 2. Normalize Stage

**Purpose**: Clean and extract structured content from raw HTML

**Components**:
- `HTMLCleaner`: Removes unwanted elements and normalizes HTML
- `ContentExtractor`: Extracts main content using readability algorithms

**Key Features**:
- Advanced HTML cleaning and sanitization
- Main content extraction using multiple algorithms
- Quality scoring and readability analysis
- Language detection
- Metadata extraction (title, author, date, etc.)

### 3. Dedup Stage

**Purpose**: Identify and handle duplicate or near-duplicate content

**Components**:
- `SimHashDeduplicator`: SimHash-based similarity detection
- `LSHIndex`: MinHash LSH for efficient similarity search

**Key Features**:
- Dual-algorithm approach (SimHash + MinHash LSH)
- Configurable similarity thresholds
- <2% false positive rate
- Efficient indexing for large content volumes
- Cluster detection for related content

### 4. Enrich Stage

**Purpose**: Enhance content with summaries, cross-links, and metadata

**Components**:
- `ContentSummarizer`: AI-powered content summarization
- `CrossLinker`: Automatic cross-linking between related articles

**Key Features**:
- Multi-provider AI summarization (OpenAI, Anthropic)
- Intelligent cross-linking based on content similarity
- Keyword and topic extraction
- Content classification and tagging
- Human-in-the-loop review for quality assurance

### 5. Publish Stage

**Purpose**: Generate markdown files with proper frontmatter for the Astro site

**Components**:
- `MarkdownGenerator`: Creates markdown files with frontmatter

**Key Features**:
- Astro content collection compatibility
- Zod schema validation
- Automatic taxonomy generation
- SEO optimization
- Source attribution and licensing

## CLI Usage

The pipeline provides a comprehensive CLI interface:

### Run Complete Pipeline
```bash
# Run all stages
python run_graph.py run

# Run with filters
python run_graph.py run --source arxiv_cs_ai --stage normalize,enrich

# Custom configuration
python run_graph.py run --batch-size 20 --max-retries 5 --name "daily_run"
```

### Individual Stages
```bash
# Ingest content
python run_graph.py ingest --source arxiv_cs_ai

# Publish articles
python run_graph.py publish --all
python run_graph.py publish --article article-id-1 --article article-id-2
```

### Monitoring and Status
```bash
# Show pipeline status
python run_graph.py status
python run_graph.py status --recent 20
python run_graph.py status --run-id specific-run-id

# Manage sources
python run_graph.py sources --list
python run_graph.py sources --sync

# Show workflow
python run_graph.py workflow
```

## Monitoring and Observability

### Logging

The pipeline uses structured logging with multiple output formats:

```python
from pipelines import setup_pipeline_logging, get_pipeline_logger

# Setup logging (call once at startup)
setup_pipeline_logging()

# Get logger for component
logger = get_pipeline_logger("my_component")
logger.info("Processing started", article_id="123", stage="normalize")
```

**Log Files**:
- `logs/pipeline.log`: Main application log
- `logs/errors.log`: Error-only log with stack traces  
- `logs/performance.log`: Performance metrics and timing

### Metrics

Prometheus metrics are available at `/metrics` endpoint:

**Key Metrics**:
- `pipeline_articles_processed_total`: Articles processed by stage/status
- `pipeline_stage_duration_seconds`: Processing time per stage
- `pipeline_errors_total`: Error counts by type/stage
- `pipeline_quality_score_average`: Average content quality score

### Health Checks

Built-in health checking monitors:
- Database connectivity and performance
- Content source availability
- Disk space usage
- Recent pipeline performance
- Error rates and trends

```python
from pipelines import health_checker

# Get health status
health_status = await health_checker.check_health()
```

## Development

### Project Structure

```
pipelines/
├── __init__.py              # Package initialization
├── config.py               # Configuration management
├── requirements.txt        # Python dependencies
├── run_graph.py           # CLI interface
├── logging_config.py      # Logging setup
├── exceptions.py          # Custom exceptions
├── monitoring.py          # Metrics and monitoring
├── database/              # Database models and connection
│   ├── models.py         # SQLAlchemy models
│   └── connection.py     # Database connection management
├── ingest/               # Content ingestion
│   ├── sources.yaml      # Source configuration
│   ├── scraper.py       # Web scraping
│   ├── rss_parser.py    # RSS/feed parsing
│   └── source_manager.py # Source orchestration
├── normalize/            # Content normalization
│   ├── html_cleaner.py  # HTML cleaning
│   └── content_extractor.py # Content extraction
├── dedup/               # Deduplication
│   ├── simhash.py       # SimHash implementation
│   └── lsh_index.py     # MinHash LSH
├── enrich/              # Content enrichment
│   ├── summarizer.py    # AI summarization
│   └── cross_linker.py  # Cross-linking
├── publish/             # Publishing
│   └── markdown_generator.py # Markdown generation
└── orchestrators/       # Workflow orchestration
    └── langgraph/       # LangGraph implementation
        ├── workflow.py  # Workflow definition
        └── nodes.py     # Processing nodes
```

### Adding New Sources

1. **Add to sources.yaml**:
```yaml
sources:
  - name: "my_new_source"
    type: "rss"
    base_url: "https://example.com/feed.xml"
    config:
      categories: ["example"]
      crawl_frequency: 7200
    active: true
```

2. **Sync to database**:
```bash
python run_graph.py sources --sync
```

### Extending Pipeline Stages

To add custom processing to a stage:

```python
from pipelines.orchestrators.langgraph.nodes import PipelineNodes

class CustomPipelineNodes(PipelineNodes):
    async def custom_normalize_step(self, article):
        # Custom normalization logic
        return processed_article

# Use in workflow
custom_nodes = CustomPipelineNodes()
```

### Testing

```bash
# Run unit tests
pytest tests/

# Run integration tests
pytest tests/integration/

# Run with coverage
pytest --cov=pipelines tests/
```

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY pipelines/ ./pipelines/
COPY requirements.txt .

RUN pip install -r requirements.txt

CMD ["python", "pipelines/run_graph.py", "run"]
```

### Kubernetes Deployment

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: content-pipeline
spec:
  schedule: "0 */6 * * *"  # Every 6 hours
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: pipeline
            image: ai-knowledge/content-pipeline:latest
            command: ["python", "run_graph.py", "run"]
            env:
            - name: DATABASE__HOST
              value: "postgres-service"
```

## Performance Tuning

### Database Optimization

```sql
-- Recommended indexes
CREATE INDEX idx_article_stage_status ON articles(current_stage, status);
CREATE INDEX idx_article_source_url ON articles(source_id, url);
CREATE INDEX idx_article_created_at ON articles(created_at);
CREATE INDEX idx_pipeline_runs_started ON pipeline_runs(started_at);
```

### Memory Optimization

- Adjust `batch_size` based on available memory
- Use `max_concurrent_requests` to control scraping load
- Configure database connection pooling

### Scaling

- Run multiple pipeline instances with different source filters
- Use Redis for distributed deduplication indices
- Implement horizontal scaling with message queues

## Troubleshooting

### Common Issues

**Database Connection Errors**:
```bash
# Check database connectivity
python run_graph.py status

# Verify configuration
python -c "from pipelines.config import config; print(config.database)"
```

**High Memory Usage**:
- Reduce batch_size in configuration
- Enable memory profiling: `PYTHONMALLOC=debug`
- Monitor memory usage: `python run_graph.py status`

**Slow Processing**:
- Check database performance and indices
- Adjust concurrent processing limits
- Monitor metrics for bottlenecks

**Content Quality Issues**:
- Review quality scores: `python run_graph.py status`
- Adjust quality thresholds in configuration
- Check source content and extraction rules

### Debug Mode

```bash
# Enable debug logging
python run_graph.py run --debug

# Verbose output
python run_graph.py run --verbose

# Single article processing
python run_graph.py normalize --article article-id
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Run quality checks: `black`, `isort`, `mypy`
5. Submit pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting guide
- Review logs in the `logs/` directory
- Monitor pipeline health with `python run_graph.py status`