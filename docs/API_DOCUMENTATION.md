# API Documentation

This document provides comprehensive documentation for the AI Knowledge Website API endpoints, including the content pipeline API, webhook endpoints, and administrative interfaces.

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Pipeline API](#pipeline-api)
4. [Content Management API](#content-management-api)
5. [Administrative API](#administrative-api)
6. [Webhook Endpoints](#webhook-endpoints)
7. [Error Handling](#error-handling)
8. [Rate Limiting](#rate-limiting)
9. [SDK and Examples](#sdk-and-examples)

## Overview

The AI Knowledge Website exposes several REST APIs for content pipeline management, administrative operations, and webhook integrations. All APIs follow REST conventions and return JSON responses.

**Base URL**: `https://api.ai-knowledge.org/v1`

**Content Types**:
- Request: `application/json`
- Response: `application/json`

**API Versioning**: Version specified in URL path (`/v1/`)

## Authentication

### API Key Authentication

All API endpoints require authentication using API keys passed in the `Authorization` header.

```http
Authorization: Bearer YOUR_API_KEY
```

### Token Types

- **Pipeline Token**: Access to content pipeline operations
- **Admin Token**: Full administrative access
- **Webhook Token**: Limited access for webhook endpoints

### Getting API Keys

API keys are managed through the administrative interface or can be generated using the CLI:

```bash
# Generate pipeline API key
python -m pipelines.cli generate-key --type pipeline

# Generate admin API key (requires existing admin access)
python -m pipelines.cli generate-key --type admin --user admin@example.com
```

## Pipeline API

### Trigger Content Ingestion

Initiates content ingestion from configured sources.

```http
POST /v1/pipeline/ingest
```

**Request Body:**

```json
{
  "source_ids": ["arxiv", "github-trending"],
  "priority": "high",
  "filters": {
    "date_range": {
      "start": "2024-01-01",
      "end": "2024-01-31"
    },
    "categories": ["machine-learning", "nlp"]
  }
}
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| source_ids | array | No | Specific sources to ingest (default: all active) |
| priority | string | No | Job priority: `low`, `normal`, `high` (default: `normal`) |
| filters | object | No | Additional filtering criteria |

**Response:**

```json
{
  "job_id": "ingest_20240106_123456",
  "status": "queued",
  "created_at": "2024-01-06T12:34:56Z",
  "estimated_duration": "15m",
  "sources": ["arxiv", "github-trending"]
}
```

### Get Pipeline Status

Retrieves the status of a specific pipeline job.

```http
GET /v1/pipeline/jobs/{job_id}
```

**Response:**

```json
{
  "job_id": "ingest_20240106_123456",
  "status": "running",
  "progress": {
    "current_stage": "normalize",
    "completed_stages": ["ingest"],
    "total_items": 150,
    "processed_items": 75,
    "failed_items": 2
  },
  "started_at": "2024-01-06T12:35:00Z",
  "estimated_completion": "2024-01-06T12:50:00Z",
  "logs": [
    {
      "timestamp": "2024-01-06T12:35:30Z",
      "level": "info",
      "message": "Started processing source: arxiv"
    }
  ]
}
```

### List Pipeline Jobs

Retrieves a list of pipeline jobs with optional filtering.

```http
GET /v1/pipeline/jobs
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| status | string | Filter by job status |
| limit | integer | Number of results (default: 50, max: 200) |
| offset | integer | Pagination offset |
| date_from | string | Filter jobs from date (ISO 8601) |
| date_to | string | Filter jobs to date (ISO 8601) |

**Response:**

```json
{
  "jobs": [
    {
      "job_id": "ingest_20240106_123456",
      "status": "completed",
      "created_at": "2024-01-06T12:34:56Z",
      "completed_at": "2024-01-06T12:48:23Z",
      "duration": "13m27s",
      "items_processed": 148,
      "items_failed": 2
    }
  ],
  "total": 25,
  "limit": 50,
  "offset": 0
}
```

### Cancel Pipeline Job

Cancels a running or queued pipeline job.

```http
DELETE /v1/pipeline/jobs/{job_id}
```

**Response:**

```json
{
  "job_id": "ingest_20240106_123456",
  "status": "cancelled",
  "cancelled_at": "2024-01-06T12:40:15Z"
}
```

### Validate Content

Validates content against schema requirements before processing.

```http
POST /v1/pipeline/validate
```

**Request Body:**

```json
{
  "content_type": "article",
  "data": {
    "title": "Introduction to Transformers",
    "content": "Content body here...",
    "author": "John Doe",
    "source_url": "https://example.com/article"
  }
}
```

**Response:**

```json
{
  "valid": true,
  "errors": [],
  "warnings": [
    "Missing optional field: publish_date"
  ],
  "normalized_data": {
    "title": "Introduction to Transformers",
    "content": "Content body here...",
    "author": "John Doe",
    "source_url": "https://example.com/article",
    "content_hash": "sha256:abc123..."
  }
}
```

## Content Management API

### List Content Items

Retrieves a list of content items with filtering and search capabilities.

```http
GET /v1/content
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| type | string | Content type: `article`, `glossary` |
| status | string | Content status: `published`, `draft`, `archived` |
| search | string | Full-text search query |
| category | string | Filter by category |
| tags | string | Comma-separated list of tags |
| limit | integer | Number of results (default: 50) |
| offset | integer | Pagination offset |

**Response:**

```json
{
  "items": [
    {
      "id": "art_20240106_001",
      "type": "article",
      "title": "Introduction to Transformers",
      "status": "published",
      "created_at": "2024-01-06T10:15:30Z",
      "updated_at": "2024-01-06T10:20:45Z",
      "tags": ["nlp", "transformers", "deep-learning"],
      "category": "machine-learning",
      "author": "John Doe",
      "source_url": "https://example.com/article"
    }
  ],
  "total": 1250,
  "limit": 50,
  "offset": 0
}
```

### Get Content Item

Retrieves detailed information about a specific content item.

```http
GET /v1/content/{item_id}
```

**Response:**

```json
{
  "id": "art_20240106_001",
  "type": "article",
  "title": "Introduction to Transformers",
  "content": "Full article content...",
  "status": "published",
  "metadata": {
    "word_count": 1500,
    "reading_time": "6 minutes",
    "quality_score": 0.92,
    "last_updated": "2024-01-06T10:20:45Z"
  },
  "taxonomies": {
    "category": "machine-learning",
    "tags": ["nlp", "transformers", "deep-learning"],
    "topics": ["artificial-intelligence"]
  },
  "relationships": {
    "related_articles": ["art_20240105_042", "art_20240104_128"],
    "glossary_terms": ["transformer", "attention-mechanism"],
    "cited_sources": ["https://arxiv.org/abs/1706.03762"]
  },
  "processing_info": {
    "ingested_at": "2024-01-06T10:15:30Z",
    "processed_at": "2024-01-06T10:18:22Z",
    "pipeline_version": "1.2.3",
    "simhash": "18364758544493064720"
  }
}
```

### Update Content Status

Updates the publication status of a content item.

```http
PATCH /v1/content/{item_id}/status
```

**Request Body:**

```json
{
  "status": "published",
  "reason": "Content review completed"
}
```

**Response:**

```json
{
  "id": "art_20240106_001",
  "status": "published",
  "updated_at": "2024-01-06T14:30:00Z"
}
```

### Delete Content Item

Removes a content item from the system.

```http
DELETE /v1/content/{item_id}
```

**Response:**

```json
{
  "id": "art_20240106_001",
  "status": "deleted",
  "deleted_at": "2024-01-06T14:35:00Z"
}
```

## Administrative API

### System Health

Retrieves system health and status information.

```http
GET /v1/admin/health
```

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2024-01-06T14:30:00Z",
  "services": {
    "database": {
      "status": "healthy",
      "response_time": "5ms",
      "connections": {
        "active": 12,
        "idle": 18,
        "total": 30
      }
    },
    "redis": {
      "status": "healthy",
      "response_time": "2ms",
      "memory_usage": "45%"
    },
    "pipeline": {
      "status": "healthy",
      "active_jobs": 3,
      "queue_size": 15
    }
  },
  "metrics": {
    "content_items": 5420,
    "successful_ingestions": 1245,
    "duplicate_detection_accuracy": 0.987
  }
}
```

### Configuration Management

Retrieves or updates system configuration.

```http
GET /v1/admin/config
```

**Response:**

```json
{
  "scraping": {
    "request_delay": 1.0,
    "concurrent_requests": 5,
    "respect_robots_txt": true
  },
  "deduplication": {
    "simhash_threshold": 3,
    "minhash_threshold": 0.85
  },
  "sources": {
    "active_count": 15,
    "total_count": 20
  }
}
```

```http
PATCH /v1/admin/config
```

**Request Body:**

```json
{
  "scraping": {
    "request_delay": 1.5
  }
}
```

### Source Management

Manage content sources.

```http
GET /v1/admin/sources
```

**Response:**

```json
{
  "sources": [
    {
      "id": "arxiv",
      "name": "ArXiv Papers",
      "type": "rss",
      "url": "http://export.arxiv.org/rss/cs.AI",
      "status": "active",
      "last_crawled": "2024-01-06T12:00:00Z",
      "success_rate": 0.95,
      "items_ingested": 450
    }
  ]
}
```

```http
POST /v1/admin/sources
```

**Request Body:**

```json
{
  "name": "New AI Blog",
  "type": "rss",
  "url": "https://blog.ai-example.com/feed.xml",
  "config": {
    "update_frequency": "daily",
    "category": "blog-posts"
  }
}
```

## Webhook Endpoints

### Content Published

Triggered when new content is published.

```http
POST /webhooks/content/published
```

**Payload:**

```json
{
  "event": "content.published",
  "timestamp": "2024-01-06T14:30:00Z",
  "data": {
    "id": "art_20240106_001",
    "type": "article",
    "title": "Introduction to Transformers",
    "url": "https://ai-knowledge.org/articles/introduction-to-transformers",
    "tags": ["nlp", "transformers"],
    "category": "machine-learning"
  }
}
```

### Pipeline Completed

Triggered when a pipeline job completes.

```http
POST /webhooks/pipeline/completed
```

**Payload:**

```json
{
  "event": "pipeline.completed",
  "timestamp": "2024-01-06T12:48:23Z",
  "data": {
    "job_id": "ingest_20240106_123456",
    "status": "completed",
    "duration": "13m27s",
    "statistics": {
      "items_processed": 148,
      "items_published": 142,
      "duplicates_detected": 6,
      "items_failed": 2
    }
  }
}
```

### Quality Alert

Triggered when content quality issues are detected.

```http
POST /webhooks/quality/alert
```

**Payload:**

```json
{
  "event": "quality.alert",
  "timestamp": "2024-01-06T14:45:00Z",
  "data": {
    "severity": "warning",
    "type": "duplicate_detection_accuracy",
    "message": "Duplicate detection accuracy below threshold",
    "metrics": {
      "current_accuracy": 0.96,
      "threshold": 0.98,
      "affected_items": 15
    }
  }
}
```

## Error Handling

### Error Response Format

All API errors follow a consistent format:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The request contains invalid parameters",
    "details": [
      {
        "field": "source_ids",
        "message": "Invalid source ID: invalid-source"
      }
    ],
    "request_id": "req_20240106_001234"
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| INVALID_REQUEST | 400 | Request contains invalid parameters |
| UNAUTHORIZED | 401 | Authentication required or invalid |
| FORBIDDEN | 403 | Insufficient permissions |
| NOT_FOUND | 404 | Requested resource not found |
| RATE_LIMITED | 429 | Rate limit exceeded |
| INTERNAL_ERROR | 500 | Internal server error |
| SERVICE_UNAVAILABLE | 503 | Service temporarily unavailable |

### Retry Logic

For 5xx errors and rate limiting (429), implement exponential backoff:

```python
import time
import requests
from typing import Optional

def api_request_with_retry(url: str, max_retries: int = 3) -> Optional[requests.Response]:
    for attempt in range(max_retries):
        try:
            response = requests.get(url)
            if response.status_code < 500:
                return response
        except requests.RequestException:
            pass
        
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # Exponential backoff
    
    return None
```

## Rate Limiting

API endpoints are rate-limited to ensure fair usage and system stability.

### Rate Limit Headers

All responses include rate limit information:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 995
X-RateLimit-Reset: 1704552000
X-RateLimit-Window: 3600
```

### Rate Limit Tiers

| API Type | Requests per Hour | Burst Limit |
|----------|-------------------|-------------|
| Pipeline API | 100 | 20 |
| Content API | 1000 | 50 |
| Admin API | 500 | 25 |
| Webhooks | 10000 | 100 |

### Rate Limit Exceeded

When rate limits are exceeded, the API returns:

```http
HTTP/1.1 429 Too Many Requests
Content-Type: application/json

{
  "error": {
    "code": "RATE_LIMITED",
    "message": "Rate limit exceeded",
    "details": {
      "limit": 1000,
      "window": 3600,
      "reset_at": "2024-01-06T15:00:00Z"
    }
  }
}
```

## SDK and Examples

### Python SDK

Install the official Python SDK:

```bash
pip install ai-knowledge-sdk
```

**Basic Usage:**

```python
from ai_knowledge_sdk import AIKnowledgeClient

# Initialize client
client = AIKnowledgeClient(api_key="your_api_key")

# Trigger content ingestion
job = client.pipeline.ingest(source_ids=["arxiv"])
print(f"Job ID: {job.job_id}")

# Monitor job progress
status = client.pipeline.get_job_status(job.job_id)
print(f"Status: {status.status}")

# List content items
articles = client.content.list(type="article", limit=10)
for article in articles:
    print(f"Title: {article.title}")
```

### JavaScript/TypeScript SDK

Install the JavaScript SDK:

```bash
npm install @ai-knowledge/sdk
```

**Basic Usage:**

```typescript
import { AIKnowledgeClient } from '@ai-knowledge/sdk';

// Initialize client
const client = new AIKnowledgeClient({
  apiKey: 'your_api_key',
  baseUrl: 'https://api.ai-knowledge.org/v1'
});

// Trigger content ingestion
const job = await client.pipeline.ingest({
  sourceIds: ['arxiv'],
  priority: 'high'
});

console.log(`Job ID: ${job.jobId}`);

// List content items
const articles = await client.content.list({
  type: 'article',
  limit: 10
});

articles.items.forEach(article => {
  console.log(`Title: ${article.title}`);
});
```

### cURL Examples

**Trigger Ingestion:**

```bash
curl -X POST https://api.ai-knowledge.org/v1/pipeline/ingest \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "source_ids": ["arxiv"],
    "priority": "high"
  }'
```

**Get Job Status:**

```bash
curl https://api.ai-knowledge.org/v1/pipeline/jobs/ingest_20240106_123456 \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**List Articles:**

```bash
curl "https://api.ai-knowledge.org/v1/content?type=article&limit=10" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## Support

- **API Issues**: Report issues on GitHub or contact api-support@ai-knowledge.org
- **Rate Limit Increases**: Contact support with usage requirements
- **SDK Documentation**: Visit the SDK-specific documentation repositories

---

**Last Updated**: January 2024  
**API Version**: v1.0.0