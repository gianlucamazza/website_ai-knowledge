# Monitoring Guide

This comprehensive guide covers monitoring, observability, and alerting for the AI Knowledge Website system, including metrics, logging, tracing, and operational procedures.

## Table of Contents

1. [Monitoring Overview](#monitoring-overview)
2. [Metrics and KPIs](#metrics-and-kpis)
3. [Logging Strategy](#logging-strategy)
4. [Alerting Framework](#alerting-framework)
5. [Dashboards and Visualization](#dashboards-and-visualization)
6. [Performance Monitoring](#performance-monitoring)
7. [Application Monitoring](#application-monitoring)
8. [Infrastructure Monitoring](#infrastructure-monitoring)
9. [Security Monitoring](#security-monitoring)
10. [Operational Procedures](#operational-procedures)

## Monitoring Overview

### Monitoring Architecture

The monitoring system follows a three-pillars approach: **Metrics**, **Logging**, and **Tracing**.

```
┌─────────────────────────────────────────────────────────────────┐
│                      Data Collection                            │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐  │
│  │     Metrics     │ │      Logs       │ │     Traces      │  │
│  │   (Prometheus)  │ │ (Elasticsearch) │ │    (Jaeger)     │  │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│                    Data Processing                              │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐  │
│  │    Grafana      │ │     Kibana      │ │  Alert Manager  │  │
│  │ (Visualization) │ │  (Log Analysis) │ │   (Alerting)    │  │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│                      Notification                              │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐  │
│  │     Slack       │ │     Email       │ │   PagerDuty     │  │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Monitoring Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Metrics** | Prometheus + Grafana | Time-series metrics and visualization |
| **Logging** | Fluentd + Elasticsearch + Kibana | Log aggregation and analysis |
| **Tracing** | Jaeger + OpenTelemetry | Distributed tracing |
| **Alerting** | AlertManager + PagerDuty | Alert routing and escalation |
| **Uptime** | Pingdom + StatusPage | External monitoring and status |

## Metrics and KPIs

### System Health Metrics

**Infrastructure Metrics:**
```yaml
# Kubernetes cluster health
cluster_node_count: "Number of active nodes"
cluster_pod_count: "Total pods running"
cluster_cpu_usage: "CPU utilization percentage"
cluster_memory_usage: "Memory utilization percentage"
cluster_disk_usage: "Disk usage percentage"

# Application metrics
app_instances_running: "Number of healthy app instances"
app_response_time_p95: "95th percentile response time"
app_error_rate: "Error rate percentage"
app_throughput: "Requests per second"
```

**Business Metrics:**
```yaml
# Content pipeline metrics
pipeline_articles_processed_total: "Total articles processed"
pipeline_articles_published_total: "Total articles published" 
pipeline_processing_duration_seconds: "Processing time per stage"
pipeline_error_rate: "Pipeline error percentage"
pipeline_duplicate_detection_accuracy: "Duplicate detection accuracy"

# Content quality metrics
content_quality_score_average: "Average content quality score"
content_broken_links_count: "Number of broken links detected"
content_freshness_score: "Content freshness metric"

# User engagement metrics
api_requests_total: "Total API requests"
api_unique_users_daily: "Daily active API users"
content_views_total: "Total content views"
search_queries_total: "Total search queries"
```

### Performance KPIs

**Service Level Objectives (SLOs):**

| Service | SLO | Target | Measurement |
|---------|-----|--------|-------------|
| **API Availability** | 99.9% | <0.1% error rate | `sum(rate(http_requests_total{code=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))` |
| **API Latency** | P95 < 200ms | 95% of requests | `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))` |
| **Pipeline Success** | 95% | Success rate | `pipeline_jobs_success_total / pipeline_jobs_total` |
| **Content Freshness** | 24h | Daily updates | `time() - content_last_updated_timestamp` |

### Prometheus Metrics Implementation

```python
# Application metrics instrumentation
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import functools

# Define metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

ACTIVE_CONNECTIONS = Gauge(
    'database_connections_active',
    'Active database connections'
)

PIPELINE_PROCESSING_TIME = Histogram(
    'pipeline_processing_duration_seconds',
    'Pipeline processing duration',
    ['stage', 'source']
)

CONTENT_QUALITY_SCORE = Gauge(
    'content_quality_score',
    'Content quality score',
    ['content_id', 'source']
)

# Metric collection decorators
def track_requests(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        method = kwargs.get('method', 'GET')
        endpoint = kwargs.get('endpoint', 'unknown')
        
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            status = 'success'
            REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
            return result
        except Exception as e:
            status = 'error'
            REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
            raise
        finally:
            REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(
                time.time() - start_time
            )
    return wrapper

def track_pipeline_stage(stage_name):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            source = kwargs.get('source', 'unknown')
            
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                PIPELINE_PROCESSING_TIME.labels(stage=stage_name, source=source).observe(duration)
        return wrapper
    return decorator

# Usage example
@track_requests
async def get_content_item(content_id: str, method='GET', endpoint='/content'):
    # API logic here
    pass

@track_pipeline_stage('normalize')
async def normalize_content(content_item, source='arxiv'):
    # Normalization logic here
    pass
```

## Logging Strategy

### Structured Logging

**Log Format and Standards:**
```python
import structlog
import json
from datetime import datetime

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Application logging
logger = structlog.get_logger("ai_knowledge")

# Example structured log entries
logger.info(
    "content_processed",
    content_id="art_20240101_001",
    stage="normalize",
    duration_seconds=2.34,
    quality_score=0.92,
    source="arxiv"
)

logger.error(
    "pipeline_failed",
    error_type="APIError",
    error_message="Rate limit exceeded",
    content_id="art_20240101_002",
    stage="enrich",
    retry_attempt=3
)

logger.warning(
    "content_quality_low",
    content_id="art_20240101_003",
    quality_score=0.45,
    threshold=0.7,
    action="manual_review_required"
)
```

### Log Categories and Levels

**Log Categories:**
```yaml
categories:
  application:
    - api_requests: "All API request/response logs"
    - business_logic: "Core business logic execution"
    - data_processing: "Content pipeline processing"
    - performance: "Performance-related events"
  
  security:
    - authentication: "Login attempts and token validation"
    - authorization: "Permission checks and violations"
    - audit: "Security-relevant actions and changes"
    - threats: "Security threat detection events"
  
  infrastructure:
    - deployment: "Application deployment events"
    - scaling: "Auto-scaling and resource events"
    - network: "Network connectivity and errors"
    - storage: "Database and file storage events"
  
  integration:
    - external_apis: "Third-party API interactions"
    - webhooks: "Webhook delivery and processing"
    - message_queue: "Message queue operations"
```

**Log Levels:**
- **DEBUG**: Detailed diagnostic information
- **INFO**: General operational messages
- **WARNING**: Potentially harmful situations
- **ERROR**: Error events that don't stop operation
- **CRITICAL**: Serious errors that may cause termination

### Log Aggregation and Storage

**Fluentd Configuration:**
```yaml
# fluentd.conf
<source>
  @type tail
  @id input_tail
  path /var/log/containers/*.log
  pos_file /var/log/fluentd-containers.log.pos
  tag kubernetes.*
  format json
  refresh_interval 5
</source>

<filter kubernetes.**>
  @type kubernetes_metadata
</filter>

<match kubernetes.var.log.containers.**ai-knowledge**.log>
  @type elasticsearch
  host elasticsearch.monitoring.svc.cluster.local
  port 9200
  index_name ai-knowledge-logs
  type_name _doc
  include_tag_key true
  tag_key @log_name
  flush_interval 1s
</match>
```

### Log Retention and Management

**Retention Policies:**
```yaml
retention_policies:
  application_logs:
    hot_tier: "7 days"    # Fast SSD storage
    warm_tier: "30 days"  # Standard storage
    cold_tier: "90 days"  # Archive storage
    deletion: "1 year"    # Permanent deletion
  
  security_logs:
    hot_tier: "30 days"
    warm_tier: "180 days"
    cold_tier: "2 years" 
    deletion: "7 years"   # Compliance requirement
  
  debug_logs:
    hot_tier: "1 day"
    warm_tier: "7 days"
    cold_tier: "30 days"
    deletion: "90 days"
```

## Alerting Framework

### Alert Categories and Severity

**Severity Levels:**

| Severity | Response Time | Escalation | Examples |
|----------|---------------|------------|----------|
| **Critical** | 5 minutes | Immediate page | System down, data loss |
| **High** | 15 minutes | Phone call | High error rate, security breach |
| **Medium** | 1 hour | Slack/email | Performance degradation |
| **Low** | 4 hours | Email only | Non-critical warnings |

### Prometheus Alert Rules

```yaml
# alerts.yml
groups:
  - name: ai-knowledge.rules
    rules:
      # Critical alerts
      - alert: ServiceDown
        expr: up{job="ai-knowledge-api"} == 0
        for: 1m
        labels:
          severity: critical
          team: platform
        annotations:
          summary: "AI Knowledge API service is down"
          description: "Service {{ $labels.instance }} has been down for more than 1 minute"
          runbook: "https://docs.ai-knowledge.org/runbooks/service-down"
      
      - alert: HighErrorRate
        expr: |
          (
            sum(rate(http_requests_total{code=~"5.."}[5m])) /
            sum(rate(http_requests_total[5m]))
          ) > 0.05
        for: 5m
        labels:
          severity: critical
          team: platform
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}"
      
      - alert: DatabaseConnectionFailure
        expr: database_connections_active < 1
        for: 2m
        labels:
          severity: critical
          team: platform
        annotations:
          summary: "Database connection failure"
          description: "No active database connections for 2 minutes"
      
      # High priority alerts
      - alert: HighLatency
        expr: |
          histogram_quantile(0.95, 
            rate(http_request_duration_seconds_bucket[5m])
          ) > 0.5
        for: 10m
        labels:
          severity: high
          team: platform
        annotations:
          summary: "High API latency detected"
          description: "95th percentile latency is {{ $value }}s"
      
      - alert: PipelineFailureRate
        expr: |
          (
            sum(rate(pipeline_jobs_failed_total[30m])) /
            sum(rate(pipeline_jobs_total[30m]))
          ) > 0.1
        for: 15m
        labels:
          severity: high
          team: content
        annotations:
          summary: "Pipeline failure rate too high"
          description: "Pipeline failure rate is {{ $value | humanizePercentage }}"
      
      # Medium priority alerts
      - alert: HighMemoryUsage
        expr: |
          (
            container_memory_usage_bytes{pod=~"ai-knowledge-.*"} /
            container_spec_memory_limit_bytes
          ) > 0.8
        for: 15m
        labels:
          severity: medium
          team: platform
        annotations:
          summary: "High memory usage"
          description: "Pod {{ $labels.pod }} memory usage is {{ $value | humanizePercentage }}"
      
      - alert: ContentQualityDegradation
        expr: avg(content_quality_score) < 0.7
        for: 1h
        labels:
          severity: medium
          team: content
        annotations:
          summary: "Content quality score below threshold"
          description: "Average quality score is {{ $value }}"
      
      # Low priority alerts
      - alert: SlowDatabaseQuery
        expr: |
          histogram_quantile(0.95, 
            rate(database_query_duration_seconds_bucket[10m])
          ) > 1.0
        for: 30m
        labels:
          severity: low
          team: platform
        annotations:
          summary: "Slow database queries detected"
          description: "95th percentile query time is {{ $value }}s"
```

### Alert Manager Configuration

```yaml
# alertmanager.yml
global:
  slack_api_url: '{{ .SlackURL }}'
  
route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'default-receiver'
  routes:
  - match:
      severity: critical
    receiver: 'critical-alerts'
    group_wait: 0s
    repeat_interval: 5m
  - match:
      severity: high
    receiver: 'high-alerts'
    repeat_interval: 15m
  - match:
      team: security
    receiver: 'security-alerts'

receivers:
- name: 'default-receiver'
  email_configs:
  - to: 'devops@ai-knowledge.org'
    subject: '[AI Knowledge] {{ .GroupLabels.SortedPairs }}'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      {{ end }}

- name: 'critical-alerts'
  pagerduty_configs:
  - service_key: '{{ .PagerDutyServiceKey }}'
    description: '{{ .GroupLabels.SortedPairs }}'
  slack_configs:
  - channel: '#alerts-critical'
    title: 'Critical Alert'
    text: |
      {{ range .Alerts }}
      *Alert:* {{ .Annotations.summary }}
      *Description:* {{ .Annotations.description }}
      *Runbook:* {{ .Annotations.runbook }}
      {{ end }}

- name: 'security-alerts'
  email_configs:
  - to: 'security@ai-knowledge.org'
    subject: '[SECURITY] {{ .GroupLabels.SortedPairs }}'
  slack_configs:
  - channel: '#security-alerts'
    title: 'Security Alert'
```

## Dashboards and Visualization

### Grafana Dashboard Structure

**Main Dashboards:**

1. **System Overview Dashboard**
   - Service health status
   - Key performance indicators
   - Error rates and latencies
   - Infrastructure resource usage

2. **Application Performance Dashboard**
   - API response times
   - Throughput metrics
   - Error analysis
   - User activity patterns

3. **Content Pipeline Dashboard**
   - Pipeline execution status
   - Processing stages performance
   - Content quality metrics
   - Source ingestion rates

4. **Infrastructure Dashboard**
   - Kubernetes cluster health
   - Node and pod metrics
   - Network performance
   - Storage utilization

### Sample Dashboard JSON

```json
{
  "dashboard": {
    "title": "AI Knowledge - System Overview",
    "panels": [
      {
        "title": "Service Health",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=\"ai-knowledge-api\"}",
            "legendFormat": "API Service"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": {
              "mode": "thresholds"
            },
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "green", "value": 1}
              ]
            }
          }
        }
      },
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total[5m]))",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{code=~\"5..\"}[5m])) / sum(rate(http_requests_total[5m]))",
            "legendFormat": "Error Rate %"
          }
        ]
      }
    ]
  }
}
```

## Performance Monitoring

### Application Performance Monitoring (APM)

**Key Performance Metrics:**
```python
# Performance monitoring implementation
import time
from contextlib import contextmanager
from typing import Dict, Any

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}
    
    @contextmanager
    def measure_time(self, operation_name: str, **labels):
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_timing(operation_name, duration, labels)
    
    def record_timing(self, operation: str, duration: float, labels: Dict[str, Any]):
        PIPELINE_PROCESSING_TIME.labels(
            stage=operation,
            source=labels.get('source', 'unknown')
        ).observe(duration)
        
        # Also log detailed performance data
        logger.info(
            "performance_metric",
            operation=operation,
            duration_seconds=duration,
            **labels
        )

# Usage example
perf_monitor = PerformanceMonitor()

async def process_content_item(item, source='arxiv'):
    with perf_monitor.measure_time('content_processing', source=source, stage='total'):
        with perf_monitor.measure_time('normalize', source=source):
            normalized_item = await normalize_content(item)
        
        with perf_monitor.measure_time('enrich', source=source):
            enriched_item = await enrich_content(normalized_item)
        
        return enriched_item
```

### Database Performance Monitoring

```sql
-- PostgreSQL performance queries for monitoring
-- Slow query monitoring
SELECT
    query,
    calls,
    total_time,
    mean_time,
    rows,
    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;

-- Connection monitoring
SELECT
    pid,
    usename,
    application_name,
    client_addr,
    state,
    query_start,
    now() - query_start AS duration,
    query
FROM pg_stat_activity
WHERE state != 'idle'
ORDER BY duration DESC;

-- Lock monitoring
SELECT
    blocked_locks.pid AS blocked_pid,
    blocked_activity.usename AS blocked_user,
    blocking_locks.pid AS blocking_pid,
    blocking_activity.usename AS blocking_user,
    blocked_activity.query AS blocked_statement,
    blocking_activity.query AS current_statement_in_blocking_process
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;
```

## Security Monitoring

### Security Event Detection

**Security Metrics:**
```python
# Security monitoring metrics
SECURITY_EVENTS = Counter(
    'security_events_total',
    'Total security events',
    ['event_type', 'severity', 'source']
)

AUTHENTICATION_ATTEMPTS = Counter(
    'authentication_attempts_total',
    'Authentication attempts',
    ['result', 'user_type', 'source_ip']
)

PERMISSION_VIOLATIONS = Counter(
    'permission_violations_total',
    'Permission violations',
    ['user_id', 'resource', 'action']
)

def log_security_event(event_type: str, severity: str, **details):
    """Log security events with structured data."""
    SECURITY_EVENTS.labels(
        event_type=event_type,
        severity=severity,
        source=details.get('source', 'unknown')
    ).inc()
    
    security_logger = structlog.get_logger("security")
    security_logger.warning(
        "security_event",
        event_type=event_type,
        severity=severity,
        timestamp=time.time(),
        **details
    )

# Security event examples
log_security_event(
    "failed_authentication",
    "medium",
    user_id="user_123",
    source_ip="192.168.1.100",
    user_agent="curl/7.68.0",
    reason="invalid_password"
)

log_security_event(
    "privilege_escalation_attempt",
    "high", 
    user_id="user_456",
    requested_permission="admin:write",
    current_role="reader",
    resource="/admin/users"
)
```

### Intrusion Detection Rules

```yaml
# Security alert rules
- alert: BruteForceAttack
  expr: |
    sum(rate(authentication_attempts_total{result="failed"}[5m])) by (source_ip) > 10
  for: 2m
  labels:
    severity: high
    team: security
  annotations:
    summary: "Brute force attack detected"
    description: "IP {{ $labels.source_ip }} has {{ $value }} failed login attempts"

- alert: UnusualTrafficPattern
  expr: |
    sum(rate(http_requests_total[5m])) > 
    (avg_over_time(sum(rate(http_requests_total[5m]))[1h:5m]) * 3)
  for: 5m
  labels:
    severity: medium
    team: security
  annotations:
    summary: "Unusual traffic pattern detected"
    description: "Request rate is {{ $value }} times higher than normal"

- alert: PrivilegeEscalation
  expr: increase(permission_violations_total[5m]) > 0
  for: 0m
  labels:
    severity: high
    team: security
  annotations:
    summary: "Privilege escalation attempt"
    description: "User attempted unauthorized access"
```

## Operational Procedures

### Runbooks and Playbooks

**Incident Response Playbook:**

1. **Alert Triage (0-5 minutes)**
   ```bash
   # Check service health
   kubectl get pods -n ai-knowledge-prod
   curl -f https://api.ai-knowledge.org/health
   
   # Check recent deployments
   kubectl rollout history deployment/ai-knowledge-api -n ai-knowledge-prod
   
   # Review recent alerts
   # Check Grafana dashboard for anomalies
   ```

2. **Investigation (5-15 minutes)**
   ```bash
   # Gather logs
   kubectl logs -f deployment/ai-knowledge-api -n ai-knowledge-prod --since=15m
   
   # Check metrics
   # Query Prometheus for relevant metrics
   # Review error rates and latencies
   
   # Database health
   kubectl exec -it deployment/postgresql -n ai-knowledge-prod -- psql -c "SELECT 1"
   ```

3. **Mitigation (15-30 minutes)**
   ```bash
   # Scale up if needed
   kubectl scale deployment ai-knowledge-api --replicas=5 -n ai-knowledge-prod
   
   # Rollback if recent deployment caused issue
   kubectl rollout undo deployment/ai-knowledge-api -n ai-knowledge-prod
   
   # Enable maintenance mode if severe
   kubectl apply -f k8s/maintenance-mode.yaml
   ```

### Monitoring Checklists

**Daily Monitoring Checklist:**
- [ ] Check system health dashboard
- [ ] Review error rates and latencies
- [ ] Verify pipeline execution success
- [ ] Check content quality metrics
- [ ] Review security alerts
- [ ] Validate backup completion
- [ ] Monitor resource utilization

**Weekly Monitoring Checklist:**
- [ ] Review alert effectiveness and tune thresholds
- [ ] Analyze performance trends
- [ ] Check log retention and storage usage
- [ ] Review and update dashboards
- [ ] Validate monitoring tool health
- [ ] Test alert notification delivery
- [ ] Review security monitoring coverage

### Monitoring Tool Maintenance

**Prometheus Maintenance:**
```bash
# Check Prometheus health
curl http://prometheus:9090/-/healthy

# Check configuration
curl http://prometheus:9090/api/v1/status/config

# Reload configuration
curl -X POST http://prometheus:9090/-/reload

# Check targets
curl http://prometheus:9090/api/v1/targets

# Query metrics retention
curl 'http://prometheus:9090/api/v1/query?query=prometheus_tsdb_retention_limit_seconds'
```

**Log Management:**
```bash
# Check Elasticsearch cluster health
curl -X GET "elasticsearch:9200/_cluster/health?pretty"

# Check index status
curl -X GET "elasticsearch:9200/_cat/indices/ai-knowledge-*?v"

# Clean up old indices
curator --config config.yml delete_indices.yml

# Check Fluentd status
kubectl logs daemonset/fluentd -n kube-system
```

---

**Monitoring Contacts:**
- **Platform Team**: platform@ai-knowledge.org
- **On-Call Engineer**: +1-XXX-XXX-XXXX
- **Escalation**: engineering-managers@ai-knowledge.org

**Monitoring Resources:**
- Grafana: https://grafana.ai-knowledge.org
- Prometheus: https://prometheus.ai-knowledge.org
- Kibana: https://kibana.ai-knowledge.org
- Status Page: https://status.ai-knowledge.org

**Last Updated**: January 2024  
**Review Cycle**: Monthly  
**Owner**: Platform Engineering Team