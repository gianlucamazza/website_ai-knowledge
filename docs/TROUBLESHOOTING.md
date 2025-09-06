# Troubleshooting Guide

This comprehensive troubleshooting guide provides solutions for common issues encountered in the AI Knowledge Website system, covering frontend, backend pipeline, infrastructure, and operational problems.

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Frontend Issues](#frontend-issues)
3. [Backend Pipeline Issues](#backend-pipeline-issues)
4. [Database Issues](#database-issues)
5. [Infrastructure Issues](#infrastructure-issues)
6. [Performance Issues](#performance-issues)
7. [Security Issues](#security-issues)
8. [Content Issues](#content-issues)
9. [Monitoring and Alerting](#monitoring-and-alerting)
10. [Emergency Procedures](#emergency-procedures)

## Quick Diagnostics

### System Health Check

Run the automated health check to identify issues:

```bash
# Quick system status
make health-check

# Comprehensive diagnostic
make diagnose

# Check specific components
make check-database
make check-redis
make check-pipeline
make check-frontend
```

### Log Access

Access logs for different components:

```bash
# Frontend logs
kubectl logs -f deployment/ai-knowledge-frontend -n ai-knowledge-prod

# Pipeline logs
kubectl logs -f deployment/ai-knowledge-pipeline -n ai-knowledge-prod

# Database logs (if self-hosted)
kubectl logs -f deployment/postgresql -n ai-knowledge-prod

# Ingress logs
kubectl logs -f deployment/nginx-ingress-controller -n ingress-nginx
```

### Common Status Commands

```bash
# Pod status
kubectl get pods -n ai-knowledge-prod
kubectl describe pod <pod-name> -n ai-knowledge-prod

# Service status  
kubectl get svc -n ai-knowledge-prod
kubectl describe svc <service-name> -n ai-knowledge-prod

# Ingress status
kubectl get ingress -n ai-knowledge-prod
kubectl describe ingress ai-knowledge-ingress -n ai-knowledge-prod
```

## Frontend Issues

### Build Failures

**Problem**: Astro build fails during CI/CD

**Symptoms**:
- Build process exits with error code
- Missing static files
- TypeScript compilation errors

**Diagnosis**:
```bash
# Local build test
cd apps/site
npm install
npm run build

# Check build logs
npm run build 2>&1 | tee build.log

# Verify dependencies
npm audit
npm outdated
```

**Solutions**:

1. **Dependency Issues**:
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Update dependencies
npm update
npm audit fix
```

2. **TypeScript Errors**:
```bash
# Run type checking
npm run type-check

# Fix common type errors
# Check src/env.d.ts for missing type declarations
# Verify Zod schemas in src/content/config.ts
```

3. **Content Validation Errors**:
```bash
# Validate content against schemas
npm run lint

# Check specific content files
npx astro check
```

### Deployment Issues

**Problem**: Frontend not accessible after deployment

**Symptoms**:
- 502/503 errors
- Page not loading
- Static assets missing

**Diagnosis**:
```bash
# Check pod status
kubectl get pods -n ai-knowledge-prod -l component=frontend

# Check service endpoints
kubectl get endpoints -n ai-knowledge-prod

# Test internal connectivity
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  curl http://ai-knowledge-frontend:3000/health
```

**Solutions**:

1. **Pod Not Ready**:
```bash
# Check pod logs
kubectl logs -f deployment/ai-knowledge-frontend -n ai-knowledge-prod

# Check resource limits
kubectl describe deployment ai-knowledge-frontend -n ai-knowledge-prod

# Scale up if needed
kubectl scale deployment ai-knowledge-frontend --replicas=3 -n ai-knowledge-prod
```

2. **Service Configuration**:
```bash
# Verify service selector
kubectl get svc ai-knowledge-frontend -n ai-knowledge-prod -o yaml

# Check port configuration
kubectl port-forward svc/ai-knowledge-frontend 3000:3000 -n ai-knowledge-prod
```

### Content Not Updating

**Problem**: New content not appearing on site

**Symptoms**:
- Old content still visible
- Missing articles/glossary entries
- Stale cache

**Diagnosis**:
```bash
# Check content files
ls -la apps/site/src/content/articles/
ls -la apps/site/src/content/glossary/

# Check build process
kubectl logs deployment/ai-knowledge-pipeline -n ai-knowledge-prod | grep "publish"

# Check CDN cache status
curl -I https://ai-knowledge.org/ | grep -i cache
```

**Solutions**:

1. **Build Process**:
```bash
# Trigger manual build
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  python -m pipelines.publish.build_site

# Check build logs
kubectl logs -f deployment/ai-knowledge-pipeline -n ai-knowledge-prod --since=10m
```

2. **Cache Invalidation**:
```bash
# Clear CDN cache
curl -X POST https://api.cloudflare.com/client/v4/zones/ZONE_ID/purge_cache \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  --data '{"purge_everything":true}'

# Clear local Redis cache
kubectl exec -it deployment/redis -n ai-knowledge-prod -- redis-cli FLUSHALL
```

## Backend Pipeline Issues

### Pipeline Jobs Failing

**Problem**: Content ingestion pipeline consistently fails

**Symptoms**:
- Jobs stuck in "running" state
- High error rates in logs
- Content not being processed

**Diagnosis**:
```bash
# Check job status
python -m pipelines.cli job-status

# Check LangGraph workflow state
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  python -c "
from pipelines.orchestrators.langgraph.workflow import get_workflow_state
print(get_workflow_state())
"

# Check worker processes
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- ps aux
```

**Solutions**:

1. **Memory Issues**:
```bash
# Check memory usage
kubectl top pods -n ai-knowledge-prod

# Increase memory limits
kubectl patch deployment ai-knowledge-pipeline -n ai-knowledge-prod \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"pipeline","resources":{"limits":{"memory":"4Gi"}}}]}}}}'

# Restart deployment
kubectl rollout restart deployment/ai-knowledge-pipeline -n ai-knowledge-prod
```

2. **External API Issues**:
```bash
# Test external API connectivity
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  python -c "
import requests
try:
    response = requests.get('https://api.openai.com/v1/models', 
                          headers={'Authorization': 'Bearer $OPENAI_API_KEY'})
    print(f'OpenAI API Status: {response.status_code}')
except Exception as e:
    print(f'OpenAI API Error: {e}')
"

# Check rate limiting
grep "rate.limit" /var/log/pipeline/error.log
```

3. **Database Connectivity**:
```bash
# Test database connection
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  python -c "
from pipelines.database.connection import get_connection
try:
    conn = get_connection()
    print(f'Database connection: {conn.info.status}')
    conn.close()
except Exception as e:
    print(f'Database error: {e}')
"
```

### Duplicate Detection Issues

**Problem**: High false positive rate in duplicate detection

**Symptoms**:
- Valid content being marked as duplicates
- Similar content not being detected
- Inconsistent deduplication results

**Diagnosis**:
```bash
# Check duplicate detection metrics
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  python -m pipelines.dedup.analyze_performance

# Review recent duplicate detections
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  python -c "
from pipelines.database.models import Duplicate
from pipelines.database.connection import get_session
with get_session() as session:
    recent_dupes = session.query(Duplicate).order_by(Duplicate.created_at.desc()).limit(10)
    for dupe in recent_dupes:
        print(f'Similarity: {dupe.similarity_score}, Method: {dupe.detection_method}')
"
```

**Solutions**:

1. **Adjust Thresholds**:
```python
# Update configuration
# In pipelines/config.py
DEDUP_SIMHASH_THRESHOLD = 4  # Increase for less strict matching
DEDUP_MINHASH_THRESHOLD = 0.75  # Decrease for less strict matching
```

2. **Clear False Positives**:
```bash
# Review and clear false positives
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  python -m pipelines.dedup.review_duplicates --clear-false-positives

# Rebuild LSH index
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  python -m pipelines.dedup.rebuild_index
```

### Content Enrichment Failures

**Problem**: AI-powered content enrichment not working

**Symptoms**:
- Missing summaries
- No cross-links generated  
- Tags not being applied

**Diagnosis**:
```bash
# Check AI API keys
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  python -c "
import os
print('OpenAI key present:', bool(os.getenv('OPENAI_API_KEY')))
print('Anthropic key present:', bool(os.getenv('ANTHROPIC_API_KEY')))
"

# Check enrichment logs
kubectl logs deployment/ai-knowledge-pipeline -n ai-knowledge-prod | grep enrich
```

**Solutions**:

1. **API Key Issues**:
```bash
# Update API keys
kubectl create secret generic ai-knowledge-secrets \
  --from-literal=OPENAI_API_KEY="your-new-key" \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart deployment to pick up new secrets
kubectl rollout restart deployment/ai-knowledge-pipeline -n ai-knowledge-prod
```

2. **Rate Limiting**:
```bash
# Check rate limit status
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  python -m pipelines.enrich.check_rate_limits

# Implement exponential backoff
# Review pipelines/enrich/summarizer.py for retry logic
```

## Database Issues

### Connection Pool Exhaustion

**Problem**: "Connection pool is exhausted" errors

**Symptoms**:
- Database connection timeouts
- Slow query performance
- Pipeline jobs failing

**Diagnosis**:
```bash
# Check active connections
kubectl exec -it deployment/postgresql -n ai-knowledge-prod -- \
  psql -c "SELECT count(*) as active_connections FROM pg_stat_activity WHERE state = 'active';"

# Check connection pool configuration
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  python -c "
from pipelines.config import config
print(f'Pool size: {config.database.pool_size}')
print(f'Max overflow: {config.database.max_overflow}')
"
```

**Solutions**:

1. **Increase Connection Pool**:
```python
# Update pipelines/config.py
DATABASE_POOL_SIZE = 30  # Increase from 20
DATABASE_MAX_OVERFLOW = 50  # Increase from 30
```

2. **Optimize Long-Running Queries**:
```sql
-- Check for long-running queries
SELECT pid, now() - pg_stat_activity.query_start AS duration, query 
FROM pg_stat_activity 
WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes';

-- Kill problematic queries if necessary
SELECT pg_terminate_backend(pid) FROM pg_stat_activity 
WHERE (now() - pg_stat_activity.query_start) > interval '10 minutes';
```

3. **Connection Cleanup**:
```bash
# Restart pipeline to clear connections
kubectl rollout restart deployment/ai-knowledge-pipeline -n ai-knowledge-prod

# Check if connections are properly closed
kubectl logs deployment/ai-knowledge-pipeline -n ai-knowledge-prod | grep "connection"
```

### Database Lock Issues

**Problem**: Database queries hanging due to locks

**Symptoms**:
- Queries timing out
- Pipeline hanging during deduplication
- Slow database performance

**Diagnosis**:
```sql
-- Check for locks
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

**Solutions**:

1. **Terminate Blocking Queries**:
```sql
-- Identify and terminate long-running queries
SELECT pg_terminate_backend(<blocking_pid>);
```

2. **Optimize Locking Strategy**:
```python
# In pipelines/database/operations.py
# Use advisory locks for critical sections
def with_advisory_lock(session, lock_id):
    session.execute(f"SELECT pg_advisory_lock({lock_id})")
    try:
        yield
    finally:
        session.execute(f"SELECT pg_advisory_unlock({lock_id})")
```

### Data Corruption Issues

**Problem**: Inconsistent or corrupted data

**Symptoms**:
- Foreign key constraint violations
- Duplicate primary keys
- Inconsistent content_hash values

**Diagnosis**:
```sql
-- Check for orphaned records
SELECT COUNT(*) FROM content_items ci 
LEFT JOIN sources s ON ci.source_id = s.id 
WHERE s.id IS NULL;

-- Check for duplicate hashes
SELECT content_hash, COUNT(*) 
FROM content_items 
GROUP BY content_hash 
HAVING COUNT(*) > 1;

-- Verify foreign key constraints
SELECT conname, conrelid::regclass, confrelid::regclass
FROM pg_constraint 
WHERE contype = 'f' AND NOT convalidated;
```

**Solutions**:

1. **Data Cleanup**:
```sql
-- Remove orphaned records
DELETE FROM content_items 
WHERE source_id NOT IN (SELECT id FROM sources);

-- Fix duplicate hashes by regenerating
UPDATE content_items 
SET content_hash = md5(title || content || source_url) 
WHERE content_hash IN (
    SELECT content_hash FROM content_items 
    GROUP BY content_hash HAVING COUNT(*) > 1
);
```

2. **Restore from Backup**:
```bash
# If corruption is severe, restore from backup
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  python -m pipelines.database.restore --backup-name "daily-backup-20240106"
```

## Infrastructure Issues

### Kubernetes Pod Failures

**Problem**: Pods crashing or failing to start

**Symptoms**:
- CrashLoopBackOff status
- ImagePullBackOff errors
- Pods stuck in Pending state

**Diagnosis**:
```bash
# Check pod status
kubectl get pods -n ai-knowledge-prod -o wide

# Describe problematic pod
kubectl describe pod <pod-name> -n ai-knowledge-prod

# Check events
kubectl get events -n ai-knowledge-prod --sort-by='.lastTimestamp'
```

**Solutions**:

1. **Image Pull Issues**:
```bash
# Check image exists
docker pull ai-knowledge/pipeline:v1.2.3

# Update image pull policy
kubectl patch deployment ai-knowledge-pipeline -n ai-knowledge-prod \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"pipeline","imagePullPolicy":"Always"}]}}}}'

# Add image pull secrets if needed
kubectl create secret docker-registry regcred \
  --docker-server=<registry-url> \
  --docker-username=<username> \
  --docker-password=<password>
```

2. **Resource Constraints**:
```bash
# Check node resources
kubectl describe nodes

# Increase resource limits
kubectl patch deployment ai-knowledge-pipeline -n ai-knowledge-prod \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"pipeline","resources":{"limits":{"cpu":"2","memory":"4Gi"},"requests":{"cpu":"500m","memory":"1Gi"}}}]}}}}'
```

3. **ConfigMap/Secret Issues**:
```bash
# Verify ConfigMap exists
kubectl get configmap ai-knowledge-config -n ai-knowledge-prod

# Verify Secret exists
kubectl get secret ai-knowledge-secrets -n ai-knowledge-prod

# Check secret values (base64 decode)
kubectl get secret ai-knowledge-secrets -n ai-knowledge-prod -o jsonpath='{.data.DATABASE_URL}' | base64 -d
```

### Networking Issues

**Problem**: Services cannot communicate with each other

**Symptoms**:
- Connection timeouts between services
- DNS resolution failures
- Ingress not routing traffic

**Diagnosis**:
```bash
# Test DNS resolution
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  nslookup ai-knowledge-frontend.ai-knowledge-prod.svc.cluster.local

# Test service connectivity
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  curl -v http://ai-knowledge-frontend:3000/health

# Check network policies
kubectl get networkpolicy -n ai-knowledge-prod
```

**Solutions**:

1. **Service Discovery**:
```bash
# Check service endpoints
kubectl get endpoints -n ai-knowledge-prod

# Verify service selector matches pod labels
kubectl get svc ai-knowledge-frontend -n ai-knowledge-prod -o yaml
kubectl get pods -n ai-knowledge-prod --show-labels
```

2. **Ingress Configuration**:
```bash
# Check ingress controller
kubectl get pods -n ingress-nginx

# Verify ingress rules
kubectl get ingress -n ai-knowledge-prod -o yaml

# Test ingress connectivity
curl -H "Host: ai-knowledge.org" http://<ingress-ip>/
```

### Storage Issues

**Problem**: Persistent volume issues

**Symptoms**:
- Pods stuck mounting volumes
- Data not persisting across restarts
- Storage full errors

**Diagnosis**:
```bash
# Check persistent volumes
kubectl get pv
kubectl get pvc -n ai-knowledge-prod

# Check storage usage
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- df -h

# Check volume mounts
kubectl describe pod <pod-name> -n ai-knowledge-prod | grep -A 10 Volumes
```

**Solutions**:

1. **Volume Mounting**:
```bash
# Check if PVC is bound
kubectl get pvc -n ai-knowledge-prod

# If stuck, delete and recreate PVC
kubectl delete pvc <pvc-name> -n ai-knowledge-prod
kubectl apply -f k8s/storage/pvc.yaml
```

2. **Storage Expansion**:
```bash
# Expand PVC (if storage class supports it)
kubectl patch pvc <pvc-name> -n ai-knowledge-prod \
  -p '{"spec":{"resources":{"requests":{"storage":"100Gi"}}}}'
```

## Performance Issues

### High Latency

**Problem**: API responses taking too long

**Symptoms**:
- Response times >1 second
- Users experiencing delays
- Timeouts in frontend

**Diagnosis**:
```bash
# Check response times
curl -w "@curl-format.txt" -o /dev/null -s https://api.ai-knowledge.org/health

# Monitor resource usage
kubectl top pods -n ai-knowledge-prod
kubectl top nodes
```

**Solutions**:

1. **Database Optimization**:
```sql
-- Check slow queries
SELECT query, mean_time, calls, rows 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;

-- Add missing indexes
CREATE INDEX CONCURRENTLY idx_content_items_created_at 
ON content_items(created_at);
```

2. **Cache Implementation**:
```python
# Add Redis caching to expensive operations
from pipelines.cache import cache_manager

@cache_manager.cached(timeout=3600)
def get_article_list():
    # Expensive database query
    return session.query(Article).all()
```

3. **Load Balancing**:
```bash
# Scale up deployment
kubectl scale deployment ai-knowledge-pipeline --replicas=5 -n ai-knowledge-prod

# Enable HPA
kubectl autoscale deployment ai-knowledge-pipeline \
  --min=3 --max=10 --cpu-percent=70 -n ai-knowledge-prod
```

### Memory Issues

**Problem**: High memory usage or memory leaks

**Symptoms**:
- OOMKilled pod restarts
- Gradually increasing memory usage
- System slowness

**Diagnosis**:
```bash
# Check memory usage
kubectl top pods -n ai-knowledge-prod --sort-by=memory

# Check for memory leaks
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  python -c "
import psutil, os
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB')
print(f'Memory percent: {process.memory_percent():.2f}%')
"
```

**Solutions**:

1. **Memory Limits**:
```bash
# Increase memory limits
kubectl patch deployment ai-knowledge-pipeline -n ai-knowledge-prod \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"pipeline","resources":{"limits":{"memory":"4Gi"}}}]}}}}'
```

2. **Memory Optimization**:
```python
# In pipeline code, implement memory-efficient processing
def process_content_batch(items, batch_size=100):
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        yield process_batch(batch)
        # Force garbage collection
        import gc
        gc.collect()
```

### Database Performance

**Problem**: Slow database queries

**Diagnosis**:
```sql
-- Enable query logging
ALTER SYSTEM SET log_min_duration_statement = 1000; -- Log queries > 1s
SELECT pg_reload_conf();

-- Check table statistics
SELECT 
    schemaname,
    tablename,
    n_tup_ins as inserts,
    n_tup_upd as updates,
    n_tup_del as deletes,
    n_live_tup as live_tuples,
    n_dead_tup as dead_tuples
FROM pg_stat_user_tables 
ORDER BY n_live_tup DESC;
```

**Solutions**:

1. **Query Optimization**:
```sql
-- Add indexes for frequent queries
CREATE INDEX CONCURRENTLY idx_content_items_simhash ON content_items USING hash(simhash);
CREATE INDEX CONCURRENTLY idx_content_items_status_created ON content_items(status, created_at);

-- Optimize joins
EXPLAIN ANALYZE SELECT * FROM content_items ci 
JOIN sources s ON ci.source_id = s.id 
WHERE ci.status = 'published';
```

2. **Table Maintenance**:
```bash
# Schedule regular VACUUM and ANALYZE
kubectl create cronjob db-maintenance -n ai-knowledge-prod \
  --image=postgres:14 \
  --schedule="0 2 * * *" \
  -- psql "$DATABASE_URL" -c "VACUUM ANALYZE;"
```

## Security Issues

### Authentication Failures

**Problem**: API authentication not working

**Symptoms**:
- 401 Unauthorized errors
- Invalid token messages
- Authentication bypass attempts

**Diagnosis**:
```bash
# Check API key configuration
kubectl get secret ai-knowledge-secrets -n ai-knowledge-prod -o yaml

# Test authentication
curl -H "Authorization: Bearer invalid-token" https://api.ai-knowledge.org/health

# Check authentication logs
kubectl logs deployment/ai-knowledge-pipeline -n ai-knowledge-prod | grep auth
```

**Solutions**:

1. **Token Validation**:
```bash
# Generate new API token
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  python -m pipelines.auth.generate_token --role admin

# Update client configurations
# Rotate tokens regularly
```

2. **Authentication Debugging**:
```python
# Add detailed authentication logging
import logging
logger = logging.getLogger('auth')

def verify_token(token):
    logger.info(f"Verifying token: {token[:10]}...")
    try:
        # Token verification logic
        pass
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        raise
```

### Rate Limiting Issues

**Problem**: Rate limiting not working correctly

**Diagnosis**:
```bash
# Check rate limiting headers
curl -I https://api.ai-knowledge.org/health

# Check Redis for rate limiting data
kubectl exec -it deployment/redis -n ai-knowledge-prod -- \
  redis-cli keys "rate_limit:*"
```

**Solutions**:

1. **Rate Limit Configuration**:
```python
# Update rate limiting configuration
RATE_LIMITS = {
    'api': '100/hour',
    'pipeline': '10/hour',
    'admin': '1000/hour'
}
```

2. **Rate Limit Monitoring**:
```bash
# Monitor rate limiting effectiveness
kubectl logs deployment/ai-knowledge-pipeline -n ai-knowledge-prod | grep "rate.limit"
```

## Content Issues

### Content Not Publishing

**Problem**: Content stuck in processing state

**Symptoms**:
- Articles not appearing on site
- Processing status not updating
- Pipeline completing without errors

**Diagnosis**:
```bash
# Check content status in database
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  python -c "
from pipelines.database.models import ContentItem
from pipelines.database.connection import get_session
with get_session() as session:
    pending = session.query(ContentItem).filter_by(status='processing').count()
    print(f'Items in processing: {pending}')
"

# Check file system
ls -la apps/site/src/content/articles/ | head -10
```

**Solutions**:

1. **Manual Content Publishing**:
```bash
# Force publish specific content
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  python -m pipelines.publish.force_publish --content-id <content-id>

# Republish all content
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  python -m pipelines.publish.republish_all
```

2. **Content Status Reset**:
```sql
-- Reset stuck content to pending
UPDATE content_items 
SET status = 'pending', updated_at = NOW() 
WHERE status = 'processing' 
AND updated_at < NOW() - INTERVAL '1 hour';
```

### Content Quality Issues

**Problem**: Low-quality content being published

**Diagnosis**:
```bash
# Check quality scores
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  python -c "
from pipelines.database.models import ContentItem
from pipelines.database.connection import get_session
with get_session() as session:
    low_quality = session.query(ContentItem).filter(ContentItem.quality_score < 0.7).count()
    print(f'Low quality items: {low_quality}')
"
```

**Solutions**:

1. **Quality Threshold Adjustment**:
```python
# Increase quality threshold in config
QUALITY_SCORE_THRESHOLD = 0.8  # Increase from 0.7
```

2. **Manual Quality Review**:
```bash
# Set up manual review queue for borderline content
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  python -m pipelines.quality.create_review_queue --threshold 0.75
```

## Monitoring and Alerting

### Missing Metrics

**Problem**: Prometheus metrics not being collected

**Diagnosis**:
```bash
# Check Prometheus targets
kubectl port-forward svc/prometheus 9090:9090 -n monitoring
# Visit http://localhost:9090/targets

# Check metrics endpoint
curl http://ai-knowledge-pipeline:8000/metrics
```

**Solutions**:

1. **Metrics Configuration**:
```python
# Ensure metrics are properly exported
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests')
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
```

2. **Prometheus Configuration**:
```yaml
# Update prometheus.yml
scrape_configs:
- job_name: 'ai-knowledge-pipeline'
  static_configs:
  - targets: ['ai-knowledge-pipeline:8000']
  metrics_path: '/metrics'
  scrape_interval: 30s
```

### Alert Fatigue

**Problem**: Too many false positive alerts

**Solutions**:

1. **Alert Tuning**:
```yaml
# Adjust alert thresholds
- alert: HighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05  # Reduced from 0.1
  for: 10m  # Increased from 5m
```

2. **Alert Grouping**:
```yaml
# Group related alerts
route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
```

## Emergency Procedures

### System Outage

**Immediate Response**:

1. **Assess Impact**:
```bash
# Check overall system status
make health-check

# Check critical services
kubectl get pods -n ai-knowledge-prod
curl -I https://ai-knowledge.org/
curl -I https://api.ai-knowledge.org/health
```

2. **Enable Maintenance Mode**:
```bash
# Deploy maintenance page
kubectl apply -f k8s/maintenance/maintenance-mode.yaml

# Update DNS to point to maintenance page
# Update CDN configuration
```

3. **Investigate Root Cause**:
```bash
# Check recent changes
git log --oneline --since="1 hour ago"
kubectl rollout history deployment/ai-knowledge-pipeline -n ai-knowledge-prod

# Review alerts and logs
kubectl logs -f deployment/ai-knowledge-pipeline -n ai-knowledge-prod --since=1h
```

### Data Loss Incident

**Response Procedure**:

1. **Stop All Writes**:
```bash
# Scale down pipeline to prevent further data changes
kubectl scale deployment ai-knowledge-pipeline --replicas=0 -n ai-knowledge-prod
```

2. **Assess Data Loss**:
```bash
# Check database integrity
kubectl exec -it deployment/postgresql -n ai-knowledge-prod -- \
  psql -c "SELECT COUNT(*) FROM content_items;"

# Compare with backup
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  python -m pipelines.database.compare_backup --backup-name latest
```

3. **Restore from Backup**:
```bash
# Restore from most recent backup
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  python -m pipelines.database.restore --backup-name "$(date -d 'yesterday' +%Y%m%d)"

# Verify restoration
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  python -m pipelines.database.verify_restore
```

### Security Incident

**Response Steps**:

1. **Isolate Affected Systems**:
```bash
# Block suspicious traffic
kubectl apply -f k8s/security/emergency-network-policy.yaml

# Rotate API keys immediately
kubectl delete secret ai-knowledge-secrets -n ai-knowledge-prod
kubectl create secret generic ai-knowledge-secrets --from-env-file=.env.emergency
```

2. **Investigate Breach**:
```bash
# Check access logs
kubectl logs deployment/ai-knowledge-pipeline -n ai-knowledge-prod | grep -E "(unauthorized|failed|error)"

# Review authentication attempts
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  python -m pipelines.security.audit_logs --since "1 hour ago"
```

3. **Contact Security Team**:
```bash
# Send automated security alert
python scripts/security_alert.py --incident-type "data-breach" --severity "critical"
```

---

**Emergency Contacts**:
- **On-call Engineer**: +1-XXX-XXX-XXXX
- **Security Team**: security@ai-knowledge.org
- **Infrastructure Team**: infra@ai-knowledge.org

**Last Updated**: January 2024  
**Version**: 1.0.0