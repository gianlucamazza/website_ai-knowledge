# Deployment Guide

This guide provides comprehensive instructions for deploying the AI Knowledge Website to production environments, including infrastructure setup, configuration management, and operational procedures.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Infrastructure Setup](#infrastructure-setup)
4. [Environment Configuration](#environment-configuration)
5. [Deployment Procedures](#deployment-procedures)
6. [Database Migrations](#database-migrations)
7. [Monitoring Setup](#monitoring-setup)
8. [SSL/TLS Configuration](#ssltls-configuration)
9. [Scaling and Load Balancing](#scaling-and-load-balancing)
10. [Troubleshooting](#troubleshooting)

## Overview

The AI Knowledge Website deployment follows a containerized, multi-tier architecture optimized for high availability and scalability. The deployment supports multiple environments (staging, production) with automated CI/CD pipelines.

### Architecture Components

- **Frontend**: Astro-based static site deployed to CDN
- **Backend Pipeline**: Containerized Python services on Kubernetes
- **Database**: Managed PostgreSQL with read replicas
- **Cache**: Redis cluster for session and application caching
- **Load Balancer**: Nginx with SSL termination
- **Monitoring**: Prometheus, Grafana, and centralized logging

## Prerequisites

### Required Software

- Docker 20.10+
- Kubernetes 1.25+ (or cloud equivalent)
- kubectl configured with cluster access
- Helm 3.8+ for package management
- Terraform 1.3+ for infrastructure as code

### Required Services

- **Container Registry**: Docker Hub, GitHub Container Registry, or cloud equivalent
- **Database**: PostgreSQL 14+ (managed service recommended)
- **Cache**: Redis 7+ (managed service or cluster)
- **DNS**: Managed DNS service (Cloudflare, Route 53)
- **Storage**: Object storage for backups and assets

### Access Requirements

- Cluster admin access to Kubernetes
- DNS management permissions
- Container registry push/pull permissions
- Database administration access
- SSL certificate management access

## Infrastructure Setup

### Cloud Infrastructure (Terraform)

Create infrastructure using the provided Terraform configurations:

```bash
# Navigate to infrastructure directory
cd infrastructure/terraform

# Initialize Terraform
terraform init

# Review planned changes
terraform plan -var-file="production.tfvars"

# Apply infrastructure changes
terraform apply -var-file="production.tfvars"
```

**Key Infrastructure Components:**

```hcl
# infrastructure/terraform/main.tf
resource "kubernetes_cluster" "ai_knowledge" {
  name     = "ai-knowledge-${var.environment}"
  version  = "1.25"
  
  node_pool {
    name         = "pipeline-pool"
    machine_type = "e2-standard-4"
    min_size     = 2
    max_size     = 10
    
    labels = {
      workload = "pipeline"
    }
  }
  
  node_pool {
    name         = "web-pool"
    machine_type = "e2-standard-2"
    min_size     = 1
    max_size     = 5
    
    labels = {
      workload = "web"
    }
  }
}

resource "postgresql_database" "ai_knowledge" {
  name     = "ai_knowledge_${var.environment}"
  version  = "14"
  tier     = var.db_tier
  
  backup_configuration {
    enabled    = true
    start_time = "02:00"
    retention_days = 30
  }
}
```

### Kubernetes Namespace Setup

Create dedicated namespaces for different environments:

```yaml
# k8s/namespaces/production.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ai-knowledge-prod
  labels:
    environment: production
    app: ai-knowledge
---
apiVersion: v1
kind: Namespace
metadata:
  name: ai-knowledge-staging
  labels:
    environment: staging
    app: ai-knowledge
```

Apply namespaces:

```bash
kubectl apply -f k8s/namespaces/production.yaml
```

## Environment Configuration

### Configuration Management

Use Kubernetes ConfigMaps and Secrets for environment-specific configuration:

**ConfigMap for Application Settings:**

```yaml
# k8s/configs/production-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ai-knowledge-config
  namespace: ai-knowledge-prod
data:
  ENVIRONMENT: "production"
  DEBUG: "false"
  LOG_LEVEL: "INFO"
  SCRAPING_REQUEST_DELAY: "1.0"
  SCRAPING_CONCURRENT_REQUESTS: "5"
  DEDUP_SIMHASH_THRESHOLD: "3"
  DEDUP_MINHASH_THRESHOLD: "0.85"
```

**Secrets for Sensitive Data:**

```yaml
# k8s/secrets/production-secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: ai-knowledge-secrets
  namespace: ai-knowledge-prod
type: Opaque
data:
  DATABASE_URL: <base64-encoded-database-url>
  REDIS_URL: <base64-encoded-redis-url>
  OPENAI_API_KEY: <base64-encoded-api-key>
  ANTHROPIC_API_KEY: <base64-encoded-api-key>
  SECRET_KEY: <base64-encoded-secret-key>
```

Create secrets securely:

```bash
# Create database URL secret
kubectl create secret generic ai-knowledge-secrets \
  --namespace=ai-knowledge-prod \
  --from-literal=DATABASE_URL="postgresql://user:pass@host:5432/dbname" \
  --from-literal=REDIS_URL="redis://redis-cluster:6379/0"

# Or use sealed secrets for GitOps
echo -n "your-secret-value" | base64
```

### Environment Variables

**Production Environment Variables:**

```bash
# Production environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
DATABASE_URL=postgresql://prod-user:password@prod-db:5432/ai_knowledge_prod
REDIS_URL=redis://redis-prod:6379/0
SECRET_KEY=your-production-secret-key
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key

# Scraping Configuration
SCRAPING_RESPECT_ROBOTS_TXT=true
SCRAPING_USER_AGENT="AI-Knowledge-Bot/1.0 (+https://ai-knowledge.org/bot)"
SCRAPING_REQUEST_DELAY=1.0
SCRAPING_MAX_RETRIES=3
SCRAPING_TIMEOUT=30
SCRAPING_CONCURRENT_REQUESTS=5

# Performance Configuration
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30
REDIS_MAX_CONNECTIONS=100
WORKER_CONCURRENCY=4
```

**Staging Environment Variables:**

```bash
# Staging environment
ENVIRONMENT=staging
DEBUG=true
LOG_LEVEL=DEBUG
DATABASE_URL=postgresql://staging-user:password@staging-db:5432/ai_knowledge_staging
REDIS_URL=redis://redis-staging:6379/0
SECRET_KEY=your-staging-secret-key

# Reduced resource usage for staging
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=15
REDIS_MAX_CONNECTIONS=50
WORKER_CONCURRENCY=2
SCRAPING_CONCURRENT_REQUESTS=2
```

## Deployment Procedures

### Container Build and Push

Build and push containers to the registry:

```bash
# Build all containers
make docker-build

# Tag for production
docker tag ai-knowledge/pipeline:latest ai-knowledge/pipeline:v1.2.3
docker tag ai-knowledge/frontend:latest ai-knowledge/frontend:v1.2.3

# Push to registry
docker push ai-knowledge/pipeline:v1.2.3
docker push ai-knowledge/frontend:v1.2.3
```

### Kubernetes Deployment

**Pipeline Deployment:**

```yaml
# k8s/deployments/pipeline-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-knowledge-pipeline
  namespace: ai-knowledge-prod
  labels:
    app: ai-knowledge
    component: pipeline
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-knowledge
      component: pipeline
  template:
    metadata:
      labels:
        app: ai-knowledge
        component: pipeline
    spec:
      containers:
      - name: pipeline
        image: ai-knowledge/pipeline:v1.2.3
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: 250m
            memory: 512Mi
          limits:
            cpu: 1000m
            memory: 2Gi
        env:
        - name: ENVIRONMENT
          valueFrom:
            configMapKeyRef:
              name: ai-knowledge-config
              key: ENVIRONMENT
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: ai-knowledge-secrets
              key: DATABASE_URL
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      nodeSelector:
        workload: pipeline
```

**Frontend Deployment:**

```yaml
# k8s/deployments/frontend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-knowledge-frontend
  namespace: ai-knowledge-prod
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ai-knowledge
      component: frontend
  template:
    metadata:
      labels:
        app: ai-knowledge
        component: frontend
    spec:
      containers:
      - name: frontend
        image: ai-knowledge/frontend:v1.2.3
        ports:
        - containerPort: 3000
        resources:
          requests:
            cpu: 100m
            memory: 256Mi
          limits:
            cpu: 500m
            memory: 1Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 15
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 5
      nodeSelector:
        workload: web
```

### Service Configuration

**Pipeline Service:**

```yaml
# k8s/services/pipeline-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ai-knowledge-pipeline
  namespace: ai-knowledge-prod
spec:
  selector:
    app: ai-knowledge
    component: pipeline
  ports:
  - name: http
    port: 8000
    targetPort: 8000
  type: ClusterIP
```

**Frontend Service:**

```yaml
# k8s/services/frontend-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ai-knowledge-frontend
  namespace: ai-knowledge-prod
spec:
  selector:
    app: ai-knowledge
    component: frontend
  ports:
  - name: http
    port: 3000
    targetPort: 3000
  type: ClusterIP
```

### Ingress Configuration

```yaml
# k8s/ingress/production-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ai-knowledge-ingress
  namespace: ai-knowledge-prod
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - ai-knowledge.org
    - api.ai-knowledge.org
    secretName: ai-knowledge-tls
  rules:
  - host: ai-knowledge.org
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ai-knowledge-frontend
            port:
              number: 3000
  - host: api.ai-knowledge.org
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ai-knowledge-pipeline
            port:
              number: 8000
```

### Deployment Commands

Apply all configurations:

```bash
# Create namespace
kubectl apply -f k8s/namespaces/production.yaml

# Apply configurations and secrets
kubectl apply -f k8s/configs/production-config.yaml
kubectl apply -f k8s/secrets/production-secrets.yaml

# Deploy applications
kubectl apply -f k8s/deployments/
kubectl apply -f k8s/services/
kubectl apply -f k8s/ingress/

# Verify deployment
kubectl get pods -n ai-knowledge-prod
kubectl get services -n ai-knowledge-prod
kubectl get ingress -n ai-knowledge-prod
```

## Database Migrations

### Initial Database Setup

```bash
# Create database and initial schema
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  python -m pipelines.database.init_db

# Run initial migrations
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  python -m pipelines.database.migrate
```

### Schema Migration Process

1. **Create Migration Script:**

```python
# migrations/003_add_quality_score.py
def upgrade():
    """Add quality_score column to content_items table."""
    return [
        """
        ALTER TABLE content_items 
        ADD COLUMN quality_score FLOAT DEFAULT 0.0;
        """,
        """
        CREATE INDEX idx_content_items_quality_score 
        ON content_items(quality_score DESC);
        """
    ]

def downgrade():
    """Remove quality_score column."""
    return [
        "DROP INDEX IF EXISTS idx_content_items_quality_score;",
        "ALTER TABLE content_items DROP COLUMN IF EXISTS quality_score;"
    ]
```

2. **Test Migration in Staging:**

```bash
# Run in staging first
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-staging -- \
  python -m pipelines.database.migrate --target 003

# Verify migration success
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-staging -- \
  python -m pipelines.database.verify_migration 003
```

3. **Run Production Migration:**

```bash
# Backup database before migration
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  python -m pipelines.database.backup --name "pre-migration-$(date +%Y%m%d)"

# Run migration
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  python -m pipelines.database.migrate --target 003

# Verify migration
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  python -m pipelines.database.verify_migration 003
```

## Monitoring Setup

### Prometheus Configuration

```yaml
# k8s/monitoring/prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'ai-knowledge-pipeline'
      kubernetes_sd_configs:
      - role: pod
        namespaces:
          names: ['ai-knowledge-prod']
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_component]
        action: keep
        regex: pipeline
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
```

### Grafana Dashboard

Import the provided Grafana dashboard:

```bash
# Import dashboard
kubectl create configmap grafana-dashboard \
  --from-file=monitoring/grafana-dashboard.json \
  -n monitoring
```

### Alerting Rules

```yaml
# k8s/monitoring/alert-rules.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: ai-knowledge-alerts
  namespace: ai-knowledge-prod
spec:
  groups:
  - name: ai-knowledge.rules
    rules:
    - alert: HighErrorRate
      expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "High error rate detected"
        
    - alert: PipelineJobFailure
      expr: pipeline_job_failures_total > 0
      for: 1m
      labels:
        severity: warning
      annotations:
        summary: "Pipeline job failed"
        
    - alert: DatabaseConnectionFailure
      expr: database_connections_failed_total > 0
      for: 1m
      labels:
        severity: critical
      annotations:
        summary: "Database connection failure"
```

## SSL/TLS Configuration

### Certificate Management with cert-manager

1. **Install cert-manager:**

```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
```

2. **Configure ClusterIssuer:**

```yaml
# k8s/ssl/cluster-issuer.yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@ai-knowledge.org
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
```

3. **Certificate will be automatically provisioned via Ingress annotations**

### Manual SSL Configuration

For custom certificates:

```bash
# Create TLS secret from certificate files
kubectl create secret tls ai-knowledge-tls \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key \
  -n ai-knowledge-prod
```

## Scaling and Load Balancing

### Horizontal Pod Autoscaler

```yaml
# k8s/autoscaling/pipeline-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-knowledge-pipeline-hpa
  namespace: ai-knowledge-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-knowledge-pipeline
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Vertical Pod Autoscaler

```yaml
# k8s/autoscaling/pipeline-vpa.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: ai-knowledge-pipeline-vpa
  namespace: ai-knowledge-prod
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-knowledge-pipeline
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: pipeline
      maxAllowed:
        cpu: 2
        memory: 4Gi
      minAllowed:
        cpu: 100m
        memory: 256Mi
```

### Load Balancer Configuration

```yaml
# k8s/networking/load-balancer.yaml
apiVersion: v1
kind: Service
metadata:
  name: ai-knowledge-lb
  namespace: ai-knowledge-prod
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    service.beta.kubernetes.io/aws-load-balancer-ssl-cert: "arn:aws:acm:region:account:certificate/cert-id"
spec:
  type: LoadBalancer
  selector:
    app: ai-knowledge
  ports:
  - name: https
    port: 443
    targetPort: 8000
  - name: http
    port: 80
    targetPort: 8000
```

## Troubleshooting

### Common Deployment Issues

**1. Pod Startup Failures:**

```bash
# Check pod status
kubectl get pods -n ai-knowledge-prod

# View pod logs
kubectl logs -f deployment/ai-knowledge-pipeline -n ai-knowledge-prod

# Describe pod for events
kubectl describe pod <pod-name> -n ai-knowledge-prod
```

**2. Configuration Issues:**

```bash
# Check ConfigMap
kubectl get configmap ai-knowledge-config -n ai-knowledge-prod -o yaml

# Check Secrets
kubectl get secret ai-knowledge-secrets -n ai-knowledge-prod

# Verify environment variables in pod
kubectl exec -it <pod-name> -n ai-knowledge-prod -- env | grep -E "(DATABASE|REDIS)"
```

**3. Network Connectivity:**

```bash
# Test service connectivity
kubectl exec -it <pod-name> -n ai-knowledge-prod -- curl http://ai-knowledge-pipeline:8000/health

# Check ingress status
kubectl get ingress -n ai-knowledge-prod
kubectl describe ingress ai-knowledge-ingress -n ai-knowledge-prod
```

**4. Database Connection Issues:**

```bash
# Test database connectivity
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  python -c "from pipelines.database.connection import get_connection; print(get_connection().info)"

# Check database logs (if using cloud SQL)
gcloud sql instances describe ai-knowledge-prod --format="value(backendType,state)"
```

### Performance Issues

**1. High Memory Usage:**

```bash
# Check resource usage
kubectl top pods -n ai-knowledge-prod

# Increase memory limits if needed
kubectl patch deployment ai-knowledge-pipeline -n ai-knowledge-prod \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"pipeline","resources":{"limits":{"memory":"4Gi"}}}]}}}}'
```

**2. Database Performance:**

```bash
# Check active connections
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  python -c "
import psycopg2
conn = psycopg2.connect('$DATABASE_URL')
cur = conn.cursor()
cur.execute('SELECT count(*) FROM pg_stat_activity;')
print(f'Active connections: {cur.fetchone()[0]}')
"
```

### Rollback Procedures

**1. Deployment Rollback:**

```bash
# View deployment history
kubectl rollout history deployment/ai-knowledge-pipeline -n ai-knowledge-prod

# Rollback to previous version
kubectl rollout undo deployment/ai-knowledge-pipeline -n ai-knowledge-prod

# Rollback to specific revision
kubectl rollout undo deployment/ai-knowledge-pipeline -n ai-knowledge-prod --to-revision=2
```

**2. Database Rollback:**

```bash
# List available backups
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  python -m pipelines.database.list_backups

# Restore from backup
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  python -m pipelines.database.restore --backup-name "backup-20240106-120000"
```

### Health Checks

**Application Health:**

```bash
# Pipeline health
curl https://api.ai-knowledge.org/health

# Frontend health
curl https://ai-knowledge.org/health

# Database health
kubectl exec -it deployment/ai-knowledge-pipeline -n ai-knowledge-prod -- \
  python -m pipelines.health_check --component database
```

**Infrastructure Health:**

```bash
# Cluster health
kubectl get nodes
kubectl get componentstatuses

# Resource usage
kubectl top nodes
kubectl top pods -n ai-knowledge-prod
```

---

**Last Updated**: January 2024  
**Version**: 1.0.0