# GitHub Secrets Configuration Guide

## Overview
This document lists all GitHub secrets required for the AI Knowledge Website CI/CD pipelines.

## Required Secrets by Priority

### 🔴 **Critical (Required for Basic CI/CD)**

#### 1. **GITHUB_TOKEN**
- **Description**: GitHub token for repository operations
- **Used in**: Most workflows
- **Note**: Usually auto-provided by GitHub Actions, no manual configuration needed
- **Configuration**: Automatic

### 🟡 **Important (Required for Core Features)**

#### 2. **OPENAI_API_KEY**
- **Description**: OpenAI API key for content enrichment
- **Used in**: Content pipeline, testing
- **Required for**: AI-powered summarization and enrichment
- **How to get**: https://platform.openai.com/api-keys
- **Format**: `sk-...`

```bash
gh secret set OPENAI_API_KEY --body "sk-your-api-key-here"
```

#### 3. **ANTHROPIC_API_KEY**
- **Description**: Anthropic Claude API key for content processing
- **Used in**: Content pipeline, testing
- **Required for**: Alternative AI provider for enrichment
- **How to get**: https://console.anthropic.com/
- **Format**: `sk-ant-...`

```bash
gh secret set ANTHROPIC_API_KEY --body "sk-ant-your-api-key-here"
```

#### 4. **PIPELINE_DB_PASSWORD**
- **Description**: PostgreSQL password for pipeline database
- **Used in**: Content pipeline
- **Required for**: Pipeline state persistence
- **Format**: Strong password

```bash
gh secret set PIPELINE_DB_PASSWORD --body "your-secure-password"
```

### 🟢 **Optional (For Production Deployment)**

#### 5. **PRODUCTION_DATABASE_URL**
- **Description**: Production PostgreSQL connection string
- **Used in**: Production deployment
- **Format**: `postgresql://user:password@host:5432/dbname`
- **Example**: `postgresql://prod_user:pass@db.example.com:5432/ai_knowledge`

```bash
gh secret set PRODUCTION_DATABASE_URL --body "postgresql://..."
```

#### 6. **STAGING_DATABASE_URL**
- **Description**: Staging PostgreSQL connection string
- **Used in**: Staging deployment
- **Format**: Same as production

```bash
gh secret set STAGING_DATABASE_URL --body "postgresql://..."
```

#### 7. **PRODUCTION_CLOUDFRONT_ID** / **STAGING_CLOUDFRONT_ID**
- **Description**: AWS CloudFront distribution IDs
- **Used in**: CDN cache invalidation
- **Format**: CloudFront distribution ID
- **Example**: `E1234567890ABC`

```bash
gh secret set PRODUCTION_CLOUDFRONT_ID --body "E1234567890ABC"
gh secret set STAGING_CLOUDFRONT_ID --body "E0987654321XYZ"
```

### 🔵 **Optional (For Monitoring & Alerts)**

#### 8. **SLACK_WEBHOOK_URL**
- **Description**: Slack webhook for deployment notifications
- **Used in**: Deployment workflow
- **How to get**: https://api.slack.com/messaging/webhooks
- **Format**: `https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXX`

```bash
gh secret set SLACK_WEBHOOK_URL --body "https://hooks.slack.com/services/..."
```

#### 9. **ALERT_WEBHOOK_URL**
- **Description**: Generic webhook for alerts
- **Used in**: Content pipeline, deployment
- **Format**: URL to your alerting service

```bash
gh secret set ALERT_WEBHOOK_URL --body "https://your-alerting-service/webhook"
```

#### 10. **SECURITY_WEBHOOK_URL**
- **Description**: Webhook for security alerts
- **Used in**: Security scanning workflow
- **Format**: URL to security monitoring service

```bash
gh secret set SECURITY_WEBHOOK_URL --body "https://security-service/webhook"
```

#### 11. **MONITORING_WEBHOOK_URL**
- **Description**: Webhook for monitoring alerts
- **Used in**: Deployment workflow
- **Format**: URL to monitoring service

```bash
gh secret set MONITORING_WEBHOOK_URL --body "https://monitoring-service/webhook"
```

#### 12. **NOTIFICATION_WEBHOOK_URL**
- **Description**: General notification webhook
- **Used in**: Dependency updates
- **Format**: URL to notification service

```bash
gh secret set NOTIFICATION_WEBHOOK_URL --body "https://notification-service/webhook"
```

#### 13. **METRICS_ENDPOINT**
- **Description**: Endpoint for sending metrics
- **Used in**: Content pipeline
- **Format**: URL to metrics collection service

```bash
gh secret set METRICS_ENDPOINT --body "https://metrics-service/api/metrics"
```

#### 14. **LHCI_GITHUB_APP_TOKEN**
- **Description**: Token for Lighthouse CI GitHub App
- **Used in**: CI workflow for performance testing
- **How to get**: https://github.com/apps/lighthouse-ci
- **Format**: GitHub App token

```bash
gh secret set LHCI_GITHUB_APP_TOKEN --body "your-lhci-token"
```

## Quick Setup Commands

### Minimal Setup (To Get CI/CD Working)
```bash
# Set essential secrets for basic functionality
gh secret set OPENAI_API_KEY --body "sk-your-openai-key"
gh secret set ANTHROPIC_API_KEY --body "sk-ant-your-anthropic-key"
gh secret set PIPELINE_DB_PASSWORD --body "secure-password-here"
```

### Production Setup
```bash
# Add production deployment secrets
gh secret set PRODUCTION_DATABASE_URL --body "postgresql://..."
gh secret set STAGING_DATABASE_URL --body "postgresql://..."
gh secret set PRODUCTION_CLOUDFRONT_ID --body "E1234567890ABC"
gh secret set STAGING_CLOUDFRONT_ID --body "E0987654321XYZ"
```

### Full Monitoring Setup
```bash
# Add all monitoring and alert webhooks
gh secret set SLACK_WEBHOOK_URL --body "https://hooks.slack.com/..."
gh secret set ALERT_WEBHOOK_URL --body "https://..."
gh secret set SECURITY_WEBHOOK_URL --body "https://..."
gh secret set MONITORING_WEBHOOK_URL --body "https://..."
gh secret set NOTIFICATION_WEBHOOK_URL --body "https://..."
gh secret set METRICS_ENDPOINT --body "https://..."
```

## Verification

Check which secrets are configured:
```bash
gh secret list
```

## Security Best Practices

1. **Never commit secrets** to the repository
2. **Use strong passwords** for database credentials
3. **Rotate API keys** regularly
4. **Limit secret access** to required workflows only
5. **Use environment-specific secrets** (staging vs production)
6. **Monitor secret usage** in GitHub audit logs

## Troubleshooting

### Workflow fails with "secret not found"
- Check if the secret name matches exactly (case-sensitive)
- Verify the secret is set: `gh secret list`
- Ensure the workflow has permission to access the secret

### API key errors
- Verify the API key is valid and not expired
- Check API rate limits
- Ensure proper formatting (no extra spaces or quotes)

### Database connection errors
- Verify the connection string format
- Check network accessibility from GitHub Actions
- Ensure database allows connections from GitHub IPs

## Environment Variables vs Secrets

Some values can be set as environment variables in workflows instead of secrets if they're not sensitive:

- `NODE_VERSION`: Already set in workflows
- `PYTHON_VERSION`: Already set in workflows
- `DATABASE_HOST`: Can be public if using cloud services
- `DATABASE_PORT`: Usually 5432 for PostgreSQL
- `DATABASE_NAME`: Can be public

## Next Steps

1. Configure minimal secrets to get CI/CD working
2. Add production secrets when ready to deploy
3. Set up monitoring webhooks for better observability
4. Consider using GitHub Environments for better secret management