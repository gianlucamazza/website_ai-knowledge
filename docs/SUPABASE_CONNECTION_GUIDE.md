# Supabase Connection Guide

## Connection Methods

Supabase provides different connection methods:

### 1. Direct Connection (Port 5432)
```
postgresql://postgres:[YOUR-PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres
```
- Best for: Long-running servers
- Connection limit: 50 concurrent connections

### 2. Connection Pooler - Transaction Mode (Port 6543)
```
postgresql://postgres.[PROJECT-REF]:[YOUR-PASSWORD]@aws-0-eu-central-1.pooler.supabase.com:6543/postgres?pgbouncer=true
```
- Best for: Serverless functions, high concurrency
- Connection limit: 10,000 concurrent connections
- Note: Some session-based features not available

### 3. Connection Pooler - Session Mode (Port 5432)
```
postgresql://postgres.[PROJECT-REF]:[YOUR-PASSWORD]@aws-0-eu-central-1.pooler.supabase.com:5432/postgres
```
- Best for: Applications needing session features
- Connection limit: 500 concurrent connections

## For Our Project

Based on your Supabase project `dhidxgdfmknosqlthyzm`, the connection URLs would be:

### Direct Connection (IPv6 issue)
```
postgresql://postgres:mdMTtu5CDtvWbkNj@db.dhidxgdfmknosqlthyzm.supabase.co:5432/postgres
```

### Pooler Transaction Mode (RECOMMENDED)
```
postgresql://postgres.dhidxgdfmknosqlthyzm:mdMTtu5CDtvWbkNj@aws-0-eu-central-1.pooler.supabase.com:6543/postgres?pgbouncer=true
```

### Pooler Session Mode
```
postgresql://postgres.dhidxgdfmknosqlthyzm:mdMTtu5CDtvWbkNj@aws-0-eu-central-1.pooler.supabase.com:5432/postgres
```

## How to Find Your Connection String

1. Go to your Supabase Dashboard
2. Navigate to Settings → Database
3. Look for "Connection string" section
4. You'll see tabs for:
   - URI (direct connection)
   - Pooling (connection pooler)
   - Connection parameters

## Troubleshooting

### IPv6 Connection Issues
If you're getting "No route to host" errors, it's likely an IPv6 issue. Solutions:
1. Use the connection pooler (recommended)
2. Add `?sslmode=require` to force SSL
3. Use the pooler hostname which resolves to IPv4

### SSL Certificate Errors
Add `?sslmode=require` or `?sslmode=disable` (not recommended for production)

### Connection Timeout
- Check if your IP is whitelisted (if using IP restrictions)
- Try the connection pooler instead of direct connection
- Verify the password doesn't contain special characters that need encoding

## Update GitHub Secrets

Once you have the correct connection string, update:

```bash
# Update with pooler URL
gh secret set STAGING_DATABASE_URL --body "postgresql://postgres.dhidxgdfmknosqlthyzm:mdMTtu5CDtvWbkNj@aws-0-eu-central-1.pooler.supabase.com:6543/postgres"

gh secret set PRODUCTION_DATABASE_URL --body "postgresql://postgres.dhidxgdfmknosqlthyzm:mdMTtu5CDtvWbkNj@aws-0-eu-central-1.pooler.supabase.com:6543/postgres"
```