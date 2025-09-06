# Database Options for AI Knowledge Website

## Recommended Free Cloud Database Options

### 🥇 **Option 1: Supabase (RECOMMENDED)**
- **Free Tier**: 500 MB database, 2 GB bandwidth, 50 MB file storage
- **PostgreSQL**: Full PostgreSQL 15 support
- **Pros**: 
  - Real PostgreSQL instance
  - Built-in authentication
  - REST API included
  - Excellent for our pipeline needs
- **Setup Time**: 5 minutes
- **URL Format**: `postgresql://postgres:[YOUR-PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres`
- **Sign up**: https://supabase.com

### 🥈 **Option 2: Neon**
- **Free Tier**: 3 GB storage with 1 compute hour/day
- **PostgreSQL**: Serverless PostgreSQL
- **Pros**:
  - Serverless (auto-suspend when not in use)
  - Branching for dev/staging
  - Good for development
- **Cons**: Limited compute hours might affect pipeline runs
- **URL Format**: `postgresql://[user]:[password]@[endpoint].neon.tech/[database]`
- **Sign up**: https://neon.tech

### 🥉 **Option 3: Aiven**
- **Free Tier**: 1 month free trial, then requires credit card
- **PostgreSQL**: Managed PostgreSQL
- **Pros**: Production-ready, reliable
- **Cons**: Only 1 month free
- **Sign up**: https://aiven.io

### **Option 4: ElephantSQL**
- **Free Tier**: 20 MB database (Tiny Turtle plan)
- **PostgreSQL**: Shared PostgreSQL
- **Pros**: Simple, no credit card required
- **Cons**: Very limited storage (20 MB)
- **URL Format**: `postgresql://[user]:[password]@[server].db.elephantsql.com/[database]`
- **Sign up**: https://www.elephantsql.com

### **Option 5: CockroachDB**
- **Free Tier**: 5 GB storage, 250M request units/month
- **PostgreSQL**: PostgreSQL-compatible
- **Pros**: Distributed, scalable
- **Cons**: Some PostgreSQL features not supported
- **Sign up**: https://www.cockroachlabs.com/

## Quick Setup Guide - Supabase (Recommended)

### Step 1: Create Supabase Account
1. Go to https://supabase.com
2. Sign up with GitHub (recommended) or email
3. Create a new project
4. Choose a region close to you (e.g., eu-central-1 for Europe)
5. Generate a strong database password or use the generated one
6. Wait for project to be ready (~2 minutes)

### Step 2: Get Connection Details
1. Go to Settings → Database
2. Find "Connection string" section
3. Copy the "URI" connection string
4. It will look like:
   ```
   postgresql://postgres:[YOUR-PASSWORD]@db.xxxxxxxxxxxx.supabase.co:5432/postgres
   ```

### Step 3: Configure GitHub Secrets
```bash
# For development/staging
gh secret set STAGING_DATABASE_URL --body "postgresql://postgres:[YOUR-PASSWORD]@db.xxxxxxxxxxxx.supabase.co:5432/postgres"

# For production (can use same for now)
gh secret set PRODUCTION_DATABASE_URL --body "postgresql://postgres:[YOUR-PASSWORD]@db.xxxxxxxxxxxx.supabase.co:5432/postgres"

# For pipeline database password (just the password part)
gh secret set PIPELINE_DB_PASSWORD --body "[YOUR-PASSWORD]"
```

### Step 4: Create Required Tables
Once connected, run our migration scripts:

```sql
-- Create schema for pipeline
CREATE SCHEMA IF NOT EXISTS pipeline;

-- Create articles table
CREATE TABLE IF NOT EXISTS pipeline.articles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    url VARCHAR(512) UNIQUE NOT NULL,
    title VARCHAR(512),
    content TEXT,
    summary TEXT,
    source_id VARCHAR(256),
    published_at TIMESTAMP,
    ingested_at TIMESTAMP DEFAULT NOW(),
    processed_at TIMESTAMP,
    quality_score FLOAT,
    status VARCHAR(50) DEFAULT 'pending',
    metadata JSONB
);

-- Create sources table
CREATE TABLE IF NOT EXISTS pipeline.sources (
    id VARCHAR(256) PRIMARY KEY,
    name VARCHAR(256) NOT NULL,
    url VARCHAR(512),
    type VARCHAR(50),
    active BOOLEAN DEFAULT true,
    last_fetched TIMESTAMP,
    config JSONB
);

-- Create duplicates table
CREATE TABLE IF NOT EXISTS pipeline.duplicates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id UUID REFERENCES pipeline.articles(id),
    duplicate_of UUID REFERENCES pipeline.articles(id),
    similarity_score FLOAT,
    detected_at TIMESTAMP DEFAULT NOW()
);

-- Create pipeline_runs table
CREATE TABLE IF NOT EXISTS pipeline.pipeline_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    stage VARCHAR(50),
    status VARCHAR(50),
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    error_message TEXT,
    metrics JSONB
);

-- Create indexes for performance
CREATE INDEX idx_articles_url ON pipeline.articles(url);
CREATE INDEX idx_articles_status ON pipeline.articles(status);
CREATE INDEX idx_articles_source ON pipeline.articles(source_id);
CREATE INDEX idx_pipeline_runs_stage ON pipeline.pipeline_runs(stage);
CREATE INDEX idx_pipeline_runs_status ON pipeline.pipeline_runs(status);
```

## Alternative: Local Development with Docker

If you prefer local development first:

```yaml
# docker-compose.yml
version: '3.8'
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: aiknowledge
      POSTGRES_PASSWORD: localdev123
      POSTGRES_DB: ai_knowledge
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

Then use:
```bash
# Local connection string
postgresql://aiknowledge:localdev123@localhost:5432/ai_knowledge
```

## Cost Considerations

For our AI Knowledge Website:
- **Expected storage**: ~100 MB initially, growing to ~500 MB
- **Expected bandwidth**: ~1-2 GB/month
- **Pipeline runs**: ~30-50 per day

**Supabase Free Tier** is perfect for our needs with room to grow.

## Next Steps

1. Create Supabase account
2. Set up project
3. Configure GitHub secrets with connection strings
4. Run migration scripts
5. Test connection with pipeline

## Security Notes

- Never commit database URLs to the repository
- Use different databases for staging/production
- Enable SSL connections (Supabase has this by default)
- Regularly backup your data
- Monitor usage to stay within free tier limits