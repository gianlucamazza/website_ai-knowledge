-- Initialize test database for Act local testing
-- This script runs when the PostgreSQL container starts

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create test schema
CREATE SCHEMA IF NOT EXISTS test_schema;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE ai_knowledge_test TO postgres;
GRANT ALL PRIVILEGES ON SCHEMA test_schema TO postgres;

-- Create test tables (simplified versions of production tables)
CREATE TABLE IF NOT EXISTS test_schema.content_sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_url TEXT NOT NULL UNIQUE,
    source_title TEXT,
    content_type VARCHAR(50),
    last_crawled TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS test_schema.glossary_entries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    slug VARCHAR(255) NOT NULL UNIQUE,
    title TEXT NOT NULL,
    summary TEXT,
    content TEXT,
    tags TEXT[],
    aliases TEXT[],
    related_entries TEXT[],
    source_id UUID REFERENCES test_schema.content_sources(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_content_sources_url ON test_schema.content_sources(source_url);
CREATE INDEX IF NOT EXISTS idx_glossary_entries_slug ON test_schema.glossary_entries(slug);
CREATE INDEX IF NOT EXISTS idx_glossary_entries_tags ON test_schema.glossary_entries USING gin(tags);

-- Insert test data
INSERT INTO test_schema.content_sources (source_url, source_title, content_type, is_active) VALUES
    ('https://example.com/test1', 'Test Source 1', 'article', true),
    ('https://example.com/test2', 'Test Source 2', 'glossary', true),
    ('https://example.com/test3', 'Test Source 3', 'documentation', false)
ON CONFLICT (source_url) DO NOTHING;

INSERT INTO test_schema.glossary_entries (slug, title, summary, tags, aliases) VALUES
    ('test-entry-1', 'Test Entry 1', 'A test glossary entry for unit testing', ARRAY['test', 'unit'], ARRAY['test1']),
    ('test-entry-2', 'Test Entry 2', 'Another test glossary entry', ARRAY['test', 'integration'], ARRAY['test2']),
    ('artificial-intelligence', 'Artificial Intelligence', 'The simulation of human intelligence in machines', ARRAY['ai', 'machine-learning'], ARRAY['ai', 'ml'])
ON CONFLICT (slug) DO NOTHING;

-- Create test user for application
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_user WHERE usename = 'app_user') THEN
        CREATE USER app_user WITH PASSWORD 'test_password';
        GRANT CONNECT ON DATABASE ai_knowledge_test TO app_user;
        GRANT USAGE ON SCHEMA test_schema TO app_user;
        GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA test_schema TO app_user;
        GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA test_schema TO app_user;
    END IF;
END $$;