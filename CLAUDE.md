# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI knowledge website project that provides a glossary and articles about Artificial Intelligence. The project uses:

- **Content**: Markdown/MDX files with Astro Content Collections + Zod validation
- **Site**: Astro-based presentation layer
- **Pipeline**: Automated AI-powered content processing (ingest → normalize → dedup → enrich → publish)
- **CI/CD**: GitHub Actions for automated quality checks and deployment

## Core Architecture

### Directory Structure

- `apps/site/` - Astro presentation layer with content collections
- `pipelines/` - ETL pipeline modules (ingest, normalize, dedup, enrich, publish)
- `orchestrators/langgraph/` - LangGraph-based workflow orchestration
- `scripts/` - CLI utilities and migrations
- `tests/` - Unit/integration tests for pipelines and schemas
- `data/sources/` - Raw HTML/JSON snapshots
- `data/curated/` - Normalized/enriched output

### Content Management

- All content in Markdown with required frontmatter validated by Zod schemas
- Schema definitions in `apps/site/src/content/config.ts`
- Controlled taxonomy in `apps/site/src/content/taxonomies/`
- Duplicate detection using simhash/LSH algorithms
- Cross-linking between related glossary entries

## Common Commands

Development workflow:

```bash
make install     # Install all dependencies
make dev        # Start Astro dev server
make build      # Build and validate entire project
make test       # Run pipeline tests
```

Content pipeline:

```bash
make ingest     # Ingest from external sources
make publish    # Generate MD files from curated data
python pipelines/run_graph.py --flow [ingest|normalize|dedup|enrich|publish|full]
```

Quality checks (all must pass for PR):

```bash
# These are run automatically in CI
npm run build          # Astro build + Zod validation
npm run lint           # Markdown linting
scripts/link_check.py  # Check all links
pytest                 # Pipeline tests
scripts/dedup_check.py # Duplicate detection
```

## Content Creation Rules

### Glossary Entries

- File: `apps/site/src/content/glossary/<slug>.md`
- Required frontmatter: `title`, `slug` (kebab-case), `summary` (120-160 words), `tags`, `updated`
- Optional: `aliases`, `related`, `sources` (with license info)
- Always run `make build` to validate schema compliance

### External Source Integration

1. Add configuration to `pipelines/ingest/sources.yaml`
2. Run ingest pipeline to fetch content
3. Check for duplicates with dedup tools
4. Generate markdown via publish pipeline
5. If near-duplicate detected, mark PR title with `merge-candidate`

## Development Guidelines

### File Operations

- Always check if slug/filename already exists before creating
- Prefer editing existing files over creating new ones
- Keep changes atomic and focused
- Show clear diffs for refactoring proposals

### Content Quality

- Validate all frontmatter against Zod schemas
- Use controlled taxonomy for tags
- Include proper source attribution with licenses
- Maintain cross-references between related entries
- No promotional language in summaries

### Pipeline Development

- Make commands idempotent and safe
- Use canonical URL normalization for sources
- Implement proper deduplication (simhash + MinHash LSH)
- Respect robots.txt and implement ethical scraping practices

## Schema Migration Process

When updating content schemas:

1. Modify `apps/site/src/content/config.ts`
2. Run `scripts/migrate_frontmatter.py` for bulk updates
3. Update corresponding tests in `tests/content/`
4. Verify with `make build`

## PR Requirements

Every PR must pass:

- [ ] Astro build + Zod validation
- [ ] Markdown linting
- [ ] Link checking
- [ ] No duplicate content (simhash/LSH)
- [ ] Source licensing compliance
- [ ] Pipeline tests

## Content Frontmatter Examples

Glossary entry:

```yaml
title: Agent (AI)
slug: agent
aliases: ["autonomous agent", "tool-using agent"]
summary: An agent is an AI system that pursues goals, maintains state and uses external tools in a perceive→plan→act→observe cycle.
tags: ["agents", "orchestration", "ai-engineering"]
related: ["langgraph", "toolformer"]
updated: "2025-09-06"
sources:
  - source_url: "https://langchain-ai.github.io/langgraph/"
    source_title: "LangGraph docs"
    license: "proprietary"
```

## Commit Conventions

- Prefixes: `feat(site)`, `feat(pipeline)`, `fix`, `docs`, `chore`, `refactor`, `test`, `ci`
- Present tense, 72 char wrap
- Include detailed body when necessary

## Working Scope

Only work within:

- `apps/site/**` - presentation and content
- `pipelines/**` - data processing
- `scripts/**` - utilities
- `tests/**` - test files
- `.github/workflows/**` - CI/CD

Do not create files outside these directories without explicit permission.
