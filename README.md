# AI Knowledge Website

[![Private Repository](https://img.shields.io/badge/Repository-Private-red)](https://github.com/gianlucamazza/website_ai-knowledge)
[![GitHub Issues](https://img.shields.io/badge/Issues-Enabled-green)](https://github.com/gianlucamazza/website_ai-knowledge/issues)
[![GitHub Discussions](https://img.shields.io/badge/Discussions-Enabled-blue)](https://github.com/gianlucamazza/website_ai-knowledge/discussions)
[![Protected Branch](https://img.shields.io/badge/Main-Protected-yellow)](https://github.com/gianlucamazza/website_ai-knowledge)

A comprehensive, enterprise-grade platform for automated AI knowledge aggregation, curation, and publication. This project combines an Astro-based frontend with a sophisticated Python content pipeline to deliver high-quality, continuously updated AI knowledge resources.

## Quick Start

### Prerequisites

- Node.js 18+ and npm 8+
- Python 3.9+ with pip
- PostgreSQL 14+
- Redis 7+
- Git

### Development Setup

```bash
# Clone the repository (private - requires authentication)
git clone https://github.com/gianlucamazza/website_ai-knowledge.git
cd website_ai-knowledge

# Install dependencies
make install

# Set up environment
cp .env.example .env
# Edit .env with your configuration

# Initialize database
make db-setup

# Start development services
make dev
```

Access the site at `http://localhost:4321`

## Project Structure

```
website_ai-knowledge/
├── apps/site/              # Astro frontend application
│   ├── src/
│   │   ├── components/     # Reusable UI components
│   │   ├── content/        # Content collections (articles, glossary)
│   │   ├── layouts/        # Page layout templates
│   │   └── pages/          # Route definitions
│   └── tests/              # Frontend tests
├── pipelines/              # Python content processing pipeline
│   ├── ingest/             # Content source ingestion
│   ├── normalize/          # Data cleaning and standardization
│   ├── dedup/              # Duplicate detection algorithms
│   ├── enrich/             # Content enhancement
│   ├── publish/            # Output generation
│   └── orchestrators/      # LangGraph workflow management
├── security/               # Security modules and compliance
├── tests/                  # Python test suite
├── scripts/                # Automation and utility scripts
└── docs/                   # Comprehensive documentation
```

## Core Features

### Content Pipeline

- **Automated Ingestion**: Ethical web scraping with rate limiting and robots.txt compliance
- **Duplicate Detection**: Advanced SimHash and LSH algorithms with >98% accuracy
- **Content Enrichment**: AI-powered summarization, tagging, and cross-linking
- **Quality Assurance**: Multi-stage validation and schema compliance
- **Workflow Orchestration**: LangGraph-based pipeline management

### Frontend Application

- **Static Site Generation**: Optimized Astro-based site with excellent performance
- **Content Collections**: Zod-validated content with structured metadata
- **Search & Navigation**: Full-text search and intelligent content discovery
- **Responsive Design**: Mobile-first design with accessibility compliance
- **SEO Optimization**: Structured data and meta tag management

### Enterprise Features

- **Security**: Zero-trust architecture with comprehensive input validation
- **Monitoring**: Prometheus metrics, structured logging, and alerting
- **Scalability**: Horizontal scaling with Kubernetes deployment
- **Compliance**: GDPR, copyright, and ethical AI compliance
- **CI/CD**: Automated testing, quality gates, and deployment pipelines

## Documentation

### Getting Started
- [Development Guide](docs/DEVELOPMENT_GUIDE.md) - Local development setup and workflow
- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md) - Production deployment procedures

### Technical Documentation
- [Architecture Overview](ARCHITECTURE.md) - Comprehensive system architecture
- [API Documentation](docs/API_DOCUMENTATION.md) - Pipeline API reference
- [Code Standards](docs/CODE_STANDARDS.md) - Code quality and style guidelines

### Operational Documentation
- [CI/CD Documentation](docs/ci-cd/) - Continuous integration and deployment
- [Monitoring Guide](docs/MONITORING_GUIDE.md) - System monitoring and alerting
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions
- [Maintenance Schedule](docs/MAINTENANCE_SCHEDULE.md) - Regular maintenance tasks

### Security Documentation
- [Security Overview](docs/SECURITY_OVERVIEW.md) - Security architecture and practices
- [Incident Response](docs/INCIDENT_RESPONSE.md) - Security incident procedures
- [Compliance Guide](docs/COMPLIANCE_GUIDE.md) - Regulatory compliance procedures

## Technology Stack

### Frontend
- **Framework**: Astro 4.x with TypeScript
- **Styling**: Tailwind CSS
- **Validation**: Zod schemas
- **Testing**: Vitest, Playwright

### Backend Pipeline
- **Language**: Python 3.9+
- **Orchestration**: LangGraph
- **Database**: PostgreSQL 14+
- **Cache**: Redis 7+
- **Processing**: Celery, FastAPI

### Infrastructure
- **Containerization**: Docker
- **Orchestration**: Kubernetes
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus, Grafana
- **Logging**: Structured JSON logging

## Performance Metrics

- **Duplicate Detection**: >98% precision, <2% false positives
- **Pipeline Processing**: <30 minutes for full content refresh
- **Site Build Time**: <5 minutes for incremental builds
- **API Response Time**: <200ms (95th percentile)
- **Uptime**: >99.9% availability target

## Quality Standards

- **Test Coverage**: >95% code coverage requirement
- **Security**: Zero critical vulnerabilities policy
- **Performance**: Core Web Vitals compliance
- **Accessibility**: WCAG 2.1 AA compliance
- **Code Quality**: Automated linting and type checking

## Common Tasks

### Content Management
```bash
# Trigger content ingestion
make ingest

# Validate existing content
make validate

# Check for broken links
make link-check

# Run duplicate detection
make dedup-check
```

### Development
```bash
# Run all tests
make test

# Run specific test suite
make test-unit
make test-integration
make test-performance

# Code quality checks
make lint
make type-check
make security-check
```

### Deployment
```bash
# Deploy to staging
make deploy-staging

# Deploy to production (requires approval)
make deploy-production

# Rollback deployment
make rollback
```

## Contributing

We welcome contributions from the community. Please read our [Contributing Guide](docs/CONTRIBUTING.md) for detailed information on:

- Code style and standards
- Development workflow
- Testing requirements
- Pull request process
- Issue reporting

## Support

- **Documentation**: Comprehensive guides in the `docs/` directory
- **Issue Tracking**: GitHub Issues for bug reports and feature requests
- **Security Issues**: Report to security@example.com (not public issues)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- OpenAI for GPT models used in content processing
- Anthropic for Claude models used in summarization
- The open-source community for the excellent tools and libraries

---

**Project Status**: Production Ready
**Last Updated**: $(date)
**Documentation Version**: 1.0.0