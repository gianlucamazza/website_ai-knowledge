# CI/CD Documentation

This directory contains all documentation related to Continuous Integration and Continuous Deployment for the AI Knowledge Website project.

## 📖 Documentation Index

### Core Documentation
- **[CI_CD_README.md](CI_CD_README.md)** - Overview of CI/CD architecture and workflows
- **[CI_CD_PHASE1_IMPLEMENTATION.md](CI_CD_PHASE1_IMPLEMENTATION.md)** - Phase 1 implementation details and decisions
- **[CI_CD_TEST_FIXES.md](CI_CD_TEST_FIXES.md)** - Testing fixes and improvements

### Workflow Configuration
- **[../.github/workflows/](../.github/workflows/)** - GitHub Actions workflow definitions
  - `ci.yml` - Main CI/CD pipeline
  - `security.yml` - Security scanning
  - `deploy.yml` - Deployment workflows
  - `test-and-coverage.yml` - Comprehensive testing with multi-version support

### Development Tools
- **[../scripts/port-allocator.js](../scripts/port-allocator.js)** - Dynamic port allocation for parallel CI jobs
- **[../act-installation.sh](../act-installation.sh)** - Local GitHub Actions testing setup
- **[../.actrc](../.actrc)** - Act configuration

## 🚀 Quick Start

1. **Local Testing**: Use `act` to test workflows locally
   ```bash
   ./act-installation.sh  # Setup act
   act -j quality-checks  # Test specific job
   ```

2. **Security Setup**: Configure secrets for local testing
   ```bash
   scripts/setup-real-secrets.sh
   ```

3. **Port Allocation**: For parallel testing
   ```bash
   node scripts/port-allocator.js allocate test-job-1
   ```

## 📋 Workflow Overview

Our CI/CD pipeline includes:
- **Quality Gates**: Code formatting, linting, type checking
- **Security Scanning**: Vulnerability detection, secrets scanning
- **Testing**: Unit, integration, and E2E tests
- **Performance**: Lighthouse CI, load testing
- **Deployment**: Automated deployment to production

## 🔒 Security

- All sensitive files are properly gitignored
- Secrets are managed through GitHub Secrets
- Security scanning runs on every PR
- Vulnerability alerts are automated

---

For more details, see the individual documentation files above.