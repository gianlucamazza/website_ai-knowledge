# Act Developer Guide

This guide covers how to use Act (GitHub Actions local runner) for local CI/CD validation in the AI Knowledge Website project.

## Overview

Act allows you to run GitHub Actions workflows locally using Docker, enabling:
- **Fast feedback loops**: Validate changes before pushing
- **Debug CI issues**: Test fixes locally without consuming GitHub minutes
- **Offline development**: Work without internet connectivity
- **Cost savings**: Reduce GitHub Actions usage

## Quick Start

### 1. Installation

```bash
# Install Act and setup environment
make act-install
make act-setup
```

### 2. Configuration

Edit the `.secrets` file (created from `.secrets.example`):

```bash
# Required for database tests
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ai_knowledge_test

# Test environment
TEST_ENVIRONMENT=local
TEST_MOCK_AI_APIS=true

# API keys (use placeholders for local testing)
OPENAI_API_KEY=test-key-placeholder
ANTHROPIC_API_KEY=test-key-placeholder
```

### 3. Basic Usage

```bash
# List available workflows
make act-list

# Run unit tests locally
make act-test

# Run all tests
make act-test-all

# Simulate PR workflow
make act-pr
```

## Available Commands

### Test Commands

| Command | Description | Use Case |
|---------|-------------|----------|
| `make act-test` | Unit tests only | Quick feedback during development |
| `make act-test-all` | All test types | Pre-commit validation |
| `make act-test-security` | Security scans | Security-focused changes |
| `make act-test-quality` | Code quality checks | Linting and formatting |
| `make act-test-integration` | Integration tests | Database-related changes |

### Workflow Simulation

| Command | Description | Use Case |
|---------|-------------|----------|
| `make act-pr` | Simulate PR workflow | Before creating PR |
| `make act-push` | Simulate push workflow | Before merging |
| `make act-dry` | Dry run (no execution) | See what would run |

### Maintenance

| Command | Description | Use Case |
|---------|-------------|----------|
| `make act-clean` | Clean containers/cache | Free up disk space |
| `make act-setup` | Re-setup environment | After Docker issues |

## Development Workflow

### 1. Pre-Commit Validation

Before committing changes:

```bash
# Quick validation (recommended)
make pre-commit

# Full validation (comprehensive)
make ci-simulate
```

### 2. Feature Development

When working on new features:

```bash
# Start with unit tests
make act-test

# Add integration tests if needed
make act-test-integration

# Final validation
make act-test-all
```

### 3. Security-Sensitive Changes

For security-related modifications:

```bash
# Run security scans
make act-test-security

# Full security workflow
make act-pr
```

## Workflow Configurations

### Act-Compatible Test Workflow

The `.github/workflows/act-compatible-test.yml` workflow is optimized for local execution:

- **Modular jobs**: Test different aspects independently
- **No external dependencies**: Works offline
- **Fast execution**: Optimized for local development
- **Resource efficient**: Uses minimal Docker resources

### Supported Test Types

| Type | Jobs | Dependencies | Duration |
|------|------|--------------|----------|
| `unit` | Python/Node.js unit tests | None | ~2-3 min |
| `security` | Bandit, Safety, pip-audit | None | ~1-2 min |
| `quality` | Linting, formatting, type checking | None | ~1-2 min |
| `integration` | Database tests | PostgreSQL | ~3-5 min |
| `all` | All of the above | PostgreSQL | ~5-10 min |

## Troubleshooting

### Common Issues

#### 1. Docker Permission Issues

```bash
# On Linux/macOS, ensure Docker daemon is running
sudo systemctl start docker

# Add user to docker group (Linux)
sudo usermod -aG docker $USER
```

#### 2. Port Conflicts

If PostgreSQL port 5432 is in use:

```bash
# Stop local PostgreSQL
brew services stop postgresql  # macOS
sudo systemctl stop postgresql  # Linux

# Or use a different port in .secrets
DATABASE_URL=postgresql://postgres:postgres@localhost:5433/ai_knowledge_test
```

#### 3. Memory Issues

For memory-constrained systems:

```bash
# Run tests one at a time
make act-test          # Unit tests
make act-test-security # Security tests
make act-test-quality  # Quality tests
```

#### 4. Network Issues

Act works offline, but some workflows may need adjustments:

```bash
# Skip external API calls
export TEST_MOCK_AI_APIS=true

# Use cached Docker images
docker image prune -f  # Clear unused images
```

### Debug Mode

For debugging workflow issues:

```bash
# Verbose output
act -v workflow_dispatch -W .github/workflows/act-compatible-test.yml

# Interactive shell access
act --shell workflow_dispatch

# Keep containers after failure
act --rm=false workflow_dispatch
```

## Performance Optimization

### Docker Image Management

```bash
# Pull required images once
docker pull catthehacker/ubuntu:act-latest
docker pull postgres:15

# Clean up unused images
make act-clean
```

### Resource Limits

Configure Docker resource limits in Docker Desktop:
- **Memory**: 4GB minimum, 8GB recommended
- **Disk**: 20GB minimum for container images
- **Swap**: 2GB recommended

### Selective Testing

Use specific test commands instead of `act-test-all` during development:

```bash
# For Python changes
make act-test && make act-test-integration

# For frontend changes
make act-test  # Includes Node.js unit tests

# For security changes
make act-test-security
```

## Integration with Existing Tools

### Git Hooks

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
make pre-commit
```

### IDE Integration

For VS Code, add to `.vscode/tasks.json`:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Act: Run Unit Tests",
      "type": "shell",
      "command": "make act-test",
      "group": "test",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "panel": "new"
      }
    }
  ]
}
```

## Best Practices

### 1. Test Selection Strategy

- **Development**: Use `make act-test` for fast feedback
- **Pre-commit**: Use `make pre-commit` for comprehensive validation
- **Pre-PR**: Use `make act-pr` to simulate full PR workflow
- **Pre-merge**: Use `make ci-simulate` for full pipeline validation

### 2. Resource Management

- Clean up regularly with `make act-clean`
- Monitor Docker disk usage
- Use specific test commands to save time

### 3. Secrets Management

- Never commit `.secrets` file
- Use placeholder values for local testing
- Keep `.secrets.example` updated

### 4. Workflow Maintenance

- Keep act workflows synchronized with main workflows
- Test act compatibility when modifying GitHub Actions
- Update Docker images regularly

## Advanced Usage

### Custom Workflow Testing

Run specific workflows:

```bash
# Test specific workflow file
act -W .github/workflows/security.yml

# Test specific job
act -j python-tests

# Test with custom event
act workflow_dispatch --input test_type=security
```

### Environment Customization

Create custom environment files:

```bash
# .env.local
NODE_ENV=development
DEBUG=true
TEST_PARALLEL=false

# Use with act
act --env-file .env.local
```

### Matrix Testing

Test multiple configurations:

```bash
# Test different Python versions (if configured)
act -W .github/workflows/act-compatible-test.yml \
    --matrix python-version:3.9 \
    --matrix python-version:3.11
```

## Migration from CI-Only Testing

### Step 1: Assess Current Workflows

Review existing workflows for local compatibility:
- External dependencies
- Service requirements
- Secret usage
- Resource requirements

### Step 2: Create Act-Optimized Versions

- Simplify complex workflows
- Mock external services
- Use local databases
- Remove deployment steps

### Step 3: Gradual Adoption

- Start with unit tests
- Add integration tests
- Include security scans
- Full workflow simulation

## Contributing

When modifying Act configurations:

1. Test changes locally first
2. Update documentation
3. Verify Docker image compatibility
4. Test on different operating systems
5. Update troubleshooting guides

## Support

For issues or questions:
- Check troubleshooting section above
- Review Act documentation: https://github.com/nektos/act
- File issues in project repository
- Discuss in team channels