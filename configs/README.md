# Configuration Files

This directory contains centralized configuration files for the AI Knowledge Website project.

## 📁 Configuration Files

### Code Quality & Testing
- **`.coveragerc`** - Python code coverage configuration
- **`.bandit`** - Python security linting configuration

### Development Tools  
- **`.actrc`** - Act (GitHub Actions local runner) configuration
- **`.env.act`** - Environment variables for Act testing

### Related Configuration
- **`../pyproject.toml`** - Python project configuration (packaging, tools)
- **`../pytest.ini`** - Python testing configuration
- **`../.pre-commit-config.yaml`** - Pre-commit hooks configuration
- **`../codecov.yml`** - Code coverage reporting configuration

## 🔗 Symlinks

For backward compatibility, symlinks are maintained in the project root:
- `.coveragerc` → `configs/.coveragerc`
- `.actrc` → `configs/.actrc` 
- `.env.act` → `configs/.env.act`
- `.bandit` → `configs/.bandit`

## 🎯 Configuration Management

### Python Tools Configuration
All Python-related configurations are consolidated:
- **Testing**: pytest.ini, .coveragerc
- **Code Quality**: pyproject.toml (black, isort, mypy)
- **Security**: .bandit

### Local Development
- **Act Testing**: .actrc, .env.act for local GitHub Actions simulation
- **Pre-commit**: Automated code quality checks

### Best Practices
1. **Centralization**: Keep configurations organized in this directory
2. **Backward Compatibility**: Maintain symlinks for tools expecting root location
3. **Documentation**: Update this README when adding new configurations
4. **Security**: Never include sensitive values directly in config files

---

For tool-specific configuration details, see the individual files above.