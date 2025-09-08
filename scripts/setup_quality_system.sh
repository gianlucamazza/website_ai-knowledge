#!/bin/bash
# Enterprise Markdown Quality Control System Setup
# Complete installation and configuration script

set -e

echo "🎯 Setting up Enterprise Markdown Quality Control System..."
echo "================================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        log_error "Node.js is not installed. Please install Node.js 18+ and try again."
        exit 1
    fi
    
    NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
    if [ "$NODE_VERSION" -lt 18 ]; then
        log_error "Node.js version 18 or higher required. Found: $(node -v)"
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed. Please install Python 3.8+ and try again."
        exit 1
    fi
    
    # Check npm
    if ! command -v npm &> /dev/null; then
        log_error "npm is not installed. Please install npm and try again."
        exit 1
    fi
    
    log_success "System requirements check passed"
}

# Install Node dependencies
install_node_dependencies() {
    log_info "Installing Node.js dependencies..."
    
    cd apps/site
    
    # Install base dependencies if not already installed
    if [ ! -d "node_modules" ]; then
        npm ci
    fi
    
    # Ensure markdownlint dependencies are installed
    npm install --save-dev markdownlint-cli markdownlint-cli2 markdownlint-cli2-formatter-pretty
    
    cd ../..
    
    log_success "Node.js dependencies installed"
}

# Install Python dependencies
install_python_dependencies() {
    log_info "Installing Python dependencies..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        log_info "Created virtual environment"
    fi
    
    # Activate virtual environment and install dependencies
    source venv/bin/activate
    pip install --upgrade pip
    pip install python-frontmatter pyyaml
    
    log_success "Python dependencies installed"
}

# Set up pre-commit hooks
setup_pre_commit() {
    log_info "Setting up pre-commit hooks..."
    
    # Install pre-commit if not already installed
    if ! command -v pre-commit &> /dev/null; then
        pip install pre-commit
    fi
    
    # Install pre-commit hooks
    pre-commit install
    
    log_success "Pre-commit hooks installed"
}

# Make scripts executable
make_scripts_executable() {
    log_info "Making scripts executable..."
    
    chmod +x scripts/migrate_current_violations.py
    chmod +x scripts/markdown_quality_fixer.py
    chmod +x scripts/markdown_quality_hook.py
    chmod +x scripts/quality_dashboard.py
    chmod +x scripts/validate_frontmatter_hook.py
    chmod +x scripts/pre-commit-link-check.sh
    
    log_success "Scripts made executable"
}

# Run initial quality check
run_quality_check() {
    log_info "Running initial quality check..."
    
    cd apps/site
    
    # Run linting
    if npm run lint; then
        log_success "✅ All markdown files pass quality checks!"
    else
        log_warning "⚠️  Some quality issues remain. Use the migration script to fix them:"
        echo "   python scripts/migrate_current_violations.py apps/site/src/content --apply"
    fi
    
    cd ../..
}

# Generate quality dashboard
generate_dashboard() {
    log_info "Generating quality dashboard..."
    
    python scripts/quality_dashboard.py apps/site/src/content --html-output quality_report.html
    
    if [ -f "quality_report.html" ]; then
        log_success "Quality dashboard generated: quality_report.html"
    fi
}

# Display setup completion message
show_completion_message() {
    echo ""
    echo "🎉 ENTERPRISE MARKDOWN QUALITY CONTROL SYSTEM SETUP COMPLETE!"
    echo "=============================================================="
    echo ""
    echo "📋 What was installed:"
    echo "   ✅ Enhanced markdownlint configuration"
    echo "   ✅ Pre-commit hooks for local quality gates"
    echo "   ✅ GitHub Actions workflow for CI/CD"
    echo "   ✅ Migration scripts for fixing violations"
    echo "   ✅ Quality dashboard for monitoring"
    echo ""
    echo "🛠️  Available Commands:"
    echo "   • Check quality:        npm run lint"
    echo "   • Fix violations:       python scripts/migrate_current_violations.py apps/site/src/content --apply"
    echo "   • Generate dashboard:   python scripts/quality_dashboard.py apps/site/src/content"
    echo "   • Run pre-commit:       pre-commit run --all-files"
    echo ""
    echo "📊 Quality Status:"
    python scripts/quality_dashboard.py apps/site/src/content --quiet
    echo ""
    echo "📖 Next Steps:"
    echo "   1. Review the quality report: open quality_report.html"
    echo "   2. Run 'git add .' and 'git commit' to test pre-commit hooks"
    echo "   3. Push changes to trigger GitHub Actions workflow"
    echo "   4. Set up team guidelines for markdown quality standards"
    echo ""
    echo "🔗 Documentation:"
    echo "   • Quality standards: See .markdownlint.json"
    echo "   • Pre-commit config: See .pre-commit-config.yaml"
    echo "   • CI/CD workflow: See .github/workflows/markdown-quality.yml"
    echo ""
    log_success "System is ready for production use!"
}

# Main execution flow
main() {
    echo "Starting enterprise markdown quality control system setup..."
    echo ""
    
    check_requirements
    install_node_dependencies
    install_python_dependencies
    make_scripts_executable
    setup_pre_commit
    run_quality_check
    generate_dashboard
    show_completion_message
}

# Run main function
main "$@"