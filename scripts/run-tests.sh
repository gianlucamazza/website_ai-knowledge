#!/bin/bash
# Comprehensive test runner script for local development and CI/CD
set -e

# Configuration
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export TEST_ENV="local"
DB_NAME="ai_knowledge_test"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
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

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        log_error "Node.js is required but not installed"
        exit 1
    fi
    
    # Check PostgreSQL
    if ! command -v psql &> /dev/null; then
        log_warning "PostgreSQL client not found. Database tests may fail."
    fi
    
    log_success "Dependencies check completed"
}

# Setup test environment
setup_test_env() {
    log_info "Setting up test environment..."
    
    # Create test directories
    mkdir -p tests/temp tests/logs tests/reports
    
    # Install Python dependencies
    if [ -f "pipelines/requirements.txt" ]; then
        pip install -r pipelines/requirements.txt > /dev/null 2>&1
    fi
    
    if [ -f "tests/requirements.txt" ]; then
        pip install -r tests/requirements.txt > /dev/null 2>&1
    fi
    
    # Install Node.js dependencies
    if [ -d "apps/site" ] && [ -f "apps/site/package.json" ]; then
        cd apps/site
        npm install > /dev/null 2>&1
        cd ../..
    fi
    
    log_success "Test environment setup completed"
}

# Setup test database
setup_test_database() {
    log_info "Setting up test database..."
    
    # Check if PostgreSQL is running
    if pg_isready -h localhost -p 5432 > /dev/null 2>&1; then
        # Drop and recreate test database
        dropdb --if-exists $DB_NAME > /dev/null 2>&1 || true
        createdb $DB_NAME > /dev/null 2>&1
        
        export DATABASE_URL="postgresql://localhost:5432/$DB_NAME"
        log_success "Test database setup completed"
    else
        log_warning "PostgreSQL not running. Using SQLite for tests."
        export DATABASE_URL="sqlite:///tests/temp/test.db"
    fi
}

# Run Python unit tests
run_python_unit_tests() {
    log_info "Running Python unit tests..."
    
    pytest tests/unit/ \
        --cov=pipelines \
        --cov-report=html:tests/reports/htmlcov-unit \
        --cov-report=xml:tests/reports/coverage-unit.xml \
        --cov-report=term \
        --junit-xml=tests/reports/junit-unit.xml \
        --verbose \
        --tb=short
    
    if [ $? -eq 0 ]; then
        log_success "Python unit tests passed"
    else
        log_error "Python unit tests failed"
        return 1
    fi
}

# Run Python integration tests
run_python_integration_tests() {
    log_info "Running Python integration tests..."
    
    pytest tests/integration/ \
        --cov=pipelines \
        --cov-append \
        --cov-report=html:tests/reports/htmlcov-integration \
        --cov-report=xml:tests/reports/coverage-integration.xml \
        --cov-report=term \
        --junit-xml=tests/reports/junit-integration.xml \
        --verbose \
        --tb=short
    
    if [ $? -eq 0 ]; then
        log_success "Python integration tests passed"
    else
        log_error "Python integration tests failed"
        return 1
    fi
}

# Run performance tests
run_performance_tests() {
    log_info "Running performance tests..."
    
    pytest tests/performance/ \
        --benchmark-only \
        --benchmark-json=tests/reports/benchmark-results.json \
        --benchmark-html=tests/reports/benchmark-report.html \
        --verbose
    
    if [ $? -eq 0 ]; then
        log_success "Performance tests completed"
    else
        log_warning "Some performance benchmarks may have failed"
    fi
}

# Run security tests
run_security_tests() {
    log_info "Running security tests..."
    
    pytest tests/security/ \
        -m security \
        --junit-xml=tests/reports/junit-security.xml \
        --verbose
    
    if [ $? -eq 0 ]; then
        log_success "Security tests passed"
    else
        log_error "Security tests failed"
        return 1
    fi
}

# Run frontend tests
run_frontend_tests() {
    log_info "Running frontend tests..."
    
    if [ -d "apps/site" ]; then
        cd apps/site
        
        # Unit tests
        npm run test:unit -- --coverage --reporter=junit --outputFile=../tests/reports/junit-frontend.xml
        
        if [ $? -eq 0 ]; then
            log_success "Frontend unit tests passed"
        else
            log_error "Frontend unit tests failed"
            cd ../..
            return 1
        fi
        
        # E2E tests (if not in headless environment)
        if [ "${DISPLAY:-}" != "" ] || [ "${CI:-}" = "true" ]; then
            npx playwright install --with-deps > /dev/null 2>&1
            npm run test:e2e
            
            if [ $? -eq 0 ]; then
                log_success "E2E tests passed"
            else
                log_warning "E2E tests failed or skipped"
            fi
        else
            log_info "Skipping E2E tests (no display available)"
        fi
        
        cd ../..
    else
        log_info "No frontend directory found, skipping frontend tests"
    fi
}

# Check coverage thresholds
check_coverage() {
    log_info "Checking coverage thresholds..."
    
    # Python coverage
    coverage report --fail-under=95 > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        log_success "Python coverage meets threshold (≥95%)"
    else
        log_error "Python coverage below threshold (95%)"
        coverage report --show-missing
        return 1
    fi
    
    # Frontend coverage (if available)
    if [ -f "apps/site/coverage/lcov-report/index.html" ]; then
        # Extract coverage percentage (this is a simplified check)
        COVERAGE_PERCENT=$(grep -o 'Functions</span><span class="strong">[0-9.]*%' apps/site/coverage/lcov-report/index.html | grep -o '[0-9.]*' | head -1)
        if (( $(echo "$COVERAGE_PERCENT >= 95" | bc -l) )); then
            log_success "Frontend coverage meets threshold (≥95%)"
        else
            log_warning "Frontend coverage may be below threshold"
        fi
    fi
}

# Generate coverage reports
generate_reports() {
    log_info "Generating test reports..."
    
    # Combine coverage reports
    coverage combine > /dev/null 2>&1 || true
    coverage html -d tests/reports/htmlcov-combined
    coverage xml -o tests/reports/coverage-combined.xml
    coverage json -o tests/reports/coverage-combined.json
    
    # Create summary report
    cat > tests/reports/test-summary.html << EOF
<!DOCTYPE html>
<html>
<head>
    <title>AI Knowledge Test Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .success { color: green; }
        .warning { color: orange; }
        .error { color: red; }
        .report-link { margin: 10px 0; }
        .report-link a { text-decoration: none; color: #0066cc; }
    </style>
</head>
<body>
    <h1>AI Knowledge Test Results</h1>
    <h2>Coverage Reports</h2>
    <div class="report-link"><a href="htmlcov-combined/index.html">Combined Coverage Report</a></div>
    <div class="report-link"><a href="htmlcov-unit/index.html">Unit Tests Coverage</a></div>
    <div class="report-link"><a href="htmlcov-integration/index.html">Integration Tests Coverage</a></div>
    
    <h2>Performance Reports</h2>
    <div class="report-link"><a href="benchmark-report.html">Performance Benchmarks</a></div>
    
    <h2>Test Results</h2>
    <p>Generated on: $(date)</p>
</body>
</html>
EOF
    
    log_success "Test reports generated in tests/reports/"
}

# Cleanup
cleanup() {
    log_info "Cleaning up test environment..."
    
    # Remove temporary files
    rm -rf tests/temp/* > /dev/null 2>&1 || true
    
    # Drop test database
    if pg_isready -h localhost -p 5432 > /dev/null 2>&1; then
        dropdb --if-exists $DB_NAME > /dev/null 2>&1 || true
    fi
    
    log_success "Cleanup completed"
}

# Main execution
main() {
    local RUN_ALL=true
    local RUN_UNIT=false
    local RUN_INTEGRATION=false
    local RUN_PERFORMANCE=false
    local RUN_SECURITY=false
    local RUN_FRONTEND=false
    local SKIP_SETUP=false
    local SKIP_COVERAGE=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --unit)
                RUN_ALL=false
                RUN_UNIT=true
                shift
                ;;
            --integration)
                RUN_ALL=false
                RUN_INTEGRATION=true
                shift
                ;;
            --performance)
                RUN_ALL=false
                RUN_PERFORMANCE=true
                shift
                ;;
            --security)
                RUN_ALL=false
                RUN_SECURITY=true
                shift
                ;;
            --frontend)
                RUN_ALL=false
                RUN_FRONTEND=true
                shift
                ;;
            --skip-setup)
                SKIP_SETUP=true
                shift
                ;;
            --skip-coverage)
                SKIP_COVERAGE=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --unit          Run only unit tests"
                echo "  --integration   Run only integration tests"
                echo "  --performance   Run only performance tests"
                echo "  --security      Run only security tests"
                echo "  --frontend      Run only frontend tests"
                echo "  --skip-setup    Skip environment setup"
                echo "  --skip-coverage Skip coverage checks"
                echo "  --help          Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    echo "=========================="
    echo "AI Knowledge Test Runner"
    echo "=========================="
    
    # Setup
    if [ "$SKIP_SETUP" = false ]; then
        check_dependencies
        setup_test_env
        setup_test_database
    fi
    
    # Track overall success
    local OVERALL_SUCCESS=true
    
    # Run tests based on options
    if [ "$RUN_ALL" = true ] || [ "$RUN_UNIT" = true ]; then
        run_python_unit_tests || OVERALL_SUCCESS=false
    fi
    
    if [ "$RUN_ALL" = true ] || [ "$RUN_INTEGRATION" = true ]; then
        run_python_integration_tests || OVERALL_SUCCESS=false
    fi
    
    if [ "$RUN_ALL" = true ] || [ "$RUN_PERFORMANCE" = true ]; then
        run_performance_tests || true  # Don't fail on performance issues
    fi
    
    if [ "$RUN_ALL" = true ] || [ "$RUN_SECURITY" = true ]; then
        run_security_tests || OVERALL_SUCCESS=false
    fi
    
    if [ "$RUN_ALL" = true ] || [ "$RUN_FRONTEND" = true ]; then
        run_frontend_tests || OVERALL_SUCCESS=false
    fi
    
    # Coverage check
    if [ "$SKIP_COVERAGE" = false ]; then
        check_coverage || OVERALL_SUCCESS=false
    fi
    
    # Generate reports
    generate_reports
    
    # Cleanup
    cleanup
    
    # Final result
    echo "=========================="
    if [ "$OVERALL_SUCCESS" = true ]; then
        log_success "All tests completed successfully!"
        echo "📊 View reports: tests/reports/test-summary.html"
        exit 0
    else
        log_error "Some tests failed!"
        exit 1
    fi
}

# Handle script interruption
trap cleanup EXIT INT TERM

# Run main function with all arguments
main "$@"