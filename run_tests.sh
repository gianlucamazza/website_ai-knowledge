#!/bin/bash
set -euo pipefail

# AI Knowledge Website - Comprehensive Test Runner
# This script runs the complete test suite matching the CI/CD pipeline

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COVERAGE_THRESHOLD=95
PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
NODE_VERSION=$(node --version | sed 's/v//' | cut -d'.' -f1)

# Default values
RUN_FRONTEND=true
RUN_BACKEND=true
RUN_SLOW_TESTS=false
RUN_E2E=true
RUN_SECURITY=true
RUN_PERFORMANCE=false
PARALLEL=false
VERBOSE=false
CLEAN_FIRST=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --frontend-only)
            RUN_FRONTEND=true
            RUN_BACKEND=false
            shift
            ;;
        --backend-only)
            RUN_FRONTEND=false
            RUN_BACKEND=true
            shift
            ;;
        --skip-frontend)
            RUN_FRONTEND=false
            shift
            ;;
        --skip-backend)
            RUN_BACKEND=false
            shift
            ;;
        --skip-e2e)
            RUN_E2E=false
            shift
            ;;
        --skip-security)
            RUN_SECURITY=false
            shift
            ;;
        --include-slow)
            RUN_SLOW_TESTS=true
            shift
            ;;
        --include-performance)
            RUN_PERFORMANCE=true
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --clean)
            CLEAN_FIRST=true
            shift
            ;;
        --help|-h)
            echo "AI Knowledge Website Test Runner"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --frontend-only     Run only frontend tests"
            echo "  --backend-only      Run only backend/Python tests"
            echo "  --skip-frontend     Skip frontend tests"
            echo "  --skip-backend      Skip backend tests"
            echo "  --skip-e2e          Skip E2E tests"
            echo "  --skip-security     Skip security tests"
            echo "  --include-slow      Include slow-running tests"
            echo "  --include-performance  Include performance benchmarks"
            echo "  --parallel          Run tests in parallel where possible"
            echo "  --verbose, -v       Verbose output"
            echo "  --clean             Clean previous test artifacts"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                          # Run all tests"
            echo "  $0 --frontend-only          # Run only frontend tests"
            echo "  $0 --skip-e2e --verbose     # Skip E2E tests, verbose output"
            echo "  $0 --include-slow --parallel # Include slow tests, run in parallel"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print configuration
print_header() {
    echo -e "${BLUE}=================================="
    echo -e "AI Knowledge Website Test Runner"
    echo -e "==================================${NC}"
    echo ""
    echo "Configuration:"
    echo "  Python version: $PYTHON_VERSION"
    echo "  Node.js version: $NODE_VERSION"
    echo "  Coverage threshold: ${COVERAGE_THRESHOLD}%"
    echo "  Frontend tests: $([ $RUN_FRONTEND = true ] && echo 'Yes' || echo 'No')"
    echo "  Backend tests: $([ $RUN_BACKEND = true ] && echo 'Yes' || echo 'No')"
    echo "  E2E tests: $([ $RUN_E2E = true ] && echo 'Yes' || echo 'No')"
    echo "  Security tests: $([ $RUN_SECURITY = true ] && echo 'Yes' || echo 'No')"
    echo "  Performance tests: $([ $RUN_PERFORMANCE = true ] && echo 'Yes' || echo 'No')"
    echo "  Include slow tests: $([ $RUN_SLOW_TESTS = true ] && echo 'Yes' || echo 'No')"
    echo "  Parallel execution: $([ $PARALLEL = true ] && echo 'Yes' || echo 'No')"
    echo ""
}

# Print section header
print_section() {
    echo -e "${YELLOW}=== $1 ===${NC}"
}

# Print success message
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Print error message
print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Print warning message
print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Clean previous artifacts
clean_artifacts() {
    if [ $CLEAN_FIRST = true ]; then
        print_section "Cleaning Previous Artifacts"
        
        # Python artifacts
        rm -rf coverage/ htmlcov/ .coverage .coverage.* .pytest_cache/ __pycache__/
        find . -name "*.pyc" -delete 2>/dev/null || true
        find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
        
        # Frontend artifacts
        rm -rf apps/site/coverage/ apps/site/playwright-report/ apps/site/test-results/
        rm -rf apps/site/node_modules/.cache/ 2>/dev/null || true
        
        # Test logs and temp files
        rm -rf tests/logs/ tests/temp/ tests/output/
        
        print_success "Artifacts cleaned"
        echo ""
    fi
}

# Setup test environment
setup_environment() {
    print_section "Setting up Test Environment"
    
    # Set environment variables
    export TEST_ENVIRONMENT=local
    export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
    export NODE_ENV=test
    
    if [ $VERBOSE = true ]; then
        export TEST_VERBOSE=true
        export TEST_DEBUG=true
    fi
    
    if [ $RUN_SLOW_TESTS = true ]; then
        export TEST_RUN_SLOW=true
    fi
    
    # Create required directories
    mkdir -p coverage tests/logs tests/temp tests/output
    
    # Initialize test configuration
    python -c "
import sys
sys.path.append('.')
from tests.config.test_settings import setup_test_environment
setup_test_environment()
print('Test environment initialized')
    "
    
    print_success "Environment setup completed"
    echo ""
}

# Check dependencies
check_dependencies() {
    print_section "Checking Dependencies"
    
    # Check Python version
    if ! python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
        print_error "Python 3.10+ required, found $PYTHON_VERSION"
        exit 1
    fi
    
    # Check Node.js version
    if [ $NODE_VERSION -lt 18 ]; then
        print_error "Node.js 18+ required, found $NODE_VERSION"
        exit 1
    fi
    
    # Check Python dependencies
    if [ $RUN_BACKEND = true ]; then
        if [ ! -f "tests/requirements.txt" ]; then
            print_error "tests/requirements.txt not found"
            exit 1
        fi
        
        print_warning "Installing Python dependencies..."
        pip install -q -r pipelines/requirements.txt
        pip install -q -r tests/requirements.txt
    fi
    
    # Check Node.js dependencies
    if [ $RUN_FRONTEND = true ]; then
        if [ ! -f "apps/site/package.json" ]; then
            print_error "apps/site/package.json not found"
            exit 1
        fi
        
        cd apps/site
        if [ ! -d "node_modules" ]; then
            print_warning "Installing Node.js dependencies..."
            npm ci
        fi
        cd ../..
    fi
    
    print_success "Dependencies checked"
    echo ""
}

# Run Python/Backend tests
run_backend_tests() {
    if [ $RUN_BACKEND = false ]; then
        return 0
    fi
    
    print_section "Running Backend Tests"
    
    local pytest_args="--cov=pipelines --cov-branch --cov-report=term --cov-report=html:coverage/htmlcov --cov-report=xml:coverage/coverage.xml --cov-report=lcov:coverage/lcov.info"
    
    if [ $VERBOSE = true ]; then
        pytest_args="$pytest_args --verbose --tb=long"
    else
        pytest_args="$pytest_args --tb=short"
    fi
    
    if [ $PARALLEL = true ]; then
        pytest_args="$pytest_args -n auto"
    fi
    
    # Unit tests
    echo "Running unit tests..."
    if [ $RUN_SLOW_TESTS = true ]; then
        pytest tests/unit/ -m "unit" $pytest_args
    else
        pytest tests/unit/ -m "unit and not slow" $pytest_args
    fi
    
    # Integration tests
    echo "Running integration tests..."
    if [ $RUN_SLOW_TESTS = true ]; then
        pytest tests/integration/ -m "integration" --cov-append $pytest_args
    else
        pytest tests/integration/ -m "integration and not slow" --cov-append $pytest_args
    fi
    
    # Database tests
    echo "Running database tests..."
    if [ $RUN_SLOW_TESTS = true ]; then
        pytest tests/database/ -m "database" --cov-append $pytest_args
    else
        pytest tests/database/ -m "database and not slow" --cov-append $pytest_args
    fi
    
    # Security tests
    if [ $RUN_SECURITY = true ]; then
        echo "Running security tests..."
        if [ $RUN_SLOW_TESTS = true ]; then
            pytest tests/security/ -m "security" --cov-append $pytest_args
        else
            pytest tests/security/ -m "security and not slow" --cov-append $pytest_args
        fi
    fi
    
    # Performance tests
    if [ $RUN_PERFORMANCE = true ]; then
        echo "Running performance tests..."
        if [ $RUN_SLOW_TESTS = true ]; then
            pytest tests/performance/ -m "performance" --cov-append $pytest_args --benchmark-only
        else
            pytest tests/performance/ -m "performance and not slow" --cov-append $pytest_args
        fi
    fi
    
    # Check coverage
    echo "Checking coverage..."
    coverage report --show-missing --precision=2
    
    if ! coverage report --fail-under=$COVERAGE_THRESHOLD >/dev/null 2>&1; then
        print_error "Python coverage below ${COVERAGE_THRESHOLD}%"
        return 1
    fi
    
    print_success "Backend tests completed"
    echo ""
}

# Run Frontend tests
run_frontend_tests() {
    if [ $RUN_FRONTEND = false ]; then
        return 0
    fi
    
    print_section "Running Frontend Tests"
    
    cd apps/site
    
    # Type checking
    echo "Running type checks..."
    npm run type-check
    
    # Unit tests with coverage
    echo "Running unit tests with coverage..."
    npm run test:unit:coverage
    
    # Check coverage
    echo "Checking frontend coverage..."
    if [ -f "coverage/coverage-summary.json" ]; then
        LINES_PCT=$(jq -r '.total.lines.pct' coverage/coverage-summary.json 2>/dev/null || echo "0")
        FUNCTIONS_PCT=$(jq -r '.total.functions.pct' coverage/coverage-summary.json 2>/dev/null || echo "0")
        BRANCHES_PCT=$(jq -r '.total.branches.pct' coverage/coverage-summary.json 2>/dev/null || echo "0")
        STATEMENTS_PCT=$(jq -r '.total.statements.pct' coverage/coverage-summary.json 2>/dev/null || echo "0")
        
        echo "Coverage Results:"
        echo "  Lines: ${LINES_PCT}%"
        echo "  Functions: ${FUNCTIONS_PCT}%"
        echo "  Branches: ${BRANCHES_PCT}%"
        echo "  Statements: ${STATEMENTS_PCT}%"
        
        # Check thresholds (using bc for float comparison if available, otherwise basic check)
        if command -v bc >/dev/null 2>&1; then
            if (( $(echo "$LINES_PCT < $COVERAGE_THRESHOLD" | bc -l) )) || \
               (( $(echo "$FUNCTIONS_PCT < $COVERAGE_THRESHOLD" | bc -l) )) || \
               (( $(echo "$BRANCHES_PCT < $COVERAGE_THRESHOLD" | bc -l) )) || \
               (( $(echo "$STATEMENTS_PCT < $COVERAGE_THRESHOLD" | bc -l) )); then
                print_error "Frontend coverage below ${COVERAGE_THRESHOLD}%"
                cd ../..
                return 1
            fi
        else
            # Basic integer comparison fallback
            LINES_INT=${LINES_PCT%.*}
            if [ "$LINES_INT" -lt "$COVERAGE_THRESHOLD" ]; then
                print_error "Frontend coverage below ${COVERAGE_THRESHOLD}%"
                cd ../..
                return 1
            fi
        fi
    else
        print_warning "Coverage summary not found"
    fi
    
    # E2E tests
    if [ $RUN_E2E = true ]; then
        echo "Installing Playwright browsers..."
        npx playwright install --with-deps chromium
        
        echo "Building site for E2E tests..."
        npm run build
        
        echo "Running E2E tests..."
        # Start dev server and run E2E tests
        npm run dev &
        DEV_PID=$!
        
        # Wait for server
        sleep 5
        
        # Run E2E tests
        if ! npm run test:e2e; then
            kill $DEV_PID 2>/dev/null || true
            cd ../..
            return 1
        fi
        
        kill $DEV_PID 2>/dev/null || true
    fi
    
    cd ../..
    
    print_success "Frontend tests completed"
    echo ""
}

# Generate final report
generate_report() {
    print_section "Test Results Summary"
    
    echo "📊 Coverage Reports:"
    if [ $RUN_BACKEND = true ] && [ -d "coverage" ]; then
        echo "  Python: coverage/htmlcov/index.html"
    fi
    
    if [ $RUN_FRONTEND = true ] && [ -d "apps/site/coverage" ]; then
        echo "  Frontend: apps/site/coverage/index.html"
    fi
    
    echo ""
    echo "🎯 Quality Metrics:"
    echo "  Coverage Target: ${COVERAGE_THRESHOLD}%"
    echo "  Tests Run: $([ $RUN_BACKEND = true ] && echo -n 'Backend ' || true)$([ $RUN_FRONTEND = true ] && echo -n 'Frontend ' || true)$([ $RUN_E2E = true ] && echo -n 'E2E ' || true)$([ $RUN_SECURITY = true ] && echo -n 'Security ' || true)$([ $RUN_PERFORMANCE = true ] && echo -n 'Performance' || true)"
    echo ""
}

# Main execution
main() {
    local start_time=$(date +%s)
    
    print_header
    
    # Trap to ensure cleanup on exit
    trap 'kill $(jobs -p) 2>/dev/null || true' EXIT
    
    # Clean artifacts if requested
    clean_artifacts
    
    # Setup
    setup_environment
    check_dependencies
    
    # Run tests
    local backend_result=0
    local frontend_result=0
    
    if ! run_backend_tests; then
        backend_result=1
    fi
    
    if ! run_frontend_tests; then
        frontend_result=1
    fi
    
    # Results
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo -e "${BLUE}=================================="
    if [ $backend_result -eq 0 ] && [ $frontend_result -eq 0 ]; then
        print_success "All tests passed! 🎉"
    else
        print_error "Some tests failed!"
    fi
    
    echo "Total runtime: ${duration}s"
    echo -e "==================================${NC}"
    
    generate_report
    
    # Exit with error if any tests failed
    if [ $backend_result -ne 0 ] || [ $frontend_result -ne 0 ]; then
        exit 1
    fi
}

# Run main function
main "$@"