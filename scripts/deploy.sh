#!/bin/bash
set -euo pipefail

# AI Knowledge Website Deployment Script
# 
# This script handles deployment with rollback capabilities, health checks,
# and zero-downtime deployment strategies.
#
# Usage:
#   ./deploy.sh [staging|production] [--rollback] [--dry-run]

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_ID=$(date +"%Y%m%d_%H%M%S")_$(git rev-parse --short HEAD)

# Default values
ENVIRONMENT="staging"
DRY_RUN=false
ROLLBACK=false
BACKUP_RETENTION_DAYS=30
HEALTH_CHECK_TIMEOUT=300
HEALTH_CHECK_INTERVAL=10

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ❌ $1${NC}"
}

# Error handling
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_error "Deployment failed with exit code $exit_code"
        log "Cleaning up temporary files..."
        # Add cleanup logic here
        rm -f "$PROJECT_ROOT/deployment_lock" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            staging|production)
                ENVIRONMENT="$1"
                shift
                ;;
            --rollback)
                ROLLBACK=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

show_help() {
    cat << EOF
AI Knowledge Website Deployment Script

Usage: $0 [ENVIRONMENT] [OPTIONS]

ENVIRONMENT:
  staging       Deploy to staging environment (default)
  production    Deploy to production environment

OPTIONS:
  --rollback    Rollback to previous deployment
  --dry-run     Show what would be deployed without making changes
  --help, -h    Show this help message

Examples:
  $0 staging                    # Deploy to staging
  $0 production                 # Deploy to production
  $0 production --rollback      # Rollback production
  $0 staging --dry-run          # Dry run staging deployment

Environment Variables:
  STAGING_SERVER        Staging server hostname
  PRODUCTION_SERVER     Production server hostname
  DATABASE_URL_STAGING  Staging database connection
  DATABASE_URL_PROD     Production database connection
  BACKUP_BUCKET         S3 bucket for backups
EOF
}

# Load environment-specific configuration
load_config() {
    log "Loading configuration for $ENVIRONMENT environment..."
    
    case $ENVIRONMENT in
        staging)
            SERVER_HOST="${STAGING_SERVER:-staging.ai-knowledge.example.com}"
            SERVER_USER="${STAGING_USER:-deploy}"
            DATABASE_URL="${DATABASE_URL_STAGING:-}"
            SITE_URL="${STAGING_URL:-https://staging.ai-knowledge.example.com}"
            BACKUP_PREFIX="staging"
            ;;
        production)
            SERVER_HOST="${PRODUCTION_SERVER:-ai-knowledge.example.com}"
            SERVER_USER="${PRODUCTION_USER:-deploy}"
            DATABASE_URL="${DATABASE_URL_PROD:-}"
            SITE_URL="${PRODUCTION_URL:-https://ai-knowledge.example.com}"
            BACKUP_PREFIX="production"
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            exit 1
            ;;
    esac
    
    # Validate required configuration
    if [ -z "$SERVER_HOST" ] || [ -z "$SERVER_USER" ]; then
        log_error "Missing required server configuration"
        exit 1
    fi
    
    log "Configuration loaded:"
    log "  Environment: $ENVIRONMENT"
    log "  Server: $SERVER_USER@$SERVER_HOST"
    log "  Site URL: $SITE_URL"
    log "  Deployment ID: $DEPLOYMENT_ID"
}

# Check prerequisites
check_prerequisites() {
    log "Checking deployment prerequisites..."
    
    # Check if deployment is locked
    if [ -f "$PROJECT_ROOT/deployment_lock" ]; then
        log_error "Another deployment is in progress. Remove deployment_lock file if this is incorrect."
        exit 1
    fi
    
    # Create deployment lock
    echo "$DEPLOYMENT_ID" > "$PROJECT_ROOT/deployment_lock"
    
    # Check required tools
    local required_tools=("node" "npm" "git" "rsync" "ssh")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            log_error "Required tool not found: $tool"
            exit 1
        fi
    done
    
    # Check Git status
    if [ -n "$(git status --porcelain)" ]; then
        log_warning "Working directory has uncommitted changes"
        if [ "$ENVIRONMENT" = "production" ]; then
            log_error "Production deployment requires clean working directory"
            exit 1
        fi
    fi
    
    # Check if we're on the right branch
    local current_branch=$(git rev-parse --abbrev-ref HEAD)
    if [ "$ENVIRONMENT" = "production" ] && [ "$current_branch" != "main" ]; then
        log_error "Production deployment must be from main branch (current: $current_branch)"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Build the application
build_application() {
    if [ "$ROLLBACK" = true ]; then
        log "Skipping build for rollback deployment"
        return 0
    fi
    
    log "Building application for $ENVIRONMENT..."
    
    cd "$PROJECT_ROOT/apps/site"
    
    # Install dependencies
    log "Installing Node.js dependencies..."
    if [ "$DRY_RUN" = false ]; then
        npm ci --only=production
    fi
    
    # Run quality checks
    log "Running quality checks..."
    if [ "$DRY_RUN" = false ]; then
        npm run astro check
        npm run lint || log_warning "Linting issues found but continuing deployment"
    fi
    
    # Build the site
    log "Building Astro site..."
    if [ "$DRY_RUN" = false ]; then
        NODE_ENV=production npm run build
        
        # Verify build output
        if [ ! -d "dist" ] || [ ! -f "dist/index.html" ]; then
            log_error "Build failed: missing output files"
            exit 1
        fi
        
        # Calculate build size
        local build_size=$(du -sh dist | cut -f1)
        log "Build completed successfully (size: $build_size)"
    fi
}

# Create backup
create_backup() {
    if [ "$ROLLBACK" = true ]; then
        log "Skipping backup for rollback deployment"
        return 0
    fi
    
    log "Creating backup of current deployment..."
    
    local backup_name="${BACKUP_PREFIX}_backup_$(date +%Y%m%d_%H%M%S)"
    
    if [ "$DRY_RUN" = false ]; then
        # Backup current deployment
        ssh "$SERVER_USER@$SERVER_HOST" "
            if [ -d /var/www/html ]; then
                sudo tar -czf /var/backups/$backup_name.tar.gz -C /var/www html
                echo 'Backup created: /var/backups/$backup_name.tar.gz'
            else
                echo 'No existing deployment to backup'
            fi
        "
        
        # Backup database if configured
        if [ -n "$DATABASE_URL" ]; then
            log "Creating database backup..."
            ssh "$SERVER_USER@$SERVER_HOST" "
                pg_dump '$DATABASE_URL' | gzip > /var/backups/${backup_name}_db.sql.gz
                echo 'Database backup created: /var/backups/${backup_name}_db.sql.gz'
            "
        fi
    fi
    
    log_success "Backup completed: $backup_name"
    echo "$backup_name" > "$PROJECT_ROOT/.last_backup"
}

# Deploy application
deploy_application() {
    if [ "$ROLLBACK" = true ]; then
        perform_rollback
        return 0
    fi
    
    log "Deploying application to $ENVIRONMENT..."
    
    local source_dir="$PROJECT_ROOT/apps/site/dist/"
    local target_dir="/var/www/html/"
    
    if [ "$DRY_RUN" = true ]; then
        log "DRY RUN: Would deploy $source_dir to $SERVER_USER@$SERVER_HOST:$target_dir"
        rsync -avz --dry-run --delete "$source_dir" "$SERVER_USER@$SERVER_HOST:$target_dir"
    else
        # Deploy with rsync
        log "Syncing files to server..."
        rsync -avz --delete \
            --exclude='.git*' \
            --exclude='node_modules' \
            --exclude='*.log' \
            "$source_dir" "$SERVER_USER@$SERVER_HOST:$target_dir"
        
        # Set proper permissions
        ssh "$SERVER_USER@$SERVER_HOST" "
            sudo chown -R www-data:www-data $target_dir
            sudo find $target_dir -type f -exec chmod 644 {} \;
            sudo find $target_dir -type d -exec chmod 755 {} \;
        "
        
        # Create deployment marker
        ssh "$SERVER_USER@$SERVER_HOST" "
            echo '$DEPLOYMENT_ID' | sudo tee $target_dir/.deployment_id
            echo '$(date -u +"%Y-%m-%d %H:%M:%S UTC")' | sudo tee $target_dir/.deployment_time
        "
    fi
    
    log_success "Application deployment completed"
}

# Perform rollback
perform_rollback() {
    log "Performing rollback for $ENVIRONMENT..."
    
    # Get last backup name
    local last_backup=""
    if [ -f "$PROJECT_ROOT/.last_backup" ]; then
        last_backup=$(cat "$PROJECT_ROOT/.last_backup")
    fi
    
    if [ -z "$last_backup" ]; then
        log_error "No backup information found for rollback"
        exit 1
    fi
    
    if [ "$DRY_RUN" = true ]; then
        log "DRY RUN: Would rollback to backup: $last_backup"
        return 0
    fi
    
    # Restore from backup
    ssh "$SERVER_USER@$SERVER_HOST" "
        if [ -f /var/backups/$last_backup.tar.gz ]; then
            sudo rm -rf /var/www/html.old
            sudo mv /var/www/html /var/www/html.old
            sudo mkdir -p /var/www
            sudo tar -xzf /var/backups/$last_backup.tar.gz -C /var/www
            sudo chown -R www-data:www-data /var/www/html
            echo 'Rollback completed from: $last_backup'
        else
            echo 'Backup file not found: /var/backups/$last_backup.tar.gz'
            exit 1
        fi
    "
    
    # Rollback database if backup exists
    if [ -n "$DATABASE_URL" ]; then
        ssh "$SERVER_USER@$SERVER_HOST" "
            if [ -f /var/backups/${last_backup}_db.sql.gz ]; then
                echo 'Rolling back database...'
                gunzip -c /var/backups/${last_backup}_db.sql.gz | psql '$DATABASE_URL'
                echo 'Database rollback completed'
            fi
        "
    fi
    
    log_success "Rollback completed"
}

# Run database migrations
run_migrations() {
    if [ "$ROLLBACK" = true ] || [ -z "$DATABASE_URL" ]; then
        return 0
    fi
    
    log "Running database migrations..."
    
    if [ "$DRY_RUN" = true ]; then
        log "DRY RUN: Would run database migrations"
        return 0
    fi
    
    # Run migrations via SSH
    ssh "$SERVER_USER@$SERVER_HOST" "
        cd /opt/ai-knowledge/pipelines
        python -m alembic upgrade head
    " || log_warning "Database migrations failed or not configured"
}

# Restart services
restart_services() {
    if [ "$ROLLBACK" = true ]; then
        log "Restarting services after rollback..."
    else
        log "Restarting services..."
    fi
    
    if [ "$DRY_RUN" = true ]; then
        log "DRY RUN: Would restart web services"
        return 0
    fi
    
    # Restart web server
    ssh "$SERVER_USER@$SERVER_HOST" "
        sudo systemctl reload nginx || sudo systemctl restart nginx
        sudo systemctl status nginx --no-pager
    "
    
    # Restart application services if they exist
    ssh "$SERVER_USER@$SERVER_HOST" "
        if systemctl is-enabled ai-knowledge-pipeline >/dev/null 2>&1; then
            sudo systemctl restart ai-knowledge-pipeline
        fi
    " || true
    
    log_success "Services restarted"
}

# Health checks
run_health_checks() {
    log "Running health checks..."
    
    local start_time=$(date +%s)
    local max_time=$((start_time + HEALTH_CHECK_TIMEOUT))
    
    while [ $(date +%s) -lt $max_time ]; do
        if curl -sf --max-time 10 "$SITE_URL" >/dev/null 2>&1; then
            log_success "Health check passed: $SITE_URL is responding"
            
            # Additional health checks
            local status_code=$(curl -s -o /dev/null -w "%{http_code}" "$SITE_URL")
            if [ "$status_code" = "200" ]; then
                log_success "HTTP status check passed (200 OK)"
            else
                log_warning "HTTP status check: received $status_code"
            fi
            
            # Check specific pages
            local test_pages=("/about" "/articles" "/glossary")
            for page in "${test_pages[@]}"; do
                if curl -sf --max-time 5 "$SITE_URL$page" >/dev/null 2>&1; then
                    log "✓ $SITE_URL$page is responding"
                else
                    log_warning "✗ $SITE_URL$page is not responding"
                fi
            done
            
            return 0
        fi
        
        log "Health check failed, retrying in ${HEALTH_CHECK_INTERVAL}s..."
        sleep $HEALTH_CHECK_INTERVAL
    done
    
    log_error "Health checks failed after ${HEALTH_CHECK_TIMEOUT}s timeout"
    return 1
}

# Invalidate CDN cache
invalidate_cache() {
    if [ "$DRY_RUN" = true ]; then
        log "DRY RUN: Would invalidate CDN cache"
        return 0
    fi
    
    log "Invalidating CDN cache..."
    
    # CloudFront invalidation (if configured)
    if [ -n "${CLOUDFRONT_DISTRIBUTION_ID:-}" ]; then
        aws cloudfront create-invalidation \
            --distribution-id "$CLOUDFRONT_DISTRIBUTION_ID" \
            --paths "/*" >/dev/null 2>&1 || log_warning "CDN cache invalidation failed"
        log_success "CDN cache invalidation initiated"
    else
        log "CDN cache invalidation not configured"
    fi
}

# Cleanup old backups
cleanup_old_backups() {
    if [ "$DRY_RUN" = true ]; then
        log "DRY RUN: Would cleanup backups older than ${BACKUP_RETENTION_DAYS} days"
        return 0
    fi
    
    log "Cleaning up old backups (older than ${BACKUP_RETENTION_DAYS} days)..."
    
    ssh "$SERVER_USER@$SERVER_HOST" "
        find /var/backups -name '${BACKUP_PREFIX}_backup_*.tar.gz' -mtime +${BACKUP_RETENTION_DAYS} -delete
        find /var/backups -name '${BACKUP_PREFIX}_backup_*_db.sql.gz' -mtime +${BACKUP_RETENTION_DAYS} -delete
        echo 'Old backups cleaned up'
    "
}

# Send deployment notification
send_notification() {
    local status=$1
    local message="AI Knowledge Website deployment $status"
    
    if [ "$ROLLBACK" = true ]; then
        message="AI Knowledge Website rollback $status"
    fi
    
    log "Sending deployment notification..."
    
    # Slack notification (if configured)
    if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
        local color="good"
        local icon=":white_check_mark:"
        
        if [ "$status" = "failed" ]; then
            color="danger"
            icon=":x:"
        fi
        
        curl -sf -X POST "$SLACK_WEBHOOK_URL" \
            -H 'Content-type: application/json' \
            --data "{
                \"attachments\": [{
                    \"color\": \"$color\",
                    \"title\": \"$message\",
                    \"fields\": [
                        {\"title\": \"Environment\", \"value\": \"$ENVIRONMENT\", \"short\": true},
                        {\"title\": \"Deployment ID\", \"value\": \"$DEPLOYMENT_ID\", \"short\": true},
                        {\"title\": \"Site URL\", \"value\": \"$SITE_URL\", \"short\": false}
                    ]
                }]
            }" >/dev/null 2>&1 || log_warning "Slack notification failed"
    fi
}

# Main deployment function
main() {
    parse_args "$@"
    load_config
    
    log "Starting deployment process..."
    log "Environment: $ENVIRONMENT"
    log "Deployment ID: $DEPLOYMENT_ID"
    log "Dry run: $DRY_RUN"
    log "Rollback: $ROLLBACK"
    
    check_prerequisites
    
    if [ "$ROLLBACK" = false ]; then
        build_application
        create_backup
    fi
    
    deploy_application
    run_migrations
    restart_services
    
    if run_health_checks; then
        invalidate_cache
        cleanup_old_backups
        send_notification "completed successfully"
        
        # Remove deployment lock
        rm -f "$PROJECT_ROOT/deployment_lock"
        
        log_success "Deployment completed successfully!"
        log "Site URL: $SITE_URL"
        log "Deployment ID: $DEPLOYMENT_ID"
    else
        send_notification "failed"
        log_error "Deployment failed health checks"
        
        if [ "$ENVIRONMENT" = "production" ]; then
            log "Consider running rollback: $0 production --rollback"
        fi
        
        exit 1
    fi
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi