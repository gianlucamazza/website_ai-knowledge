# AI Knowledge Website - Development Commands
# 
# This Makefile provides convenient commands for developing, building, and testing
# the AI Knowledge website built with Astro and Content Collections.

.PHONY: help install dev build test clean lint preview check deploy

# Default target
help: ## Show this help message
	@echo "AI Knowledge Website - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Common Development Workflow:"
	@echo "  1. make install     # Install dependencies"
	@echo "  2. make dev         # Start development server"
	@echo "  3. make build       # Build for production"
	@echo "  4. make test        # Run all tests"

# Installation and Setup
install: ## Install all dependencies
	@echo "🔧 Installing dependencies..."
	cd apps/site && npm install
	@echo "✅ Dependencies installed successfully!"

# Development Commands
dev: ## Start Astro development server with hot reload
	@echo "🚀 Starting development server..."
	@echo "📍 Site will be available at http://localhost:4321"
	cd apps/site && npm run dev

start: dev ## Alias for 'dev' command

# Build Commands
build: ## Build the site for production with validation
	@echo "🏗️  Building site for production..."
	cd apps/site && npm run build
	@echo "✅ Build completed successfully!"

preview: build ## Build and preview the production site locally
	@echo "🔍 Starting preview server..."
	@echo "📍 Production preview at http://localhost:4322"
	cd apps/site && npm run preview

# Quality and Testing
check: ## Run Astro type checking and Content Collections validation
	@echo "🔍 Running type checking and schema validation..."
	cd apps/site && npm run astro check
	@echo "✅ Type checking completed!"

lint: ## Run markdown linting on all content
	@echo "📝 Linting markdown files..."
	cd apps/site && npm run lint
	@echo "✅ Markdown linting completed!"

lint-fix: ## Fix auto-fixable markdown linting issues
	@echo "🔧 Fixing markdown linting issues..."
	cd apps/site && npm run lint:fix
	@echo "✅ Markdown fixes applied!"

test: check lint ## Run all quality checks (type checking + linting)
	@echo "🧪 All tests passed! Ready for production."

# Content Management (Future Pipeline Integration)
ingest: ## Run content ingestion pipeline (placeholder)
	@echo "🔄 Content ingestion pipeline not yet implemented"
	@echo "📝 This will fetch content from external sources"

normalize: ## Run content normalization pipeline (placeholder)
	@echo "🔄 Content normalization pipeline not yet implemented"
	@echo "📝 This will standardize content format and metadata"

dedup: ## Run duplicate detection pipeline (placeholder)
	@echo "🔄 Duplicate detection pipeline not yet implemented"
	@echo "📝 This will identify and handle duplicate content"

enrich: ## Run content enrichment pipeline (placeholder)
	@echo "🔄 Content enrichment pipeline not yet implemented"
	@echo "📝 This will add cross-references and additional metadata"

publish: ## Generate markdown files from curated data (placeholder)
	@echo "🔄 Content publishing pipeline not yet implemented"
	@echo "📝 This will generate final markdown files"

pipeline: ## Run full content pipeline (placeholder)
	@echo "🔄 Full content pipeline not yet implemented"
	@echo "📝 This will run: ingest → normalize → dedup → enrich → publish"

# Maintenance Commands
clean: ## Clean build artifacts and node_modules
	@echo "🧹 Cleaning build artifacts..."
	cd apps/site && rm -rf dist/ node_modules/ .astro/
	@echo "✅ Cleanup completed!"

reinstall: clean install ## Clean install all dependencies

# Deployment (Placeholder)
deploy: build test ## Build and deploy to production (placeholder)
	@echo "🚀 Deployment not yet configured"
	@echo "📝 This will deploy the built site to production"

# Content Validation
validate-content: ## Validate all content against schemas
	@echo "🔍 Validating content schemas..."
	cd apps/site && npm run astro check
	@echo "✅ Content validation completed!"

validate-links: ## Check all links in content (placeholder)
	@echo "🔗 Link validation not yet implemented"
	@echo "📝 This will check all internal and external links"

# Development Utilities
stats: ## Show project statistics
	@echo "📊 AI Knowledge Website Statistics:"
	@echo ""
	@echo "📁 Content Files:"
	@find apps/site/src/content -name "*.md" | wc -l | xargs -I {} echo "   Glossary entries: {}"
	@echo ""
	@echo "📦 Dependencies:"
	@cd apps/site && npm list --depth=0 2>/dev/null | grep -E "^[├└]" | wc -l | xargs -I {} echo "   Packages: {}"
	@echo ""
	@echo "📈 Build Info:"
	@if [ -d "apps/site/dist" ]; then \
		echo "   Last build: $$(stat -f %Sm apps/site/dist 2>/dev/null || stat -c %y apps/site/dist 2>/dev/null || echo 'Unknown')"; \
		echo "   Build size: $$(du -sh apps/site/dist 2>/dev/null | cut -f1 || echo 'Unknown')"; \
	else \
		echo "   Status: Not built"; \
	fi

info: stats ## Alias for 'stats' command

# Git Integration (Optional)
git-status: ## Show git status with content focus
	@echo "📋 Repository Status:"
	@git status --porcelain | grep -E '\.(md|astro|ts|js)$$' || echo "No content changes detected"

# Node.js specific commands
node-version: ## Check Node.js and npm versions
	@echo "📋 Environment Info:"
	@echo "Node.js: $$(node --version)"
	@echo "npm: $$(npm --version)"
	@echo "Working directory: $$(pwd)"

# Emergency commands
fix-permissions: ## Fix file permissions (Unix/Linux/macOS)
	@echo "🔧 Fixing file permissions..."
	find . -type f -name "*.md" -exec chmod 644 {} \;
	find . -type f -name "*.json" -exec chmod 644 {} \;
	find . -type f -name "*.js" -exec chmod 644 {} \;
	find . -type f -name "*.ts" -exec chmod 644 {} \;
	@echo "✅ File permissions fixed!"

# CI/CD Integration
ci-install: ## Install dependencies in CI environment
	cd apps/site && npm ci

ci-test: ## Run tests optimized for CI
	@echo "🧪 Running CI tests..."
	cd apps/site && npm run build && npm run lint
	@echo "✅ All CI tests passed!"

# Documentation
docs: ## Generate documentation (placeholder)
	@echo "📚 Documentation generation not yet implemented"
	@echo "📝 This will generate API docs and developer guides"