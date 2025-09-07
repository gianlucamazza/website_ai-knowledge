/**
 * Coverage configuration for the AI Knowledge Website project
 * Defines coverage thresholds and reporting settings for both frontend and backend
 */

module.exports = {
  // Global coverage settings
  coverageThreshold: {
    global: {
      branches: 95,
      functions: 95,
      lines: 95,
      statements: 95
    },
    
    // Frontend-specific thresholds
    './apps/site/src/': {
      branches: 90,
      functions: 90,
      lines: 90,
      statements: 90
    },
    
    // Pipeline modules - higher threshold for critical components
    './pipelines/': {
      branches: 95,
      functions: 95,
      lines: 95,
      statements: 95
    },
    
    // Individual component thresholds
    './pipelines/ingest/': {
      branches: 95,
      functions: 95,
      lines: 95,
      statements: 95
    },
    
    './pipelines/normalize/': {
      branches: 95,
      functions: 95,
      lines: 95,
      statements: 95
    },
    
    './pipelines/dedup/': {
      branches: 98,  // Critical for data quality
      functions: 98,
      lines: 98,
      statements: 98
    },
    
    './pipelines/enrich/': {
      branches: 90,  // Lower threshold due to AI API dependencies
      functions: 90,
      lines: 90,
      statements: 90
    },
    
    './pipelines/publish/': {
      branches: 95,
      functions: 95,
      lines: 95,
      statements: 95
    }
  },
  
  // Coverage reporters
  coverageReporters: [
    'text',
    'text-summary',
    'lcov',
    'html',
    'json',
    'clover'
  ],
  
  // Coverage directory
  coverageDirectory: 'coverage',
  
  // Files to collect coverage from
  collectCoverageFrom: [
    // Frontend coverage
    'apps/site/src/**/*.{js,ts,tsx,astro}',
    
    // Python pipeline coverage (for JS tooling)
    'pipelines/**/*.py',
    
    // Exclude patterns
    '!**/*.d.ts',
    '!**/*.test.{js,ts,tsx,py}',
    '!**/__tests__/**',
    '!**/tests/**',
    '!**/*.stories.{js,ts,tsx}',
    '!**/node_modules/**',
    '!**/dist/**',
    '!**/.astro/**',
    '!**/coverage/**',
    '!**/__pycache__/**',
    '!**/*.pyc',
    '!**/env.d.ts',
    '!**/vite.config.*',
    '!**/astro.config.*',
    
    // Exclude configuration files
    '!**/*.config.{js,ts}',
    '!**/.eslintrc.*',
    '!**/prettier.config.*',
    '!**/tailwind.config.*'
  ],
  
  // Coverage path mapping for better organization
  coveragePathIgnorePatterns: [
    '/node_modules/',
    '/dist/',
    '/.astro/',
    '/__pycache__/',
    '/coverage/',
    '\\.d\\.ts$',
    '\\.config\\.(js|ts)$',
    '\\.stories\\.(js|ts|tsx)$'
  ],
  
  // HTML coverage report options
  coverageOptions: {
    html: {
      skipEmpty: false,
      subdir: 'html-report',
      includeAllSources: true
    },
    
    lcov: {
      outputFile: 'lcov.info',
      includeAllSources: true
    },
    
    json: {
      outputFile: 'coverage.json'
    },
    
    text: {
      maxCols: 100,
      skipEmpty: false,
      skipFull: false
    }
  },
  
  // Quality gates
  qualityGates: {
    // Fail CI if coverage drops below these absolute minimums
    minimumCoverage: {
      branches: 85,
      functions: 85,
      lines: 85,
      statements: 85
    },
    
    // Fail CI if coverage drops more than this percentage
    coverageDecrease: {
      branches: 2,
      functions: 2,
      lines: 2,
      statements: 2
    }
  },
  
  // Coverage exclusions for specific patterns
  excludePatterns: [
    // Test files
    '**/*.test.*',
    '**/*.spec.*',
    '**/test_*',
    '**/tests/**',
    '**/__tests__/**',
    
    // Configuration files
    '**/*.config.*',
    '**/.*rc.*',
    
    // Build and dist files
    '**/dist/**',
    '**/build/**',
    '**/.astro/**',
    
    // Dependencies
    '**/node_modules/**',
    '**/__pycache__/**',
    
    // Documentation
    '**/*.md',
    '**/docs/**',
    
    // Environment and type definitions
    '**/*.d.ts',
    '**/env.*',
    
    // Migration and script files
    '**/migrations/**',
    '**/scripts/**/*.py',  // Exclude utility scripts
    
    // Mock and fixture files
    '**/mocks/**',
    '**/fixtures/**',
    '**/__mocks__/**'
  ],
  
  // Per-file coverage requirements
  perFileThreshold: {
    // Core pipeline modules require higher coverage
    'pipelines/dedup/similarity_detector.py': {
      branches: 100,
      functions: 100,
      lines: 100,
      statements: 100
    },
    
    'pipelines/database/operations.py': {
      branches: 95,
      functions: 95,
      lines: 95,
      statements: 95
    },
    
    'pipelines/normalize/content_extractor.py': {
      branches: 95,
      functions: 95,
      lines: 95,
      statements: 95
    }
  }
};