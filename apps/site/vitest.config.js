/**
 * Vitest Configuration for AI Knowledge Website Frontend Tests
 * 
 * Comprehensive testing setup for Astro components, utilities, and integrations.
 */

import { defineConfig } from 'vitest/config'
import { resolve } from 'path'

export default defineConfig({
  test: {
    // Test environment configuration
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./tests/setup.js'],
    
    // Test discovery
    include: [
      'tests/**/*.{test,spec}.{js,ts}',
      'src/**/*.{test,spec}.{js,ts}'
    ],
    exclude: [
      'node_modules/**',
      'dist/**',
      '.astro/**',
      'tests/e2e/**' // E2E tests run separately
    ],
    
    // Coverage configuration
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html', 'lcov'],
      reportsDirectory: './coverage',
      exclude: [
        'node_modules/**',
        'tests/**',
        'dist/**',
        '.astro/**',
        '**/*.config.js',
        '**/*.config.ts',
        'src/env.d.ts',
        '**/*.d.ts'
      ],
      thresholds: {
        global: {
          branches: 95,
          functions: 95,
          lines: 95,
          statements: 95
        }
      },
      // Files that must be covered
      include: [
        'src/components/**',
        'src/layouts/**',
        'src/utils/**',
        'src/lib/**'
      ],
      // Per-file coverage requirements
      perFile: true
    },
    
    // Test execution configuration
    testTimeout: 10000,
    hookTimeout: 10000,
    teardownTimeout: 5000,
    
    // Parallel execution
    threads: true,
    maxThreads: 4,
    minThreads: 1,
    
    // Watch mode configuration
    watch: {
      clearScreen: false
    },
    
    // Reporter configuration
    reporter: process.env.CI ? ['verbose', 'json', 'junit'] : ['verbose'],
    outputFile: {
      json: './test-results.json',
      junit: './junit.xml'
    },
    
    // Retry configuration for flaky tests
    retry: process.env.CI ? 2 : 0,
    
    // Bail on first failure in CI
    bail: process.env.CI ? 1 : 0
  },
  
  // Resolve configuration for module imports
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
      '@components': resolve(__dirname, './src/components'),
      '@layouts': resolve(__dirname, './src/layouts'),
      '@utils': resolve(__dirname, './src/utils'),
      '@content': resolve(__dirname, './src/content'),
      '@test-utils': resolve(__dirname, './tests/utils')
    }
  },
  
  // Define global variables
  define: {
    'import.meta.env.MODE': JSON.stringify('test'),
    'import.meta.env.PROD': false,
    'import.meta.env.DEV': true,
    'import.meta.env.SSR': false
  },
  
  // Optimize deps for testing
  optimizeDeps: {
    include: [
      '@testing-library/jest-dom',
      '@testing-library/react',
      'vitest/globals'
    ]
  },
  
  // Mock configuration
  server: {
    deps: {
      // Mock Astro components and modules
      external: ['astro', 'astro/*']
    }
  }
})