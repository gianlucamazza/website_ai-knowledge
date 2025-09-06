import { defineConfig } from 'vitest/config'
import { resolve } from 'path'

export default defineConfig({
  test: {
    // Test environment
    environment: 'jsdom',
    
    // Global setup
    globals: true,
    setupFiles: ['./tests/setup.js'],
    
    // Test patterns
    include: [
      'tests/**/*.{test,spec}.{js,ts}',
      'src/**/*.{test,spec}.{js,ts}'
    ],
    exclude: [
      'node_modules/**',
      'dist/**',
      '.astro/**'
    ],
    
    // Coverage configuration
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html', 'lcov'],
      reportsDirectory: './tests/coverage',
      include: [
        'src/**/*.{js,ts,astro}',
        'middleware/**/*.{js,ts}'
      ],
      exclude: [
        'tests/**',
        'node_modules/**',
        'dist/**',
        '.astro/**',
        '**/*.d.ts',
        '**/*.config.{js,ts}',
        'src/env.d.ts'
      ],
      thresholds: {
        global: {
          branches: 80,
          functions: 80,
          lines: 80,
          statements: 80
        }
      }
    },
    
    // Test timeout
    testTimeout: 10000,
    hookTimeout: 10000,
    
    // Reporter configuration
    reporter: ['verbose', 'json', 'html'],
    outputFile: {
      json: './tests/reports/vitest-report.json',
      html: './tests/reports/vitest-report.html'
    },
    
    // Mock configuration
    deps: {
      inline: ['@astrojs/internal-helpers']
    },
    
    // Browser testing (for e2e tests)
    browser: {
      enabled: false, // Enable for browser testing
      name: 'chromium',
      provider: 'playwright',
      headless: true
    }
  },
  
  // Resolve configuration
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
      '@components': resolve(__dirname, './src/components'),
      '@layouts': resolve(__dirname, './src/layouts'),
      '@pages': resolve(__dirname, './src/pages'),
      '@content': resolve(__dirname, './src/content'),
      '@utils': resolve(__dirname, './src/utils'),
      '@tests': resolve(__dirname, './tests')
    }
  },
  
  // Define global constants
  define: {
    __TEST__: true,
    'import.meta.env.MODE': '"test"'
  }
})