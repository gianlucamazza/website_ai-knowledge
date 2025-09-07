/**
 * Jest configuration for legacy or additional test scenarios
 * Primary testing is done with Vitest, but Jest config is provided for compatibility
 */

module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'jsdom',
  
  // Test file patterns
  testMatch: [
    '<rootDir>/tests/**/*.test.{js,ts,tsx}',
    '<rootDir>/src/**/__tests__/**/*.{js,ts,tsx}'
  ],
  
  // Module resolution
  moduleNameMapping: {
    '^@/(.*)$': '<rootDir>/src/$1',
    '^~/(.*)$': '<rootDir>/$1'
  },
  
  // File extensions to consider
  moduleFileExtensions: ['js', 'jsx', 'ts', 'tsx', 'json'],
  
  // Transform configuration
  transform: {
    '^.+\\.(js|jsx|ts|tsx)$': 'ts-jest',
    '^.+\\.astro$': '@astrojs/compiler/jest',
    '^.+\\.(css|scss|sass|less|styl|stylus|pcss|postcss)$': 'identity-obj-proxy'
  },
  
  // Setup files
  setupFilesAfterEnv: ['<rootDir>/tests/setup.js'],
  
  // Coverage configuration
  collectCoverageFrom: [
    'src/**/*.{js,ts,tsx,astro}',
    '!src/**/*.d.ts',
    '!src/**/*.stories.{js,ts,tsx}',
    '!src/**/*.test.{js,ts,tsx}',
    '!src/env.d.ts'
  ],
  
  coverageDirectory: 'coverage',
  coverageReporters: ['text', 'lcov', 'html'],
  
  coverageThreshold: {
    global: {
      branches: 90,
      functions: 90,
      lines: 90,
      statements: 90
    }
  },
  
  // Test environment options
  testEnvironmentOptions: {
    url: 'http://localhost:4321'
  },
  
  // Globals for Astro components
  globals: {
    'ts-jest': {
      useESM: true,
      tsconfig: './tsconfig.json'
    }
  },
  
  // Ignore patterns
  testPathIgnorePatterns: [
    '<rootDir>/node_modules/',
    '<rootDir>/dist/',
    '<rootDir>/.astro/'
  ],
  
  // Module directories
  moduleDirectories: ['node_modules', '<rootDir>/src'],
  
  // Clear mocks between tests
  clearMocks: true,
  
  // Timeout for tests
  testTimeout: 10000,
  
  // Verbose output
  verbose: true
};