/**
 * Vitest global setup for frontend testing
 * 
 * Configures testing environment, mocks, and utilities for Astro site testing.
 */

import { beforeAll, beforeEach, afterEach, afterAll, vi } from 'vitest'
import { cleanup } from '@testing-library/react'
import '@testing-library/jest-dom'

// Extend expect with custom matchers
import * as matchers from '@testing-library/jest-dom/matchers'
expect.extend(matchers)

// Global test setup
beforeAll(() => {
  // Setup global test environment
  console.log('🧪 Setting up test environment...')
  
  // Mock browser APIs that might not be available in jsdom
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: vi.fn().mockImplementation(query => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: vi.fn(), // deprecated
      removeListener: vi.fn(), // deprecated
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  })

  // Mock IntersectionObserver
  global.IntersectionObserver = vi.fn().mockImplementation((callback) => ({
    observe: vi.fn(),
    unobserve: vi.fn(),
    disconnect: vi.fn(),
  }))

  // Mock ResizeObserver
  global.ResizeObserver = vi.fn().mockImplementation((callback) => ({
    observe: vi.fn(),
    unobserve: vi.fn(),
    disconnect: vi.fn(),
  }))

  // Mock requestAnimationFrame
  global.requestAnimationFrame = vi.fn().mockImplementation((callback) => {
    return setTimeout(callback, 16)
  })

  global.cancelAnimationFrame = vi.fn().mockImplementation((id) => {
    clearTimeout(id)
  })

  // Mock localStorage
  const localStorageMock = {
    getItem: vi.fn(),
    setItem: vi.fn(),
    removeItem: vi.fn(),
    clear: vi.fn(),
    length: 0,
    key: vi.fn(),
  }
  Object.defineProperty(window, 'localStorage', {
    value: localStorageMock,
    writable: true,
  })

  // Mock sessionStorage
  const sessionStorageMock = {
    getItem: vi.fn(),
    setItem: vi.fn(),
    removeItem: vi.fn(),
    clear: vi.fn(),
    length: 0,
    key: vi.fn(),
  }
  Object.defineProperty(window, 'sessionStorage', {
    value: sessionStorageMock,
    writable: true,
  })

  // Mock fetch for API testing
  global.fetch = vi.fn()

  // Mock URL constructor for Node.js compatibility
  if (typeof URL === 'undefined') {
    global.URL = class URL {
      constructor(url, base) {
        this.href = url
        this.origin = base || ''
        this.pathname = url.replace(/^https?:\/\/[^/]+/, '') || '/'
        this.search = ''
        this.hash = ''
      }
    }
  }

  // Mock import.meta for Vite/Astro compatibility
  if (typeof globalThis.importMeta === 'undefined') {
    globalThis.importMeta = {
      env: {
        MODE: 'test',
        BASE_URL: '/',
        PROD: false,
        DEV: false,
        SSR: false,
      },
      glob: vi.fn().mockReturnValue({}),
      hot: {
        accept: vi.fn(),
        decline: vi.fn(),
        dispose: vi.fn(),
        invalidate: vi.fn(),
        on: vi.fn(),
        send: vi.fn(),
      },
    }
  }
})

// Setup before each test
beforeEach(() => {
  // Reset all mocks before each test
  vi.clearAllMocks()
  
  // Reset fetch mock
  global.fetch.mockClear()
  
  // Reset storage mocks
  window.localStorage.clear()
  window.sessionStorage.clear()
  
  // Reset DOM
  document.body.innerHTML = ''
  document.head.innerHTML = ''
})

// Cleanup after each test
afterEach(() => {
  // Cleanup React Testing Library
  cleanup()
  
  // Clear all timers
  vi.clearAllTimers()
  
  // Restore all mocks
  vi.restoreAllMocks()
})

// Global teardown
afterAll(() => {
  console.log('🧹 Cleaning up test environment...')
})

// Custom test utilities
export const testUtils = {
  // Wait for async operations
  waitFor: async (callback, timeout = 5000) => {
    const startTime = Date.now()
    while (Date.now() - startTime < timeout) {
      try {
        const result = await callback()
        if (result) return result
      } catch (error) {
        // Continue waiting
      }
      await new Promise(resolve => setTimeout(resolve, 10))
    }
    throw new Error(`Timeout after ${timeout}ms`)
  },

  // Create mock Astro component props
  createMockAstroProps: (overrides = {}) => ({
    url: new URL('http://localhost:3000'),
    params: {},
    props: {},
    ...overrides,
  }),

  // Mock Astro.glob results
  mockAstroGlob: (files = []) => {
    return vi.fn().mockResolvedValue(files.map(file => ({
      url: file.url || `/test/${file.slug}`,
      file: file.file || `src/content/${file.slug}.md`,
      frontmatter: file.frontmatter || {},
      ...file,
    })))
  },

  // Mock content collection
  mockContentCollection: (entries = []) => {
    return entries.map((entry, index) => ({
      id: entry.id || `test-entry-${index}`,
      slug: entry.slug || `test-entry-${index}`,
      body: entry.body || 'Test content body',
      collection: entry.collection || 'articles',
      data: entry.data || {
        title: `Test Entry ${index + 1}`,
        description: `Test description for entry ${index + 1}`,
        publishDate: new Date().toISOString(),
      },
      ...entry,
    }))
  },

  // Create test article data
  createTestArticle: (overrides = {}) => ({
    id: 'test-article-1',
    slug: 'test-article-1',
    title: 'Test Article',
    description: 'This is a test article for testing purposes',
    publishDate: '2024-01-15T10:00:00Z',
    updateDate: '2024-01-15T10:00:00Z',
    category: 'test',
    tags: ['test', 'article'],
    author: 'Test Author',
    readingTime: 5,
    wordCount: 500,
    difficulty: 'beginner',
    body: 'This is the test article content.',
    ...overrides,
  }),

  // Create test glossary entry
  createTestGlossaryEntry: (overrides = {}) => ({
    id: 'test-term',
    slug: 'test-term',
    term: 'Test Term',
    definition: 'This is a test term definition',
    category: 'test-category',
    tags: ['test'],
    relatedTerms: [],
    examples: ['Example 1', 'Example 2'],
    ...overrides,
  }),

  // Mock search functionality
  mockSearch: (results = []) => {
    return vi.fn().mockImplementation((query) => {
      return Promise.resolve({
        query,
        results: results.filter(result => 
          result.title?.toLowerCase().includes(query.toLowerCase()) ||
          result.content?.toLowerCase().includes(query.toLowerCase())
        ),
        total: results.length,
      })
    })
  },

  // Mock navigation
  mockNavigation: () => ({
    navigate: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    reload: vi.fn(),
  }),
}

// Make test utilities globally available
globalThis.testUtils = testUtils

// Error handling for unhandled promise rejections
process.on('unhandledRejection', (error) => {
  console.error('Unhandled promise rejection:', error)
})

// Silence console warnings in tests unless debugging
if (!process.env.DEBUG_TESTS) {
  console.warn = vi.fn()
}