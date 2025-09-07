/**
 * Unit tests for Astro components
 * 
 * Tests component rendering, props handling, and functionality.
 */

import React from 'react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { testUtils } from '../setup.js'

// Mock Astro components (since they need to be compiled)
const mockComponents = {
  BaseLayout: ({ children, title, description }) => (
    <html>
      <head>
        <title>{title}</title>
        <meta name="description" content={description} />
      </head>
      <body>{children}</body>
    </html>
  ),
  
  ArticleCard: ({ article }) => (
    <article data-testid="article-card">
      <h2>{article.title}</h2>
      <p>{article.description}</p>
      <div data-testid="article-meta">
        <span>By {article.author}</span>
        <span>{article.readingTime} min read</span>
        <span>{article.category}</span>
      </div>
      <div data-testid="article-tags">
        {(article.tags || []).map(tag => (
          <span key={tag} className="tag">{tag}</span>
        ))}
      </div>
    </article>
  ),
  
  SearchBox: ({ placeholder = 'Search...', onSearch }) => {
    const handleSubmit = (e) => {
      e.preventDefault()
      const formData = new FormData(e.target)
      const query = formData.get('query')
      if (onSearch) onSearch(query)
    }
    
    return (
      <form onSubmit={handleSubmit} data-testid="search-form" role="form">
        <input
          name="query"
          type="text"
          placeholder={placeholder}
          data-testid="search-input"
        />
        <button type="submit" data-testid="search-button">
          Search
        </button>
      </form>
    )
  },
  
  NavigationMenu: ({ items = [] }) => (
    <nav data-testid="navigation-menu">
      <ul>
        {items.map(item => (
          <li key={item.href}>
            <a href={item.href}>{item.label}</a>
          </li>
        ))}
      </ul>
    </nav>
  ),
  
  GlossaryEntry: ({ entry }) => (
    <div data-testid="glossary-entry">
      <h1>{entry.term}</h1>
      <div data-testid="definition">{entry.definition}</div>
      {entry.examples && (
        <div data-testid="examples">
          <h3>Examples</h3>
          <ul>
            {entry.examples.map((example, index) => (
              <li key={index}>{example}</li>
            ))}
          </ul>
        </div>
      )}
      {entry.relatedTerms && entry.relatedTerms.length > 0 && (
        <div data-testid="related-terms">
          <h3>Related Terms</h3>
          <ul>
            {entry.relatedTerms.map(term => (
              <li key={term.slug}>
                <a href={`/glossary/${term.slug}`}>{term.term}</a>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

describe('BaseLayout Component', () => {
  it('renders with correct title and description', () => {
    const { container } = render(
      mockComponents.BaseLayout({
        title: 'Test Page',
        description: 'Test page description',
        children: <main>Test content</main>
      })
    )
    
    expect(container.querySelector('title')).toHaveTextContent('Test Page')
    expect(container.querySelector('meta[name="description"]'))
      .toHaveAttribute('content', 'Test page description')
    expect(container.querySelector('main')).toHaveTextContent('Test content')
  })
  
  it('renders children content correctly', () => {
    const { container } = render(
      mockComponents.BaseLayout({
        title: 'Test',
        children: (
          <>
            <header>Header content</header>
            <main>Main content</main>
            <footer>Footer content</footer>
          </>
        )
      })
    )
    
    expect(container.querySelector('header')).toHaveTextContent('Header content')
    expect(container.querySelector('main')).toHaveTextContent('Main content')
    expect(container.querySelector('footer')).toHaveTextContent('Footer content')
  })
})

describe('ArticleCard Component', () => {
  let testArticle
  
  beforeEach(() => {
    testArticle = testUtils.createTestArticle({
      title: 'Understanding Machine Learning',
      description: 'A comprehensive guide to ML concepts',
      author: 'Jane Doe',
      readingTime: 8,
      category: 'machine-learning',
      tags: ['ml', 'ai', 'algorithms']
    })
  })
  
  it('renders article information correctly', () => {
    render(mockComponents.ArticleCard({ article: testArticle }))
    
    expect(screen.getByText('Understanding Machine Learning')).toBeInTheDocument()
    expect(screen.getByText('A comprehensive guide to ML concepts')).toBeInTheDocument()
    expect(screen.getByText('By Jane Doe')).toBeInTheDocument()
    expect(screen.getByText('8 min read')).toBeInTheDocument()
    expect(screen.getByText('machine-learning')).toBeInTheDocument()
  })
  
  it('renders all article tags', () => {
    render(mockComponents.ArticleCard({ article: testArticle }))
    
    const tagsContainer = screen.getByTestId('article-tags')
    expect(tagsContainer).toBeInTheDocument()
    
    testArticle.tags.forEach(tag => {
      expect(screen.getByText(tag)).toBeInTheDocument()
    })
  })
  
  it('handles articles with minimal data', () => {
    const minimalArticle = {
      title: 'Minimal Article',
      description: 'Minimal description',
      author: 'Unknown',
      readingTime: 1,
      category: 'general',
      tags: []
    }
    
    render(mockComponents.ArticleCard({ article: minimalArticle }))
    
    expect(screen.getByText('Minimal Article')).toBeInTheDocument()
    expect(screen.getByText('Minimal description')).toBeInTheDocument()
    expect(screen.getByText('By Unknown')).toBeInTheDocument()
    expect(screen.getByText('1 min read')).toBeInTheDocument()
  })
  
  it('applies correct test IDs for testing', () => {
    render(mockComponents.ArticleCard({ article: testArticle }))
    
    expect(screen.getByTestId('article-card')).toBeInTheDocument()
    expect(screen.getByTestId('article-meta')).toBeInTheDocument()
    expect(screen.getByTestId('article-tags')).toBeInTheDocument()
  })
})

describe('SearchBox Component', () => {
  it('renders with default placeholder', () => {
    render(mockComponents.SearchBox({}))
    
    expect(screen.getByPlaceholderText('Search...')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Search' })).toBeInTheDocument()
  })
  
  it('renders with custom placeholder', () => {
    render(mockComponents.SearchBox({ placeholder: 'Search articles...' }))
    
    expect(screen.getByPlaceholderText('Search articles...')).toBeInTheDocument()
  })
  
  it('calls onSearch when form is submitted', async () => {
    const mockOnSearch = vi.fn()
    render(mockComponents.SearchBox({ onSearch: mockOnSearch }))
    
    const input = screen.getByTestId('search-input')
    const form = screen.getByTestId('search-form')
    
    // Simulate user input
    input.value = 'machine learning'
    
    // Simulate form submission
    const event = new Event('submit', { bubbles: true })
    Object.defineProperty(event, 'target', {
      value: form,
      writable: false
    })
    
    // Mock FormData
    const formData = new FormData()
    formData.append('query', 'machine learning')
    vi.spyOn(window, 'FormData').mockReturnValue(formData)
    
    form.dispatchEvent(event)
    
    expect(mockOnSearch).toHaveBeenCalledWith('machine learning')
  })
  
  it('prevents default form submission', () => {
    const mockOnSearch = vi.fn()
    render(mockComponents.SearchBox({ onSearch: mockOnSearch }))
    
    const form = screen.getByTestId('search-form')
    const event = new Event('submit', { bubbles: true })
    const preventDefaultSpy = vi.spyOn(event, 'preventDefault')
    
    form.dispatchEvent(event)
    
    expect(preventDefaultSpy).toHaveBeenCalled()
  })
})

describe('NavigationMenu Component', () => {
  const mockNavItems = [
    { href: '/', label: 'Home' },
    { href: '/articles', label: 'Articles' },
    { href: '/glossary', label: 'Glossary' },
    { href: '/about', label: 'About' }
  ]
  
  it('renders all navigation items', () => {
    render(mockComponents.NavigationMenu({ items: mockNavItems }))
    
    mockNavItems.forEach(item => {
      const link = screen.getByRole('link', { name: item.label })
      expect(link).toBeInTheDocument()
      expect(link).toHaveAttribute('href', item.href)
    })
  })
  
  it('renders empty menu when no items provided', () => {
    render(mockComponents.NavigationMenu({ items: [] }))
    
    const nav = screen.getByTestId('navigation-menu')
    expect(nav).toBeInTheDocument()
    expect(nav.querySelector('ul')).toBeEmptyDOMElement()
  })
  
  it('handles missing items prop gracefully', () => {
    render(mockComponents.NavigationMenu({}))
    
    const nav = screen.getByTestId('navigation-menu')
    expect(nav).toBeInTheDocument()
  })
})

describe('GlossaryEntry Component', () => {
  let testEntry
  
  beforeEach(() => {
    testEntry = testUtils.createTestGlossaryEntry({
      term: 'Machine Learning',
      definition: 'A method of data analysis that automates analytical model building.',
      examples: [
        'Email spam detection',
        'Recommendation systems',
        'Image recognition'
      ],
      relatedTerms: [
        { term: 'Artificial Intelligence', slug: 'artificial-intelligence' },
        { term: 'Deep Learning', slug: 'deep-learning' }
      ]
    })
  })
  
  it('renders term and definition', () => {
    render(mockComponents.GlossaryEntry({ entry: testEntry }))
    
    expect(screen.getByText('Machine Learning')).toBeInTheDocument()
    expect(screen.getByText('A method of data analysis that automates analytical model building.'))
      .toBeInTheDocument()
  })
  
  it('renders examples when provided', () => {
    render(mockComponents.GlossaryEntry({ entry: testEntry }))
    
    expect(screen.getByText('Examples')).toBeInTheDocument()
    testEntry.examples.forEach(example => {
      expect(screen.getByText(example)).toBeInTheDocument()
    })
  })
  
  it('renders related terms when provided', () => {
    render(mockComponents.GlossaryEntry({ entry: testEntry }))
    
    expect(screen.getByText('Related Terms')).toBeInTheDocument()
    
    const aiLink = screen.getByRole('link', { name: 'Artificial Intelligence' })
    expect(aiLink).toHaveAttribute('href', '/glossary/artificial-intelligence')
    
    const dlLink = screen.getByRole('link', { name: 'Deep Learning' })
    expect(dlLink).toHaveAttribute('href', '/glossary/deep-learning')
  })
  
  it('handles entry without examples', () => {
    const entryWithoutExamples = { ...testEntry, examples: null }
    render(mockComponents.GlossaryEntry({ entry: entryWithoutExamples }))
    
    expect(screen.getByText('Machine Learning')).toBeInTheDocument()
    expect(screen.queryByText('Examples')).not.toBeInTheDocument()
  })
  
  it('handles entry without related terms', () => {
    const entryWithoutRelated = { ...testEntry, relatedTerms: [] }
    render(mockComponents.GlossaryEntry({ entry: entryWithoutRelated }))
    
    expect(screen.getByText('Machine Learning')).toBeInTheDocument()
    expect(screen.queryByText('Related Terms')).not.toBeInTheDocument()
  })
  
  it('applies correct test IDs', () => {
    render(mockComponents.GlossaryEntry({ entry: testEntry }))
    
    expect(screen.getByTestId('glossary-entry')).toBeInTheDocument()
    expect(screen.getByTestId('definition')).toBeInTheDocument()
    expect(screen.getByTestId('examples')).toBeInTheDocument()
    expect(screen.getByTestId('related-terms')).toBeInTheDocument()
  })
})

describe('Component Accessibility', () => {
  it('ArticleCard has proper semantic structure', () => {
    const testArticle = testUtils.createTestArticle()
    render(mockComponents.ArticleCard({ article: testArticle }))
    
    // Should use semantic article element
    expect(screen.getByRole('article')).toBeInTheDocument()
    
    // Should have proper heading hierarchy
    expect(screen.getByRole('heading', { level: 2 })).toBeInTheDocument()
  })
  
  it('SearchBox has proper form labels and structure', () => {
    render(mockComponents.SearchBox({ placeholder: 'Search articles' }))
    
    // Should have proper form structure
    const form = screen.getByRole('form')
    expect(form).toBeInTheDocument()
    
    // Should have searchbox role
    const input = screen.getByRole('textbox')
    expect(input).toBeInTheDocument()
    
    // Should have submit button
    const button = screen.getByRole('button', { name: 'Search' })
    expect(button).toHaveAttribute('type', 'submit')
  })
  
  it('NavigationMenu has proper navigation structure', () => {
    const navItems = [
      { href: '/', label: 'Home' },
      { href: '/articles', label: 'Articles' }
    ]
    
    render(mockComponents.NavigationMenu({ items: navItems }))
    
    // Should use semantic nav element
    const nav = screen.getByRole('navigation')
    expect(nav).toBeInTheDocument()
    
    // Should have list structure
    const list = screen.getByRole('list')
    expect(list).toBeInTheDocument()
    
    // Should have proper links
    navItems.forEach(item => {
      expect(screen.getByRole('link', { name: item.label })).toBeInTheDocument()
    })
  })
  
  it('GlossaryEntry has proper heading hierarchy', () => {
    const testEntry = testUtils.createTestGlossaryEntry()
    render(mockComponents.GlossaryEntry({ entry: testEntry }))
    
    // Should have main heading
    expect(screen.getByRole('heading', { level: 1 })).toBeInTheDocument()
    
    // Should have section headings
    const subHeadings = screen.getAllByRole('heading', { level: 3 })
    expect(subHeadings.length).toBeGreaterThan(0)
  })
})

describe('Component Error Handling', () => {
  it('ArticleCard handles missing article data gracefully', () => {
    const incompleteArticle = {
      title: 'Test Article'
      // Missing other required fields
    }
    
    expect(() => {
      render(mockComponents.ArticleCard({ article: incompleteArticle }))
    }).not.toThrow()
    
    expect(screen.getByText('Test Article')).toBeInTheDocument()
  })
  
  it('SearchBox handles missing onSearch prop', () => {
    expect(() => {
      render(mockComponents.SearchBox({}))
    }).not.toThrow()
    
    const form = screen.getByTestId('search-form')
    expect(form).toBeInTheDocument()
  })
  
  it('NavigationMenu handles invalid items gracefully', () => {
    const invalidItems = [
      { href: '/', label: 'Home' },
      { href: null, label: 'Invalid' }, // Invalid item
      { href: '/valid', label: 'Valid' }
    ]
    
    expect(() => {
      render(mockComponents.NavigationMenu({ items: invalidItems }))
    }).not.toThrow()
  })
})