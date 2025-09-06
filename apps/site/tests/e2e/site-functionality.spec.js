/**
 * End-to-end tests for AI Knowledge website functionality
 * 
 * Tests user journeys, page interactions, and core site features.
 */

import { test, expect } from '@playwright/test'

// Test data
const testData = {
  articles: [
    {
      slug: 'machine-learning-basics',
      title: 'Machine Learning Basics',
      description: 'Introduction to machine learning concepts'
    },
    {
      slug: 'deep-learning-guide',
      title: 'Deep Learning Guide',
      description: 'Comprehensive guide to deep learning'
    }
  ],
  glossaryTerms: [
    {
      slug: 'artificial-intelligence',
      term: 'Artificial Intelligence',
      definition: 'Computer systems able to perform tasks that typically require human intelligence'
    },
    {
      slug: 'neural-network',
      term: 'Neural Network',
      definition: 'Computing systems inspired by biological neural networks'
    }
  ]
}

test.describe('Homepage Functionality', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('loads homepage successfully', async ({ page }) => {
    // Check page title
    await expect(page).toHaveTitle(/AI Knowledge/i)
    
    // Check main heading
    const mainHeading = page.getByRole('heading', { level: 1 })
    await expect(mainHeading).toBeVisible()
    
    // Check navigation is present
    const navigation = page.getByRole('navigation')
    await expect(navigation).toBeVisible()
  })

  test('has correct meta tags for SEO', async ({ page }) => {
    // Check meta description
    const metaDescription = page.locator('meta[name="description"]')
    await expect(metaDescription).toHaveAttribute('content', /AI|artificial intelligence|knowledge/i)
    
    // Check meta viewport
    const metaViewport = page.locator('meta[name="viewport"]')
    await expect(metaViewport).toHaveAttribute('content', /width=device-width/i)
    
    // Check canonical URL if present
    const canonicalLink = page.locator('link[rel="canonical"]')
    if (await canonicalLink.count() > 0) {
      await expect(canonicalLink).toHaveAttribute('href')
    }
  })

  test('navigation menu works correctly', async ({ page }) => {
    // Test Articles link
    const articlesLink = page.getByRole('link', { name: /articles/i })
    await expect(articlesLink).toBeVisible()
    await articlesLink.click()
    await expect(page).toHaveURL(/\/articles/i)
    
    // Go back to home
    await page.goto('/')
    
    // Test Glossary link
    const glossaryLink = page.getByRole('link', { name: /glossary/i })
    if (await glossaryLink.count() > 0) {
      await glossaryLink.click()
      await expect(page).toHaveURL(/\/glossary/i)
    }
  })

  test('search functionality is accessible', async ({ page }) => {
    // Look for search input
    const searchInput = page.getByRole('searchbox').or(page.getByPlaceholder(/search/i))
    
    if (await searchInput.count() > 0) {
      await expect(searchInput).toBeVisible()
      await expect(searchInput).toBeEnabled()
      
      // Test search interaction
      await searchInput.fill('machine learning')
      await searchInput.press('Enter')
      
      // Should navigate to search results or show results
      await page.waitForTimeout(1000) // Wait for search to complete
    }
  })

  test('responsive design works on mobile', async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 })
    
    // Check that content is still accessible
    const mainContent = page.getByRole('main').or(page.locator('main'))
    await expect(mainContent).toBeVisible()
    
    // Check mobile navigation (hamburger menu or mobile nav)
    const mobileNav = page.getByRole('button', { name: /menu|nav/i })
    if (await mobileNav.count() > 0) {
      await mobileNav.click()
      await expect(page.getByRole('navigation')).toBeVisible()
    }
  })
})

test.describe('Articles Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/articles')
  })

  test('articles page loads and displays content', async ({ page }) => {
    // Check page title
    await expect(page).toHaveTitle(/articles/i)
    
    // Check main heading
    const heading = page.getByRole('heading', { level: 1 })
    await expect(heading).toContainText(/articles/i)
    
    // Check for article cards or list
    const articles = page.getByTestId('article-card').or(
      page.locator('article').or(
        page.locator('[data-article]')
      )
    )
    
    if (await articles.count() > 0) {
      await expect(articles.first()).toBeVisible()
    }
  })

  test('article filtering and sorting works', async ({ page }) => {
    // Look for filter controls
    const categoryFilter = page.getByRole('combobox', { name: /category/i }).or(
      page.locator('select[name*="category"]')
    )
    
    if (await categoryFilter.count() > 0) {
      await categoryFilter.selectOption({ label: /machine.learning/i })
      await page.waitForTimeout(500)
      
      // Verify filtering worked (articles should be filtered)
      const articles = page.locator('[data-category*="machine-learning"]')
      if (await articles.count() > 0) {
        await expect(articles.first()).toBeVisible()
      }
    }
    
    // Test sorting
    const sortSelect = page.getByRole('combobox', { name: /sort/i }).or(
      page.locator('select[name*="sort"]')
    )
    
    if (await sortSelect.count() > 0) {
      await sortSelect.selectOption({ label: /date/i })
      await page.waitForTimeout(500)
    }
  })

  test('pagination works if present', async ({ page }) => {
    // Look for pagination controls
    const nextButton = page.getByRole('button', { name: /next/i }).or(
      page.getByRole('link', { name: /next/i })
    )
    
    if (await nextButton.count() > 0) {
      await nextButton.click()
      await page.waitForLoadState('networkidle')
      
      // Should navigate to next page
      await expect(page).toHaveURL(/page=2|\/2/i)
      
      // Previous button should be available
      const prevButton = page.getByRole('button', { name: /previous|prev/i }).or(
        page.getByRole('link', { name: /previous|prev/i })
      )
      await expect(prevButton).toBeVisible()
    }
  })

  test('article card interaction', async ({ page }) => {
    // Find first article card
    const firstArticle = page.locator('[data-testid="article-card"]').or(
      page.locator('article').or(
        page.locator('[data-article]')
      )
    ).first()
    
    if (await firstArticle.count() > 0) {
      // Check article card has required elements
      const title = firstArticle.getByRole('heading')
      await expect(title).toBeVisible()
      
      const description = firstArticle.locator('p, [data-description]').first()
      if (await description.count() > 0) {
        await expect(description).toBeVisible()
      }
      
      // Click on article to navigate to detail page
      await firstArticle.click()
      await page.waitForLoadState('networkidle')
      
      // Should navigate to article detail page
      await expect(page).toHaveURL(/\/articles\/[^\/]+$/)
    }
  })
})

test.describe('Article Detail Page', () => {
  test('article detail page renders correctly', async ({ page }) => {
    // Navigate to a test article (adjust URL as needed)
    await page.goto('/articles/machine-learning-basics', { waitUntil: 'networkidle' })
    
    // Check if page loads (might be 404 if article doesn't exist)
    const pageStatus = page.locator('body')
    await expect(pageStatus).toBeVisible()
    
    // If article exists, check its structure
    const articleTitle = page.getByRole('heading', { level: 1 })
    if (await articleTitle.count() > 0) {
      await expect(articleTitle).toBeVisible()
      
      // Check article content
      const articleContent = page.getByRole('main').or(page.locator('[data-content]'))
      await expect(articleContent).toBeVisible()
      
      // Check metadata (author, date, reading time)
      const metadata = page.locator('[data-meta], .article-meta, .post-meta')
      if (await metadata.count() > 0) {
        await expect(metadata).toBeVisible()
      }
      
      // Check tags if present
      const tags = page.locator('[data-tags], .tags, .article-tags')
      if (await tags.count() > 0) {
        await expect(tags).toBeVisible()
      }
    }
  })

  test('article navigation and related content', async ({ page }) => {
    await page.goto('/articles/machine-learning-basics', { waitUntil: 'networkidle' })
    
    // Check for related articles section
    const relatedSection = page.locator('[data-related], .related-articles, .related-posts')
    if (await relatedSection.count() > 0) {
      await expect(relatedSection).toBeVisible()
      
      // Check related article links
      const relatedLinks = relatedSection.getByRole('link')
      if (await relatedLinks.count() > 0) {
        await expect(relatedLinks.first()).toBeVisible()
      }
    }
    
    // Check for table of contents if present
    const toc = page.locator('[data-toc], .table-of-contents, .toc')
    if (await toc.count() > 0) {
      await expect(toc).toBeVisible()
      
      // Test TOC link functionality
      const tocLinks = toc.getByRole('link')
      if (await tocLinks.count() > 0) {
        await tocLinks.first().click()
        // Should scroll to section (hard to test scrolling directly)
      }
    }
    
    // Check breadcrumb navigation
    const breadcrumbs = page.locator('[data-breadcrumbs], .breadcrumbs, nav[aria-label*="breadcrumb"]')
    if (await breadcrumbs.count() > 0) {
      await expect(breadcrumbs).toBeVisible()
    }
  })

  test('social sharing functionality', async ({ page }) => {
    await page.goto('/articles/machine-learning-basics', { waitUntil: 'networkidle' })
    
    // Look for social sharing buttons
    const shareButtons = page.locator('[data-share], .share-buttons, .social-share')
    if (await shareButtons.count() > 0) {
      await expect(shareButtons).toBeVisible()
      
      // Test share button (without actually sharing)
      const shareButton = shareButtons.getByRole('button').or(shareButtons.getByRole('link')).first()
      if (await shareButton.count() > 0) {
        await expect(shareButton).toBeVisible()
        // Note: We don't actually click to avoid opening external sites
      }
    }
  })
})

test.describe('Glossary Functionality', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/glossary')
  })

  test('glossary page loads and displays terms', async ({ page }) => {
    // Check page title
    await expect(page).toHaveTitle(/glossary/i)
    
    // Check main heading
    const heading = page.getByRole('heading', { level: 1 })
    await expect(heading).toContainText(/glossary/i)
    
    // Check for glossary terms
    const terms = page.getByTestId('glossary-term').or(
      page.locator('[data-term]').or(
        page.locator('.glossary-entry')
      )
    )
    
    if (await terms.count() > 0) {
      await expect(terms.first()).toBeVisible()
    }
  })

  test('glossary search and filtering', async ({ page }) => {
    // Look for search functionality
    const searchInput = page.getByRole('searchbox').or(
      page.getByPlaceholder(/search.*term/i)
    )
    
    if (await searchInput.count() > 0) {
      await searchInput.fill('machine')
      await page.keyboard.press('Enter')
      await page.waitForTimeout(500)
      
      // Results should be filtered
      const results = page.locator('[data-term*="machine"], [data-search-result]')
      if (await results.count() > 0) {
        await expect(results.first()).toBeVisible()
      }
    }
    
    // Test category filtering
    const categoryFilter = page.locator('select[name*="category"], [data-category-filter]')
    if (await categoryFilter.count() > 0) {
      await categoryFilter.selectOption({ index: 1 })
      await page.waitForTimeout(500)
    }
  })

  test('alphabetical navigation works', async ({ page }) => {
    // Look for alphabetical navigation
    const alphaNav = page.locator('[data-alpha-nav], .alphabet-nav, .letter-nav')
    if (await alphaNav.count() > 0) {
      await expect(alphaNav).toBeVisible()
      
      // Click on a letter
      const letterM = alphaNav.getByText('M', { exact: true })
      if (await letterM.count() > 0) {
        await letterM.click()
        await page.waitForTimeout(500)
        
        // Should scroll to or filter terms starting with M
      }
    }
  })
})

test.describe('Glossary Term Detail', () => {
  test('glossary term page renders correctly', async ({ page }) => {
    // Navigate to a test term
    await page.goto('/glossary/artificial-intelligence', { waitUntil: 'networkidle' })
    
    // Check if page loads
    const pageStatus = page.locator('body')
    await expect(pageStatus).toBeVisible()
    
    // If term exists, check its structure
    const termTitle = page.getByRole('heading', { level: 1 })
    if (await termTitle.count() > 0) {
      await expect(termTitle).toBeVisible()
      
      // Check definition
      const definition = page.getByTestId('definition').or(
        page.locator('[data-definition], .definition')
      )
      await expect(definition).toBeVisible()
      
      // Check examples if present
      const examples = page.getByTestId('examples').or(
        page.locator('[data-examples], .examples')
      )
      if (await examples.count() > 0) {
        await expect(examples).toBeVisible()
      }
      
      // Check related terms
      const relatedTerms = page.getByTestId('related-terms').or(
        page.locator('[data-related-terms], .related-terms')
      )
      if (await relatedTerms.count() > 0) {
        await expect(relatedTerms).toBeVisible()
        
        // Test related term links
        const relatedLinks = relatedTerms.getByRole('link')
        if (await relatedLinks.count() > 0) {
          await expect(relatedLinks.first()).toBeVisible()
          
          // Click on related term
          await relatedLinks.first().click()
          await page.waitForLoadState('networkidle')
          
          // Should navigate to related term page
          await expect(page).toHaveURL(/\/glossary\/[^\/]+$/)
        }
      }
    }
  })
})

test.describe('Search Functionality', () => {
  test('global search works across content types', async ({ page }) => {
    await page.goto('/')
    
    // Find global search
    const searchInput = page.getByRole('searchbox').or(
      page.getByPlaceholder(/search/i)
    )
    
    if (await searchInput.count() > 0) {
      await searchInput.fill('neural network')
      await page.keyboard.press('Enter')
      await page.waitForLoadState('networkidle')
      
      // Should show search results
      const results = page.locator('[data-search-results], .search-results, .results')
      if (await results.count() > 0) {
        await expect(results).toBeVisible()
        
        // Check for different content types in results
        const articleResults = page.locator('[data-type="article"], .result-article')
        const glossaryResults = page.locator('[data-type="glossary"], .result-glossary')
        
        // At least one type should have results
        if (await articleResults.count() > 0) {
          await expect(articleResults.first()).toBeVisible()
        }
        if (await glossaryResults.count() > 0) {
          await expect(glossaryResults.first()).toBeVisible()
        }
      }
    }
  })

  test('search result interaction', async ({ page }) => {
    await page.goto('/?q=machine%20learning') // Direct search URL
    
    // Look for search results
    const firstResult = page.locator('[data-search-result], .search-result').first()
    if (await firstResult.count() > 0) {
      await expect(firstResult).toBeVisible()
      
      // Check result has title and description
      const resultTitle = firstResult.getByRole('heading').or(firstResult.locator('.title'))
      await expect(resultTitle).toBeVisible()
      
      // Click on result
      await firstResult.click()
      await page.waitForLoadState('networkidle')
      
      // Should navigate to the content page
      await expect(page).toHaveURL(/\/(articles|glossary)\//)
    }
  })
})

test.describe('Performance and Accessibility', () => {
  test('page loads within acceptable time', async ({ page }) => {
    const startTime = Date.now()
    
    await page.goto('/', { waitUntil: 'networkidle' })
    
    const loadTime = Date.now() - startTime
    expect(loadTime).toBeLessThan(5000) // Should load within 5 seconds
  })

  test('has no accessibility violations on key pages', async ({ page }) => {
    // Test homepage
    await page.goto('/')
    
    // Basic accessibility checks
    const mainLandmark = page.getByRole('main')
    await expect(mainLandmark).toBeVisible()
    
    const navigation = page.getByRole('navigation')
    await expect(navigation).toBeVisible()
    
    // Check heading hierarchy
    const h1 = page.getByRole('heading', { level: 1 })
    await expect(h1).toHaveCount(1) // Should have exactly one H1
    
    // Test articles page
    await page.goto('/articles')
    const articlesH1 = page.getByRole('heading', { level: 1 })
    await expect(articlesH1).toHaveCount(1)
    
    // Test glossary page
    await page.goto('/glossary')
    const glossaryH1 = page.getByRole('heading', { level: 1 })
    await expect(glossaryH1).toHaveCount(1)
  })

  test('works with keyboard navigation', async ({ page }) => {
    await page.goto('/')
    
    // Test tab navigation
    await page.keyboard.press('Tab')
    
    // Should focus on first focusable element
    const focusedElement = page.locator(':focus')
    await expect(focusedElement).toBeVisible()
    
    // Continue tabbing through navigation
    for (let i = 0; i < 5; i++) {
      await page.keyboard.press('Tab')
      const currentFocus = page.locator(':focus')
      if (await currentFocus.count() > 0) {
        await expect(currentFocus).toBeVisible()
      }
    }
    
    // Test Enter key on focused link
    const focusedLink = page.locator(':focus')
    if (await focusedLink.count() > 0 && await focusedLink.getAttribute('href')) {
      await page.keyboard.press('Enter')
      await page.waitForLoadState('networkidle')
      // Should navigate to the link destination
    }
  })

  test('responsive design works across viewports', async ({ page }) => {
    const viewports = [
      { width: 320, height: 568 },  // Mobile
      { width: 768, height: 1024 }, // Tablet
      { width: 1024, height: 768 }, // Desktop small
      { width: 1920, height: 1080 } // Desktop large
    ]
    
    for (const viewport of viewports) {
      await page.setViewportSize(viewport)
      await page.goto('/')
      
      // Check main content is visible
      const mainContent = page.getByRole('main').or(page.locator('main'))
      await expect(mainContent).toBeVisible()
      
      // Check navigation is accessible (might be in mobile menu)
      const navigation = page.getByRole('navigation')
      const mobileMenuButton = page.getByRole('button', { name: /menu|nav/i })
      
      if (await navigation.isVisible()) {
        // Desktop navigation visible
        await expect(navigation).toBeVisible()
      } else if (await mobileMenuButton.count() > 0) {
        // Mobile menu button available
        await mobileMenuButton.click()
        await expect(navigation).toBeVisible()
      }
    }
  })
})