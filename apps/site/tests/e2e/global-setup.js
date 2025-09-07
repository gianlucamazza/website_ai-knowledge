/**
 * Playwright Global Setup
 * 
 * Runs once before all test suites. Sets up test environment, database, and services.
 */

import { chromium } from '@playwright/test'

async function globalSetup(config) {
  console.log('🚀 Setting up E2E test environment...')
  
  try {
    // Launch browser for setup tasks
    const browser = await chromium.launch()
    const page = await browser.newPage()
    
    // Wait for the dev server to be ready
    const baseURL = config.projects[0].use.baseURL || 'http://localhost:4321'
    console.log(`📡 Waiting for dev server at ${baseURL}...`)
    
    // Poll until server is ready (max 2 minutes)
    const maxRetries = 24 // 2 minutes with 5-second intervals
    let retries = 0
    let serverReady = false
    
    while (retries < maxRetries && !serverReady) {
      try {
        const response = await page.goto(baseURL, { 
          waitUntil: 'networkidle',
          timeout: 5000 
        })
        
        if (response && response.ok()) {
          serverReady = true
          console.log('✅ Dev server is ready!')
        } else {
          throw new Error(`Server responded with ${response?.status()}`)
        }
      } catch (error) {
        retries++
        console.log(`⏳ Server not ready, retry ${retries}/${maxRetries}...`)
        
        if (retries < maxRetries) {
          await new Promise(resolve => setTimeout(resolve, 5000))
        }
      }
    }
    
    if (!serverReady) {
      throw new Error('Dev server failed to start within timeout period')
    }
    
    // Perform any additional setup tasks
    await setupTestData(page, baseURL)
    
    await browser.close()
    
    console.log('✅ Global setup completed successfully')
    
  } catch (error) {
    console.error('❌ Global setup failed:', error)
    throw error
  }
}

async function setupTestData(page, baseURL) {
  console.log('📊 Setting up test data...')
  
  try {
    // Check if essential pages exist
    const pages = [
      '/',
      '/articles',
      '/glossary',
      '/about'
    ]
    
    for (const path of pages) {
      try {
        const response = await page.goto(`${baseURL}${path}`, { 
          waitUntil: 'networkidle',
          timeout: 10000 
        })
        
        if (response && !response.ok()) {
          console.warn(`⚠️  Page ${path} returned ${response.status()}`)
        } else {
          console.log(`✅ Verified page: ${path}`)
        }
      } catch (error) {
        console.warn(`⚠️  Could not verify page ${path}:`, error.message)
      }
    }
    
    // Check for content availability
    await page.goto(`${baseURL}/articles`, { waitUntil: 'networkidle' })
    
    const articleCards = page.locator('[data-testid="article-card"], article, [data-article]')
    const articleCount = await articleCards.count()
    console.log(`📄 Found ${articleCount} articles on articles page`)
    
    await page.goto(`${baseURL}/glossary`, { waitUntil: 'networkidle' })
    
    const glossaryTerms = page.locator('[data-testid="glossary-term"], [data-term], .glossary-entry')
    const termCount = await glossaryTerms.count()
    console.log(`📖 Found ${termCount} terms on glossary page`)
    
    // Create test-specific data if needed
    if (process.env.CREATE_TEST_DATA) {
      console.log('🔧 Creating test-specific data...')
      // Add any test data creation logic here
    }
    
    console.log('✅ Test data setup completed')
    
  } catch (error) {
    console.warn('⚠️  Test data setup had issues:', error.message)
    // Don't fail setup for data issues, just warn
  }
}

export default globalSetup