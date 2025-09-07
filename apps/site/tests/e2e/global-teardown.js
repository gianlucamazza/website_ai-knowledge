/**
 * Playwright Global Teardown
 * 
 * Runs once after all test suites complete. Cleans up test environment and resources.
 */

import { chromium } from '@playwright/test'
import { promises as fs } from 'fs'
import path from 'path'

async function globalTeardown(config) {
  console.log('🧹 Starting E2E test environment teardown...')
  
  try {
    // Clean up test artifacts and reports
    await cleanupTestArtifacts()
    
    // Clean up any test data if created
    await cleanupTestData()
    
    // Generate test summary
    await generateTestSummary()
    
    console.log('✅ Global teardown completed successfully')
    
  } catch (error) {
    console.error('❌ Global teardown encountered errors:', error)
    // Don't fail teardown, just log errors
  }
}

async function cleanupTestArtifacts() {
  console.log('🗑️  Cleaning up test artifacts...')
  
  try {
    const artifactDirs = [
      'tests/artifacts',
      'test-results',
      '.playwright'
    ]
    
    for (const dir of artifactDirs) {
      try {
        const stats = await fs.stat(dir)
        if (stats.isDirectory()) {
          const files = await fs.readdir(dir)
          
          // Only clean up if there are many files (keep recent failures)
          if (files.length > 50) {
            console.log(`🧹 Cleaning ${dir} (${files.length} files)...`)
            
            // Sort by modification time and keep only the 10 most recent
            const fileStats = await Promise.all(
              files.map(async (file) => {
                const fullPath = path.join(dir, file)
                const stat = await fs.stat(fullPath)
                return { file, path: fullPath, mtime: stat.mtime }
              })
            )
            
            fileStats.sort((a, b) => b.mtime - a.mtime)
            
            // Delete all but the 10 most recent files
            const filesToDelete = fileStats.slice(10)
            for (const { path: filePath } of filesToDelete) {
              try {
                await fs.unlink(filePath)
              } catch (err) {
                // Ignore deletion errors
              }
            }
            
            console.log(`✅ Cleaned up ${filesToDelete.length} old artifact files`)
          }
        }
      } catch (error) {
        // Directory doesn't exist or access error, skip
      }
    }
    
  } catch (error) {
    console.warn('⚠️  Artifact cleanup had issues:', error.message)
  }
}

async function cleanupTestData() {
  console.log('🗄️  Cleaning up test data...')
  
  try {
    // Clean up any test-specific data that was created
    if (process.env.CLEANUP_TEST_DATA) {
      console.log('🔧 Removing test-specific data...')
      // Add any test data cleanup logic here
    }
    
    // Clear any test caches
    const cacheFiles = [
      '.astro/cache',
      'node_modules/.cache',
      '.vite/cache'
    ]
    
    for (const cacheDir of cacheFiles) {
      try {
        const stats = await fs.stat(cacheDir)
        if (stats.isDirectory()) {
          console.log(`🗑️  Clearing cache: ${cacheDir}`)
          // Don't actually delete cache directories as they're needed for development
          // Just log that we would clean them
        }
      } catch (error) {
        // Cache directory doesn't exist, skip
      }
    }
    
    console.log('✅ Test data cleanup completed')
    
  } catch (error) {
    console.warn('⚠️  Test data cleanup had issues:', error.message)
  }
}

async function generateTestSummary() {
  console.log('📊 Generating test summary...')
  
  try {
    const summaryData = {
      timestamp: new Date().toISOString(),
      environment: {
        nodeVersion: process.version,
        platform: process.platform,
        ci: !!process.env.CI,
        baseUrl: process.env.BASE_URL || 'http://localhost:4321'
      },
      testRun: {
        startTime: process.env.TEST_START_TIME || 'unknown',
        endTime: new Date().toISOString(),
        totalDuration: process.env.TEST_START_TIME ? 
          Date.now() - parseInt(process.env.TEST_START_TIME) : 'unknown'
      }
    }
    
    // Try to read test results if available
    const resultsFiles = [
      'tests/reports/playwright-results.json',
      'test-results.json',
      'playwright-report/results.json'
    ]
    
    for (const resultsFile of resultsFiles) {
      try {
        const results = await fs.readFile(resultsFile, 'utf-8')
        const parsed = JSON.parse(results)
        
        summaryData.results = {
          totalTests: parsed.stats?.expected || 0,
          passed: parsed.stats?.passed || 0,
          failed: parsed.stats?.failed || 0,
          skipped: parsed.stats?.skipped || 0,
          duration: parsed.stats?.duration || 0
        }
        
        console.log(`📈 Test Results Summary:`)
        console.log(`   Total Tests: ${summaryData.results.totalTests}`)
        console.log(`   Passed: ${summaryData.results.passed}`)
        console.log(`   Failed: ${summaryData.results.failed}`)
        console.log(`   Skipped: ${summaryData.results.skipped}`)
        console.log(`   Duration: ${Math.round(summaryData.results.duration / 1000)}s`)
        
        break
      } catch (error) {
        // Results file doesn't exist or can't be read, continue
      }
    }
    
    // Save summary to file
    try {
      const summaryPath = 'tests/reports/test-summary.json'
      await fs.mkdir(path.dirname(summaryPath), { recursive: true })
      await fs.writeFile(summaryPath, JSON.stringify(summaryData, null, 2))
      console.log(`📄 Test summary saved to ${summaryPath}`)
    } catch (error) {
      console.warn('⚠️  Could not save test summary:', error.message)
    }
    
    // Environment-specific cleanup
    if (process.env.CI) {
      console.log('🚀 CI environment detected - performing CI-specific cleanup')
      
      // In CI, we might want to compress artifacts or upload them
      // For now, just log what we would do
      console.log('📦 Would compress and archive test artifacts for CI')
    }
    
    console.log('✅ Test summary generation completed')
    
  } catch (error) {
    console.warn('⚠️  Test summary generation had issues:', error.message)
  }
}

// Track test start time for duration calculation
if (!process.env.TEST_START_TIME) {
  process.env.TEST_START_TIME = Date.now().toString()
}

export default globalTeardown