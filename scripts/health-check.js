#!/usr/bin/env node

/**
 * Robust Health Check Utility for CI/CD
 * 
 * Performs comprehensive health checks on web servers with retries,
 * proper error handling, and detailed diagnostics.
 */

const http = require('http');
const https = require('https');
const url = require('url');

class HealthChecker {
  constructor(options = {}) {
    this.timeout = options.timeout || 30000;
    this.retries = options.retries || 10;
    this.retryDelay = options.retryDelay || 3000;
    this.expectedStatus = options.expectedStatus || 200;
    this.verbose = options.verbose || false;
  }

  /**
   * Log message if verbose mode is enabled
   */
  log(message) {
    if (this.verbose) {
      console.log(`[HealthCheck] ${message}`);
    }
  }

  /**
   * Make HTTP request with proper error handling
   */
  makeRequest(targetUrl) {
    return new Promise((resolve, reject) => {
      const parsedUrl = url.parse(targetUrl);
      const client = parsedUrl.protocol === 'https:' ? https : http;
      
      const options = {
        hostname: parsedUrl.hostname,
        port: parsedUrl.port,
        path: parsedUrl.path || '/',
        method: 'GET',
        timeout: this.timeout,
        headers: {
          'User-Agent': 'HealthCheck/1.0',
          'Accept': 'text/html,application/json,*/*',
          'Cache-Control': 'no-cache'
        }
      };

      const req = client.request(options, (res) => {
        let data = '';
        
        res.on('data', (chunk) => {
          data += chunk;
        });
        
        res.on('end', () => {
          resolve({
            statusCode: res.statusCode,
            headers: res.headers,
            data: data.substring(0, 500), // First 500 chars
            length: data.length
          });
        });
      });

      req.on('timeout', () => {
        req.destroy();
        reject(new Error(`Request timeout after ${this.timeout}ms`));
      });

      req.on('error', (error) => {
        reject(error);
      });

      req.end();
    });
  }

  /**
   * Perform health check with retries
   */
  async checkHealth(targetUrl) {
    const startTime = Date.now();
    let lastError = null;
    
    console.log(`🔍 Starting health check for: ${targetUrl}`);
    console.log(`⚙️  Configuration: ${this.retries} retries, ${this.retryDelay}ms delay, ${this.timeout}ms timeout`);
    
    for (let attempt = 1; attempt <= this.retries; attempt++) {
      try {
        this.log(`Attempt ${attempt}/${this.retries}...`);
        
        const response = await this.makeRequest(targetUrl);
        
        if (response.statusCode === this.expectedStatus) {
          const duration = Date.now() - startTime;
          console.log(`✅ Health check passed on attempt ${attempt}!`);
          console.log(`📊 Status: ${response.statusCode}`);
          console.log(`⏱️  Total time: ${duration}ms`);
          console.log(`📏 Response size: ${response.length} bytes`);
          console.log(`🔗 Content-Type: ${response.headers['content-type'] || 'not set'}`);
          
          if (this.verbose && response.data) {
            console.log(`📄 Response preview: ${response.data.substring(0, 200)}...`);
          }
          
          return {
            success: true,
            attempts: attempt,
            duration,
            response
          };
        } else {
          throw new Error(`Unexpected status code: ${response.statusCode} (expected ${this.expectedStatus})`);
        }
        
      } catch (error) {
        lastError = error;
        
        console.log(`❌ Attempt ${attempt}/${this.retries} failed: ${error.message}`);
        
        if (attempt < this.retries) {
          this.log(`⏳ Waiting ${this.retryDelay}ms before retry...`);
          await new Promise(resolve => setTimeout(resolve, this.retryDelay));
        }
      }
    }
    
    // All retries failed
    const duration = Date.now() - startTime;
    console.log(`💥 Health check failed after ${this.retries} attempts (${duration}ms total)`);
    console.log(`❌ Final error: ${lastError?.message || 'Unknown error'}`);
    
    return {
      success: false,
      attempts: this.retries,
      duration,
      error: lastError
    };
  }

  /**
   * Check if a port is listening (basic connectivity test)
   */
  async checkPort(host, port) {
    return new Promise((resolve) => {
      const socket = require('net').createConnection({ port, host }, () => {
        socket.end();
        resolve(true);
      });
      
      socket.on('error', () => {
        resolve(false);
      });
      
      socket.setTimeout(5000, () => {
        socket.destroy();
        resolve(false);
      });
    });
  }
}

// CLI Interface
async function main() {
  const args = process.argv.slice(2);
  
  if (args.length === 0) {
    console.error('❌ Usage: node health-check.js <url> [options]');
    console.error('   Options:');
    console.error('     --timeout <ms>     Request timeout (default: 30000)');
    console.error('     --retries <n>      Number of retries (default: 10)');
    console.error('     --delay <ms>       Delay between retries (default: 3000)');
    console.error('     --status <code>    Expected status code (default: 200)');
    console.error('     --verbose          Verbose output');
    console.error('');
    console.error('   Examples:');
    console.error('     node health-check.js http://localhost:4321');
    console.error('     node health-check.js http://localhost:4194 --retries 20 --verbose');
    process.exit(1);
  }
  
  const targetUrl = args[0];
  const options = {
    verbose: args.includes('--verbose')
  };
  
  // Parse options
  const timeoutIndex = args.indexOf('--timeout');
  if (timeoutIndex !== -1 && args[timeoutIndex + 1]) {
    options.timeout = parseInt(args[timeoutIndex + 1]);
  }
  
  const retriesIndex = args.indexOf('--retries');
  if (retriesIndex !== -1 && args[retriesIndex + 1]) {
    options.retries = parseInt(args[retriesIndex + 1]);
  }
  
  const delayIndex = args.indexOf('--delay');
  if (delayIndex !== -1 && args[delayIndex + 1]) {
    options.retryDelay = parseInt(args[delayIndex + 1]);
  }
  
  const statusIndex = args.indexOf('--status');
  if (statusIndex !== -1 && args[statusIndex + 1]) {
    options.expectedStatus = parseInt(args[statusIndex + 1]);
  }
  
  try {
    // Validate URL
    const parsedUrl = url.parse(targetUrl);
    if (!parsedUrl.hostname || !parsedUrl.port) {
      throw new Error('Invalid URL format. Must include hostname and port.');
    }
    
    const checker = new HealthChecker(options);
    
    // First check basic port connectivity
    console.log(`🔌 Testing port connectivity to ${parsedUrl.hostname}:${parsedUrl.port}...`);
    const portOpen = await checker.checkPort(parsedUrl.hostname, parsedUrl.port);
    
    if (!portOpen) {
      console.log(`❌ Port ${parsedUrl.port} is not accessible on ${parsedUrl.hostname}`);
      process.exit(1);
    }
    
    console.log(`✅ Port ${parsedUrl.port} is accessible`);
    
    // Perform full health check
    const result = await checker.checkHealth(targetUrl);
    
    if (result.success) {
      console.log(`🎉 Health check completed successfully!`);
      process.exit(0);
    } else {
      console.log(`💥 Health check failed!`);
      process.exit(1);
    }
    
  } catch (error) {
    console.error(`❌ Error: ${error.message}`);
    process.exit(1);
  }
}

// Export for use as module
module.exports = HealthChecker;

// Run CLI if called directly
if (require.main === module) {
  main();
}