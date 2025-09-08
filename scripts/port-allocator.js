#!/usr/bin/env node

/**
 * Dynamic Port Allocator for CI/CD
 * 
 * This utility finds available ports dynamically to avoid conflicts
 * between parallel CI jobs running Node 18 and Node 20.
 */

const net = require('net');
const fs = require('fs');
const path = require('path');

class PortAllocator {
  constructor() {
    this.basePort = 4000;
    this.maxPort = 65535;
    this.reservedPorts = new Set([
      3000, 3001, 4321, 4322, 4323, 4324, 4325, // Common dev ports
      5432, 5433, // PostgreSQL
      8080, 8081, 8082, // Common HTTP
      9000, 9001, 9002  // Common alt HTTP
    ]);
  }

  /**
   * Check if a port is available
   */
  async isPortAvailable(port) {
    return new Promise((resolve) => {
      const server = net.createServer();
      
      server.listen(port, '127.0.0.1', () => {
        server.once('close', () => {
          resolve(true);
        });
        server.close();
      });
      
      server.on('error', (err) => {
        if (err.code === 'EADDRINUSE') {
          console.log(`  Port ${port} is already in use`);
        } else {
          console.log(`  Port ${port} check failed: ${err.message}`);
        }
        resolve(false);
      });
    });
  }

  /**
   * Check if a port is actively serving HTTP content
   */
  async isPortServing(port, timeout = 5000) {
    return new Promise((resolve) => {
      const http = require('http');
      
      const req = http.request({
        hostname: '127.0.0.1',
        port: port,
        path: '/',
        method: 'GET',
        timeout: timeout
      }, (res) => {
        resolve(res.statusCode >= 200 && res.statusCode < 500);
      });
      
      req.on('error', () => {
        resolve(false);
      });
      
      req.on('timeout', () => {
        req.destroy();
        resolve(false);
      });
      
      req.end();
    });
  }

  /**
   * Find next available port starting from a base port
   */
  async findAvailablePort(basePort = this.basePort) {
    let port = basePort;
    
    while (port <= this.maxPort) {
      if (!this.reservedPorts.has(port)) {
        const available = await this.isPortAvailable(port);
        if (available) {
          return port;
        }
      }
      port++;
    }
    
    throw new Error(`No available ports found starting from ${basePort}`);
  }

  /**
   * Allocate ports for a CI job
   */
  async allocatePorts(jobId = 'default') {
    // Generate deterministic but unique base port for job
    const hash = this.hashString(jobId);
    const basePort = 4000 + (hash % 1000);
    
    const devPort = await this.findAvailablePort(basePort);
    const previewPort = await this.findAvailablePort(devPort + 1);
    const testPort = await this.findAvailablePort(previewPort + 1);
    
    return {
      dev: devPort,
      preview: previewPort,
      test: testPort,
      jobId
    };
  }

  /**
   * Simple hash function for job ID
   */
  hashString(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash);
  }

  /**
   * Save port allocation to file
   */
  async savePortAllocation(allocation, outputFile = null) {
    const filePath = outputFile || path.join(require('os').tmpdir(), '.port-allocation.json');
    
    try {
      fs.writeFileSync(filePath, JSON.stringify(allocation, null, 2));
      console.log(`✅ Port allocation saved to: ${filePath}`);
      return filePath;
    } catch (error) {
      console.error(`❌ Failed to save port allocation: ${error.message}`);
      throw error;
    }
  }

  /**
   * Load port allocation from file
   */
  loadPortAllocation(inputFile = null) {
    const filePath = inputFile || path.join(require('os').tmpdir(), '.port-allocation.json');
    
    try {
      if (fs.existsSync(filePath)) {
        const content = fs.readFileSync(filePath, 'utf8');
        return JSON.parse(content);
      }
      return null;
    } catch (error) {
      console.error(`❌ Failed to load port allocation: ${error.message}`);
      return null;
    }
  }
}

// CLI Interface
async function main() {
  const allocator = new PortAllocator();
  
  const args = process.argv.slice(2);
  const command = args[0] || 'allocate';
  
  try {
    switch (command) {
      case 'allocate': {
        const jobId = args[1] || process.env.GITHUB_RUN_ID || 'local';
        const allocation = await allocator.allocatePorts(jobId);
        
        console.log('🎯 Port allocation for job:', jobId);
        console.log('📊 Allocated ports:');
        console.log(`  • Dev server: ${allocation.dev}`);
        console.log(`  • Preview server: ${allocation.preview}`);
        console.log(`  • Test server: ${allocation.test}`);
        
        // Save to file for use by other scripts
        await allocator.savePortAllocation(allocation);
        
        // Export environment variables for GitHub Actions
        if (process.env.GITHUB_ENV) {
          const envVars = [
            `DEV_SERVER_PORT=${allocation.dev}`,
            `PREVIEW_SERVER_PORT=${allocation.preview}`,
            `TEST_SERVER_PORT=${allocation.test}`
          ].join('\n');
          
          fs.appendFileSync(process.env.GITHUB_ENV, envVars + '\n');
          console.log('✅ Environment variables exported to GITHUB_ENV');
        }
        
        return allocation;
      }
      
      case 'check': {
        const port = parseInt(args[1]);
        if (!port) {
          console.error('❌ Port number required for check command');
          process.exit(1);
        }
        
        const available = await allocator.isPortAvailable(port);
        console.log(`Port ${port}: ${available ? '✅ Available' : '❌ In use'}`);
        return { port, available };
      }
      
      case 'verify': {
        const port = parseInt(args[1]);
        if (!port) {
          console.error('❌ Port number required for verify command');
          process.exit(1);
        }
        
        console.log(`🔍 Verifying server is running on port ${port}...`);
        const serving = await allocator.isPortServing(port);
        console.log(`Server on port ${port}: ${serving ? '✅ Active and serving' : '❌ Not responding'}`);
        
        if (!serving) {
          process.exit(1);
        }
        
        return { port, serving };
      }
      
      case 'load': {
        const allocation = allocator.loadPortAllocation();
        if (allocation) {
          console.log('📋 Loaded port allocation:');
          console.log(JSON.stringify(allocation, null, 2));
        } else {
          console.log('❌ No port allocation file found');
        }
        return allocation;
      }
      
      default:
        console.error(`❌ Unknown command: ${command}`);
        console.log('📚 Available commands:');
        console.log('  • allocate [jobId] - Allocate ports for CI job');
        console.log('  • check <port>     - Check if port is available');
        console.log('  • verify <port>    - Verify server is responding on port');
        console.log('  • load             - Load existing allocation');
        process.exit(1);
    }
  } catch (error) {
    console.error(`❌ Error: ${error.message}`);
    process.exit(1);
  }
}

// Export for use as module
module.exports = PortAllocator;

// Run CLI if called directly
if (require.main === module) {
  main();
}