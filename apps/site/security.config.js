/**
 * Security Configuration for Astro Site
 * 
 * Configures Content Security Policy, security headers, and other
 * security measures for the AI knowledge website.
 */

const isDevelopment = process.env.NODE_ENV === 'development';

// Content Security Policy configuration
const cspConfig = {
  // Basic CSP directives
  directives: {
    'default-src': ["'self'"],
    
    // Scripts - Astro needs inline scripts for hydration
    'script-src': [
      "'self'",
      "'unsafe-inline'", // Required for Astro hydration
      ...(isDevelopment ? ["'unsafe-eval'", "localhost:*"] : []),
      // Add any CDNs or external script sources here
      'https://cdn.jsdelivr.net',
      'https://unpkg.com'
    ],
    
    // Styles - Allow inline styles and Google Fonts
    'style-src': [
      "'self'",
      "'unsafe-inline'", // Required for component styles
      'https://fonts.googleapis.com',
      'https://cdn.jsdelivr.net'
    ],
    
    // Images - Allow data URIs and HTTPS images
    'img-src': [
      "'self'",
      'data:',
      'https:',
      'blob:'
    ],
    
    // Fonts
    'font-src': [
      "'self'",
      'https://fonts.gstatic.com',
      'data:'
    ],
    
    // Connections - API calls and WebSocket for dev
    'connect-src': [
      "'self'",
      ...(isDevelopment ? ['ws://localhost:*', 'http://localhost:*'] : []),
      // Add API endpoints here
      'https://api.openai.com',
      'https://api.anthropic.com'
    ],
    
    // Media
    'media-src': ["'self'"],
    
    // Objects - Disable for security
    'object-src': ["'none'"],
    
    // Frames - Disable for security  
    'frame-src': ["'none'"],
    'frame-ancestors': ["'none'"],
    
    // Base URI restriction
    'base-uri': ["'self'"],
    
    // Form actions
    'form-action': ["'self'"],
    
    // Workers
    'worker-src': ["'self'", 'blob:'],
    
    // Manifest
    'manifest-src': ["'self'"]
  },
  
  // Report violations in development
  reportOnly: isDevelopment,
  reportUri: isDevelopment ? '/api/security/csp-report' : null,
  
  // Upgrade insecure requests in production
  upgradeInsecureRequests: !isDevelopment
};

// Security headers configuration
const securityHeaders = {
  // HTTP Strict Transport Security
  'Strict-Transport-Security': 
    isDevelopment ? null : 'max-age=31536000; includeSubDomains; preload',
  
  // Prevent clickjacking
  'X-Frame-Options': 'DENY',
  
  // Prevent MIME sniffing
  'X-Content-Type-Options': 'nosniff',
  
  // XSS protection for older browsers
  'X-XSS-Protection': '1; mode=block',
  
  // Referrer policy
  'Referrer-Policy': 'strict-origin-when-cross-origin',
  
  // Permissions policy
  'Permissions-Policy': [
    'geolocation=()',
    'microphone=()',
    'camera=()',
    'payment=()',
    'usb=()',
    'magnetometer=()',
    'gyroscope=()',
    'accelerometer=()',
    'fullscreen=(self)'
  ].join(', '),
  
  // Cross-Origin policies
  'Cross-Origin-Embedder-Policy': 'require-corp',
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Resource-Policy': 'same-site',
  
  // Hide server information
  'Server': ''
};

// Build CSP header value
function buildCSPHeader(config) {
  const directives = Object.entries(config.directives)
    .map(([directive, sources]) => `${directive} ${sources.join(' ')}`)
    .join('; ');
  
  let csp = directives;
  
  if (config.upgradeInsecureRequests) {
    csp += '; upgrade-insecure-requests';
  }
  
  if (config.reportUri) {
    csp += `; report-uri ${config.reportUri}`;
  }
  
  return csp;
}

// Get all security headers
function getSecurityHeaders() {
  const headers = {};
  
  // Add CSP header
  const cspHeaderName = cspConfig.reportOnly 
    ? 'Content-Security-Policy-Report-Only'
    : 'Content-Security-Policy';
  
  headers[cspHeaderName] = buildCSPHeader(cspConfig);
  
  // Add other security headers
  Object.entries(securityHeaders).forEach(([name, value]) => {
    if (value !== null) {
      headers[name] = value;
    }
  });
  
  return headers;
}

// Validate CSP configuration
function validateCSP() {
  const issues = [];
  const { directives } = cspConfig;
  
  // Check for unsafe directives
  const unsafeSources = ["'unsafe-inline'", "'unsafe-eval'"];
  Object.entries(directives).forEach(([directive, sources]) => {
    sources.forEach(source => {
      if (unsafeSources.includes(source) && directive === 'script-src') {
        issues.push(`Warning: ${source} in ${directive} reduces security`);
      }
    });
  });
  
  // Check for missing important directives
  const important = ['default-src', 'script-src', 'object-src', 'base-uri'];
  important.forEach(directive => {
    if (!directives[directive]) {
      issues.push(`Warning: Missing ${directive} directive`);
    }
  });
  
  // Check object-src
  if (directives['object-src'] && !directives['object-src'].includes("'none'")) {
    issues.push("Warning: object-src should be 'none' for security");
  }
  
  return issues;
}

// Security middleware for API routes
function createSecurityMiddleware() {
  return (request, response, next) => {
    // Add security headers
    const headers = getSecurityHeaders();
    Object.entries(headers).forEach(([name, value]) => {
      response.setHeader(name, value);
    });
    
    // Add cache control for sensitive routes
    if (request.url.includes('/admin') || request.url.includes('/api')) {
      response.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
      response.setHeader('Pragma', 'no-cache');
      response.setHeader('Expires', '0');
    }
    
    next();
  };
}

// CSP report handler
function handleCSPReport(reportData) {
  try {
    if (reportData && reportData['csp-report']) {
      const violation = reportData['csp-report'];
      
      console.warn('CSP Violation:', {
        directive: violation['violated-directive'],
        blockedURI: violation['blocked-uri'],
        documentURI: violation['document-uri'],
        sourceFile: violation['source-file'],
        lineNumber: violation['line-number'],
        columnNumber: violation['column-number']
      });
      
      // In production, you might want to log this to a security monitoring service
      if (!isDevelopment) {
        // Log to security monitoring service
        // securityLogger.logCSPViolation(violation);
      }
    }
  } catch (error) {
    console.error('Error processing CSP report:', error);
  }
}

// Content sanitization rules for static content
const contentSanitization = {
  // Allowed HTML tags in markdown content
  allowedTags: [
    'a', 'abbr', 'acronym', 'b', 'blockquote', 'br', 'code', 'dd', 'dl', 'dt',
    'em', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'hr', 'i', 'img', 'li', 'ol',
    'p', 'pre', 'strong', 'sub', 'sup', 'table', 'tbody', 'td', 'th', 'thead',
    'tr', 'ul', 'span', 'div'
  ],
  
  // Allowed attributes per tag
  allowedAttributes: {
    'a': ['href', 'title', 'rel', 'target'],
    'img': ['src', 'alt', 'width', 'height', 'title'],
    'abbr': ['title'],
    'acronym': ['title'],
    'span': ['class'],
    'div': ['class'],
    'table': ['class'],
    'td': ['colspan', 'rowspan'],
    'th': ['colspan', 'rowspan']
  },
  
  // Allowed protocols
  allowedProtocols: ['http', 'https', 'mailto'],
  
  // Transform functions
  transformTags: {
    'a': function(tagName, attribs) {
      // Add security attributes to external links
      if (attribs.href && (attribs.href.startsWith('http://') || attribs.href.startsWith('https://'))) {
        // Check if it's an external link
        const url = new URL(attribs.href);
        if (url.hostname !== 'localhost' && !url.hostname.endsWith('.ai-knowledge.com')) {
          attribs.rel = 'nofollow noopener noreferrer';
          attribs.target = '_blank';
        }
      }
      return { tagName, attribs };
    }
  }
};

// Export configuration
export {
  cspConfig,
  securityHeaders,
  getSecurityHeaders,
  validateCSP,
  createSecurityMiddleware,
  handleCSPReport,
  contentSanitization,
  isDevelopment
};

// Validate configuration on load
if (isDevelopment) {
  const issues = validateCSP();
  if (issues.length > 0) {
    console.warn('CSP Configuration Issues:');
    issues.forEach(issue => console.warn(`  - ${issue}`));
  }
}