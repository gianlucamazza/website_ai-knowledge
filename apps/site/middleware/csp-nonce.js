/**
 * CSP Nonce Generation Middleware
 * 
 * Generates a unique nonce for each request to allow specific inline scripts
 * while maintaining strong Content Security Policy.
 */

import crypto from 'crypto';

/**
 * Generate a cryptographically secure nonce
 * @returns {string} Base64 encoded nonce
 */
export function generateNonce() {
  return crypto.randomBytes(16).toString('base64');
}

/**
 * Middleware to add CSP nonce to requests
 */
export function cspNonceMiddleware() {
  return (request, response, next) => {
    // Generate nonce for this request
    const nonce = generateNonce();
    
    // Store nonce in request for use in templates
    request.cspNonce = nonce;
    
    // Also store in response locals for template engines
    if (response.locals) {
      response.locals.cspNonce = nonce;
    }
    
    next();
  };
}

/**
 * Add nonce to CSP header
 * @param {string} cspHeader - Original CSP header
 * @param {string} nonce - Nonce to add
 * @returns {string} Updated CSP header with nonce
 */
export function addNonceToCSP(cspHeader, nonce) {
  // Add nonce to script-src directive
  const scriptSrcRegex = /(script-src[^;]*)/;
  const updatedHeader = cspHeader.replace(scriptSrcRegex, (match) => {
    // Don't add nonce in development if unsafe-inline is present
    if (process.env.NODE_ENV === 'development' && match.includes("'unsafe-inline'")) {
      return match;
    }
    
    // Add nonce after 'self'
    return match.replace("'self'", `'self' 'nonce-${nonce}'`);
  });
  
  // Also add to style-src if needed
  const styleSrcRegex = /(style-src[^;]*)/;
  const finalHeader = updatedHeader.replace(styleSrcRegex, (match) => {
    // Only add nonce if we're removing unsafe-inline
    if (!match.includes("'unsafe-inline'")) {
      return match.replace("'self'", `'self' 'nonce-${nonce}'`);
    }
    return match;
  });
  
  return finalHeader;
}

/**
 * Astro integration for CSP nonce
 */
export function astroCSPNonce() {
  return {
    name: 'csp-nonce',
    hooks: {
      'astro:server:setup': ({ server }) => {
        server.middlewares.use((req, res, next) => {
          const nonce = generateNonce();
          req.cspNonce = nonce;
          
          // Store for use in Astro components
          if (!req.locals) req.locals = {};
          req.locals.cspNonce = nonce;
          
          next();
        });
      },
      'astro:build:ssr': ({ manifest }) => {
        // For SSR builds, we need to handle nonce generation differently
        // This would be implemented based on your deployment strategy
        console.log('CSP nonce configuration for SSR build');
      }
    }
  };
}

/**
 * Helper to add nonce attribute to script tags in HTML
 * @param {string} html - HTML content
 * @param {string} nonce - Nonce value
 * @returns {string} HTML with nonce attributes added
 */
export function addNonceToScripts(html, nonce) {
  // Add nonce to all script tags that don't have src attribute
  const scriptRegex = /<script(?![^>]*\ssrc=)([^>]*)>/gi;
  return html.replace(scriptRegex, (match, attributes) => {
    // Don't add if nonce already exists
    if (attributes.includes('nonce=')) {
      return match;
    }
    return `<script nonce="${nonce}"${attributes}>`;
  });
}

/**
 * Helper to add nonce attribute to style tags in HTML
 * @param {string} html - HTML content
 * @param {string} nonce - Nonce value
 * @returns {string} HTML with nonce attributes added
 */
export function addNonceToStyles(html, nonce) {
  // Add nonce to all style tags
  const styleRegex = /<style([^>]*)>/gi;
  return html.replace(styleRegex, (match, attributes) => {
    // Don't add if nonce already exists
    if (attributes.includes('nonce=')) {
      return match;
    }
    return `<style nonce="${nonce}"${attributes}>`;
  });
}

/**
 * Process HTML to add nonces
 * @param {string} html - HTML content
 * @param {string} nonce - Nonce value
 * @returns {string} Processed HTML
 */
export function processHTMLWithNonce(html, nonce) {
  let processed = html;
  processed = addNonceToScripts(processed, nonce);
  processed = addNonceToStyles(processed, nonce);
  return processed;
}

export default {
  generateNonce,
  cspNonceMiddleware,
  addNonceToCSP,
  astroCSPNonce,
  addNonceToScripts,
  addNonceToStyles,
  processHTMLWithNonce
};