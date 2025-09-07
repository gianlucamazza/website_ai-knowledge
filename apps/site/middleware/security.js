/**
 * Security Middleware for Astro Site
 * 
 * Provides runtime security protections including input validation,
 * rate limiting, and security monitoring for the Astro application.
 */

import { getSecurityHeaders, handleCSPReport } from '../security.config.js';

// Simple in-memory rate limiting (use Redis in production)
const rateLimitStore = new Map();
const RATE_LIMIT_WINDOW = 60 * 1000; // 1 minute
const MAX_REQUESTS_PER_WINDOW = 100;
const MAX_REQUESTS_PER_IP = 1000;

// Security event logging
const securityEvents = [];
const MAX_SECURITY_EVENTS = 1000;

/**
 * Rate limiting middleware
 */
function rateLimitMiddleware(request) {
  const clientIP = getClientIP(request);
  const now = Date.now();
  const windowStart = now - RATE_LIMIT_WINDOW;
  
  // Clean old entries
  if (rateLimitStore.has(clientIP)) {
    const requests = rateLimitStore.get(clientIP);
    const validRequests = requests.filter(timestamp => timestamp > windowStart);
    rateLimitStore.set(clientIP, validRequests);
  }
  
  // Check current request count
  const currentRequests = rateLimitStore.get(clientIP) || [];
  
  if (currentRequests.length >= MAX_REQUESTS_PER_WINDOW) {
    logSecurityEvent({
      type: 'RATE_LIMIT_EXCEEDED',
      ip: clientIP,
      userAgent: request.headers.get('user-agent'),
      url: request.url,
      timestamp: now
    });
    
    return new Response('Rate limit exceeded', {
      status: 429,
      headers: {
        'Retry-After': '60',
        'X-RateLimit-Limit': MAX_REQUESTS_PER_WINDOW.toString(),
        'X-RateLimit-Remaining': '0',
        'X-RateLimit-Reset': Math.ceil((now + RATE_LIMIT_WINDOW) / 1000).toString()
      }
    });
  }
  
  // Add current request
  currentRequests.push(now);
  rateLimitStore.set(clientIP, currentRequests);
  
  return null; // Continue processing
}

/**
 * Input validation middleware
 */
function inputValidationMiddleware(request) {
  const url = new URL(request.url);
  const userAgent = request.headers.get('user-agent') || '';
  
  // Block suspicious user agents
  const suspiciousPatterns = [
    /sqlmap/i,
    /nikto/i,
    /nessus/i,
    /masscan/i,
    /nmap/i,
    /dirb/i,
    /dirbuster/i,
    /gobuster/i,
    /burpsuite/i,
    /owasp.zap/i
  ];
  
  for (const pattern of suspiciousPatterns) {
    if (pattern.test(userAgent)) {
      logSecurityEvent({
        type: 'SUSPICIOUS_USER_AGENT',
        ip: getClientIP(request),
        userAgent: userAgent,
        url: request.url,
        timestamp: Date.now()
      });
      
      return new Response('Forbidden', { status: 403 });
    }
  }
  
  // Validate URL parameters
  for (const [key, value] of url.searchParams.entries()) {
    if (!isValidInput(value)) {
      logSecurityEvent({
        type: 'INVALID_INPUT',
        ip: getClientIP(request),
        parameter: key,
        value: value.substring(0, 100), // Limit logged value length
        url: request.url,
        timestamp: Date.now()
      });
      
      return new Response('Invalid input detected', { status: 400 });
    }
  }
  
  // Check for path traversal attempts
  if (url.pathname.includes('..') || url.pathname.includes('%2e%2e')) {
    logSecurityEvent({
      type: 'PATH_TRAVERSAL_ATTEMPT',
      ip: getClientIP(request),
      path: url.pathname,
      timestamp: Date.now()
    });
    
    return new Response('Forbidden', { status: 403 });
  }
  
  return null; // Continue processing
}

/**
 * Security headers middleware
 */
function securityHeadersMiddleware(response) {
  const headers = getSecurityHeaders();
  
  // Apply security headers to response
  Object.entries(headers).forEach(([name, value]) => {
    response.headers.set(name, value);
  });
  
  return response;
}

/**
 * Content Security Policy violation handler
 */
async function handleCSPViolation(request) {
  try {
    const reportData = await request.json();
    
    // Log the violation
    logSecurityEvent({
      type: 'CSP_VIOLATION',
      ip: getClientIP(request),
      report: reportData,
      timestamp: Date.now()
    });
    
    // Process the report
    handleCSPReport(reportData);
    
    return new Response('', { status: 204 });
  } catch (error) {
    console.error('Error handling CSP report:', error);
    return new Response('Error processing report', { status: 400 });
  }
}

/**
 * Main security middleware function
 */
export function securityMiddleware(request, context) {
  const url = new URL(request.url);
  
  // Handle CSP violation reports
  if (url.pathname === '/api/security/csp-report' && request.method === 'POST') {
    return handleCSPViolation(request);
  }
  
  // Apply rate limiting
  const rateLimitResponse = rateLimitMiddleware(request);
  if (rateLimitResponse) {
    return rateLimitResponse;
  }
  
  // Apply input validation
  const validationResponse = inputValidationMiddleware(request);
  if (validationResponse) {
    return validationResponse;
  }
  
  // Continue to next middleware/handler
  return null;
}

/**
 * Response middleware to add security headers
 */
export function responseSecurityMiddleware(response) {
  return securityHeadersMiddleware(response);
}

/**
 * Validate input string for malicious content
 */
function isValidInput(input) {
  if (typeof input !== 'string') {
    return false;
  }
  
  // Check length
  if (input.length > 1000) {
    return false;
  }
  
  // Check for dangerous patterns
  const dangerousPatterns = [
    /<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi,
    /javascript:/i,
    /vbscript:/i,
    /on\w+\s*=/i,
    /eval\s*\(/i,
    /exec\s*\(/i,
    /union\s+select/i,
    /drop\s+table/i,
    /information_schema/i,
    /\.\.\//g,
    /<iframe/i,
    /<object/i,
    /<embed/i
  ];
  
  for (const pattern of dangerousPatterns) {
    if (pattern.test(input)) {
      return false;
    }
  }
  
  return true;
}

/**
 * Get client IP address from request
 */
function getClientIP(request) {
  // Check various headers for the real IP
  const headers = [
    'cf-connecting-ip', // Cloudflare
    'x-forwarded-for',  // Standard proxy header
    'x-real-ip',        // Nginx
    'x-client-ip'       // Generic
  ];
  
  for (const header of headers) {
    const ip = request.headers.get(header);
    if (ip) {
      // Take the first IP if there are multiple
      return ip.split(',')[0].trim();
    }
  }
  
  // Fallback to connection IP (may not be available in all environments)
  return request.headers.get('x-forwarded-for') || 'unknown';
}

/**
 * Log security events
 */
function logSecurityEvent(event) {
  // Add to in-memory store
  securityEvents.push(event);
  
  // Keep only recent events
  if (securityEvents.length > MAX_SECURITY_EVENTS) {
    securityEvents.shift();
  }
  
  // Log to console in development
  if (process.env.NODE_ENV === 'development') {
    console.warn(`Security Event [${event.type}]:`, event);
  }
  
  // In production, you might want to:
  // - Send to a SIEM system
  // - Store in a database
  // - Send alerts for critical events
  // - Rate limit logging to prevent log spam
}

/**
 * Get security metrics and recent events
 */
export function getSecurityMetrics() {
  const now = Date.now();
  const oneHourAgo = now - (60 * 60 * 1000);
  const oneDayAgo = now - (24 * 60 * 60 * 1000);
  
  const recentEvents = securityEvents.filter(event => event.timestamp > oneHourAgo);
  const dailyEvents = securityEvents.filter(event => event.timestamp > oneDayAgo);
  
  // Count events by type
  const eventCounts = {};
  dailyEvents.forEach(event => {
    eventCounts[event.type] = (eventCounts[event.type] || 0) + 1;
  });
  
  return {
    totalEvents: securityEvents.length,
    recentEvents: recentEvents.length,
    dailyEvents: dailyEvents.length,
    eventCounts,
    topIPs: getTopIPs(dailyEvents),
    rateLimitEntries: rateLimitStore.size
  };
}

/**
 * Get most active IPs from events
 */
function getTopIPs(events, limit = 10) {
  const ipCounts = {};
  
  events.forEach(event => {
    if (event.ip) {
      ipCounts[event.ip] = (ipCounts[event.ip] || 0) + 1;
    }
  });
  
  return Object.entries(ipCounts)
    .sort(([,a], [,b]) => b - a)
    .slice(0, limit)
    .map(([ip, count]) => ({ ip, count }));
}

/**
 * Clear old rate limit entries (call periodically)
 */
export function cleanupRateLimit() {
  const now = Date.now();
  const windowStart = now - RATE_LIMIT_WINDOW;
  
  for (const [ip, requests] of rateLimitStore.entries()) {
    const validRequests = requests.filter(timestamp => timestamp > windowStart);
    
    if (validRequests.length === 0) {
      rateLimitStore.delete(ip);
    } else {
      rateLimitStore.set(ip, validRequests);
    }
  }
}

/**
 * Security health check endpoint
 */
export function handleSecurityHealthCheck() {
  const metrics = getSecurityMetrics();
  
  // Determine health status
  let status = 'healthy';
  let warnings = [];
  
  // Check for high rate limit usage
  if (metrics.rateLimitEntries > 1000) {
    warnings.push('High rate limit entries');
  }
  
  // Check for high security event rate
  if (metrics.recentEvents > 100) {
    status = 'warning';
    warnings.push('High security event rate');
  }
  
  return new Response(JSON.stringify({
    status,
    warnings,
    metrics,
    timestamp: Date.now()
  }), {
    headers: {
      'Content-Type': 'application/json',
      'Cache-Control': 'no-cache'
    }
  });
}

// Cleanup rate limit entries every 5 minutes
setInterval(cleanupRateLimit, 5 * 60 * 1000);