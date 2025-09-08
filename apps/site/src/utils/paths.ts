/**
 * Helper function to create proper URLs with BASE_URL
 * Handles the trailing slash correctly
 */
export function getUrl(path: string): string {
  const base = import.meta.env.BASE_URL || '/';
  // Remove trailing slash from base if present
  const cleanBase = base.endsWith('/') ? base.slice(0, -1) : base;
  // Ensure path starts with /
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  return `${cleanBase}${cleanPath}`;
}