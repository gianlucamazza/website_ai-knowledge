"""
Ethical web scraper with robots.txt compliance and rate limiting.

Implements responsible scraping practices with respect for server resources
and website policies.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import aiohttp
from aiohttp import ClientSession, ClientTimeout
from bs4 import BeautifulSoup

from ..config import config

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter for controlling request frequency per domain."""

    def __init__(self, default_delay: float = 1.0):
        self.default_delay = default_delay
        self.last_requests: Dict[str, datetime] = {}
        self.delays: Dict[str, float] = {}

    def set_delay(self, domain: str, delay: float) -> None:
        """Set custom delay for a specific domain."""
        self.delays[domain] = delay

    async def wait(self, url: str) -> None:
        """Wait for appropriate delay before making request."""
        domain = urlparse(url).netloc
        delay = self.delays.get(domain, self.default_delay)

        if domain in self.last_requests:
            time_since_last = datetime.now() - self.last_requests[domain]
            if time_since_last.total_seconds() < delay:
                wait_time = delay - time_since_last.total_seconds()
                await asyncio.sleep(wait_time)

        self.last_requests[domain] = datetime.now()


class RobotsChecker:
    """Robots.txt checker for ethical scraping compliance."""

    def __init__(self, user_agent: str):
        self.user_agent = user_agent
        self.robots_cache: Dict[str, RobotFileParser] = {}
        self.cache_expiry: Dict[str, datetime] = {}
        self.cache_duration = timedelta(hours=24)

    async def can_fetch(self, url: str, session: ClientSession) -> bool:
        """Check if URL can be fetched according to robots.txt."""
        try:
            domain = urlparse(url).netloc

            # Check cache
            if domain in self.robots_cache and domain in self.cache_expiry:
                if datetime.now() < self.cache_expiry[domain]:
                    return self.robots_cache[domain].can_fetch(self.user_agent, url)

            # Fetch robots.txt
            robots_url = f"https://{domain}/robots.txt"

            try:
                async with session.get(robots_url) as response:
                    if response.status == 200:
                        robots_content = await response.text()

                        # Parse robots.txt
                        rp = RobotFileParser()
                        rp.set_url(robots_url)
                        rp.read_lines(robots_content.splitlines())

                        # Cache the result
                        self.robots_cache[domain] = rp
                        self.cache_expiry[domain] = datetime.now() + self.cache_duration

                        return rp.can_fetch(self.user_agent, url)
                    else:
                        # If robots.txt not found, assume allowed
                        logger.info(
                            f"No robots.txt found for {domain}, assuming allowed"
                        )
                        return True

            except Exception as e:
                logger.warning(f"Failed to fetch robots.txt for {domain}: {e}")
                return True  # Assume allowed if can't fetch robots.txt

        except Exception as e:
            logger.error(f"Error checking robots.txt for {url}: {e}")
            return True  # Default to allowed on error


class EthicalScraper:
    """Ethical web scraper with rate limiting and robots.txt compliance."""

    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self.scraping_config = config.scraping

        self.rate_limiter = RateLimiter(self.scraping_config.request_delay)
        self.robots_checker = RobotsChecker(self.scraping_config.user_agent)

        self.session: Optional[ClientSession] = None
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def __aenter__(self):
        """Async context manager entry."""
        timeout = ClientTimeout(total=self.scraping_config.timeout)

        # Optimized connector with connection pooling
        connector = aiohttp.TCPConnector(
            limit=100,  # Total connection pool size
            limit_per_host=20,  # Per-host connection limit
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
            keepalive_timeout=60,  # Keep connections alive
            enable_cleanup_closed=True,
        )

        self.session = ClientSession(
            timeout=timeout,
            headers={
                "User-Agent": self.scraping_config.user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            },
            connector=connector,
            # Enable compression
            auto_decompress=True,
        )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def fetch_url(self, url: str, allow_redirects: bool = True) -> Optional[Dict]:
        """
        Fetch a single URL with ethical scraping practices.

        Returns:
            Dict with 'content', 'status_code', 'headers', 'url' keys or None if failed
        """
        async with self.semaphore:
            try:
                # Check robots.txt compliance
                if not await self.robots_checker.can_fetch(url, self.session):
                    logger.warning(f"Robots.txt disallows fetching: {url}")
                    return None

                # Apply rate limiting
                await self.rate_limiter.wait(url)

                # Make request with retries
                for attempt in range(self.scraping_config.max_retries):
                    try:
                        async with self.session.get(
                            url, allow_redirects=allow_redirects, max_redirects=5
                        ) as response:

                            # Check content size
                            content_length = response.headers.get("content-length")
                            if (
                                content_length
                                and int(content_length)
                                > self.scraping_config.max_content_size
                            ):
                                logger.warning(
                                    f"Content too large for {url}: {content_length} bytes"
                                )
                                return None

                            # Read content
                            content = await response.text()

                            # Check actual content size
                            if (
                                len(content.encode("utf-8"))
                                > self.scraping_config.max_content_size
                            ):
                                logger.warning(f"Content too large for {url}")
                                return None

                            return {
                                "content": content,
                                "status_code": response.status,
                                "headers": dict(response.headers),
                                "url": str(response.url),  # Final URL after redirects
                                "content_type": response.headers.get(
                                    "content-type", ""
                                ),
                            }

                    except asyncio.TimeoutError:
                        logger.warning(
                            f"Timeout fetching {url} (attempt {attempt + 1})"
                        )
                        if attempt < self.scraping_config.max_retries - 1:
                            await asyncio.sleep(2**attempt)  # Exponential backoff
                        continue

                    except Exception as e:
                        logger.error(
                            f"Error fetching {url} (attempt {attempt + 1}): {e}"
                        )
                        if attempt < self.scraping_config.max_retries - 1:
                            await asyncio.sleep(2**attempt)
                        continue

                logger.error(
                    f"Failed to fetch {url} after {self.scraping_config.max_retries} attempts"
                )
                return None

            except Exception as e:
                logger.error(f"Unexpected error fetching {url}: {e}")
                return None

    async def fetch_multiple(self, urls: List[str]) -> List[Optional[Dict]]:
        """
        Fetch multiple URLs concurrently with rate limiting.

        Args:
            urls: List of URLs to fetch

        Returns:
            List of fetch results (None for failed requests)
        """
        tasks = [self.fetch_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to None
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)

        return processed_results

    def extract_links(self, html_content: str, base_url: str) -> Set[str]:
        """Extract all links from HTML content."""
        try:
            soup = BeautifulSoup(html_content, "lxml")
            links = set()

            # Find all anchor tags with href
            for link in soup.find_all("a", href=True):
                href = link["href"]

                # Convert relative URLs to absolute
                full_url = urljoin(base_url, href)

                # Filter out non-HTTP(S) links
                parsed = urlparse(full_url)
                if parsed.scheme in ("http", "https"):
                    links.add(full_url)

            return links

        except Exception as e:
            logger.error(f"Error extracting links from {base_url}: {e}")
            return set()

    async def crawl_sitemap(self, sitemap_url: str) -> List[str]:
        """
        Crawl a sitemap.xml file to discover URLs.

        Args:
            sitemap_url: URL of the sitemap.xml file

        Returns:
            List of URLs found in the sitemap
        """
        try:
            result = await self.fetch_url(sitemap_url)
            if not result:
                return []

            soup = BeautifulSoup(result["content"], "xml")
            urls = []

            # Handle regular sitemaps
            for loc in soup.find_all("loc"):
                url = loc.get_text(strip=True)
                if url:
                    urls.append(url)

            # Handle sitemap index files
            for sitemap in soup.find_all("sitemap"):
                loc = sitemap.find("loc")
                if loc:
                    sitemap_url = loc.get_text(strip=True)
                    # Recursively crawl nested sitemaps
                    nested_urls = await self.crawl_sitemap(sitemap_url)
                    urls.extend(nested_urls)

            logger.info(f"Found {len(urls)} URLs in sitemap: {sitemap_url}")
            return urls

        except Exception as e:
            logger.error(f"Error crawling sitemap {sitemap_url}: {e}")
            return []

    async def discover_content_urls(
        self, base_url: str, max_depth: int = 2
    ) -> Set[str]:
        """
        Discover content URLs through crawling with depth limit.

        Args:
            base_url: Starting URL for discovery
            max_depth: Maximum crawl depth

        Returns:
            Set of discovered URLs
        """
        discovered_urls = set()
        visited_urls = set()
        current_level = {base_url}

        for depth in range(max_depth):
            if not current_level:
                break

            logger.info(
                f"Discovering URLs at depth {depth + 1}: {len(current_level)} URLs"
            )

            # Fetch all URLs at current level
            results = await self.fetch_multiple(list(current_level))
            next_level = set()

            for url, result in zip(current_level, results):
                if result and result["status_code"] == 200:
                    discovered_urls.add(url)

                    # Extract links for next level
                    if depth < max_depth - 1:
                        links = self.extract_links(result["content"], url)

                        # Filter to same domain and unvisited URLs
                        base_domain = urlparse(base_url).netloc
                        for link in links:
                            if (
                                urlparse(link).netloc == base_domain
                                and link not in visited_urls
                                and link not in discovered_urls
                            ):
                                next_level.add(link)

                visited_urls.add(url)

            current_level = next_level

            # Respect rate limiting between levels
            if current_level:
                await asyncio.sleep(1.0)

        logger.info(f"Discovered {len(discovered_urls)} URLs total")
        return discovered_urls
