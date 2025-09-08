"""
Source management for coordinating content ingestion from multiple sources.

Handles source configuration, scheduling, and coordination of ingestion tasks.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import config
from ..database.models import Article, ContentStatus, PipelineStage, Source
from .rss_parser import RSSParser
from .scraper import EthicalScraper

logger = logging.getLogger(__name__)


class SourceManager:
    """Manages content sources and coordinates ingestion tasks."""

    def __init__(self):
        self.sources_config: Dict = {}
        self.rss_parser = RSSParser()
        self.load_sources_config()

    def load_sources_config(self) -> None:
        """Load source configuration from YAML file."""
        try:
            sources_path = Path(config.project_root) / "pipelines" / config.sources_config_path

            if not sources_path.exists():
                logger.error(f"Sources configuration file not found: {sources_path}")
                return

            with open(sources_path, "r") as f:
                self.sources_config = yaml.safe_load(f)

            logger.info(
                f"Loaded {len(self.sources_config.get('sources', []))} source configurations"
            )

        except Exception as e:
            logger.error(f"Error loading sources configuration: {e}")
            self.sources_config = {"sources": [], "global_config": {}}

    async def sync_sources_to_database(self, session: AsyncSession) -> None:
        """Synchronize source configurations to database."""
        try:
            # Get existing sources from database
            result = await session.execute(select(Source))
            existing_sources = {source.name: source for source in result.scalars().all()}

            # Process configured sources
            for source_config in self.sources_config.get("sources", []):
                source_name = source_config["name"]

                if source_name in existing_sources:
                    # Update existing source
                    source = existing_sources[source_name]
                    source.base_url = source_config["base_url"]
                    source.source_type = source_config["type"]
                    source.config = source_config.get("config", {})
                    source.is_active = source_config.get("active", True)
                    source.crawl_frequency = source_config.get("config", {}).get(
                        "crawl_frequency", 3600
                    )
                    source.max_articles_per_run = source_config.get("config", {}).get(
                        "max_articles_per_run", 100
                    )

                else:
                    # Create new source
                    source = Source(
                        name=source_name,
                        base_url=source_config["base_url"],
                        source_type=source_config["type"],
                        config=source_config.get("config", {}),
                        is_active=source_config.get("active", True),
                        crawl_frequency=source_config.get("config", {}).get(
                            "crawl_frequency", 3600
                        ),
                        max_articles_per_run=source_config.get("config", {}).get(
                            "max_articles_per_run", 100
                        ),
                    )
                    session.add(source)

            await session.commit()
            logger.info("Successfully synchronized sources to database")

        except Exception as e:
            logger.error(f"Error syncing sources to database: {e}")
            await session.rollback()
            raise

    async def get_sources_due_for_crawl(self, session: AsyncSession) -> List[Source]:
        """Get sources that are due for crawling based on their frequency."""
        try:
            current_time = datetime.utcnow()

            # Query for active sources that are due for crawling
            query = select(Source).where(
                Source.is_active == True,
                (
                    (Source.last_crawl.is_(None))
                    | (
                        Source.last_crawl + timedelta(seconds=Source.crawl_frequency)
                        <= current_time
                    )
                ),
            )

            result = await session.execute(query)
            sources = result.scalars().all()

            logger.info(f"Found {len(sources)} sources due for crawling")
            return sources

        except Exception as e:
            logger.error(f"Error getting sources due for crawl: {e}")
            return []

    async def ingest_from_source(self, source: Source, session: AsyncSession) -> Dict:
        """
        Ingest content from a specific source.

        Returns:
            Dict with ingestion statistics
        """
        stats = {
            "source_name": source.name,
            "articles_discovered": 0,
            "articles_ingested": 0,
            "articles_skipped": 0,
            "errors": 0,
            "start_time": datetime.utcnow(),
        }

        try:
            logger.info(f"Starting ingestion from source: {source.name}")

            # Dispatch to appropriate ingestion method
            articles_data = []

            if source.source_type == "rss":
                articles_data = await self._ingest_from_rss(source)
            elif source.source_type == "sitemap":
                articles_data = await self._ingest_from_sitemap(source)
            elif source.source_type == "manual":
                logger.info(f"Manual source {source.name} requires manual triggering")
                return stats
            else:
                logger.error(f"Unknown source type: {source.source_type}")
                return stats

            stats["articles_discovered"] = len(articles_data)

            # Process and store articles
            for article_data in articles_data:
                try:
                    if await self._should_ingest_article(article_data, source, session):
                        await self._create_article(article_data, source, session)
                        stats["articles_ingested"] += 1
                    else:
                        stats["articles_skipped"] += 1

                except Exception as e:
                    logger.error(
                        f"Error processing article {article_data.get('title', 'unknown')}: {e}"
                    )
                    stats["errors"] += 1

            # Update source last crawl time
            source.last_crawl = datetime.utcnow()
            await session.commit()

            stats["end_time"] = datetime.utcnow()
            stats["duration"] = (stats["end_time"] - stats["start_time"]).total_seconds()

            logger.info(
                f"Completed ingestion from {source.name}: "
                f"{stats['articles_ingested']} ingested, "
                f"{stats['articles_skipped']} skipped, "
                f"{stats['errors']} errors"
            )

            return stats

        except Exception as e:
            logger.error(f"Error ingesting from source {source.name}: {e}")
            stats["errors"] += 1
            stats["end_time"] = datetime.utcnow()
            return stats

    async def _ingest_from_rss(self, source: Source) -> List[Dict]:
        """Ingest content from RSS/Atom feed."""
        try:
            return await self.rss_parser.parse_feed(source.base_url, source.config or {})
        except Exception as e:
            logger.error(f"Error ingesting from RSS source {source.name}: {e}")
            return []

    async def _ingest_from_sitemap(self, source: Source) -> List[Dict]:
        """Ingest content from sitemap."""
        try:
            sitemap_url = source.config.get("sitemap_url", f"{source.base_url}/sitemap.xml")

            async with EthicalScraper() as scraper:
                # Get URLs from sitemap
                urls = await scraper.crawl_sitemap(sitemap_url)

                # Limit URLs based on configuration
                max_articles = source.max_articles_per_run or 100
                urls = urls[:max_articles]

                # Fetch content from URLs
                results = await scraper.fetch_multiple(urls)

                articles_data = []
                for url, result in zip(urls, results):
                    if result and result["status_code"] == 200:
                        article_data = {
                            "url": url,
                            "title": self._extract_title_from_html(result["content"]),
                            "raw_html": result["content"],
                            "content_type": source.config.get("content_type", "article"),
                            "categories": source.config.get("categories", []),
                            "language": source.config.get("language", "en"),
                            "discovered_at": datetime.utcnow(),
                        }

                        if self.rss_parser.validate_article_data(article_data):
                            articles_data.append(article_data)

                return articles_data

        except Exception as e:
            logger.error(f"Error ingesting from sitemap source {source.name}: {e}")
            return []

    def _extract_title_from_html(self, html_content: str) -> str:
        """Extract title from HTML content."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html_content, "html.parser")

            # Try different title sources
            title_sources = [
                soup.find("title"),
                soup.find("h1"),
                soup.find("meta", attrs={"property": "og:title"}),
                soup.find("meta", attrs={"name": "twitter:title"}),
            ]

            for source in title_sources:
                if source:
                    if source.name == "meta":
                        title = source.get("content", "")
                    else:
                        title = source.get_text(strip=True)

                    if title:
                        return title[:500]  # Limit title length

            return "Untitled"

        except Exception as e:
            logger.error(f"Error extracting title from HTML: {e}")
            return "Untitled"

    async def _should_ingest_article(
        self, article_data: Dict, source: Source, session: AsyncSession
    ) -> bool:
        """Determine if an article should be ingested."""
        try:
            # Check if article already exists
            query = select(Article).where(
                Article.source_id == source.id, Article.url == article_data["url"]
            )

            result = await session.execute(query)
            existing_article = result.scalar_one_or_none()

            if existing_article:
                logger.debug(f"Article already exists: {article_data['url']}")
                return False

            # Validate article data
            if not self.rss_parser.validate_article_data(article_data):
                return False

            # Check content length requirements
            content = article_data.get("raw_html", "") or article_data.get("summary", "")
            min_length = self.sources_config.get("global_config", {}).get("min_content_length", 500)

            if len(content) < min_length:
                logger.debug(f"Article content too short: {article_data.get('title', 'unknown')}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking if article should be ingested: {e}")
            return False

    async def _create_article(
        self, article_data: Dict, source: Source, session: AsyncSession
    ) -> None:
        """Create a new article record in the database."""
        try:
            # Convert article_data to Article model
            article = Article(
                source_id=source.id,
                url=article_data["url"],
                title=article_data.get("title"),
                raw_html=article_data.get("raw_html"),
                raw_text=article_data.get("raw_text"),
                content_type=article_data.get("content_type", "article"),
                language=article_data.get("language", "en"),
                tags=article_data.get("tags", []),
                categories=article_data.get("categories", []),
                meta_description=article_data.get("meta_description"),
                author=article_data.get("author"),
                publish_date=article_data.get("publish_date"),
                current_stage=PipelineStage.INGEST,
                status=ContentStatus.COMPLETED,
                discovered_at=article_data.get("discovered_at", datetime.utcnow()),
            )

            session.add(article)
            await session.flush()  # Get article ID without committing

            logger.debug(f"Created article: {article.title}")

        except Exception as e:
            logger.error(f"Error creating article: {e}")
            raise

    async def run_scheduled_ingestion(self, session: AsyncSession) -> Dict:
        """Run scheduled ingestion for all sources due for crawling."""
        try:
            sources = await self.get_sources_due_for_crawl(session)

            if not sources:
                logger.info("No sources due for crawling")
                return {"sources_processed": 0, "total_articles_ingested": 0}

            # Process sources concurrently with semaphore for rate limiting
            semaphore = asyncio.Semaphore(3)  # Limit concurrent source processing

            async def process_source(source: Source) -> Dict:
                async with semaphore:
                    return await self.ingest_from_source(source, session)

            # Create tasks for all sources
            tasks = [process_source(source) for source in sources]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Aggregate results
            total_stats = {
                "sources_processed": len(sources),
                "total_articles_discovered": 0,
                "total_articles_ingested": 0,
                "total_articles_skipped": 0,
                "total_errors": 0,
                "source_results": [],
            }

            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Source processing failed: {result}")
                    total_stats["total_errors"] += 1
                else:
                    total_stats["total_articles_discovered"] += result.get("articles_discovered", 0)
                    total_stats["total_articles_ingested"] += result.get("articles_ingested", 0)
                    total_stats["total_articles_skipped"] += result.get("articles_skipped", 0)
                    total_stats["total_errors"] += result.get("errors", 0)
                    total_stats["source_results"].append(result)

            logger.info(
                f"Scheduled ingestion completed: "
                f"{total_stats['sources_processed']} sources processed, "
                f"{total_stats['total_articles_ingested']} articles ingested"
            )

            return total_stats

        except Exception as e:
            logger.error(f"Error in scheduled ingestion: {e}")
            raise
