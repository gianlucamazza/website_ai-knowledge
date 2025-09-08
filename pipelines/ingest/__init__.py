"""
Content ingestion module with ethical web scraping and RSS parsing.

Handles content discovery and initial ingestion from various sources
while respecting robots.txt and rate limiting.
"""

from .rss_parser import RSSParser
from .scraper import EthicalScraper
from .source_manager import SourceManager

__all__ = ["RSSParser", "EthicalScraper", "SourceManager"]
