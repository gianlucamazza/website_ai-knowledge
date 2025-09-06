"""
RSS and Atom feed parser for content discovery and ingestion.

Handles parsing of RSS/Atom feeds and extraction of article metadata.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

import feedparser
from dateutil.parser import parse as parse_date

from ..database.models import Article, ContentType
from .scraper import EthicalScraper

logger = logging.getLogger(__name__)


class RSSParser:
    """RSS/Atom feed parser with content extraction capabilities."""
    
    def __init__(self):
        self.scraper: Optional[EthicalScraper] = None
    
    async def parse_feed(self, feed_url: str, source_config: Dict) -> List[Dict]:
        """
        Parse an RSS/Atom feed and extract article information.
        
        Args:
            feed_url: URL of the RSS/Atom feed
            source_config: Configuration for the source
            
        Returns:
            List of article data dictionaries
        """
        try:
            async with EthicalScraper() as scraper:
                self.scraper = scraper
                
                # Fetch the feed
                feed_result = await scraper.fetch_url(feed_url)
                if not feed_result:
                    logger.error(f"Failed to fetch feed: {feed_url}")
                    return []
                
                # Parse the feed content
                feed = feedparser.parse(feed_result['content'])
                
                if feed.bozo and feed.bozo_exception:
                    logger.warning(f"Feed parsing warning for {feed_url}: {feed.bozo_exception}")
                
                articles = []
                max_articles = source_config.get('max_articles_per_run', 100)
                
                for entry in feed.entries[:max_articles]:
                    article_data = await self._extract_article_data(entry, feed, source_config)
                    if article_data:
                        articles.append(article_data)
                
                logger.info(f"Parsed {len(articles)} articles from feed: {feed_url}")
                return articles
                
        except Exception as e:
            logger.error(f"Error parsing feed {feed_url}: {e}")
            return []
    
    async def _extract_article_data(self, entry, feed, source_config: Dict) -> Optional[Dict]:
        """Extract article data from a feed entry."""
        try:
            # Extract basic information
            title = self._get_entry_value(entry, 'title', '').strip()
            if not title:
                logger.warning("Skipping entry without title")
                return None
            
            url = self._get_entry_value(entry, 'link', '')
            if not url:
                logger.warning(f"Skipping entry without URL: {title}")
                return None
            
            # Extract content
            content = self._extract_content(entry)
            if not content or len(content) < 100:
                # Try to fetch full content from URL if summary is too short
                content = await self._fetch_full_content(url)
            
            # Extract metadata
            article_data = {
                'url': url,
                'title': title,
                'raw_html': content,
                'content_type': self._determine_content_type(entry, source_config),
                'author': self._extract_author(entry),
                'publish_date': self._extract_publish_date(entry),
                'summary': self._extract_summary(entry),
                'tags': self._extract_tags(entry, source_config),
                'categories': source_config.get('categories', []),
                'meta_description': self._extract_meta_description(entry),
                'language': source_config.get('language', 'en'),
            }
            
            return article_data
            
        except Exception as e:
            logger.error(f"Error extracting article data: {e}")
            return None
    
    def _get_entry_value(self, entry, key: str, default: str = '') -> str:
        """Safely get value from feed entry."""
        try:
            value = getattr(entry, key, default)
            return str(value) if value else default
        except Exception:
            return default
    
    def _extract_content(self, entry) -> str:
        """Extract content from feed entry."""
        content = ''
        
        # Try different content fields in order of preference
        content_fields = [
            'content',
            'summary_detail', 
            'description',
            'summary'
        ]
        
        for field in content_fields:
            if hasattr(entry, field):
                field_value = getattr(entry, field)
                
                if isinstance(field_value, list) and field_value:
                    # Handle content array (RSS 2.0)
                    content = field_value[0].get('value', '')
                elif isinstance(field_value, dict):
                    # Handle content dict (Atom)
                    content = field_value.get('value', '')
                else:
                    # Handle plain string
                    content = str(field_value) if field_value else ''
                
                if content.strip():
                    break
        
        return content
    
    async def _fetch_full_content(self, url: str) -> str:
        """Fetch full content from article URL if needed."""
        if not self.scraper:
            return ""
        
        try:
            result = await self.scraper.fetch_url(url)
            if result and result['status_code'] == 200:
                return result['content']
        except Exception as e:
            logger.warning(f"Failed to fetch full content from {url}: {e}")
        
        return ""
    
    def _extract_author(self, entry) -> Optional[str]:
        """Extract author information from feed entry."""
        # Try different author fields
        author_fields = ['author', 'author_detail', 'dc_creator']
        
        for field in author_fields:
            if hasattr(entry, field):
                author = getattr(entry, field)
                if isinstance(author, dict):
                    return author.get('name') or author.get('email', '')
                elif author:
                    return str(author)
        
        return None
    
    def _extract_publish_date(self, entry) -> Optional[datetime]:
        """Extract publish date from feed entry."""
        date_fields = [
            'published_parsed',
            'updated_parsed', 
            'published',
            'updated',
            'dc_date'
        ]
        
        for field in date_fields:
            if hasattr(entry, field):
                date_value = getattr(entry, field)
                
                if date_value:
                    try:
                        if hasattr(date_value, 'tm_year'):
                            # time.struct_time object
                            return datetime(*date_value[:6])
                        elif isinstance(date_value, str):
                            # String date
                            return parse_date(date_value)
                    except Exception as e:
                        logger.warning(f"Failed to parse date {date_value}: {e}")
                        continue
        
        return None
    
    def _extract_summary(self, entry) -> Optional[str]:
        """Extract summary/description from feed entry."""
        summary_fields = ['summary', 'description', 'subtitle']
        
        for field in summary_fields:
            if hasattr(entry, field):
                summary = getattr(entry, field)
                if summary:
                    # Clean HTML tags from summary
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(str(summary), 'html.parser')
                    clean_summary = soup.get_text(strip=True)
                    
                    # Limit summary length
                    if len(clean_summary) > 500:
                        clean_summary = clean_summary[:497] + "..."
                    
                    return clean_summary
        
        return None
    
    def _extract_tags(self, entry, source_config: Dict) -> List[str]:
        """Extract tags from feed entry."""
        tags = []
        
        # Extract from feed tags
        if hasattr(entry, 'tags'):
            for tag in entry.tags:
                if hasattr(tag, 'term'):
                    tags.append(tag.term.lower())
        
        # Extract from categories
        if hasattr(entry, 'categories'):
            tags.extend([cat.lower() for cat in entry.categories])
        
        # Add source-specific tags
        source_tags = source_config.get('tags', [])
        tags.extend(source_tags)
        
        # Remove duplicates and filter
        tags = list(set(tag.strip() for tag in tags if tag.strip()))
        
        return tags[:10]  # Limit to 10 tags
    
    def _extract_meta_description(self, entry) -> Optional[str]:
        """Extract meta description from feed entry."""
        # Use summary as meta description if available
        summary = self._extract_summary(entry)
        if summary and len(summary) <= 160:
            return summary
        elif summary:
            return summary[:157] + "..."
        
        return None
    
    def _determine_content_type(self, entry, source_config: Dict) -> ContentType:
        """Determine content type based on entry and source configuration."""
        # Check source configuration first
        config_type = source_config.get('content_type', 'article')
        
        try:
            return ContentType(config_type)
        except ValueError:
            pass
        
        # Analyze entry content for type hints
        title = self._get_entry_value(entry, 'title', '').lower()
        summary = self._extract_summary(entry) or ''
        summary_lower = summary.lower()
        
        # Tutorial indicators
        tutorial_keywords = ['tutorial', 'how to', 'guide', 'walkthrough', 'step by step']
        if any(keyword in title or keyword in summary_lower for keyword in tutorial_keywords):
            return ContentType.TUTORIAL
        
        # News indicators
        news_keywords = ['announces', 'releases', 'news', 'breaking', 'update']
        if any(keyword in title for keyword in news_keywords):
            return ContentType.NEWS
        
        # Reference indicators  
        ref_keywords = ['documentation', 'api', 'reference', 'spec', 'specification']
        if any(keyword in title for keyword in ref_keywords):
            return ContentType.REFERENCE
        
        # Default to article
        return ContentType.ARTICLE
    
    def validate_article_data(self, article_data: Dict) -> bool:
        """Validate that article data meets minimum requirements."""
        required_fields = ['url', 'title']
        
        # Check required fields
        for field in required_fields:
            if not article_data.get(field):
                logger.warning(f"Article missing required field: {field}")
                return False
        
        # Validate URL format
        parsed_url = urlparse(article_data['url'])
        if not parsed_url.scheme or not parsed_url.netloc:
            logger.warning(f"Invalid URL format: {article_data['url']}")
            return False
        
        # Check content length
        content = article_data.get('raw_html', '') or article_data.get('summary', '')
        if len(content) < 50:
            logger.warning(f"Article content too short: {article_data['title']}")
            return False
        
        return True