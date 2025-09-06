"""
HTML cleaning utilities for removing unwanted elements and normalizing content.

Provides robust HTML cleaning with configurable rules for different content types.
"""

import logging
import re
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, NavigableString, Tag
from markdownify import markdownify as md

logger = logging.getLogger(__name__)


class HTMLCleaner:
    """HTML content cleaner with configurable rules."""
    
    # Elements to remove completely
    REMOVE_ELEMENTS = {
        'script', 'style', 'meta', 'link', 'noscript', 'iframe', 'embed', 'object',
        'form', 'input', 'button', 'select', 'textarea', 'label',
        'nav', 'aside', 'footer', 'header', 'advertisement', 'ad',
        'comment', 'comments-section'
    }
    
    # Elements that typically contain navigation/non-content
    NAVIGATION_ELEMENTS = {
        'nav', 'breadcrumb', 'pagination', 'sidebar', 'menu',
        'navigation', 'navbar', 'topbar', 'bottombar'
    }
    
    # CSS classes/IDs that typically indicate non-content
    NON_CONTENT_INDICATORS = {
        'ad', 'ads', 'advertisement', 'banner', 'popup', 'modal',
        'sidebar', 'navigation', 'nav', 'menu', 'breadcrumb',
        'footer', 'header', 'social', 'share', 'related', 'recommended',
        'comments', 'comment', 'author-bio', 'tags', 'categories',
        'newsletter', 'subscribe', 'signup', 'login'
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize HTML cleaner with optional configuration."""
        self.config = config or {}
        self.base_url = self.config.get('base_url', '')
        self.preserve_links = self.config.get('preserve_links', True)
        self.preserve_images = self.config.get('preserve_images', True)
        self.min_text_length = self.config.get('min_text_length', 50)
    
    def clean(self, html_content: str, base_url: Optional[str] = None) -> Dict[str, str]:
        """
        Clean HTML content and extract structured information.
        
        Args:
            html_content: Raw HTML content to clean
            base_url: Base URL for resolving relative links
            
        Returns:
            Dict with cleaned_html, plain_text, and markdown keys
        """
        if base_url:
            self.base_url = base_url
        
        try:
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Remove unwanted elements
            self._remove_unwanted_elements(soup)
            
            # Extract main content
            main_content = self._extract_main_content(soup)
            
            if not main_content:
                logger.warning("No main content found in HTML")
                return self._empty_result()
            
            # Clean and normalize the content
            cleaned_soup = self._clean_content(main_content)
            
            # Generate different output formats
            cleaned_html = str(cleaned_soup)
            plain_text = self._extract_plain_text(cleaned_soup)
            markdown = self._convert_to_markdown(cleaned_soup)
            
            return {
                'cleaned_html': cleaned_html,
                'plain_text': plain_text,
                'markdown': markdown,
                'word_count': len(plain_text.split()),
                'reading_time': max(1, len(plain_text.split()) // 200),  # Assume 200 WPM
            }
            
        except Exception as e:
            logger.error(f"Error cleaning HTML content: {e}")
            return self._empty_result()
    
    def _remove_unwanted_elements(self, soup: BeautifulSoup) -> None:
        """Remove unwanted HTML elements."""
        # Remove by tag name
        for tag_name in self.REMOVE_ELEMENTS:
            for element in soup.find_all(tag_name):
                element.decompose()
        
        # Remove by class/ID indicators
        for element in soup.find_all(True):
            if self._is_non_content_element(element):
                element.decompose()
        
        # Remove empty elements
        self._remove_empty_elements(soup)
    
    def _is_non_content_element(self, element: Tag) -> bool:
        """Check if element is likely non-content based on class/ID."""
        if not isinstance(element, Tag):
            return False
        
        # Check class names
        classes = element.get('class', [])
        for class_name in classes:
            if any(indicator in class_name.lower() for indicator in self.NON_CONTENT_INDICATORS):
                return True
        
        # Check ID
        element_id = element.get('id', '')
        if any(indicator in element_id.lower() for indicator in self.NON_CONTENT_INDICATORS):
            return True
        
        # Check role attribute
        role = element.get('role', '')
        if role in ['banner', 'navigation', 'complementary', 'contentinfo']:
            return True
        
        return False
    
    def _remove_empty_elements(self, soup: BeautifulSoup) -> None:
        """Remove elements that are empty or contain only whitespace."""
        # Elements that should be removed if empty
        empty_removable = {'p', 'div', 'span', 'section', 'article', 'aside', 'header', 'footer'}
        
        # Keep removing empty elements until no more are found
        changed = True
        while changed:
            changed = False
            for element in soup.find_all(empty_removable):
                if self._is_empty_element(element):
                    element.decompose()
                    changed = True
    
    def _is_empty_element(self, element: Tag) -> bool:
        """Check if element is empty or contains only whitespace."""
        if not isinstance(element, Tag):
            return False
        
        # Get text content
        text = element.get_text(strip=True)
        
        # Check if element has no meaningful content
        if not text:
            # Allow elements with images or other media
            if element.find(['img', 'video', 'audio', 'iframe']):
                return False
            return True
        
        # Very short text might not be meaningful
        if len(text) < self.min_text_length and not element.find(['img', 'video', 'audio']):
            return True
        
        return False
    
    def _extract_main_content(self, soup: BeautifulSoup) -> Optional[Tag]:
        """Extract the main content area from HTML."""
        # Try different strategies to find main content
        
        # Strategy 1: Look for semantic main element
        main = soup.find('main')
        if main:
            return main
        
        # Strategy 2: Look for article element
        article = soup.find('article')
        if article:
            return article
        
        # Strategy 3: Look for content-indicating IDs/classes
        content_indicators = [
            'content', 'main', 'article', 'post', 'entry', 'body',
            'text', 'story', 'primary', 'main-content'
        ]
        
        for indicator in content_indicators:
            # Try ID first
            element = soup.find(id=re.compile(indicator, re.I))
            if element and self._is_substantial_content(element):
                return element
            
            # Try class
            element = soup.find(class_=re.compile(indicator, re.I))
            if element and self._is_substantial_content(element):
                return element
        
        # Strategy 4: Find largest content block
        return self._find_largest_content_block(soup)
    
    def _is_substantial_content(self, element: Tag) -> bool:
        """Check if element contains substantial content."""
        if not isinstance(element, Tag):
            return False
        
        text = element.get_text(strip=True)
        
        # Check text length
        if len(text) < 200:
            return False
        
        # Check for multiple paragraphs or sentences
        paragraphs = element.find_all('p')
        if len(paragraphs) >= 2:
            return True
        
        # Check sentence count
        sentences = re.split(r'[.!?]+', text)
        if len([s for s in sentences if len(s.strip()) > 10]) >= 3:
            return True
        
        return False
    
    def _find_largest_content_block(self, soup: BeautifulSoup) -> Optional[Tag]:
        """Find the largest content block by text length."""
        candidates = []
        
        # Look in common container elements
        containers = soup.find_all(['div', 'section', 'article', 'main'])
        
        for container in containers:
            text_length = len(container.get_text(strip=True))
            if text_length > 500:  # Minimum content length
                candidates.append((container, text_length))
        
        if candidates:
            # Return the container with most text
            return max(candidates, key=lambda x: x[1])[0]
        
        # Fallback to body if nothing else works
        return soup.find('body') or soup
    
    def _clean_content(self, content: Tag) -> BeautifulSoup:
        """Clean and normalize the extracted content."""
        # Create a new soup with just the content
        new_soup = BeautifulSoup('<div></div>', 'html.parser')
        container = new_soup.div
        
        # Copy content to new soup
        for element in content.children:
            if isinstance(element, (Tag, NavigableString)):
                container.append(element.extract())
        
        # Clean attributes
        self._clean_attributes(new_soup)
        
        # Normalize links
        if self.preserve_links:
            self._normalize_links(new_soup)
        
        # Normalize images
        if self.preserve_images:
            self._normalize_images(new_soup)
        
        # Remove excessive whitespace
        self._normalize_whitespace(new_soup)
        
        return new_soup
    
    def _clean_attributes(self, soup: BeautifulSoup) -> None:
        """Remove unnecessary HTML attributes."""
        # Attributes to keep
        keep_attrs = {
            'a': ['href', 'title'],
            'img': ['src', 'alt', 'title', 'width', 'height'],
            'video': ['src', 'poster', 'controls'],
            'audio': ['src', 'controls'],
            'code': ['class'],  # For syntax highlighting
            'pre': ['class'],
        }
        
        for element in soup.find_all(True):
            if isinstance(element, Tag):
                # Get allowed attributes for this tag
                allowed = keep_attrs.get(element.name, [])
                
                # Remove all other attributes
                attrs_to_remove = [attr for attr in element.attrs if attr not in allowed]
                for attr in attrs_to_remove:
                    del element[attr]
    
    def _normalize_links(self, soup: BeautifulSoup) -> None:
        """Normalize link URLs to absolute URLs."""
        if not self.base_url:
            return
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # Skip non-HTTP links
            if href.startswith(('mailto:', 'tel:', 'javascript:', '#')):
                continue
            
            # Convert to absolute URL
            try:
                absolute_url = urljoin(self.base_url, href)
                link['href'] = absolute_url
            except Exception as e:
                logger.warning(f"Failed to normalize link {href}: {e}")
    
    def _normalize_images(self, soup: BeautifulSoup) -> None:
        """Normalize image URLs to absolute URLs."""
        if not self.base_url:
            return
        
        for img in soup.find_all('img', src=True):
            src = img['src']
            
            # Convert to absolute URL
            try:
                absolute_url = urljoin(self.base_url, src)
                img['src'] = absolute_url
            except Exception as e:
                logger.warning(f"Failed to normalize image {src}: {e}")
    
    def _normalize_whitespace(self, soup: BeautifulSoup) -> None:
        """Normalize excessive whitespace in text content."""
        for element in soup.find_all(text=True):
            if isinstance(element, NavigableString):
                # Normalize whitespace
                normalized = re.sub(r'\s+', ' ', str(element))
                element.replace_with(normalized)
    
    def _extract_plain_text(self, soup: BeautifulSoup) -> str:
        """Extract plain text from cleaned HTML."""
        text = soup.get_text(separator=' ', strip=True)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive line breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text.strip()
    
    def _convert_to_markdown(self, soup: BeautifulSoup) -> str:
        """Convert cleaned HTML to Markdown."""
        try:
            # Configure markdownify
            markdown = md(
                str(soup),
                heading_style='ATX',  # Use # for headings
                bullets='-',  # Use - for bullets
                strip=['script', 'style'],
                convert=['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                        'strong', 'em', 'a', 'img', 'ul', 'ol', 'li',
                        'blockquote', 'code', 'pre', 'br'],
            )
            
            # Clean up markdown
            markdown = self._clean_markdown(markdown)
            
            return markdown
            
        except Exception as e:
            logger.error(f"Error converting to markdown: {e}")
            return self._extract_plain_text(soup)
    
    def _clean_markdown(self, markdown: str) -> str:
        """Clean and normalize markdown content."""
        # Remove excessive blank lines
        markdown = re.sub(r'\n\s*\n\s*\n+', '\n\n', markdown)
        
        # Fix spacing around headings
        markdown = re.sub(r'\n(#+\s+[^\n]+)\n+', r'\n\n\1\n\n', markdown)
        
        # Clean up list formatting
        markdown = re.sub(r'\n(\s*[-*+]\s+)', r'\n\1', markdown)
        
        # Remove trailing whitespace
        lines = [line.rstrip() for line in markdown.split('\n')]
        markdown = '\n'.join(lines)
        
        return markdown.strip()
    
    def _empty_result(self) -> Dict[str, str]:
        """Return empty result structure."""
        return {
            'cleaned_html': '',
            'plain_text': '',
            'markdown': '',
            'word_count': 0,
            'reading_time': 0,
        }