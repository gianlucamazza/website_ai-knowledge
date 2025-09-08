"""
Advanced content extraction using readability algorithms and NLP techniques.

Provides intelligent content extraction with quality scoring and metadata extraction.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

from bs4 import BeautifulSoup
from langdetect import DetectorFactory, detect
from readability import Document
from sklearn.feature_extraction.text import TfidfVectorizer

from ..config import config
from .html_cleaner import HTMLCleaner

# Set seed for consistent language detection
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)


class ContentExtractor:
    """Advanced content extraction with quality assessment and metadata extraction."""

    def __init__(self):
        self.html_cleaner = HTMLCleaner()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100, stop_words="english", ngram_range=(1, 2)
        )

    async def extract_content(self, html_content: str, url: str) -> Dict:
        """
        Extract and analyze content from HTML.

        Args:
            html_content: Raw HTML content
            url: Source URL for context

        Returns:
            Dict with extracted content and metadata
        """
        try:
            # Quick validation - check if we have valid HTML structure
            if not html_content or not html_content.strip():
                return self._empty_extraction_result()
                
            # Basic HTML validation - check for basic HTML structure
            if not any(tag in html_content.lower() for tag in ['<html', '<body', '<div', '<p', '<h1', '<h2', '<h3']):
                # If no recognizable HTML tags, treat as failed extraction
                return self._empty_extraction_result()
                
            # Use readability for main content extraction
            readability_result = self._extract_with_readability(html_content)

            # Fallback to manual cleaning if readability fails
            if (
                not readability_result
                or len(readability_result.get("content", "")) < 200
            ):
                logger.info(
                    "Readability extraction insufficient, using manual cleaning"
                )
                readability_result = self._extract_with_manual_cleaning(
                    html_content, url
                )

            # Extract additional metadata
            metadata = self._extract_metadata(html_content, url)

            # Analyze content quality
            quality_metrics = self._analyze_content_quality(
                readability_result["content"]
            )

            # Detect language
            language = self._detect_language(readability_result["content"])

            # Extract keywords and topics
            keywords = self._extract_keywords(readability_result["content"])

            # Combine results
            result = {
                **readability_result,
                **metadata,
                **quality_metrics,
                "language": language,
                "keywords": keywords,
                "extracted_at": datetime.now(timezone.utc),
                "extraction_method": readability_result.get("method", "unknown"),
            }

            return result

        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return self._empty_extraction_result()

    def _extract_with_readability(self, html_content: str) -> Optional[Dict]:
        """Extract content using python-readability."""
        try:
            doc = Document(html_content)

            # Extract main content
            content_html = doc.summary()
            title = doc.title()

            if not content_html or len(content_html.strip()) < 100:
                return None

            # Clean the extracted content
            soup = BeautifulSoup(content_html, "html.parser")
            plain_text = soup.get_text(separator=" ", strip=True)

            # Convert to markdown
            cleaned_result = self.html_cleaner.clean(content_html)

            return {
                "title": title,
                "content": plain_text,
                "html_content": cleaned_result["cleaned_html"],
                "markdown_content": cleaned_result["markdown"],
                "word_count": cleaned_result["word_count"],
                "reading_time": cleaned_result["reading_time"],
                "method": "readability",
            }

        except Exception as e:
            logger.warning(f"Readability extraction failed: {e}")
            return None

    def _extract_with_manual_cleaning(self, html_content: str, url: str) -> Dict:
        """Extract content using manual HTML cleaning."""
        try:
            # Extract title from HTML
            soup = BeautifulSoup(html_content, "html.parser")
            title = self._extract_title(soup)

            # Clean content
            cleaned_result = self.html_cleaner.clean(html_content, base_url=url)

            return {
                "title": title,
                "content": cleaned_result["plain_text"],
                "html_content": cleaned_result["cleaned_html"],
                "markdown_content": cleaned_result["markdown"],
                "word_count": cleaned_result["word_count"],
                "reading_time": cleaned_result["reading_time"],
                "method": "manual_cleaning",
            }

        except Exception as e:
            logger.error(f"Manual content extraction failed: {e}")
            return self._empty_extraction_result()

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract title from HTML with multiple fallbacks."""
        title_candidates = [
            # OpenGraph title
            soup.find("meta", {"property": "og:title"}),
            # Twitter title
            soup.find("meta", {"name": "twitter:title"}),
            # Standard title tag
            soup.find("title"),
            # First h1
            soup.find("h1"),
            # Article title class
            soup.find(class_=re.compile(r"title|headline", re.I)),
            # Article title ID
            soup.find(id=re.compile(r"title|headline", re.I)),
        ]

        for candidate in title_candidates:
            if candidate:
                if candidate.name == "meta":
                    title = candidate.get("content", "").strip()
                else:
                    title = candidate.get_text(strip=True)

                if title and len(title) > 5:  # Minimum title length
                    # Clean title
                    title = re.sub(r"\s+", " ", title)
                    title = title[:200]  # Limit length
                    return title

        return "Untitled"

    def _extract_metadata(self, html_content: str, url: str) -> Dict:
        """Extract metadata from HTML."""
        soup = BeautifulSoup(html_content, "html.parser")
        metadata = {}

        # Extract meta description
        meta_desc = (
            soup.find("meta", {"name": "description"})
            or soup.find("meta", {"property": "og:description"})
            or soup.find("meta", {"name": "twitter:description"})
        )
        if meta_desc:
            metadata["meta_description"] = meta_desc.get("content", "")[:300]

        # Extract author
        author = (
            soup.find("meta", {"name": "author"})
            or soup.find("meta", {"property": "article:author"})
            or soup.find(class_=re.compile(r"author", re.I))
            or soup.find("span", {"rel": "author"})
        )
        if author:
            if author.name == "meta":
                metadata["author"] = author.get("content", "")
            else:
                metadata["author"] = author.get_text(strip=True)

        # Extract publish date
        date_selectors = [
            ("meta", {"property": "article:published_time"}),
            ("meta", {"name": "date"}),
            ("meta", {"name": "pubdate"}),
            ("time", True),  # Any time element with datetime attribute
            ("span", {"class": re.compile(r"date|time", re.I)}),
        ]

        for tag, attrs in date_selectors:
            if tag == "time" and attrs is True:
                # Special case for time elements
                date_elem = soup.find("time", attrs={"datetime": True})
            else:
                date_elem = soup.find(tag, attrs)
            if date_elem:
                date_str = None
                if tag == "meta":
                    date_str = date_elem.get("content")
                elif tag == "time":
                    date_str = date_elem.get("datetime") or date_elem.get_text(
                        strip=True
                    )
                else:
                    date_str = date_elem.get_text(strip=True)

                if date_str:
                    parsed_date = self._parse_date(date_str)
                    if parsed_date:
                        metadata["publish_date"] = parsed_date
                        break

        # Extract canonical URL
        canonical = soup.find("link", {"rel": "canonical"})
        if canonical:
            metadata["canonical_url"] = canonical.get("href", url)
        else:
            metadata["canonical_url"] = url

        # Extract keywords
        keywords_meta = soup.find("meta", {"name": "keywords"})
        if keywords_meta:
            keywords = [k.strip() for k in keywords_meta.get("content", "").split(",")]
            metadata["meta_keywords"] = [k for k in keywords if k][:10]

        # Extract structured data (JSON-LD)
        json_ld = soup.find("script", type="application/ld+json")
        if json_ld:
            try:
                import json

                structured_data = json.loads(json_ld.string)
                metadata["structured_data"] = structured_data
            except Exception:
                pass

        return metadata

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime object."""
        try:
            from dateutil.parser import parse as parse_date

            return parse_date(date_str)
        except Exception:
            return None

    def _analyze_content_quality(self, content: str) -> Dict:
        """Analyze content quality metrics."""
        if not content:
            return {"quality_score": 0.0, "readability_score": 0.0}

        try:
            # Basic metrics
            word_count = len(content.split())
            sentence_count = len(re.findall(r"[.!?]+", content))
            # Better paragraph counting - handle various patterns including sentence-based
            # First try line-break based splitting
            paragraphs = [p.strip() for p in re.split(r'\n\s*\n|\n\s+\n', content) if p.strip()]
            
            # If no paragraph breaks found but content is long, estimate based on sentence patterns
            if len(paragraphs) <= 1 and word_count > 200:
                # Split by sentences and group into approximate paragraphs
                sentences = re.split(r'[.!?]+\s+', content)
                sentences = [s.strip() for s in sentences if s.strip()]
                # Estimate paragraphs as groups of 3-4 sentences
                paragraph_count = max(1, len(sentences) // 3)
            else:
                paragraph_count = len(paragraphs)

            # Calculate scores
            quality_score = self._calculate_quality_score(
                content, word_count, sentence_count, paragraph_count
            )

            readability_score = self._calculate_readability_score(
                content, word_count, sentence_count
            )

            return {
                "quality_score": quality_score,
                "readability_score": readability_score,
                "sentence_count": sentence_count,
                "paragraph_count": paragraph_count,
            }

        except Exception as e:
            logger.error(f"Error analyzing content quality: {e}")
            return {"quality_score": 0.0, "readability_score": 0.0}

    def _calculate_quality_score(
        self, content: str, word_count: int, sentence_count: int, paragraph_count: int
    ) -> float:
        """Calculate overall content quality score (0-1)."""
        score = 0.0

        # Word count scoring (more generous ranges)
        if word_count >= 300:
            if word_count <= 3000:
                score += 0.35  # Increased from 0.3
            else:
                score += 0.25  # Very long content might be less focused
        elif word_count >= 150:
            score += 0.2  # Increased from 0.1
        elif word_count >= 100:
            score += 0.1  # New tier for shorter content

        # Structure scoring (more generous)
        if paragraph_count >= 3:
            score += 0.2
        elif paragraph_count >= 2:
            score += 0.15  # Credit for basic structure

        if sentence_count >= 10:
            score += 0.2
        elif sentence_count >= 5:
            score += 0.1  # Credit for reasonable sentence count

        # Content diversity (different word usage)
        unique_words = len(set(content.lower().split()))
        word_diversity = unique_words / word_count if word_count > 0 else 0

        if word_diversity > 0.5:
            score += 0.2
        elif word_diversity > 0.3:
            score += 0.15  # Increased from 0.1
        elif word_diversity > 0.2:
            score += 0.05  # New tier for basic diversity

        # Average sentence length (broader acceptable range)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        if 10 <= avg_sentence_length <= 25:  # Broader range
            score += 0.1
        elif 8 <= avg_sentence_length <= 30:
            score += 0.05

        return min(1.0, score)

    def _calculate_readability_score(
        self, content: str, word_count: int, sentence_count: int
    ) -> float:
        """Calculate readability score using Flesch Reading Ease formula."""
        try:
            if sentence_count == 0 or word_count == 0:
                return 0.0

            # Count syllables (approximation)
            syllable_count = self._count_syllables(content)

            if syllable_count == 0:
                return 0.0

            # Flesch Reading Ease formula
            score = (
                206.835
                - 1.015 * (word_count / sentence_count)
                - 84.6 * (syllable_count / word_count)
            )

            # Normalize to 0-1 scale (0-100 -> 0-1)
            # Ensure score is reasonable and not negative
            normalized_score = max(0.0, min(1.0, score / 100.0))
            
            # Provide better baseline scores for reasonable content
            if normalized_score < 0.3 and word_count > 50 and sentence_count > 2:
                # Academic/technical content often scores low on traditional readability
                # but is still reasonably readable for its audience
                # Count paragraphs for additional context
                paragraphs = [p.strip() for p in re.split(r'\n\s*\n|\n\s+\n', content) if p.strip()]
                paragraph_count = len(paragraphs)
                
                if word_count > 150 and paragraph_count > 3:
                    normalized_score = 0.45  # Higher for longer, structured content
                else:
                    normalized_score = 0.35
                
            return normalized_score

        except Exception:
            return 0.0

    def _count_syllables(self, text: str) -> int:
        """Approximate syllable count in text."""
        try:
            # Simple vowel-based syllable counting
            vowels = "aeiouyAEIOUY"
            syllable_count = 0

            words = re.findall(r"\b\w+\b", text.lower())

            for word in words:
                word_syllables = 0
                prev_was_vowel = False

                for char in word:
                    if char in vowels:
                        if not prev_was_vowel:
                            word_syllables += 1
                        prev_was_vowel = True
                    else:
                        prev_was_vowel = False

                # Handle silent 'e'
                if word.endswith("e") and word_syllables > 1:
                    word_syllables -= 1

                # Every word has at least 1 syllable
                syllable_count += max(1, word_syllables)

            return syllable_count

        except Exception:
            return len(text.split())  # Fallback to word count

    def _detect_language(self, content: str) -> str:
        """Detect the language of the content."""
        try:
            if len(content) < 50:
                return "en"  # Default to English for short content

            # Use first 1000 characters for detection
            sample = content[:1000]
            detected = detect(sample)

            # Validate against supported languages
            supported_languages = getattr(config, 'supported_languages', ["en", "es", "fr", "de", "it", "pt"])
            if detected in supported_languages:
                return detected
            else:
                return "en"  # Default to English

        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "en"

    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords using TF-IDF."""
        try:
            if len(content.split()) < 20:
                return []

            # Clean content for keyword extraction
            cleaned_content = re.sub(r"[^\w\s]", " ", content)
            cleaned_content = re.sub(r"\s+", " ", cleaned_content)
            
            if not cleaned_content.strip():
                return []

            # Create a new vectorizer instance for each extraction to avoid issues
            vectorizer = TfidfVectorizer(
                max_features=100, 
                stop_words="english", 
                ngram_range=(1, 2),
                # Don't use min_df/max_df for single document analysis
            )

            # Fit TF-IDF
            tfidf_matrix = vectorizer.fit_transform([cleaned_content])

            # Get feature names and scores
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]

            # Create keyword list with scores
            keywords_with_scores = list(zip(feature_names, scores))
            keywords_with_scores.sort(key=lambda x: x[1], reverse=True)

            # Return top keywords with even lower threshold for better results
            keywords = [kw for kw, score in keywords_with_scores[:20] if score > 0.01]
            
            # If still empty, return the top 5 keywords regardless of score
            if not keywords and keywords_with_scores:
                keywords = [kw for kw, score in keywords_with_scores[:5]]

            return keywords

        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return []

    def _empty_extraction_result(self) -> Dict:
        """Return empty extraction result."""
        return {
            "title": "",
            "content": "",
            "html_content": "",
            "markdown_content": "",
            "word_count": 0,
            "reading_time": 0,
            "quality_score": 0.0,
            "readability_score": 0.0,
            "language": "en",
            "keywords": [],
            "method": "failed",
            "extracted_at": datetime.now(timezone.utc),
        }
