"""
Cross-linking system for connecting related articles and building
a knowledge graph of interconnected content.
"""

import logging
import re
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import config
from ..database.models import Article, ContentType

logger = logging.getLogger(__name__)


class CrossLinker:
    """Cross-linking system for building connections between articles."""

    def __init__(self):
        self.enrichment_config = config.enrichment
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
        )
        self.article_vectors = None
        self.article_index = {}  # Map from index to article_id

    async def build_similarity_index(self, session: AsyncSession) -> None:
        """Build similarity index from all articles in the database."""
        try:
            # Get all articles with content
            query = select(Article).where(
                Article.cleaned_content.is_not(None), Article.status == "completed"
            )

            result = await session.execute(query)
            articles = result.scalars().all()

            if not articles:
                logger.warning("No articles found for building similarity index")
                return

            logger.info(f"Building similarity index from {len(articles)} articles")

            # Prepare content for vectorization
            article_contents = []
            self.article_index = {}

            for i, article in enumerate(articles):
                content = self._prepare_content_for_similarity(article)
                article_contents.append(content)
                self.article_index[i] = str(article.id)

            # Build TF-IDF vectors
            if article_contents:
                self.article_vectors = self.tfidf_vectorizer.fit_transform(
                    article_contents
                )
                logger.info(
                    f"Similarity index built with {self.article_vectors.shape[0]} articles"
                )

        except Exception as e:
            logger.error(f"Error building similarity index: {e}")
            raise

    async def find_related_articles(
        self, article_id: str, session: AsyncSession, max_related: int = 5
    ) -> List[Dict]:
        """
        Find articles related to the given article.

        Args:
            article_id: ID of the article to find relations for
            session: Database session
            max_related: Maximum number of related articles to return

        Returns:
            List of related article info with similarity scores
        """
        try:
            # Get the target article
            query = select(Article).where(Article.id == article_id)
            result = await session.execute(query)
            target_article = result.scalar_one_or_none()

            if not target_article:
                logger.warning(f"Article not found: {article_id}")
                return []

            # If similarity index is not built, build it
            if self.article_vectors is None:
                await self.build_similarity_index(session)

            if self.article_vectors is None:
                logger.warning("Cannot find related articles - no similarity index")
                return []

            # Prepare target article content
            target_content = self._prepare_content_for_similarity(target_article)

            # Transform target content to vector space
            target_vector = self.tfidf_vectorizer.transform([target_content])

            # Calculate similarities
            similarities = cosine_similarity(
                target_vector, self.article_vectors
            ).flatten()

            # Get top similar articles (excluding the target article itself)
            similar_indices = []
            for idx in np.argsort(similarities)[::-1]:
                if self.article_index[idx] != article_id:  # Exclude self
                    similarity_score = similarities[idx]
                    if similarity_score >= self.enrichment_config.similarity_threshold:
                        similar_indices.append((idx, similarity_score))

                        if len(similar_indices) >= max_related:
                            break

            # Get article details for similar articles
            related_articles = []
            if similar_indices:
                similar_article_ids = [
                    self.article_index[idx] for idx, _ in similar_indices
                ]

                query = select(Article).where(Article.id.in_(similar_article_ids))
                result = await session.execute(query)
                articles_by_id = {
                    str(article.id): article for article in result.scalars()
                }

                for idx, similarity_score in similar_indices:
                    article_id_key = self.article_index[idx]
                    if article_id_key in articles_by_id:
                        article = articles_by_id[article_id_key]
                        related_articles.append(
                            {
                                "id": str(article.id),
                                "title": article.title,
                                "url": article.url,
                                "content_type": article.content_type,
                                "categories": article.categories or [],
                                "tags": article.tags or [],
                                "similarity_score": float(similarity_score),
                                "word_count": article.word_count or 0,
                            }
                        )

            logger.debug(
                f"Found {len(related_articles)} related articles for {article_id}"
            )
            return related_articles

        except Exception as e:
            logger.error(f"Error finding related articles for {article_id}: {e}")
            return []

    async def generate_cross_links(
        self, article_id: str, session: AsyncSession
    ) -> List[Dict]:
        """
        Generate cross-links for an article based on content analysis.

        Args:
            article_id: ID of the article to generate links for
            session: Database session

        Returns:
            List of cross-link suggestions with context
        """
        try:
            # Get related articles
            related_articles = await self.find_related_articles(
                article_id,
                session,
                max_related=self.enrichment_config.max_related_articles,
            )

            if not related_articles:
                return []

            # Get the source article for content analysis
            query = select(Article).where(Article.id == article_id)
            result = await session.execute(query)
            source_article = result.scalar_one_or_none()

            if not source_article or not source_article.cleaned_content:
                return []

            # Generate contextual links
            cross_links = []
            source_content = source_article.cleaned_content.lower()

            for related in related_articles:
                link_contexts = self._find_link_contexts(
                    source_content, related["title"], related.get("categories", [])
                )

                for context in link_contexts:
                    cross_links.append(
                        {
                            "target_article_id": related["id"],
                            "target_title": related["title"],
                            "target_url": related["url"],
                            "similarity_score": related["similarity_score"],
                            "link_context": context["context"],
                            "anchor_text": context["anchor_text"],
                            "position": context["position"],
                            "relevance_score": context["relevance_score"],
                            "link_type": self._determine_link_type(
                                related, source_article
                            ),
                        }
                    )

            # Sort by relevance score
            cross_links.sort(key=lambda x: x["relevance_score"], reverse=True)

            # Limit number of cross-links to avoid over-linking
            max_links = 10
            cross_links = cross_links[:max_links]

            logger.debug(f"Generated {len(cross_links)} cross-links for {article_id}")
            return cross_links

        except Exception as e:
            logger.error(f"Error generating cross-links for {article_id}: {e}")
            return []

    def _prepare_content_for_similarity(self, article: Article) -> str:
        """Prepare article content for similarity analysis."""
        content_parts = []

        # Add title with higher weight
        if article.title:
            content_parts.append(article.title * 3)  # Weight title more heavily

        # Add main content
        if article.cleaned_content:
            content_parts.append(article.cleaned_content)

        # Add tags and categories
        if article.tags:
            content_parts.append(" ".join(article.tags) * 2)

        if article.categories:
            content_parts.append(" ".join(article.categories) * 2)

        # Add meta description
        if article.meta_description:
            content_parts.append(article.meta_description)

        return " ".join(content_parts)

    def _find_link_contexts(
        self, content: str, target_title: str, target_categories: List[str]
    ) -> List[Dict]:
        """Find contexts where links to target article would be relevant."""
        contexts = []

        # Extract key terms from target title
        title_terms = self._extract_key_terms(target_title)

        # Find mentions of title terms in content
        for term in title_terms:
            if len(term) < 3:  # Skip very short terms
                continue

            # Find all occurrences of the term
            pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
            matches = list(pattern.finditer(content))

            for match in matches:
                context_start = max(0, match.start() - 100)
                context_end = min(len(content), match.end() + 100)
                context_text = content[context_start:context_end].strip()

                # Calculate relevance score based on context
                relevance_score = self._calculate_context_relevance(
                    context_text, term, target_categories
                )

                if relevance_score > 0.3:  # Minimum relevance threshold
                    contexts.append(
                        {
                            "context": context_text,
                            "anchor_text": match.group(),
                            "position": match.start(),
                            "relevance_score": relevance_score,
                        }
                    )

        # Remove duplicates and sort by relevance
        unique_contexts = []
        seen_positions = set()

        for context in sorted(
            contexts, key=lambda x: x["relevance_score"], reverse=True
        ):
            # Avoid overlapping contexts
            too_close = any(
                abs(context["position"] - pos) < 50 for pos in seen_positions
            )
            if not too_close:
                unique_contexts.append(context)
                seen_positions.add(context["position"])

        return unique_contexts[:3]  # Limit to top 3 contexts per article

    def _extract_key_terms(self, title: str) -> List[str]:
        """Extract key terms from article title."""
        # Remove common stop words and short words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "should",
            "could",
            "can",
            "may",
            "might",
            "must",
            "this",
            "that",
            "these",
            "those",
        }

        # Clean title and extract terms
        title = re.sub(r"[^\w\s]", " ", title.lower())
        terms = [
            term.strip()
            for term in title.split()
            if term.strip() and term.strip() not in stop_words and len(term.strip()) > 2
        ]

        # Also create multi-word phrases (bigrams)
        bigrams = []
        for i in range(len(terms) - 1):
            bigram = f"{terms[i]} {terms[i + 1]}"
            bigrams.append(bigram)

        return terms + bigrams

    def _calculate_context_relevance(
        self, context: str, term: str, target_categories: List[str]
    ) -> float:
        """Calculate relevance score for a link context."""
        relevance = 0.5  # Base relevance

        # Boost if term appears in a sentence that seems definitional
        definitional_patterns = [
            rf"\b{re.escape(term)}\s+is\s+",
            rf"\b{re.escape(term)}\s+refers\s+to",
            rf"\b{re.escape(term)}\s+means",
            rf"definition\s+of\s+{re.escape(term)}",
            rf"what\s+is\s+{re.escape(term)}",
        ]

        for pattern in definitional_patterns:
            if re.search(pattern, context, re.IGNORECASE):
                relevance += 0.3
                break

        # Boost if context contains category-related terms
        if target_categories:
            category_terms = " ".join(target_categories).lower()
            context_lower = context.lower()

            for category in target_categories:
                if category.lower() in context_lower:
                    relevance += 0.2

        # Boost if term appears near technical indicators
        technical_indicators = [
            "algorithm",
            "method",
            "technique",
            "approach",
            "framework",
            "model",
            "system",
            "implementation",
            "architecture",
        ]

        for indicator in technical_indicators:
            if indicator in context.lower():
                relevance += 0.1
                break

        # Penalize if context seems unrelated or too promotional
        negative_indicators = [
            "advertisement",
            "sponsored",
            "buy now",
            "click here",
            "subscribe",
            "newsletter",
            "contact us",
        ]

        for indicator in negative_indicators:
            if indicator in context.lower():
                relevance -= 0.3

        return max(0.0, min(1.0, relevance))

    def _determine_link_type(
        self, target_article: Dict, source_article: Article
    ) -> str:
        """Determine the type of cross-link relationship."""
        target_type = target_article.get("content_type", ContentType.ARTICLE)
        source_type = source_article.content_type

        # Type-based relationships
        if target_type == ContentType.GLOSSARY_TERM:
            return "definition"

        if target_type == ContentType.TUTORIAL and source_type == ContentType.ARTICLE:
            return "tutorial"

        if target_type == ContentType.REFERENCE:
            return "reference"

        # Category-based relationships
        source_categories = set(source_article.categories or [])
        target_categories = set(target_article.get("categories", []))

        if source_categories & target_categories:  # Intersection
            return "related"

        # Default relationship
        return "see_also"

    async def update_article_cross_links(
        self, article_id: str, session: AsyncSession
    ) -> bool:
        """Update cross-links for a specific article."""
        try:
            # Generate cross-links
            cross_links = await self.generate_cross_links(article_id, session)

            # TODO: Store cross-links in database or update article content
            # This would depend on how you want to store/display the cross-links

            logger.info(
                f"Updated {len(cross_links)} cross-links for article {article_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Error updating cross-links for {article_id}: {e}")
            return False

    async def rebuild_all_cross_links(self, session: AsyncSession) -> Dict:
        """Rebuild cross-links for all articles."""
        try:
            # First rebuild the similarity index
            await self.build_similarity_index(session)

            # Get all articles
            query = select(Article).where(Article.status == "completed")
            result = await session.execute(query)
            articles = result.scalars().all()

            stats = {"articles_processed": 0, "total_cross_links": 0, "errors": 0}

            for article in articles:
                try:
                    cross_links = await self.generate_cross_links(
                        str(article.id), session
                    )
                    stats["articles_processed"] += 1
                    stats["total_cross_links"] += len(cross_links)

                except Exception as e:
                    logger.error(f"Error processing cross-links for {article.id}: {e}")
                    stats["errors"] += 1

            logger.info(f"Rebuilt cross-links: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Error rebuilding all cross-links: {e}")
            return {"articles_processed": 0, "total_cross_links": 0, "errors": 1}
