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

    async def calculate_content_similarity(
        self, content1: str, content2: str
    ) -> float:
        """
        Calculate similarity score between two content pieces.
        
        Args:
            content1: First content to compare
            content2: Second content to compare
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            if not content1 or not content2:
                return 0.0
            
            # Use TF-IDF vectorization for similarity
            contents = [content1, content2]
            
            # Create a new vectorizer for pairwise comparison
            # Don't use min_df/max_df for just 2 documents
            from sklearn.feature_extraction.text import TfidfVectorizer
            pairwise_vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2)
            )
            
            # Fit and transform in one step for just two documents
            tfidf_matrix = pairwise_vectorizer.fit_transform(contents)
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            
            return float(similarity_matrix[0][0])
            
        except Exception as e:
            logger.error(f"Error calculating content similarity: {e}")
            return 0.0
    
    async def calculate_tag_similarity(
        self, tags1: List[str], tags2: List[str]
    ) -> float:
        """
        Calculate similarity between two sets of tags.
        
        Args:
            tags1: First set of tags
            tags2: Second set of tags
            
        Returns:
            Jaccard similarity coefficient (0 to 1)
        """
        try:
            if not tags1 or not tags2:
                return 0.0
            
            # Convert to sets for intersection/union operations
            set1 = set(tags1)
            set2 = set(tags2)
            
            # Calculate Jaccard similarity
            intersection = set1 & set2
            union = set1 | set2
            
            if not union:
                return 0.0
                
            return len(intersection) / len(union)
            
        except Exception as e:
            logger.error(f"Error calculating tag similarity: {e}")
            return 0.0
    
    async def extract_linkable_terms(
        self, content: str, max_terms: int = 20
    ) -> List[str]:
        """
        Extract terms from content that could be linked to other articles.
        
        Args:
            content: Content to extract terms from
            max_terms: Maximum number of terms to extract
            
        Returns:
            List of linkable terms
        """
        try:
            if not content:
                return []
            
            # Use TF-IDF to identify important terms
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                # Create a simple vectorizer for single document
                simple_vectorizer = TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=max_terms
                )
                
                # Fit vectorizer on single document
                tfidf_matrix = simple_vectorizer.fit_transform([content])
                
                # Get feature names and their scores
                feature_names = simple_vectorizer.get_feature_names_out()
                scores = tfidf_matrix.toarray()[0]
                
                # Create term-score pairs and sort by score
                term_scores = list(zip(feature_names, scores))
                term_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Extract top terms
                linkable_terms = [term for term, score in term_scores[:max_terms] if score > 0]
                
                return linkable_terms
                
            except Exception as e:
                logger.warning(f"TF-IDF extraction failed, using fallback: {e}")
                
                # Fallback to simple extraction
                # Extract technical terms and named entities
                terms = []
                
                # Look for capitalized terms (potential entities)
                capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
                terms.extend(capitalized)
                
                # Look for technical terms (acronyms)
                acronyms = re.findall(r'\b[A-Z]{2,}\b', content)
                terms.extend(acronyms)
                
                # Look for quoted terms
                quoted = re.findall(r'"([^"]+)"', content)
                terms.extend(quoted)
                
                # Deduplicate and limit
                unique_terms = list(dict.fromkeys(terms))
                return unique_terms[:max_terms]
                
        except Exception as e:
            logger.error(f"Error extracting linkable terms: {e}")
            return []
    
    async def generate_internal_links(
        self, content: str, related_articles: List[Dict]
    ) -> str:
        """
        Generate internal links within content to related articles.
        
        Args:
            content: Content to add links to
            related_articles: List of related articles with title and url
            
        Returns:
            Content with internal links added
        """
        try:
            if not content or not related_articles:
                return content
            
            linked_content = content
            links_added = set()  # Track what we've already linked
            
            for article in related_articles:
                title = article.get('title', '')
                url = article.get('url', '')
                
                if not title or not url or title in links_added:
                    continue
                
                # Extract key terms from title
                key_terms = self._extract_key_terms(title)
                
                for term in key_terms:
                    if term in links_added:
                        continue
                    
                    # Create link pattern (case-insensitive)
                    # Only link first occurrence of each term
                    pattern = rf'\b({re.escape(term)})\b'
                    
                    # Check if term exists in content
                    if re.search(pattern, linked_content, re.IGNORECASE):
                        # Replace first occurrence with link
                        replacement = f'[\\1]({url})'
                        linked_content = re.sub(
                            pattern, 
                            replacement, 
                            linked_content, 
                            count=1, 
                            flags=re.IGNORECASE
                        )
                        links_added.add(term)
                        break  # Only one link per article
            
            return linked_content
            
        except Exception as e:
            logger.error(f"Error generating internal links: {e}")
            return content
    
    async def suggest_tags(
        self, content: str, existing_tags: List[str] = None
    ) -> List[str]:
        """
        Suggest tags based on content analysis.
        
        Args:
            content: Content to analyze
            existing_tags: Current tags to augment
            
        Returns:
            List of suggested tags
        """
        try:
            if not content:
                return existing_tags or []
            
            # Extract key terms
            terms = await self.extract_linkable_terms(content, max_terms=30)
            
            # Filter terms to make good tags
            suggested_tags = []
            for term in terms:
                # Skip if too short or too long
                if len(term) < 3 or len(term) > 30:
                    continue
                    
                # Skip if contains special characters
                if re.search(r'[^\w\s-]', term):
                    continue
                    
                # Normalize for tag format (lowercase, hyphenated)
                tag = term.lower().replace(' ', '-')
                
                # Skip if already in existing tags
                if existing_tags and tag in existing_tags:
                    continue
                    
                suggested_tags.append(tag)
            
            # Combine with existing tags
            if existing_tags:
                all_tags = existing_tags + suggested_tags[:10]
            else:
                all_tags = suggested_tags[:15]
            
            # Deduplicate while preserving order
            seen = set()
            unique_tags = []
            for tag in all_tags:
                if tag not in seen:
                    seen.add(tag)
                    unique_tags.append(tag)
            
            return unique_tags
            
        except Exception as e:
            logger.error(f"Error suggesting tags: {e}")
            return existing_tags or []
    
    async def create_content_map(
        self, session: AsyncSession
    ) -> Dict:
        """
        Create a map of all content relationships.
        
        Returns:
            Dictionary mapping article IDs to their relationships
        """
        try:
            # Build similarity index first
            await self.build_similarity_index(session)
            
            content_map = {}
            
            # Get all articles
            query = select(Article).where(
                Article.cleaned_content.is_not(None),
                Article.status == "completed"
            )
            result = await session.execute(query)
            articles = result.scalars().all()
            
            for article in articles:
                article_id = str(article.id)
                
                # Find related articles
                related = await self.find_related_articles(
                    article_id, 
                    session, 
                    max_results=10
                )
                
                # Extract relationships
                content_map[article_id] = {
                    'title': article.title,
                    'url': article.url,
                    'tags': article.tags or [],
                    'categories': article.categories or [],
                    'related_articles': related,
                    'link_count': len(related),
                    'content_type': article.content_type.value if article.content_type else 'article'
                }
            
            logger.info(f"Created content map for {len(content_map)} articles")
            return content_map
            
        except Exception as e:
            logger.error(f"Error creating content map: {e}")
            return {}

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
