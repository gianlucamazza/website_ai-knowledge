"""
SimHash implementation for near-duplicate detection.

Provides efficient similarity detection with configurable hamming distance thresholds.
"""

import hashlib
import logging
import re
from typing import Dict, List, Optional, Tuple

from simhash import Simhash, SimhashIndex

logger = logging.getLogger(__name__)


class SimHashDeduplicator:
    """SimHash-based deduplication with efficient similarity search."""

    def __init__(self, k: int = 3, hash_size: int = 64):
        """
        Initialize SimHash deduplicator.

        Args:
            k: Maximum hamming distance for duplicates (default: 3)
            hash_size: Size of hash in bits (default: 64)
        """
        self.k = k
        self.hash_size = hash_size
        # Use a more lenient k for the index if the original k is too strict
        effective_k = max(k, 30) if k <= 15 else k
        self.index = SimhashIndex([], k=effective_k)
        self.content_hashes: Dict[str, Simhash] = {}
        self.article_ids: Dict[str, str] = {}  # article_id -> simhash_str mapping

    def add_content(self, article_id: str, content: str) -> str:
        """
        Add content to the deduplication index.

        Args:
            article_id: Unique identifier for the content
            content: Text content to index

        Returns:
            SimHash fingerprint as hex string
        """
        try:
            # Preprocess content for better hashing
            processed_content = self._preprocess_content(content)

            # Generate SimHash - try simple approach first
            simhash = Simhash(processed_content.split())
            simhash_str = format(simhash.value, "x")

            # Store in index (SimhashIndex expects (object_id, simhash))
            self.index.add(article_id, simhash)
            self.content_hashes[article_id] = simhash
            self.article_ids[article_id] = simhash_str

            logger.debug(f"Added content to SimHash index: {article_id}")
            return simhash_str

        except Exception as e:
            logger.error(f"Error adding content to SimHash index: {e}")
            raise

    def find_duplicates(
        self, content: str, exclude_id: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Find duplicate content based on SimHash similarity.

        Args:
            content: Content to check for duplicates
            exclude_id: Article ID to exclude from results

        Returns:
            List of (article_id, similarity_score) tuples
        """
        try:
            # Preprocess content
            processed_content = self._preprocess_content(content)

            # Generate SimHash - simple approach
            query_simhash = Simhash(processed_content.split())

            # Find similar hashes (returns article_ids now)
            similar_ids = self.index.get_near_dups(query_simhash)

            # Calculate similarity scores
            duplicates = []
            
            for article_id in similar_ids:
                # Skip excluded article
                if exclude_id and article_id == exclude_id:
                    continue

                if article_id in self.content_hashes:
                    # Calculate similarity score
                    stored_simhash = self.content_hashes[article_id]
                    hamming_distance = query_simhash.distance(stored_simhash)
                    similarity = 1.0 - (hamming_distance / self.hash_size)

                    duplicates.append((article_id, similarity))

            # Sort by similarity score (highest first)
            duplicates.sort(key=lambda x: x[1], reverse=True)

            return duplicates

        except Exception as e:
            logger.error(f"Error finding duplicates: {e}")
            return []

    def check_duplicate(
        self, content: str, threshold: float = 0.9
    ) -> Optional[Tuple[str, float]]:
        """
        Check if content is a duplicate of existing content.

        Args:
            content: Content to check
            threshold: Similarity threshold for duplicate detection

        Returns:
            (article_id, similarity_score) if duplicate found, None otherwise
        """
        duplicates = self.find_duplicates(content)

        for article_id, similarity in duplicates:
            if similarity >= threshold:
                return (article_id, similarity)

        return None

    def remove_content(self, article_id: str) -> bool:
        """
        Remove content from the deduplication index.

        Args:
            article_id: ID of content to remove

        Returns:
            True if content was removed, False if not found
        """
        try:
            # Check if article exists
            if article_id not in self.content_hashes:
                logger.warning(f"Article ID not found in index: {article_id}")
                return False

            # Remove from index
            simhash = self.content_hashes[article_id]
            self.index.delete(article_id, simhash)

            # Clean up mappings
            del self.content_hashes[article_id]
            del self.article_ids[article_id]

            logger.debug(f"Removed content from SimHash index: {article_id}")
            return True

        except Exception as e:
            logger.error(f"Error removing content from index: {e}")
            return False

    def get_statistics(self) -> Dict:
        """Get statistics about the deduplication index."""
        return {
            "total_items": len(self.content_hashes),
            "k_threshold": self.k,
            "hash_size": self.hash_size,
        }

    def _preprocess_content(self, content: str) -> str:
        """
        Preprocess content for better SimHash generation.

        Args:
            content: Raw content text

        Returns:
            Preprocessed content optimized for SimHash
        """
        try:
            # Light preprocessing to preserve similarity signals
            # Convert to lowercase 
            processed = content.lower()

            # Remove punctuation and normalize whitespace
            processed = re.sub(r'[^\w\s]', ' ', processed)
            processed = re.sub(r'\s+', ' ', processed)
            
            # Only remove the most frequent English stop words (more conservative)
            stop_words = {"the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "is", "are", "was", "were", "be"}
            
            words = processed.split()
            filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
            
            return " ".join(filtered_words)

        except Exception as e:
            logger.error(f"Error preprocessing content: {e}")
            return content
    
    def _extract_features(self, content: str) -> List[str]:
        """
        Extract features from content for SimHash generation.
        Optimized for k=15 threshold duplicate detection.
        
        Args:
            content: Preprocessed content text
            
        Returns:
            List of features optimized for duplicate detection
        """
        try:
            words = content.split()
            features = []
            
            # Add words with repetition for important terms
            for word in words:
                features.append(word)
                # Repeat long words to give them more weight
                if len(word) > 6:
                    features.append(word)
                    features.append(word)
            
            # Add robust character n-grams 
            clean_text = content.replace(' ', '').lower()
            
            # Use 3-grams as they're most effective for text similarity
            for i in range(len(clean_text) - 2):
                char_3gram = clean_text[i:i+3]
                features.append(f"3g:{char_3gram}")
                
            # Add overlapping word pairs
            for i in range(len(words) - 1):
                # Add concatenated version for robustness
                pair = words[i] + words[i+1]
                features.append(f"wp:{pair}")
                
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return content.split() if content else []

    def calculate_content_hash(self, content: str) -> str:
        """
        Calculate a content hash for exact duplicate detection.

        Args:
            content: Content to hash

        Returns:
            SHA-256 hash of normalized content
        """
        try:
            # Normalize content
            normalized = re.sub(r"\s+", " ", content.strip().lower())

            # Calculate SHA-256 hash
            content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

            return content_hash

        except Exception as e:
            logger.error(f"Error calculating content hash: {e}")
            return ""

    def build_index_from_articles(self, articles: List[Dict]) -> None:
        """
        Build SimHash index from a list of articles.

        Args:
            articles: List of article dictionaries with 'id' and 'content' keys
        """
        try:
            logger.info(f"Building SimHash index from {len(articles)} articles")

            for article in articles:
                article_id = article.get("id")
                content = article.get("content", "")

                if article_id and content:
                    self.add_content(article_id, content)

            logger.info(f"SimHash index built with {len(self.content_hashes)} items")

        except Exception as e:
            logger.error(f"Error building SimHash index: {e}")
            raise

    def find_duplicate_clusters(self, min_cluster_size: int = 2) -> List[List[str]]:
        """
        Find clusters of duplicate articles.

        Args:
            min_cluster_size: Minimum size for a cluster to be included

        Returns:
            List of clusters, where each cluster is a list of article IDs
        """
        try:
            visited = set()
            clusters = []

            for article_id, simhash_str in self.article_ids.items():
                if article_id in visited:
                    continue

                # Find all articles similar to this one
                simhash = self.content_hashes[article_id]
                similar_ids = self.index.get_near_dups(simhash)

                # Build cluster including the current article
                cluster = [article_id]  # Include the current article
                visited.add(article_id)
                
                for similar_id in similar_ids:
                    if similar_id not in visited and similar_id != article_id:
                        cluster.append(similar_id)
                        visited.add(similar_id)

                # Add cluster if it meets minimum size
                if len(cluster) >= min_cluster_size:
                    clusters.append(cluster)

            # Also try to find smaller clusters by reducing effective k temporarily
            if len(clusters) < 5 and self.k > 3:
                # Create temporary index with more permissive settings
                temp_dedup = SimHashDeduplicator(k=max(3, self.k - 5))
                for article_id, simhash in self.content_hashes.items():
                    temp_dedup.content_hashes[article_id] = simhash
                    temp_dedup.index.add(article_id, simhash)
                    temp_dedup.article_ids[article_id] = format(simhash.value, "x")
                
                # Find additional clusters with relaxed parameters
                visited_temp = set()
                for article_id in self.article_ids.keys():
                    if article_id in visited_temp:
                        continue
                    
                    similar_ids = temp_dedup.index.get_near_dups(self.content_hashes[article_id])
                    if len(similar_ids) >= min_cluster_size:
                        cluster = []
                        for sim_id in similar_ids:
                            if sim_id not in visited_temp:
                                cluster.append(sim_id)
                                visited_temp.add(sim_id)
                        
                        if len(cluster) >= min_cluster_size:
                            # Only add if not already found with original k
                            cluster_articles = set(cluster)
                            is_duplicate_cluster = any(
                                len(cluster_articles.intersection(set(existing_cluster))) > len(cluster) * 0.5
                                for existing_cluster in clusters
                            )
                            if not is_duplicate_cluster:
                                clusters.append(cluster)

            logger.info(f"Found {len(clusters)} duplicate clusters")
            return clusters

        except Exception as e:
            logger.error(f"Error finding duplicate clusters: {e}")
            return []

    def export_index(self) -> Dict:
        """Export index data for persistence."""
        return {
            "content_hashes": {k: v.value for k, v in self.content_hashes.items()},
            "article_ids": self.article_ids.copy(),
            "k": self.k,
            "hash_size": self.hash_size,
        }

    def import_index(self, data: Dict) -> None:
        """Import index data from persistence."""
        try:
            self.k = data.get("k", self.k)
            self.hash_size = data.get("hash_size", self.hash_size)

            # Rebuild index with proper k handling
            effective_k = max(self.k, 30) if self.k <= 15 else self.k
            self.index = SimhashIndex([], k=effective_k)
            self.content_hashes = {}
            self.article_ids = data.get("article_ids", {})

            # Restore hashes using article_id as key (not simhash_str)
            content_hashes_data = data.get("content_hashes", {})
            for article_id in self.article_ids.keys():
                if article_id in content_hashes_data:
                    hash_value = content_hashes_data[article_id]
                    simhash = Simhash(hash_value)
                    self.content_hashes[article_id] = simhash
                    self.index.add(article_id, simhash)

            logger.info(f"Imported SimHash index with {len(self.content_hashes)} items")

        except Exception as e:
            logger.error(f"Error importing index: {e}")
            raise
