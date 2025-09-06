"""
SimHash implementation for near-duplicate detection.

Provides efficient similarity detection with configurable hamming distance thresholds.
"""

import hashlib
import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from simhash import Simhash, SimhashIndex

from ..config import config

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
        self.index = SimhashIndex([], k=k)
        self.content_hashes: Dict[str, Simhash] = {}
        self.article_ids: Dict[str, str] = {}  # simhash -> article_id mapping
    
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
            
            # Generate SimHash
            simhash = Simhash(processed_content)
            simhash_str = format(simhash.value, 'x')
            
            # Store in index
            self.index.add(simhash_str, simhash)
            self.content_hashes[simhash_str] = simhash
            self.article_ids[simhash_str] = article_id
            
            logger.debug(f"Added content to SimHash index: {article_id}")
            return simhash_str
            
        except Exception as e:
            logger.error(f"Error adding content to SimHash index: {e}")
            raise
    
    def find_duplicates(self, content: str, exclude_id: Optional[str] = None) -> List[Tuple[str, float]]:
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
            
            # Generate SimHash
            query_simhash = Simhash(processed_content)
            
            # Find similar hashes
            similar_hashes = self.index.get_near_dups(query_simhash)
            
            # Calculate similarity scores
            duplicates = []
            for simhash_str in similar_hashes:
                if simhash_str in self.article_ids:
                    article_id = self.article_ids[simhash_str]
                    
                    # Skip excluded article
                    if exclude_id and article_id == exclude_id:
                        continue
                    
                    # Calculate similarity score
                    stored_simhash = self.content_hashes[simhash_str]
                    hamming_distance = query_simhash.distance(stored_simhash)
                    similarity = 1.0 - (hamming_distance / self.hash_size)
                    
                    duplicates.append((article_id, similarity))
            
            # Sort by similarity score (highest first)
            duplicates.sort(key=lambda x: x[1], reverse=True)
            
            return duplicates
            
        except Exception as e:
            logger.error(f"Error finding duplicates: {e}")
            return []
    
    def check_duplicate(self, content: str, threshold: float = 0.9) -> Optional[Tuple[str, float]]:
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
            # Find simhash for article ID
            simhash_str = None
            for hash_str, stored_id in self.article_ids.items():
                if stored_id == article_id:
                    simhash_str = hash_str
                    break
            
            if not simhash_str:
                logger.warning(f"Article ID not found in index: {article_id}")
                return False
            
            # Remove from index
            simhash = self.content_hashes[simhash_str]
            self.index.delete(simhash_str, simhash)
            
            # Clean up mappings
            del self.content_hashes[simhash_str]
            del self.article_ids[simhash_str]
            
            logger.debug(f"Removed content from SimHash index: {article_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing content from index: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get statistics about the deduplication index."""
        return {
            'total_items': len(self.content_hashes),
            'k_threshold': self.k,
            'hash_size': self.hash_size,
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
            # Convert to lowercase
            processed = content.lower()
            
            # Remove extra whitespace
            processed = re.sub(r'\s+', ' ', processed)
            
            # Remove common stop words for better discrimination
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
                'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                'would', 'should', 'could', 'can', 'may', 'might', 'must', 'this',
                'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
            }
            
            # Tokenize and filter stop words
            words = processed.split()
            filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
            
            # Return processed content
            return ' '.join(filtered_words)
            
        except Exception as e:
            logger.error(f"Error preprocessing content: {e}")
            return content
    
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
            normalized = re.sub(r'\s+', ' ', content.strip().lower())
            
            # Calculate SHA-256 hash
            content_hash = hashlib.sha256(normalized.encode('utf-8')).hexdigest()
            
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
                article_id = article.get('id')
                content = article.get('content', '')
                
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
            
            for simhash_str, article_id in self.article_ids.items():
                if article_id in visited:
                    continue
                
                # Find all articles similar to this one
                simhash = self.content_hashes[simhash_str]
                similar_hashes = self.index.get_near_dups(simhash)
                
                # Build cluster
                cluster = []
                for similar_hash in similar_hashes:
                    if similar_hash in self.article_ids:
                        similar_id = self.article_ids[similar_hash]
                        if similar_id not in visited:
                            cluster.append(similar_id)
                            visited.add(similar_id)
                
                # Add cluster if it meets minimum size
                if len(cluster) >= min_cluster_size:
                    clusters.append(cluster)
            
            logger.info(f"Found {len(clusters)} duplicate clusters")
            return clusters
            
        except Exception as e:
            logger.error(f"Error finding duplicate clusters: {e}")
            return []
    
    def export_index(self) -> Dict:
        """Export index data for persistence."""
        return {
            'content_hashes': {k: v.value for k, v in self.content_hashes.items()},
            'article_ids': self.article_ids.copy(),
            'k': self.k,
            'hash_size': self.hash_size,
        }
    
    def import_index(self, data: Dict) -> None:
        """Import index data from persistence."""
        try:
            self.k = data.get('k', self.k)
            self.hash_size = data.get('hash_size', self.hash_size)
            
            # Rebuild index
            self.index = SimhashIndex([], k=self.k)
            self.content_hashes = {}
            self.article_ids = data.get('article_ids', {})
            
            # Restore hashes
            for simhash_str, hash_value in data.get('content_hashes', {}).items():
                simhash = Simhash(hash_value)
                self.content_hashes[simhash_str] = simhash
                self.index.add(simhash_str, simhash)
            
            logger.info(f"Imported SimHash index with {len(self.content_hashes)} items")
            
        except Exception as e:
            logger.error(f"Error importing index: {e}")
            raise