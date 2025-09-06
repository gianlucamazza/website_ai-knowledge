"""
Locality Sensitive Hashing (LSH) implementation using MinHash for efficient similarity search.

Provides scalable near-duplicate detection with configurable similarity thresholds
and sub-linear query performance.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from datasketch import MinHashLSH, MinHash

from ..config import config

logger = logging.getLogger(__name__)


class LSHIndex:
    """MinHash LSH index for efficient similarity search."""
    
    def __init__(self, threshold: float = 0.8, num_perm: int = 256):
        """
        Initialize LSH index.
        
        Args:
            threshold: Jaccard similarity threshold for duplicates
            num_perm: Number of permutations for MinHash (higher = more accurate)
        """
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.minhashes: Dict[str, MinHash] = {}
        self.content_metadata: Dict[str, Dict] = {}
    
    def add_content(self, article_id: str, content: str, metadata: Optional[Dict] = None) -> MinHash:
        """
        Add content to the LSH index.
        
        Args:
            article_id: Unique identifier for the content
            content: Text content to index
            metadata: Optional metadata about the content
            
        Returns:
            MinHash object for the content
        """
        try:
            # Generate tokens from content
            tokens = self._tokenize_content(content)
            
            if not tokens:
                logger.warning(f"No tokens generated for article {article_id}")
                return None
            
            # Create MinHash
            minhash = MinHash(num_perm=self.num_perm)
            for token in tokens:
                minhash.update(token.encode('utf8'))
            
            # Add to LSH index
            self.lsh.insert(article_id, minhash)
            
            # Store MinHash and metadata
            self.minhashes[article_id] = minhash
            self.content_metadata[article_id] = metadata or {}
            
            logger.debug(f"Added content to LSH index: {article_id}")
            return minhash
            
        except Exception as e:
            logger.error(f"Error adding content to LSH index: {e}")
            raise
    
    def find_duplicates(self, content: str, exclude_id: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Find duplicate content using LSH similarity search.
        
        Args:
            content: Content to check for duplicates
            exclude_id: Article ID to exclude from results
            
        Returns:
            List of (article_id, jaccard_similarity) tuples
        """
        try:
            # Generate tokens and MinHash for query content
            tokens = self._tokenize_content(content)
            if not tokens:
                return []
            
            query_minhash = MinHash(num_perm=self.num_perm)
            for token in tokens:
                query_minhash.update(token.encode('utf8'))
            
            # Query LSH index
            candidates = self.lsh.query(query_minhash)
            
            # Calculate exact Jaccard similarities
            duplicates = []
            for candidate_id in candidates:
                if exclude_id and candidate_id == exclude_id:
                    continue
                
                if candidate_id in self.minhashes:
                    candidate_minhash = self.minhashes[candidate_id]
                    jaccard_sim = query_minhash.jaccard(candidate_minhash)
                    duplicates.append((candidate_id, jaccard_sim))
            
            # Sort by similarity (highest first)
            duplicates.sort(key=lambda x: x[1], reverse=True)
            
            return duplicates
            
        except Exception as e:
            logger.error(f"Error finding duplicates in LSH index: {e}")
            return []
    
    def check_duplicate(self, content: str, threshold: Optional[float] = None) -> Optional[Tuple[str, float]]:
        """
        Check if content is a duplicate of existing content.
        
        Args:
            content: Content to check
            threshold: Custom similarity threshold (uses index threshold if None)
            
        Returns:
            (article_id, similarity) if duplicate found, None otherwise
        """
        similarity_threshold = threshold or self.threshold
        duplicates = self.find_duplicates(content)
        
        for article_id, similarity in duplicates:
            if similarity >= similarity_threshold:
                return (article_id, similarity)
        
        return None
    
    def remove_content(self, article_id: str) -> bool:
        """
        Remove content from the LSH index.
        
        Args:
            article_id: ID of content to remove
            
        Returns:
            True if content was removed, False if not found
        """
        try:
            if article_id not in self.minhashes:
                logger.warning(f"Article ID not found in LSH index: {article_id}")
                return False
            
            # Remove from LSH index
            minhash = self.minhashes[article_id]
            self.lsh.remove(article_id)
            
            # Clean up stored data
            del self.minhashes[article_id]
            if article_id in self.content_metadata:
                del self.content_metadata[article_id]
            
            logger.debug(f"Removed content from LSH index: {article_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing content from LSH index: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get statistics about the LSH index."""
        return {
            'total_items': len(self.minhashes),
            'threshold': self.threshold,
            'num_perm': self.num_perm,
            'index_size': len(self.lsh.keys),
        }
    
    def _tokenize_content(self, content: str) -> Set[str]:
        """
        Tokenize content for MinHash generation.
        
        Args:
            content: Text content to tokenize
            
        Returns:
            Set of tokens (shingles)
        """
        try:
            # Clean and normalize content
            import re
            content = content.lower()
            content = re.sub(r'[^\w\s]', ' ', content)
            content = re.sub(r'\s+', ' ', content.strip())
            
            if not content:
                return set()
            
            # Generate word shingles (n-grams)
            shingle_size = 3  # 3-word shingles
            words = content.split()
            
            if len(words) < shingle_size:
                # For short content, use individual words
                return set(words)
            
            # Generate shingles
            shingles = set()
            for i in range(len(words) - shingle_size + 1):
                shingle = ' '.join(words[i:i + shingle_size])
                shingles.add(shingle)
            
            # Also include individual words for better recall
            shingles.update(words)
            
            return shingles
            
        except Exception as e:
            logger.error(f"Error tokenizing content: {e}")
            return set()
    
    def build_index_from_articles(self, articles: List[Dict]) -> None:
        """
        Build LSH index from a list of articles.
        
        Args:
            articles: List of article dictionaries with 'id' and 'content' keys
        """
        try:
            logger.info(f"Building LSH index from {len(articles)} articles")
            
            for article in articles:
                article_id = article.get('id')
                content = article.get('content', '')
                metadata = {
                    'title': article.get('title', ''),
                    'url': article.get('url', ''),
                    'word_count': len(content.split()) if content else 0
                }
                
                if article_id and content:
                    self.add_content(article_id, content, metadata)
            
            logger.info(f"LSH index built with {len(self.minhashes)} items")
            
        except Exception as e:
            logger.error(f"Error building LSH index: {e}")
            raise
    
    def find_similar_clusters(self, min_cluster_size: int = 2) -> List[List[str]]:
        """
        Find clusters of similar articles using connected components.
        
        Args:
            min_cluster_size: Minimum size for a cluster to be included
            
        Returns:
            List of clusters, where each cluster is a list of article IDs
        """
        try:
            # Build similarity graph
            similarity_graph = {}
            processed = set()
            
            for article_id in self.minhashes:
                if article_id in processed:
                    continue
                
                # Find similar articles
                if article_id in self.content_metadata:
                    # Reconstruct content from metadata or use stored MinHash
                    similar_ids = []
                    query_minhash = self.minhashes[article_id]
                    
                    # Find candidates using LSH
                    candidates = self.lsh.query(query_minhash)
                    
                    for candidate_id in candidates:
                        if candidate_id != article_id and candidate_id in self.minhashes:
                            candidate_minhash = self.minhashes[candidate_id]
                            similarity = query_minhash.jaccard(candidate_minhash)
                            
                            if similarity >= self.threshold:
                                similar_ids.append(candidate_id)
                    
                    # Add to similarity graph
                    similarity_graph[article_id] = similar_ids
                    processed.add(article_id)
            
            # Find connected components (clusters)
            clusters = self._find_connected_components(similarity_graph)
            
            # Filter by minimum cluster size
            filtered_clusters = [cluster for cluster in clusters if len(cluster) >= min_cluster_size]
            
            logger.info(f"Found {len(filtered_clusters)} similarity clusters")
            return filtered_clusters
            
        except Exception as e:
            logger.error(f"Error finding similarity clusters: {e}")
            return []
    
    def _find_connected_components(self, graph: Dict[str, List[str]]) -> List[List[str]]:
        """Find connected components in similarity graph."""
        visited = set()
        components = []
        
        def dfs(node: str, component: List[str]):
            if node in visited:
                return
            
            visited.add(node)
            component.append(node)
            
            # Visit neighbors
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, component)
        
        # Find all connected components
        for node in graph:
            if node not in visited:
                component = []
                dfs(node, component)
                if component:
                    components.append(component)
        
        return components
    
    def save_index(self, filepath: str) -> bool:
        """
        Save LSH index to disk.
        
        Args:
            filepath: Path to save the index
            
        Returns:
            True if successful, False otherwise
        """
        try:
            index_data = {
                'threshold': self.threshold,
                'num_perm': self.num_perm,
                'minhashes': self.minhashes,
                'content_metadata': self.content_metadata,
                'lsh_keys': list(self.lsh.keys),
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(index_data, f)
            
            logger.info(f"LSH index saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving LSH index: {e}")
            return False
    
    def load_index(self, filepath: str) -> bool:
        """
        Load LSH index from disk.
        
        Args:
            filepath: Path to load the index from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not Path(filepath).exists():
                logger.warning(f"LSH index file not found: {filepath}")
                return False
            
            with open(filepath, 'rb') as f:
                index_data = pickle.load(f)
            
            # Restore index configuration
            self.threshold = index_data['threshold']
            self.num_perm = index_data['num_perm']
            
            # Restore data
            self.minhashes = index_data['minhashes']
            self.content_metadata = index_data['content_metadata']
            
            # Rebuild LSH index
            self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
            
            for article_id, minhash in self.minhashes.items():
                self.lsh.insert(article_id, minhash)
            
            logger.info(f"LSH index loaded from {filepath} with {len(self.minhashes)} items")
            return True
            
        except Exception as e:
            logger.error(f"Error loading LSH index: {e}")
            return False
    
    def optimize_threshold(self, articles: List[Dict], target_precision: float = 0.98) -> float:
        """
        Optimize similarity threshold for target precision.
        
        Args:
            articles: Sample articles for optimization
            target_precision: Target precision for duplicate detection
            
        Returns:
            Optimized threshold value
        """
        try:
            if len(articles) < 10:
                logger.warning("Need at least 10 articles for threshold optimization")
                return self.threshold
            
            # Test different thresholds
            thresholds = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            best_threshold = self.threshold
            best_precision = 0.0
            
            for threshold in thresholds:
                # Create temporary index with this threshold
                temp_index = LSHIndex(threshold=threshold, num_perm=self.num_perm)
                temp_index.build_index_from_articles(articles)
                
                # Test precision on a subset
                test_articles = articles[:min(50, len(articles))]
                precision = self._calculate_precision(temp_index, test_articles)
                
                logger.info(f"Threshold {threshold}: Precision = {precision:.3f}")
                
                if precision >= target_precision and precision > best_precision:
                    best_precision = precision
                    best_threshold = threshold
            
            logger.info(f"Optimized threshold: {best_threshold} (precision: {best_precision:.3f})")
            return best_threshold
            
        except Exception as e:
            logger.error(f"Error optimizing threshold: {e}")
            return self.threshold
    
    def _calculate_precision(self, index: 'LSHIndex', test_articles: List[Dict]) -> float:
        """Calculate precision for duplicate detection."""
        try:
            true_positives = 0
            false_positives = 0
            
            for i, article in enumerate(test_articles):
                content = article.get('content', '')
                if not content:
                    continue
                
                duplicates = index.find_duplicates(content, exclude_id=article.get('id'))
                
                for dup_id, similarity in duplicates:
                    # Simple heuristic: consider it a true positive if similarity > 0.9
                    # In practice, you'd want manual validation or ground truth data
                    if similarity > 0.9:
                        true_positives += 1
                    else:
                        false_positives += 1
            
            total_positives = true_positives + false_positives
            if total_positives == 0:
                return 1.0  # No duplicates found, assume perfect precision
            
            return true_positives / total_positives
            
        except Exception as e:
            logger.error(f"Error calculating precision: {e}")
            return 0.0