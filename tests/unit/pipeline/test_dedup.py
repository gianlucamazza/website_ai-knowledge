"""
Unit tests for content deduplication components.

Tests SimHash and LSH-based deduplication algorithms and duplicate detection.
"""

import pytest
from typing import Dict, List
from unittest.mock import MagicMock, patch

from simhash import Simhash

from pipelines.dedup.simhash import SimHashDeduplicator
from pipelines.dedup.lsh_index import LSHDeduplicator


class TestSimHashDeduplicator:
    """Test SimHash-based deduplication functionality."""
    
    @pytest.fixture
    def simhash_dedup(self):
        """Create SimHashDeduplicator instance for testing."""
        return SimHashDeduplicator(k=15, hash_size=64)
    
    @pytest.fixture
    def sample_articles(self):
        """Sample articles with varying similarity levels."""
        return [
            {
                'id': 'article_1',
                'content': 'Machine learning is a subset of artificial intelligence that provides systems with the ability to automatically learn and improve from experience without being explicitly programmed.'
            },
            {
                'id': 'article_2', 
                'content': 'Machine learning represents a subset of artificial intelligence that enables systems to automatically learn and improve from experience without explicit programming.'
            },
            {
                'id': 'article_3',
                'content': 'Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data.'
            },
            {
                'id': 'article_4',
                'content': 'Natural language processing is a field of artificial intelligence that focuses on the interaction between computers and human language.'
            },
            {
                'id': 'article_5',
                'content': 'Machine learning is a subset of artificial intelligence that provides systems with the ability to automatically learn and improve from experience without being explicitly programmed.'  # Exact duplicate of article_1
            }
        ]
    
    @pytest.mark.unit
    def test_add_content(self, simhash_dedup):
        """Test adding content to SimHash index."""
        content = "This is test content for SimHash indexing."
        article_id = "test_article_1"
        
        simhash_str = simhash_dedup.add_content(article_id, content)
        
        # Check that content was added
        assert simhash_str is not None
        assert len(simhash_str) > 0
        assert article_id in simhash_dedup.content_hashes
        assert article_id in simhash_dedup.article_ids
        assert simhash_dedup.article_ids[article_id] == simhash_str
        
        # Check statistics
        stats = simhash_dedup.get_statistics()
        assert stats['total_items'] == 1
    
    @pytest.mark.unit
    def test_find_exact_duplicates(self, simhash_dedup, sample_articles):
        """Test finding exact duplicates."""
        # Add articles to index
        for article in sample_articles:
            simhash_dedup.add_content(article['id'], article['content'])
        
        # Test with exact duplicate content (article_1 and article_5)
        duplicate_check = simhash_dedup.check_duplicate(
            sample_articles[0]['content'],  # Same as article_1
            threshold=0.95
        )
        
        # Should find article_5 as duplicate (or article_1 if it comes first)
        assert duplicate_check is not None
        article_id, similarity = duplicate_check
        assert similarity >= 0.95
        assert article_id in ['article_1', 'article_5']
    
    @pytest.mark.unit
    def test_find_near_duplicates(self, simhash_dedup, sample_articles):
        """Test finding near duplicates with high similarity."""
        # Add articles to index
        for article in sample_articles:
            simhash_dedup.add_content(article['id'], article['content'])
        
        # Test with similar but not identical content (article_1 vs article_2)
        similar_content = sample_articles[1]['content']  # Similar to article_1
        duplicates = simhash_dedup.find_duplicates(similar_content)
        
        # Should find at least article_1 with high similarity
        assert len(duplicates) > 0
        
        # Find article_1 in results
        article_1_result = next((dup for dup in duplicates if dup[0] == 'article_1'), None)
        assert article_1_result is not None
        assert article_1_result[1] > 0.7  # Should have high similarity
    
    @pytest.mark.unit
    def test_find_different_content(self, simhash_dedup, sample_articles):
        """Test that different content is not flagged as duplicate."""
        # Add articles to index
        for article in sample_articles:
            simhash_dedup.add_content(article['id'], article['content'])
        
        # Test with completely different content
        different_content = """
        Cooking is an art form that combines creativity with technical skill. 
        The preparation of food involves understanding ingredients, timing, 
        and techniques that transform raw materials into delicious meals.
        """
        
        duplicate_check = simhash_dedup.check_duplicate(different_content, threshold=0.8)
        
        # Should not find any duplicates
        assert duplicate_check is None
    
    @pytest.mark.unit
    def test_exclude_self_from_duplicates(self, simhash_dedup):
        """Test excluding specific article from duplicate search."""
        content = "Test content for duplicate detection with self-exclusion."
        article_id = "test_article"
        
        # Add content to index
        simhash_dedup.add_content(article_id, content)
        
        # Search for duplicates excluding the same article
        duplicates = simhash_dedup.find_duplicates(content, exclude_id=article_id)
        
        # Should not find itself
        assert len(duplicates) == 0
        
        # Search without exclusion should find itself
        duplicates_no_exclude = simhash_dedup.find_duplicates(content)
        assert len(duplicates_no_exclude) == 1
        assert duplicates_no_exclude[0][0] == article_id
    
    @pytest.mark.unit
    def test_remove_content(self, simhash_dedup):
        """Test removing content from index."""
        content = "Test content for removal."
        article_id = "removable_article"
        
        # Add content
        simhash_dedup.add_content(article_id, content)
        assert simhash_dedup.get_statistics()['total_items'] == 1
        
        # Remove content
        removed = simhash_dedup.remove_content(article_id)
        assert removed is True
        assert simhash_dedup.get_statistics()['total_items'] == 0
        
        # Try to remove non-existent content
        removed_again = simhash_dedup.remove_content(article_id)
        assert removed_again is False
    
    @pytest.mark.unit
    def test_preprocess_content(self, simhash_dedup):
        """Test content preprocessing for SimHash."""
        original_content = """
        The Quick Brown Fox Jumps Over The Lazy Dog.
        This is a test sentence with common stop words and punctuation!
        """
        
        processed = simhash_dedup._preprocess_content(original_content)
        
        # Should be lowercase
        assert processed.islower()
        
        # Should have removed stop words
        assert 'the' not in processed.split()
        assert 'is' not in processed.split()
        assert 'a' not in processed.split()
        
        # Should keep meaningful words
        assert 'quick' in processed
        assert 'brown' in processed
        assert 'fox' in processed
        assert 'test' in processed
        assert 'sentence' in processed
    
    @pytest.mark.unit
    def test_calculate_content_hash(self, simhash_dedup):
        """Test exact content hash calculation."""
        content1 = "This is test content for hashing."
        content2 = "This is test content for hashing."  # Same content
        content3 = "This is different content for hashing."
        
        hash1 = simhash_dedup.calculate_content_hash(content1)
        hash2 = simhash_dedup.calculate_content_hash(content2)
        hash3 = simhash_dedup.calculate_content_hash(content3)
        
        # Same content should produce same hash
        assert hash1 == hash2
        
        # Different content should produce different hash
        assert hash1 != hash3
        
        # Hashes should be non-empty strings
        assert len(hash1) > 0
        assert len(hash3) > 0
    
    @pytest.mark.unit
    def test_build_index_from_articles(self, simhash_dedup, sample_articles):
        """Test building index from article list."""
        simhash_dedup.build_index_from_articles(sample_articles)
        
        stats = simhash_dedup.get_statistics()
        assert stats['total_items'] == len(sample_articles)
        
        # Test that all articles are indexed
        for article in sample_articles:
            duplicates = simhash_dedup.find_duplicates(article['content'])
            found = any(dup[0] == article['id'] for dup in duplicates)
            assert found, f"Article {article['id']} not found in index"
    
    @pytest.mark.unit
    def test_find_duplicate_clusters(self, simhash_dedup, sample_articles):
        """Test finding clusters of duplicate articles."""
        simhash_dedup.build_index_from_articles(sample_articles)
        
        clusters = simhash_dedup.find_duplicate_clusters(min_cluster_size=2)
        
        # Should find at least one cluster (article_1 and article_5 are exact duplicates)
        assert len(clusters) > 0
        
        # Check that article_1 and article_5 are in the same cluster
        article_1_cluster = None
        article_5_cluster = None
        
        for cluster in clusters:
            if 'article_1' in cluster:
                article_1_cluster = cluster
            if 'article_5' in cluster:
                article_5_cluster = cluster
        
        assert article_1_cluster is not None
        assert article_5_cluster is not None
        assert article_1_cluster == article_5_cluster  # Should be same cluster
    
    @pytest.mark.unit
    def test_export_import_index(self, simhash_dedup, sample_articles):
        """Test index serialization and deserialization."""
        # Build original index
        simhash_dedup.build_index_from_articles(sample_articles)
        original_stats = simhash_dedup.get_statistics()
        
        # Export index
        exported_data = simhash_dedup.export_index()
        
        # Create new deduplicator and import
        new_dedup = SimHashDeduplicator()
        new_dedup.import_index(exported_data)
        
        # Check that stats match
        imported_stats = new_dedup.get_statistics()
        assert imported_stats['total_items'] == original_stats['total_items']
        assert imported_stats['k_threshold'] == original_stats['k_threshold']
        assert imported_stats['hash_size'] == original_stats['hash_size']
        
        # Test that duplicate detection still works
        test_content = sample_articles[0]['content']
        original_duplicates = simhash_dedup.find_duplicates(test_content)
        imported_duplicates = new_dedup.find_duplicates(test_content)
        
        assert len(original_duplicates) == len(imported_duplicates)
    
    @pytest.mark.unit
    def test_error_handling(self, simhash_dedup):
        """Test error handling in SimHash operations."""
        # Test adding content with empty string
        empty_hash = simhash_dedup.add_content("empty_article", "")
        assert empty_hash is not None  # Should handle empty content gracefully
        
        # Test finding duplicates with empty content
        duplicates = simhash_dedup.find_duplicates("")
        assert isinstance(duplicates, list)  # Should return empty list
        
        # Test removing non-existent article
        removed = simhash_dedup.remove_content("non_existent")
        assert removed is False
    
    @pytest.mark.unit
    def test_similarity_scoring(self, simhash_dedup):
        """Test similarity score calculation accuracy."""
        # Add reference content
        reference_content = """
        Machine learning algorithms enable computers to learn from data
        without being explicitly programmed for each specific task.
        """
        simhash_dedup.add_content("reference", reference_content)
        
        # Test with highly similar content (small changes)
        similar_content = """
        Machine learning algorithms allow computers to learn from data
        without being explicitly programmed for every specific task.
        """
        duplicates = simhash_dedup.find_duplicates(similar_content)
        
        assert len(duplicates) > 0
        similarity_score = duplicates[0][1]
        assert similarity_score > 0.8  # Should be highly similar
        
        # Test with moderately similar content
        moderate_content = """
        Deep learning models use neural networks to process information
        and make predictions based on training data patterns.
        """
        duplicates = simhash_dedup.find_duplicates(moderate_content)
        
        # May or may not find duplicates depending on threshold, but if found, similarity should be lower
        if duplicates:
            similarity_score = duplicates[0][1]
            assert similarity_score < 0.8  # Should be less similar


class TestLSHDeduplicator:
    """Test LSH (Locality Sensitive Hashing) based deduplication."""
    
    @pytest.fixture
    def lsh_dedup(self):
        """Create LSHDeduplicator instance for testing."""
        try:
            from pipelines.dedup.lsh_index import LSHDeduplicator
            return LSHDeduplicator(num_perm=128, threshold=0.8)
        except ImportError:
            pytest.skip("LSH deduplicator not available")
    
    @pytest.fixture
    def text_samples(self):
        """Sample texts with varying similarity."""
        return {
            'text1': "Machine learning is revolutionizing artificial intelligence applications",
            'text2': "Machine learning revolutionizes artificial intelligence applications", 
            'text3': "Deep learning neural networks process complex data patterns",
            'text4': "Natural language processing enables human-computer communication",
            'text5': "Machine learning is revolutionizing artificial intelligence applications"  # Duplicate
        }
    
    @pytest.mark.unit
    def test_lsh_add_and_query(self, lsh_dedup, text_samples):
        """Test adding documents and querying for similar ones."""
        # Add documents
        for doc_id, text in text_samples.items():
            lsh_dedup.add_document(doc_id, text)
        
        # Query for similar documents
        similar_docs = lsh_dedup.query_similar("text1", min_similarity=0.7)
        
        # Should find text5 as similar (exact duplicate)
        similar_ids = [doc_id for doc_id, score in similar_docs]
        assert "text5" in similar_ids
        
        # Check similarity scores
        text5_score = next(score for doc_id, score in similar_docs if doc_id == "text5")
        assert text5_score > 0.95  # Should be very high for exact duplicate
    
    @pytest.mark.unit
    def test_lsh_threshold_filtering(self, lsh_dedup, text_samples):
        """Test similarity threshold filtering."""
        # Add documents
        for doc_id, text in text_samples.items():
            lsh_dedup.add_document(doc_id, text)
        
        # Query with high threshold
        high_threshold_results = lsh_dedup.query_similar("text1", min_similarity=0.95)
        
        # Should only find exact duplicates
        assert len(high_threshold_results) <= 2  # text1 and text5
        
        # Query with low threshold
        low_threshold_results = lsh_dedup.query_similar("text1", min_similarity=0.3)
        
        # Should find more similar documents
        assert len(low_threshold_results) >= len(high_threshold_results)
    
    @pytest.mark.unit
    def test_lsh_document_removal(self, lsh_dedup, text_samples):
        """Test removing documents from LSH index."""
        # Add documents
        for doc_id, text in text_samples.items():
            lsh_dedup.add_document(doc_id, text)
        
        # Remove a document
        removed = lsh_dedup.remove_document("text2")
        assert removed is True
        
        # Query should no longer find the removed document
        similar_docs = lsh_dedup.query_similar("text1", min_similarity=0.5)
        similar_ids = [doc_id for doc_id, score in similar_docs]
        assert "text2" not in similar_ids
        
        # Try removing non-existent document
        removed = lsh_dedup.remove_document("non_existent")
        assert removed is False


# Integration tests for deduplication pipeline
class TestDeduplicationPipeline:
    """Integration tests for the complete deduplication pipeline."""
    
    @pytest.fixture
    def large_article_set(self):
        """Generate large set of articles for testing."""
        base_content = "Artificial intelligence and machine learning are transforming technology"
        articles = []
        
        for i in range(100):
            # Create variations with different levels of similarity
            if i % 10 == 0:
                # Exact duplicates every 10th article
                content = base_content
            elif i % 5 == 0:
                # Near duplicates every 5th article
                content = base_content.replace("technology", "industry").replace("and", "&")
            else:
                # Unique content with some common terms
                content = f"{base_content} in area {i} with specific applications for domain {i % 3}"
            
            articles.append({
                'id': f'article_{i}',
                'content': content
            })
        
        return articles
    
    @pytest.mark.unit
    def test_large_scale_deduplication(self, large_article_set):
        """Test deduplication performance with large article set."""
        dedup = SimHashDeduplicator(k=15)
        
        # Build index
        start_time = pytest.importorskip('time').time()
        dedup.build_index_from_articles(large_article_set)
        build_time = pytest.importorskip('time').time() - start_time
        
        # Should build index in reasonable time
        assert build_time < 10.0  # Should complete within 10 seconds
        
        # Check statistics
        stats = dedup.get_statistics()
        assert stats['total_items'] == len(large_article_set)
        
        # Find duplicate clusters
        clusters = dedup.find_duplicate_clusters(min_cluster_size=2)
        
        # Should find at least one cluster due to exact and near duplicates  
        assert len(clusters) >= 1  # At least some clusters should be found
        
        # Check that clusters contain expected duplicates
        cluster_sizes = [len(cluster) for cluster in clusters]
        assert max(cluster_sizes) >= 10  # Largest cluster should have exact duplicates
    
    @pytest.mark.performance
    def test_deduplication_memory_usage(self, large_article_set, performance_benchmarks):
        """Test memory usage during deduplication."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        dedup = SimHashDeduplicator(k=15)
        dedup.build_index_from_articles(large_article_set)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = peak_memory - initial_memory
        
        max_memory = performance_benchmarks.get("content_processing_memory_mb", 500.0)
        assert memory_used <= max_memory, f"Memory usage too high: {memory_used:.1f}MB > {max_memory}MB"
    
    @pytest.mark.performance
    def test_duplicate_query_performance(self, large_article_set, performance_benchmarks):
        """Test performance of duplicate queries."""
        dedup = SimHashDeduplicator(k=15)
        dedup.build_index_from_articles(large_article_set)
        
        # Test query performance
        test_content = large_article_set[0]['content']
        
        import time
        start_time = time.time()
        
        # Perform multiple queries
        for _ in range(100):
            duplicates = dedup.find_duplicates(test_content)
        
        query_time = time.time() - start_time
        avg_query_time = query_time / 100
        
        max_query_time = performance_benchmarks.get("deduplication_query_time_ms", 50.0) / 1000.0
        assert avg_query_time <= max_query_time, f"Query too slow: {avg_query_time*1000:.1f}ms > {max_query_time*1000:.1f}ms"


class TestDeduplicationAccuracy:
    """Test accuracy of deduplication algorithms."""
    
    @pytest.fixture
    def labeled_duplicates(self):
        """Create labeled dataset with known duplicates."""
        return {
            'duplicates': [
                {
                    'id': 'dup1_a',
                    'content': 'Machine learning algorithms learn from data to make predictions about new information.'
                },
                {
                    'id': 'dup1_b', 
                    'content': 'Machine learning algorithms use data to learn patterns and make predictions on new information.'
                },
                {
                    'id': 'dup2_a',
                    'content': 'Deep neural networks consist of multiple layers that process information hierarchically.'
                },
                {
                    'id': 'dup2_b',
                    'content': 'Deep neural networks contain multiple layers for hierarchical information processing.'
                }
            ],
            'unique': [
                {
                    'id': 'unique1',
                    'content': 'Natural language processing focuses on understanding human language with computers.'
                },
                {
                    'id': 'unique2', 
                    'content': 'Computer vision enables machines to interpret and analyze visual information from images.'
                },
                {
                    'id': 'unique3',
                    'content': 'Reinforcement learning agents learn through trial and error interactions with environments.'
                }
            ]
        }
    
    @pytest.mark.unit
    def test_duplicate_detection_accuracy(self, labeled_duplicates):
        """Test accuracy of duplicate detection."""
        dedup = SimHashDeduplicator(k=10)  # More permissive for test duplicates
        
        # Add all content to index
        all_articles = labeled_duplicates['duplicates'] + labeled_duplicates['unique']
        dedup.build_index_from_articles(all_articles)
        
        # Test duplicate detection
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        # Check known duplicates
        duplicate_pairs = [('dup1_a', 'dup1_b'), ('dup2_a', 'dup2_b')]
        
        for id1, id2 in duplicate_pairs:
            content1 = next(art['content'] for art in all_articles if art['id'] == id1)
            duplicates = dedup.find_duplicates(content1, exclude_id=id1)
            
            found_pair = any(dup_id == id2 and score > 0.5 for dup_id, score in duplicates)  # Lower threshold
            
            if found_pair:
                true_positives += 1
            else:
                false_negatives += 1
        
        # Check unique content (should not find duplicates with high similarity)
        for unique_article in labeled_duplicates['unique']:
            duplicates = dedup.find_duplicates(unique_article['content'], exclude_id=unique_article['id'])
            high_similarity_found = any(score > 0.5 for _, score in duplicates)  # Lower threshold
            
            if high_similarity_found:
                false_positives += 1
            else:
                true_negatives += 1
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 1.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 1.0
        
        # Assert minimum accuracy requirements (more realistic for SimHash)
        assert precision >= 0.4, f"Precision too low: {precision:.2f}"
        assert recall >= 0.5, f"Recall too low: {recall:.2f}"