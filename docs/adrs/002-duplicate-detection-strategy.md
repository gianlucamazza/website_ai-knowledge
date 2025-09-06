# ADR-002: Duplicate Detection Strategy

## Status
Accepted

## Context

The content pipeline must detect near-duplicate content across multiple sources to:
- Prevent publishing redundant information
- Identify content that should be merged
- Maintain content quality standards
- Comply with copyright and attribution requirements

Requirements:
- **Accuracy**: <2% false positives (incorrectly flagged as duplicates)
- **Recall**: >95% true duplicates detected
- **Performance**: Process 10,000 items in <10 minutes
- **Scalability**: Handle millions of items in the database
- **Incremental**: Support real-time duplicate detection for new content

Challenges:
- Content may be paraphrased or reformatted
- Different sources may use different terminology
- HTML formatting and metadata noise
- Need to detect both exact and semantic duplicates

## Decision

We will implement a **hybrid duplicate detection system** using:

1. **SimHash for Near-Duplicate Detection**
   - Generate 64-bit SimHash fingerprints for each content item
   - Use Hamming distance to measure similarity
   - Threshold: Hamming distance ≤ 3 bits for duplicates

2. **MinHash LSH for Jaccard Similarity**
   - Create MinHash signatures with 128 hash functions
   - Use Locality-Sensitive Hashing (LSH) for efficient similarity search
   - Threshold: Jaccard similarity ≥ 0.7 for duplicates

3. **Semantic Similarity (Optional Enhancement)**
   - Use sentence embeddings for semantic comparison
   - Apply only to high-confidence near-duplicates for final validation

### Implementation Architecture

```python
from datasketch import MinHashLSH, MinHash
import hashlib
import re

class DuplicateDetector:
    def __init__(self):
        self.lsh = MinHashLSH(threshold=0.7, num_perm=128)
        self.simhash_index = {}  # item_id -> simhash
        
    def generate_simhash(self, text: str) -> int:
        """Generate SimHash fingerprint"""
        tokens = self.tokenize(text)
        hash_bits = [0] * 64
        
        for token in tokens:
            token_hash = int(hashlib.md5(token.encode()).hexdigest(), 16)
            for i in range(64):
                if token_hash & (1 << i):
                    hash_bits[i] += 1
                else:
                    hash_bits[i] -= 1
        
        fingerprint = 0
        for i, bit in enumerate(hash_bits):
            if bit > 0:
                fingerprint |= 1 << i
        
        return fingerprint
    
    def generate_minhash(self, text: str) -> MinHash:
        """Generate MinHash signature"""
        tokens = set(self.tokenize(text))
        minhash = MinHash(num_perm=128)
        for token in tokens:
            minhash.update(token.encode('utf8'))
        return minhash
    
    def find_duplicates(self, item_id: str, content: str) -> List[DuplicateMatch]:
        duplicates = []
        
        # Step 1: SimHash detection
        simhash = self.generate_simhash(content)
        for existing_id, existing_simhash in self.simhash_index.items():
            hamming_distance = bin(simhash ^ existing_simhash).count('1')
            if hamming_distance <= 3:
                duplicates.append(DuplicateMatch(
                    item_id=existing_id,
                    similarity_score=1 - (hamming_distance / 64),
                    method='simhash'
                ))
        
        # Step 2: MinHash LSH detection
        minhash = self.generate_minhash(content)
        lsh_candidates = self.lsh.query(minhash)
        for candidate_id in lsh_candidates:
            if candidate_id != item_id:
                candidate_minhash = self.get_minhash(candidate_id)
                jaccard_sim = minhash.jaccard(candidate_minhash)
                if jaccard_sim >= 0.7:
                    duplicates.append(DuplicateMatch(
                        item_id=candidate_id,
                        similarity_score=jaccard_sim,
                        method='minhash_lsh'
                    ))
        
        # Store new item
        self.simhash_index[item_id] = simhash
        self.lsh.insert(item_id, minhash)
        
        return self.deduplicate_matches(duplicates)
```

### Text Preprocessing Pipeline

```python
def preprocess_content(self, html: str) -> str:
    """Normalize content for duplicate detection"""
    # Remove HTML tags
    text = self.html_to_text(html)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs, emails, and numbers
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation except sentence boundaries
    text = re.sub(r'[^\w\s\.\!\?]', '', text)
    
    return text

def tokenize(self, text: str) -> List[str]:
    """Generate tokens for hashing"""
    # Word-level tokens
    words = text.split()
    
    # Add character n-grams for better fuzzy matching
    char_ngrams = []
    for n in [3, 4, 5]:
        for i in range(len(text) - n + 1):
            char_ngrams.append(text[i:i+n])
    
    return words + char_ngrams
```

## Consequences

### Positive
- **High Accuracy**: Dual approach reduces both false positives and negatives
- **Scalability**: LSH enables efficient similarity search in large datasets
- **Performance**: SimHash provides fast exact and near-duplicate detection
- **Flexibility**: Can tune thresholds for different content types
- **Incremental**: Supports real-time detection for new content

### Negative
- **Memory Usage**: Storing hash indices requires significant memory
- **Complexity**: Dual algorithm approach increases implementation complexity
- **Tuning Required**: Thresholds need careful calibration for different content types
- **False Positives**: Short content may trigger false positives

### Risk Mitigation

1. **Threshold Calibration**
   ```python
   # Content-type specific thresholds
   THRESHOLDS = {
       'glossary': {'simhash': 2, 'jaccard': 0.8},
       'article': {'simhash': 3, 'jaccard': 0.7},
       'news': {'simhash': 4, 'jaccard': 0.6}
   }
   ```

2. **Memory Management**
   ```python
   # LRU cache for hash indices
   from functools import lru_cache
   
   class MemoryManagedDetector(DuplicateDetector):
       @lru_cache(maxsize=10000)
       def get_cached_simhash(self, content_hash: str) -> int:
           return self.generate_simhash(content_hash)
   ```

3. **Quality Monitoring**
   ```python
   async def validate_duplicate_detection():
       # Regular validation against manually labeled dataset
       test_cases = await load_test_cases()
       results = await run_duplicate_detection(test_cases)
       
       precision = calculate_precision(results)
       recall = calculate_recall(results)
       
       if precision < 0.98 or recall < 0.95:
           await alert_quality_team("Duplicate detection quality degraded")
   ```

## Alternatives Considered

### 1. Exact Hash Matching
**Pros**: Perfect precision, very fast
**Cons**: Misses near-duplicates, too strict for content variations
**Rejected**: Insufficient for content with minor variations

### 2. TF-IDF + Cosine Similarity
**Pros**: Good for semantic similarity, well understood
**Cons**: Computationally expensive, poor scalability
**Rejected**: Doesn't scale to required throughput

### 3. Deep Learning Embeddings
**Pros**: Excellent semantic understanding, handles paraphrasing
**Cons**: High computational cost, requires GPU resources, complex deployment
**Rejected**: Overkill for current requirements, too expensive

### 4. Bloom Filters
**Pros**: Memory efficient, very fast lookups
**Cons**: Only handles exact matches, no similarity scoring
**Rejected**: Cannot detect near-duplicates

## Implementation Roadmap

### Phase 1: Basic SimHash (Week 1-2)
- Implement SimHash generation and indexing
- Create basic duplicate detection API
- Add PostgreSQL storage for simhash values

### Phase 2: MinHash LSH (Week 3-4)
- Implement MinHash signature generation
- Add LSH indexing with configurable thresholds
- Create unified duplicate detection interface

### Phase 3: Optimization (Week 5-6)
- Add content-type specific thresholds
- Implement memory management and caching
- Add comprehensive monitoring and alerting

### Phase 4: Enhancement (Week 7-8)
- Add semantic similarity validation for edge cases
- Implement automated threshold tuning
- Create duplicate resolution workflows

## Monitoring and Metrics

```python
# Key metrics to track
METRICS = [
    'duplicate_detection_latency_p95',
    'duplicate_detection_accuracy',
    'false_positive_rate',
    'false_negative_rate',
    'memory_usage_mb',
    'throughput_items_per_second'
]

# Quality alerts
ALERTS = [
    {'metric': 'false_positive_rate', 'threshold': 0.02, 'severity': 'critical'},
    {'metric': 'false_negative_rate', 'threshold': 0.05, 'severity': 'warning'},
    {'metric': 'duplicate_detection_latency_p95', 'threshold': 1000, 'severity': 'warning'}
]
```

## Review Date
This ADR should be reviewed in 3 months (December 2024) after collecting production metrics and user feedback on duplicate detection quality.