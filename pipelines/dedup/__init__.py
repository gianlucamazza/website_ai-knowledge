"""
Content deduplication module using SimHash and MinHash LSH.

Provides efficient near-duplicate detection with configurable similarity thresholds
and false positive rates below 2%.
"""

from .lsh_index import LSHIndex
from .simhash import SimHashDeduplicator

__all__ = ["LSHIndex", "SimHashDeduplicator"]
