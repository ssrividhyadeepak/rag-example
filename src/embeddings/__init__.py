"""
Embeddings Package

This package provides local embedding generation capabilities using sentence-transformers.
No external API calls required - completely offline after initial model download.

Components:
- embedding_generator: Generate embeddings from text using local models
- batch_processor: Process multiple texts efficiently
- similarity_search: Find similar embeddings using cosine similarity
- cache_manager: Cache embeddings for performance

Models supported:
- all-MiniLM-L6-v2: Fast, 384 dimensions, 22MB
- all-mpnet-base-v2: Higher quality, 768 dimensions, 420MB
"""

from .embedding_generator import EmbeddingGenerator
from .batch_processor import BatchProcessor
from .similarity_search import SimilaritySearch
from .cache_manager import CacheManager

__all__ = ['EmbeddingGenerator', 'BatchProcessor', 'SimilaritySearch', 'CacheManager']