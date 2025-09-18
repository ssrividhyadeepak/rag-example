"""
Vector Store Package

This package provides persistent vector storage and similarity search capabilities
for the ATM RAG system using a hybrid approach:

- Local MongoDB: Rich metadata storage and filtering
- FAISS: Ultra-fast vector similarity search
- Local operation: No cloud dependencies

Components:
- local_mongo_store: MongoDB operations for logs and metadata
- faiss_index: FAISS vector index management
- hybrid_store: Combined MongoDB + FAISS operations
- schema: Database schema definitions

The hybrid approach gives us the best of both worlds:
- MongoDB: Complex queries, filtering, aggregations
- FAISS: Lightning-fast vector similarity search (<1ms)
"""

from .local_mongo_store import LocalMongoStore
from .faiss_index import FAISSIndexManager
from .hybrid_store import HybridVectorStore
from .schema import ATMLogSchema, VectorMetadataSchema

__all__ = [
    'LocalMongoStore',
    'FAISSIndexManager',
    'HybridVectorStore',
    'ATMLogSchema',
    'VectorMetadataSchema'
]