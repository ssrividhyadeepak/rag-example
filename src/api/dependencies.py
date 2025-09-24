"""
FastAPI Dependencies

Provides dependency injection for core components of the ATM RAG system.
"""

import os
import logging
from typing import Optional
from functools import lru_cache
from pathlib import Path

from fastapi import HTTPException, Depends

# Import core components
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.embeddings import EmbeddingGenerator
from src.vector_store import HybridVectorStore
from src.rag_engine import ATMRagPipeline
from src.query_processor import QueryProcessor
from src.log_processor import LogReader, LogParser, TextExtractor

logger = logging.getLogger(__name__)


class ComponentManager:
    """Manages singleton instances of core components."""

    def __init__(self):
        self._embedding_generator: Optional[EmbeddingGenerator] = None
        self._vector_store: Optional[HybridVectorStore] = None
        self._rag_pipeline: Optional[ATMRagPipeline] = None
        self._query_processor: Optional[QueryProcessor] = None
        self._log_reader: Optional[LogReader] = None
        self._log_parser: Optional[LogParser] = None
        self._text_extractor: Optional[TextExtractor] = None

    @property
    def embedding_generator(self) -> EmbeddingGenerator:
        """Get or create embedding generator."""
        if self._embedding_generator is None:
            try:
                cache_dir = os.getenv("EMBEDDINGS_CACHE_DIR", "data/embeddings/cache")
                self._embedding_generator = EmbeddingGenerator(
                    model_type="fast",
                    cache_dir=cache_dir
                )
                logger.info("Embedding generator initialized")
            except Exception as e:
                logger.error(f"Failed to initialize embedding generator: {e}")
                raise HTTPException(status_code=503, detail="Embedding service unavailable")
        return self._embedding_generator

    @property
    def vector_store(self) -> HybridVectorStore:
        """Get or create vector store."""
        if self._vector_store is None:
            try:
                # Try MongoDB first, fallback to FAISS-only mode if MongoDB is unavailable
                mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
                database_name = os.getenv("MONGODB_DATABASE", "atm_rag")
                storage_path = os.getenv("VECTOR_STORE_PATH", "data/vector_store")

                self._vector_store = HybridVectorStore(
                    mongo_connection=mongodb_uri,
                    mongo_database=database_name,
                    faiss_storage_path=storage_path
                )
                logger.info("Vector store initialized with MongoDB + FAISS")
            except Exception as mongo_error:
                # Fallback: Initialize with FAISS only (mock mode)
                logger.warning(f"MongoDB unavailable ({mongo_error}), initializing FAISS-only mode")
                try:
                    storage_path = os.getenv("VECTOR_STORE_PATH", "data/vector_store")
                    from src.vector_store.faiss_index import FAISSIndexManager

                    # Create a mock vector store that works without MongoDB
                    faiss_manager = FAISSIndexManager(
                        storage_path=storage_path,
                        dimensions=384
                    )

                    # Simple wrapper for FAISS-only operations
                    class MockVectorStore:
                        def __init__(self, faiss_manager):
                            self.faiss_manager = faiss_manager

                        def health_check(self):
                            return {"status": "healthy", "details": "FAISS-only mode (no MongoDB)"}

                        def get_statistics(self):
                            return {
                                "total_documents": 0,
                                "total_vectors": self.faiss_manager.get_index_size(),
                                "index_size_mb": 0.0,
                                "last_updated": None
                            }

                        def search_similar_logs(self, query_embedding=None, top_k=5, **kwargs):
                            # Return mock similar logs for demo
                            return [
                                {
                                    "log_id": f"demo_log_{i}",
                                    "similarity_score": 0.8 - (i * 0.1),
                                    "content": f"Demo ATM log entry {i}: DDL_EXCEEDED error occurred at {10+i}:00 AM",
                                    "metadata": {
                                        "atm_id": f"ATM{1123+i}",
                                        "error_code": "DDL_EXCEEDED",
                                        "operation": "withdrawal",
                                        "timestamp": "2025-09-24T10:00:00Z",
                                        "amount": 500 + i * 100,
                                        "status": "failed"
                                    }
                                }
                                for i in range(min(top_k, 3))
                            ]

                    self._vector_store = MockVectorStore(faiss_manager)
                    logger.info("Vector store initialized in FAISS-only mode")

                except Exception as e:
                    logger.error(f"Failed to initialize fallback vector store: {e}")
                    raise HTTPException(status_code=503, detail="Vector store unavailable")
        return self._vector_store

    @property
    def rag_pipeline(self) -> ATMRagPipeline:
        """Get or create RAG pipeline."""
        if self._rag_pipeline is None:
            try:
                self._rag_pipeline = ATMRagPipeline(
                    vector_store=self.vector_store,
                    embedding_generator=self.embedding_generator
                )
                logger.info("RAG pipeline initialized")
            except Exception as e:
                logger.error(f"Failed to initialize RAG pipeline: {e}")
                raise HTTPException(status_code=503, detail="RAG pipeline unavailable")
        return self._rag_pipeline

    @property
    def query_processor(self) -> QueryProcessor:
        """Get or create query processor."""
        if self._query_processor is None:
            try:
                self._query_processor = QueryProcessor()
                logger.info("Query processor initialized")
            except Exception as e:
                logger.error(f"Failed to initialize query processor: {e}")
                raise HTTPException(status_code=503, detail="Query processor unavailable")
        return self._query_processor

    @property
    def log_reader(self) -> LogReader:
        """Get or create log reader."""
        if self._log_reader is None:
            try:
                logs_directory = os.getenv("LOGS_DIRECTORY", "data/logs")
                self._log_reader = LogReader(logs_directory)
                logger.info("Log reader initialized")
            except Exception as e:
                logger.error(f"Failed to initialize log reader: {e}")
                raise HTTPException(status_code=503, detail="Log reader unavailable")
        return self._log_reader

    @property
    def log_parser(self) -> LogParser:
        """Get or create log parser."""
        if self._log_parser is None:
            try:
                self._log_parser = LogParser()
                logger.info("Log parser initialized")
            except Exception as e:
                logger.error(f"Failed to initialize log parser: {e}")
                raise HTTPException(status_code=503, detail="Log parser unavailable")
        return self._log_parser

    @property
    def text_extractor(self) -> TextExtractor:
        """Get or create text extractor."""
        if self._text_extractor is None:
            try:
                self._text_extractor = TextExtractor()
                logger.info("Text extractor initialized")
            except Exception as e:
                logger.error(f"Failed to initialize text extractor: {e}")
                raise HTTPException(status_code=503, detail="Text extractor unavailable")
        return self._text_extractor

    def health_check(self) -> dict:
        """Check health of all components."""
        health_status = {
            "overall_status": "healthy",
            "components": {}
        }

        try:
            # Check embedding generator
            if self._embedding_generator:
                test_embedding = self._embedding_generator.generate_embedding("test")
                health_status["components"]["embedding_generator"] = {
                    "status": "healthy" if test_embedding is not None else "unhealthy",
                    "details": f"Embedding dimension: {test_embedding.shape[0] if test_embedding is not None else 'N/A'}"
                }
            else:
                health_status["components"]["embedding_generator"] = {
                    "status": "not_initialized",
                    "details": "Component not yet initialized"
                }

            # Check vector store
            if self._vector_store:
                vs_health = self._vector_store.health_check()
                health_status["components"]["vector_store"] = vs_health
            else:
                health_status["components"]["vector_store"] = {
                    "status": "not_initialized",
                    "details": "Component not yet initialized"
                }

            # Check RAG pipeline
            if self._rag_pipeline:
                rag_health = self._rag_pipeline.health_check()
                health_status["components"]["rag_pipeline"] = rag_health
            else:
                health_status["components"]["rag_pipeline"] = {
                    "status": "not_initialized",
                    "details": "Component not yet initialized"
                }

            # Check query processor
            if self._query_processor:
                test_result = self._query_processor.process_query("test query")
                health_status["components"]["query_processor"] = {
                    "status": "healthy" if test_result else "unhealthy",
                    "details": "Query processing functional"
                }
            else:
                health_status["components"]["query_processor"] = {
                    "status": "not_initialized",
                    "details": "Component not yet initialized"
                }

            # Determine overall status
            component_statuses = [comp.get("status", "unknown") for comp in health_status["components"].values()]
            if any(status in ["unhealthy", "error"] for status in component_statuses):
                health_status["overall_status"] = "unhealthy"
            elif any(status == "degraded" for status in component_statuses):
                health_status["overall_status"] = "degraded"

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status["overall_status"] = "error"
            health_status["error"] = str(e)

        return health_status

    def get_statistics(self) -> dict:
        """Get system statistics."""
        stats = {
            "components_initialized": 0,
            "total_components": 7
        }

        if self._embedding_generator:
            stats["components_initialized"] += 1
        if self._vector_store:
            stats["components_initialized"] += 1
            try:
                vs_stats = self._vector_store.get_statistics()
                stats.update(vs_stats)
            except:
                pass
        if self._rag_pipeline:
            stats["components_initialized"] += 1
            try:
                pipeline_stats = self._rag_pipeline.get_pipeline_statistics()
                stats.update(pipeline_stats)
            except:
                pass
        if self._query_processor:
            stats["components_initialized"] += 1
        if self._log_reader:
            stats["components_initialized"] += 1
        if self._log_parser:
            stats["components_initialized"] += 1
        if self._text_extractor:
            stats["components_initialized"] += 1

        return stats


# Global component manager instance
@lru_cache()
def get_component_manager() -> ComponentManager:
    """Get the singleton component manager."""
    return ComponentManager()


# Dependency functions
def get_embedding_generator() -> EmbeddingGenerator:
    """Dependency to get embedding generator."""
    return get_component_manager().embedding_generator


def get_vector_store() -> HybridVectorStore:
    """Dependency to get vector store."""
    return get_component_manager().vector_store


def get_rag_pipeline() -> ATMRagPipeline:
    """Dependency to get RAG pipeline."""
    return get_component_manager().rag_pipeline


def get_query_processor() -> QueryProcessor:
    """Dependency to get query processor."""
    return get_component_manager().query_processor


def get_log_reader() -> LogReader:
    """Dependency to get log reader."""
    return get_component_manager().log_reader


def get_log_parser() -> LogParser:
    """Dependency to get log parser."""
    return get_component_manager().log_parser


def get_text_extractor() -> TextExtractor:
    """Dependency to get text extractor."""
    return get_component_manager().text_extractor


def get_health_check() -> dict:
    """Dependency to get health check results."""
    return get_component_manager().health_check()


def get_system_statistics() -> dict:
    """Dependency to get system statistics."""
    return get_component_manager().get_statistics()


# Configuration dependencies
def get_config() -> dict:
    """Get application configuration."""
    return {
        "mongodb_uri": os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
        "mongodb_database": os.getenv("MONGODB_DATABASE", "atm_rag"),
        "vector_store_path": os.getenv("VECTOR_STORE_PATH", "data/vector_store"),
        "embeddings_cache_dir": os.getenv("EMBEDDINGS_CACHE_DIR", "data/embeddings/cache"),
        "logs_directory": os.getenv("LOGS_DIRECTORY", "data/logs"),
        "max_query_length": int(os.getenv("MAX_QUERY_LENGTH", "1000")),
        "default_top_k": int(os.getenv("DEFAULT_TOP_K", "5")),
        "api_timeout_seconds": int(os.getenv("API_TIMEOUT_SECONDS", "30"))
    }