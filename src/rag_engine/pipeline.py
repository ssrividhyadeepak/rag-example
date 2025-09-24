"""
Main RAG Pipeline for ATM Assist System

Orchestrates the complete RAG workflow: query processing, context retrieval,
response generation, and result formatting. Provides the main interface
for ATM-related question answering and troubleshooting.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

from ..vector_store.hybrid_store import HybridVectorStore
from ..embeddings.embedding_generator import EmbeddingGenerator
from .retriever import ContextRetriever
from .generator import ResponseGenerator

logger = logging.getLogger(__name__)


class ATMRagPipeline:
    """
    Complete RAG pipeline for ATM assistance and troubleshooting.

    Integrates all components to provide intelligent responses to
    ATM-related queries using retrieval-augmented generation.
    """

    def __init__(self,
                 vector_store: HybridVectorStore,
                 embedding_generator: EmbeddingGenerator,
                 enable_caching: bool = True):
        """
        Initialize RAG pipeline with all components.

        Args:
            vector_store: Hybrid vector store for log storage and search
            embedding_generator: For query and document embeddings
            enable_caching: Enable response caching for performance
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator

        # Initialize pipeline components
        self.retriever = ContextRetriever(vector_store, embedding_generator)
        self.generator = ResponseGenerator()

        # Pipeline configuration
        self.enable_caching = enable_caching
        self.response_cache = {} if enable_caching else None
        self.cache_ttl = 300  # 5 minutes

        # Performance settings
        self.default_timeout = 30.0
        self.max_concurrent_requests = 5
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_requests)

        # Pipeline statistics
        self.stats = {
            "total_queries": 0,
            "cached_responses": 0,
            "average_response_time": 0.0,
            "error_count": 0,
            "last_query_time": None
        }

        logger.info("ATM RAG Pipeline initialized successfully")

    async def process_query(self,
                           query: str,
                           filters: Optional[Dict[str, Any]] = None,
                           top_k: int = 10,
                           response_type: str = "auto",
                           timeout: float = None) -> Dict[str, Any]:
        """
        Process a user query through the complete RAG pipeline.

        Args:
            query: User's question or request
            filters: Optional MongoDB-style filters for retrieval
            top_k: Maximum number of logs to retrieve
            response_type: Type of response to generate
            timeout: Query timeout in seconds

        Returns:
            Dict containing response and metadata
        """
        start_time = time.time()
        timeout = timeout or self.default_timeout

        try:
            # Update statistics
            self.stats["total_queries"] += 1
            self.stats["last_query_time"] = datetime.utcnow()

            # Check cache first
            if self.enable_caching:
                cached_response = self._check_cache(query, filters, top_k, response_type)
                if cached_response:
                    self.stats["cached_responses"] += 1
                    return cached_response

            # Run pipeline with timeout
            try:
                response = await asyncio.wait_for(
                    self._run_pipeline(query, filters, top_k, response_type),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"Query timeout after {timeout}s: {query}")
                return self._generate_timeout_response(query, timeout)

            # Cache response
            if self.enable_caching and response.get("confidence", 0) > 0.5:
                self._cache_response(query, filters, top_k, response_type, response)

            # Update performance statistics
            response_time = time.time() - start_time
            self._update_performance_stats(response_time)

            # Add pipeline metadata
            response["pipeline_metadata"] = {
                "processing_time_ms": round(response_time * 1000, 2),
                "cached": False,
                "pipeline_version": "1.0",
                "timestamp": datetime.utcnow()
            }

            return response

        except Exception as e:
            self.stats["error_count"] += 1
            logger.error(f"Pipeline error for query '{query}': {e}")
            return self._generate_error_response(query, str(e))

    async def _run_pipeline(self,
                           query: str,
                           filters: Optional[Dict[str, Any]],
                           top_k: int,
                           response_type: str) -> Dict[str, Any]:
        """
        Execute the core RAG pipeline steps.

        Args:
            query: User query
            filters: Retrieval filters
            top_k: Number of results to retrieve
            response_type: Response type

        Returns:
            Generated response dictionary
        """
        # Step 1: Context Retrieval
        logger.debug(f"Retrieving context for query: {query}")
        context = await self._retrieve_context_async(query, filters, top_k)

        # Step 2: Response Generation
        logger.debug(f"Generating response for query: {query}")
        response = await self._generate_response_async(query, context, response_type)

        return response

    async def _retrieve_context_async(self,
                                    query: str,
                                    filters: Optional[Dict[str, Any]],
                                    top_k: int) -> Dict[str, Any]:
        """Run context retrieval asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.retriever.retrieve_context,
            query, top_k, filters
        )

    async def _generate_response_async(self,
                                     query: str,
                                     context: Dict[str, Any],
                                     response_type: str) -> Dict[str, Any]:
        """Run response generation asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.generator.generate_response,
            query, context, response_type
        )

    def process_query_sync(self,
                          query: str,
                          filters: Optional[Dict[str, Any]] = None,
                          top_k: int = 10,
                          response_type: str = "auto") -> Dict[str, Any]:
        """
        Synchronous version of query processing.

        Args:
            query: User's question
            filters: Optional retrieval filters
            top_k: Number of results to retrieve
            response_type: Response type to generate

        Returns:
            Response dictionary
        """
        try:
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, create a task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: asyncio.run(self.process_query(query, filters, top_k, response_type))
                    )
                    return future.result(timeout=30)
            except RuntimeError:
                # No running loop, safe to use asyncio.run()
                return asyncio.run(self.process_query(query, filters, top_k, response_type))
        except Exception as e:
            logger.error(f"Sync pipeline error: {e}")
            return self._generate_error_response(query, str(e))

    def troubleshoot_error(self,
                          error_code: str = None,
                          operation: str = None,
                          atm_id: str = None,
                          description: str = None) -> Dict[str, Any]:
        """
        Specialized troubleshooting for ATM errors.

        Args:
            error_code: Specific error code
            operation: ATM operation type
            atm_id: ATM machine ID
            description: Additional error description

        Returns:
            Troubleshooting response
        """
        # Build troubleshooting query
        query_parts = ["ATM troubleshooting"]

        if error_code:
            query_parts.append(f"error code {error_code}")
        if operation:
            query_parts.append(f"{operation} operation")
        if atm_id:
            query_parts.append(f"ATM {atm_id}")
        if description:
            query_parts.append(description)

        query = " ".join(query_parts)

        # Build filters for targeted retrieval
        filters = {}
        if error_code:
            filters["error_code"] = error_code
        if operation:
            filters["operation"] = operation
        if atm_id:
            filters["atm_id"] = atm_id

        return self.process_query_sync(
            query=query,
            filters=filters,
            response_type="troubleshooting",
            top_k=8
        )

    def analyze_atm_performance(self,
                               atm_id: str = None,
                               time_range: Optional[tuple] = None,
                               operation: str = None) -> Dict[str, Any]:
        """
        Analyze ATM performance and patterns.

        Args:
            atm_id: Specific ATM to analyze
            time_range: (start_date, end_date) tuple
            operation: Specific operation to analyze

        Returns:
            Analysis response
        """
        # Build analysis query
        query_parts = ["Analyze ATM performance"]

        if atm_id:
            query_parts.append(f"for ATM {atm_id}")
        if operation:
            query_parts.append(f"for {operation} operations")

        query = " ".join(query_parts)

        # Build filters
        filters = {}
        if atm_id:
            filters["atm_id"] = atm_id
        if operation:
            filters["operation"] = operation
        if time_range:
            start_date, end_date = time_range
            filters["timestamp"] = {"$gte": start_date, "$lte": end_date}

        return self.process_query_sync(
            query=query,
            filters=filters,
            response_type="analysis",
            top_k=20
        )

    def get_recent_issues(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Get recent ATM issues and problems.

        Args:
            hours_back: How many hours back to look

        Returns:
            Recent issues response
        """
        query = f"Recent ATM issues and problems in last {hours_back} hours"

        # Time filter
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=hours_back)

        filters = {
            "timestamp": {"$gte": start_date, "$lte": end_date},
            "$or": [
                {"status": {"$in": ["denied", "failed", "error", "timeout"]}},
                {"error_code": {"$exists": True, "$ne": ""}},
                {"metadata.is_error": True}
            ]
        }

        return self.process_query_sync(
            query=query,
            filters=filters,
            response_type="analysis",
            top_k=15
        )

    def search_logs(self,
                   search_text: str,
                   filters: Optional[Dict[str, Any]] = None,
                   top_k: int = 10) -> Dict[str, Any]:
        """
        Search ATM logs with semantic similarity.

        Args:
            search_text: Text to search for
            filters: Additional filters
            top_k: Number of results

        Returns:
            Search results response
        """
        query = f"Search ATM logs for: {search_text}"

        return self.process_query_sync(
            query=query,
            filters=filters,
            response_type="info",
            top_k=top_k
        )

    def batch_process_queries(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch.

        Args:
            queries: List of queries to process

        Returns:
            List of responses
        """
        responses = []

        for query in queries:
            try:
                response = self.process_query_sync(query)
                responses.append(response)
            except Exception as e:
                logger.error(f"Error processing batch query '{query}': {e}")
                responses.append(self._generate_error_response(query, str(e)))

        return responses

    def _check_cache(self,
                    query: str,
                    filters: Optional[Dict[str, Any]],
                    top_k: int,
                    response_type: str) -> Optional[Dict[str, Any]]:
        """Check if response is cached."""
        if not self.enable_caching:
            return None

        cache_key = self._build_cache_key(query, filters, top_k, response_type)

        if cache_key in self.response_cache:
            cached_item = self.response_cache[cache_key]

            # Check TTL
            if time.time() - cached_item["timestamp"] < self.cache_ttl:
                response = cached_item["response"].copy()
                response["pipeline_metadata"] = {
                    "cached": True,
                    "cache_age_seconds": round(time.time() - cached_item["timestamp"], 2)
                }
                return response
            else:
                # Remove expired cache entry
                del self.response_cache[cache_key]

        return None

    def _cache_response(self,
                       query: str,
                       filters: Optional[Dict[str, Any]],
                       top_k: int,
                       response_type: str,
                       response: Dict[str, Any]):
        """Cache a response."""
        if not self.enable_caching:
            return

        cache_key = self._build_cache_key(query, filters, top_k, response_type)

        self.response_cache[cache_key] = {
            "response": response,
            "timestamp": time.time()
        }

        # Cleanup old cache entries if cache gets too large
        if len(self.response_cache) > 100:
            self._cleanup_cache()

    def _build_cache_key(self,
                        query: str,
                        filters: Optional[Dict[str, Any]],
                        top_k: int,
                        response_type: str) -> str:
        """Build cache key from query parameters."""
        import hashlib
        import json

        cache_data = {
            "query": query.lower().strip(),
            "filters": filters,
            "top_k": top_k,
            "response_type": response_type
        }

        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()

    def _cleanup_cache(self):
        """Remove old cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, item in self.response_cache.items()
            if current_time - item["timestamp"] > self.cache_ttl
        ]

        for key in expired_keys:
            del self.response_cache[key]

    def _update_performance_stats(self, response_time: float):
        """Update performance statistics."""
        # Update average response time
        current_avg = self.stats["average_response_time"]
        total_queries = self.stats["total_queries"]

        new_avg = ((current_avg * (total_queries - 1)) + response_time) / total_queries
        self.stats["average_response_time"] = new_avg

    def _generate_timeout_response(self, query: str, timeout: float) -> Dict[str, Any]:
        """Generate response for query timeout."""
        return {
            "response": f"Query processing timed out after {timeout} seconds. Please try a more specific query or contact support.",
            "response_type": "timeout",
            "query": query,
            "confidence": 0.0,
            "sources_count": 0,
            "metadata": {"timeout_seconds": timeout},
            "generated_at": datetime.utcnow().isoformat(),
            "pipeline_metadata": {"cached": False, "timeout": True}
        }

    def _generate_error_response(self, query: str, error: str) -> Dict[str, Any]:
        """Generate response for pipeline errors."""
        return {
            "response": f"I encountered an error while processing your query. Please try again or contact support.",
            "response_type": "error",
            "query": query,
            "confidence": 0.0,
            "sources_count": 0,
            "metadata": {"error": error},
            "generated_at": datetime.utcnow().isoformat(),
            "pipeline_metadata": {"cached": False, "error": True}
        }

    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive pipeline performance statistics.

        Returns:
            Statistics dictionary
        """
        try:
            # Get component statistics
            retriever_stats = self.retriever.get_retrieval_statistics()
            vector_store_stats = self.vector_store.get_statistics()

            return {
                "pipeline_stats": self.stats.copy(),
                "retriever_stats": retriever_stats,
                "vector_store_stats": vector_store_stats,
                "cache_stats": {
                    "enabled": self.enable_caching,
                    "cached_items": len(self.response_cache) if self.response_cache else 0,
                    "cache_ttl_seconds": self.cache_ttl
                },
                "performance_settings": {
                    "default_timeout": self.default_timeout,
                    "max_concurrent_requests": self.max_concurrent_requests
                },
                "last_updated": datetime.utcnow()
            }
        except Exception as e:
            logger.error(f"Error getting pipeline statistics: {e}")
            return {"error": str(e)}

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all pipeline components.

        Returns:
            Health status dictionary
        """
        health_status = {
            "overall_status": "healthy",
            "components": {},
            "timestamp": datetime.utcnow()
        }

        try:
            # Check vector store
            vector_store_health = self.vector_store.health_check()
            health_status["components"]["vector_store"] = vector_store_health

            # Check embedding generator
            try:
                test_embedding = self.embedding_generator.generate_embedding("test")
                health_status["components"]["embedding_generator"] = {
                    "status": "healthy",
                    "test_embedding_shape": test_embedding.shape
                }
            except Exception as e:
                health_status["components"]["embedding_generator"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }

            # Check if any component is unhealthy
            if any(comp.get("status") == "unhealthy" for comp in health_status["components"].values()):
                health_status["overall_status"] = "degraded"

        except Exception as e:
            health_status["overall_status"] = "unhealthy"
            health_status["error"] = str(e)

        return health_status

    def clear_cache(self):
        """Clear response cache."""
        if self.response_cache:
            self.response_cache.clear()
            logger.info("Response cache cleared")

    def close(self):
        """Clean up pipeline resources."""
        try:
            if self.executor:
                self.executor.shutdown(wait=True)

            if hasattr(self.vector_store, 'close'):
                self.vector_store.close()

            logger.info("ATM RAG Pipeline closed successfully")
        except Exception as e:
            logger.error(f"Error closing pipeline: {e}")