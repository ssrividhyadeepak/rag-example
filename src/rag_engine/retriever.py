"""
Context Retriever for ATM RAG System

Handles intelligent retrieval of relevant ATM logs and context
for answering user queries. Combines vector similarity search
with structured filtering for optimal results.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

from ..vector_store.hybrid_store import HybridVectorStore
from ..embeddings.embedding_generator import EmbeddingGenerator

logger = logging.getLogger(__name__)


class ContextRetriever:
    """
    Intelligent context retrieval for ATM log queries.

    Combines vector similarity search with structured filtering
    to find the most relevant ATM logs for answering user questions.
    """

    def __init__(self,
                 vector_store: HybridVectorStore,
                 embedding_generator: EmbeddingGenerator):
        """
        Initialize context retriever.

        Args:
            vector_store: Hybrid vector store for search operations
            embedding_generator: For query embedding generation
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator

        # Retrieval parameters
        self.default_top_k = 10
        self.min_similarity_score = 0.3
        self.max_context_length = 5000

        logger.info("Context retriever initialized")

    def retrieve_context(self,
                        query: str,
                        top_k: int = None,
                        filters: Optional[Dict[str, Any]] = None,
                        min_score: float = None) -> Dict[str, Any]:
        """
        Retrieve relevant context for a user query.

        Args:
            query: User's question or search query
            top_k: Number of results to retrieve
            filters: MongoDB-style filters for structured search
            min_score: Minimum similarity score threshold

        Returns:
            Dict containing retrieved logs and metadata
        """
        top_k = top_k or self.default_top_k
        min_score = min_score or self.min_similarity_score

        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embedding(query)

            # Perform hybrid search
            search_results = self.vector_store.search_similar_logs(
                query_vector=query_embedding,
                top_k=top_k,
                min_score=min_score,
                filters=filters
            )

            # Process and rank results
            processed_results = self._process_search_results(
                query, search_results, top_k
            )

            # Generate context summary
            context_summary = self._generate_context_summary(processed_results)

            return {
                "query": query,
                "relevant_logs": processed_results,
                "context_summary": context_summary,
                "total_found": len(processed_results),
                "search_metadata": {
                    "min_score": min_score,
                    "filters_applied": filters,
                    "retrieval_timestamp": datetime.utcnow()
                }
            }

        except Exception as e:
            logger.error(f"Error retrieving context for query '{query}': {e}")
            return {
                "query": query,
                "relevant_logs": [],
                "context_summary": "No relevant context found due to search error.",
                "total_found": 0,
                "error": str(e)
            }

    def retrieve_error_context(self,
                              error_code: str = None,
                              operation: str = None,
                              top_k: int = 5) -> Dict[str, Any]:
        """
        Retrieve context specifically for error troubleshooting.

        Args:
            error_code: Specific error code to search for
            operation: ATM operation type
            top_k: Number of examples to retrieve

        Returns:
            Dict with error-specific context
        """
        filters = {}

        # Build error-focused filters
        if error_code:
            filters["error_code"] = error_code
        else:
            # General error filter
            filters["$or"] = [
                {"status": {"$in": ["denied", "failed", "error", "timeout"]}},
                {"error_code": {"$exists": True, "$ne": ""}},
                {"metadata.is_error": True}
            ]

        if operation:
            filters["operation"] = operation

        # Use a broad query for error retrieval
        query = f"ATM error {error_code or ''} {operation or ''} troubleshooting"

        return self.retrieve_context(
            query=query,
            top_k=top_k,
            filters=filters,
            min_score=0.1  # Lower threshold for error cases
        )

    def retrieve_operation_context(self,
                                  operation: str,
                                  status: str = None,
                                  time_range: Optional[Tuple[datetime, datetime]] = None,
                                  top_k: int = 8) -> Dict[str, Any]:
        """
        Retrieve context for specific ATM operations.

        Args:
            operation: ATM operation type (withdrawal, deposit, etc.)
            status: Operation status filter
            time_range: Tuple of (start_date, end_date)
            top_k: Number of results to retrieve

        Returns:
            Dict with operation-specific context
        """
        filters = {"operation": operation}

        if status:
            filters["status"] = status

        if time_range:
            start_date, end_date = time_range
            filters["timestamp"] = {
                "$gte": start_date,
                "$lte": end_date
            }

        query = f"{operation} operation {status or ''} ATM logs"

        return self.retrieve_context(
            query=query,
            top_k=top_k,
            filters=filters
        )

    def retrieve_atm_specific_context(self,
                                     atm_id: str,
                                     time_range: Optional[Tuple[datetime, datetime]] = None,
                                     top_k: int = 10) -> Dict[str, Any]:
        """
        Retrieve context for a specific ATM machine.

        Args:
            atm_id: ATM machine identifier
            time_range: Optional time range filter
            top_k: Number of results to retrieve

        Returns:
            Dict with ATM-specific context
        """
        filters = {"atm_id": atm_id}

        if time_range:
            start_date, end_date = time_range
            filters["timestamp"] = {
                "$gte": start_date,
                "$lte": end_date
            }

        query = f"ATM {atm_id} operations logs issues"

        return self.retrieve_context(
            query=query,
            top_k=top_k,
            filters=filters
        )

    def retrieve_recent_context(self,
                               hours_back: int = 24,
                               top_k: int = 15) -> Dict[str, Any]:
        """
        Retrieve recent ATM activity context.

        Args:
            hours_back: How many hours back to look
            top_k: Number of results to retrieve

        Returns:
            Dict with recent activity context
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=hours_back)

        filters = {
            "timestamp": {
                "$gte": start_date,
                "$lte": end_date
            }
        }

        query = f"recent ATM activity last {hours_back} hours"

        return self.retrieve_context(
            query=query,
            top_k=top_k,
            filters=filters,
            min_score=0.2  # Lower threshold for recent activity
        )

    def _process_search_results(self,
                               query: str,
                               search_results: List[Tuple[Any, float]],
                               top_k: int) -> List[Dict[str, Any]]:
        """
        Process and enhance search results with additional metadata.

        Args:
            query: Original query
            search_results: Raw search results from vector store
            top_k: Maximum number of results to return

        Returns:
            List of processed log entries with scores
        """
        processed = []

        for log_entry, score in search_results[:top_k]:
            try:
                # Convert log entry to dict if needed
                if hasattr(log_entry, 'to_dict'):
                    log_dict = log_entry.to_dict()
                else:
                    log_dict = log_entry

                # Add retrieval metadata
                log_dict["retrieval_score"] = float(score)
                log_dict["relevance_rank"] = len(processed) + 1

                # Calculate time relevance (more recent = higher score)
                if "timestamp" in log_dict:
                    time_diff = datetime.utcnow() - log_dict["timestamp"]
                    time_relevance = max(0, 1 - (time_diff.days / 30))  # Decay over 30 days
                    log_dict["time_relevance"] = time_relevance

                processed.append(log_dict)

            except Exception as e:
                logger.warning(f"Error processing search result: {e}")
                continue

        return processed

    def _generate_context_summary(self, relevant_logs: List[Dict[str, Any]]) -> str:
        """
        Generate a concise summary of retrieved context.

        Args:
            relevant_logs: List of relevant log entries

        Returns:
            Context summary string
        """
        if not relevant_logs:
            return "No relevant ATM logs found for this query."

        # Analyze retrieved logs
        operations = set()
        statuses = set()
        error_codes = set()
        atm_ids = set()

        for log in relevant_logs:
            if "operation" in log:
                operations.add(log["operation"])
            if "status" in log:
                statuses.add(log["status"])
            if "error_code" in log and log["error_code"]:
                error_codes.add(log["error_code"])
            if "atm_id" in log and log["atm_id"]:
                atm_ids.add(log["atm_id"])

        # Build summary
        summary_parts = [
            f"Found {len(relevant_logs)} relevant ATM log entries."
        ]

        if operations:
            summary_parts.append(f"Operations: {', '.join(operations)}.")

        if statuses:
            summary_parts.append(f"Statuses: {', '.join(statuses)}.")

        if error_codes:
            summary_parts.append(f"Error codes: {', '.join(error_codes)}.")

        if atm_ids:
            atm_list = list(atm_ids)[:3]  # Show max 3 ATMs
            if len(atm_ids) > 3:
                summary_parts.append(f"ATMs: {', '.join(atm_list)} and {len(atm_ids) - 3} others.")
            else:
                summary_parts.append(f"ATMs: {', '.join(atm_list)}.")

        return " ".join(summary_parts)

    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about retrieval performance.

        Returns:
            Dict with retrieval statistics
        """
        try:
            vector_stats = self.vector_store.get_statistics()

            return {
                "total_indexed_logs": vector_stats.get("mongodb_docs", 0),
                "total_vectors": vector_stats.get("faiss_vectors", 0),
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_dimensions": 384,
                "default_top_k": self.default_top_k,
                "min_similarity_score": self.min_similarity_score,
                "max_context_length": self.max_context_length
            }
        except Exception as e:
            logger.error(f"Error getting retrieval statistics: {e}")
            return {}