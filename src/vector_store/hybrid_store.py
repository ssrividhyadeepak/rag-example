"""
Hybrid Vector Store

Combines MongoDB and FAISS for the ultimate ATM RAG vector store solution:
- MongoDB: Rich metadata storage, complex queries, filtering, aggregations
- FAISS: Ultra-fast vector similarity search (<1ms)

This hybrid approach gives us enterprise-grade capabilities while staying
completely local and self-contained.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
import time

# Import our components
from .local_mongo_store import LocalMongoStore
from .faiss_index import FAISSIndexManager
from .schema import ATMLogSchema, VectorMetadataSchema

# Import from embeddings component
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from embeddings.batch_processor import LogEmbeddingData

logger = logging.getLogger(__name__)


class HybridVectorStore:
    """
    Hybrid vector store combining MongoDB and FAISS for optimal ATM RAG performance.

    Features:
    - Rich metadata queries (MongoDB)
    - Lightning-fast vector search (FAISS)
    - Automatic synchronization between stores
    - Batch processing capabilities
    - Advanced filtering and analytics
    """

    def __init__(self,
                 mongo_connection: str = "mongodb://localhost:27017",
                 mongo_database: str = "atm_rag",
                 faiss_storage_path: str = "data/vector_store",
                 faiss_dimensions: int = 384,
                 auto_save_interval: int = 100):
        """
        Initialize hybrid vector store.

        Args:
            mongo_connection (str): MongoDB connection string
            mongo_database (str): MongoDB database name
            faiss_storage_path (str): FAISS storage directory
            faiss_dimensions (int): Vector dimensions (384 for all-MiniLM-L6-v2)
            auto_save_interval (int): Auto-save FAISS index every N operations
        """
        self.auto_save_interval = auto_save_interval
        self.operations_since_save = 0

        # Initialize MongoDB store
        try:
            self.mongo_store = LocalMongoStore(
                connection_string=mongo_connection,
                database_name=mongo_database
            )
            logger.info("MongoDB store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB store: {e}")
            raise

        # Initialize FAISS index manager
        try:
            self.faiss_manager = FAISSIndexManager(
                dimensions=faiss_dimensions,
                storage_path=faiss_storage_path
            )
            logger.info("FAISS index manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS manager: {e}")
            raise

        # Verify synchronization on startup
        self._verify_synchronization()

        logger.info("Hybrid vector store initialized successfully")

    def insert_log_embeddings(self,
                             log_embeddings: List[LogEmbeddingData],
                             batch_size: int = 1000) -> Dict[str, Any]:
        """
        Insert ATM log embeddings into both MongoDB and FAISS.

        Args:
            log_embeddings (List[LogEmbeddingData]): Log embeddings from Component 2
            batch_size (int): Batch size for processing

        Returns:
            Dict[str, Any]: Insertion results
        """
        if not log_embeddings:
            return {"inserted": 0, "errors": 0, "skipped": 0}

        start_time = time.time()

        # Prepare data for MongoDB
        atm_logs = []
        vector_metadata = []

        # Prepare data for FAISS
        embeddings_array = []
        log_ids = []

        for log_data in log_embeddings:
            # Create ATM log schema
            atm_log = ATMLogSchema(
                log_id=log_data.log_id,
                timestamp=datetime.fromisoformat(log_data.original_log.get('timestamp', '').replace('Z', '+00:00'))
                         if log_data.original_log.get('timestamp') else datetime.utcnow(),
                session_id=log_data.original_log.get('session_id', ''),
                customer_session_id=log_data.original_log.get('customer_session_id'),
                operation=log_data.metadata.get('operation', ''),
                status=log_data.metadata.get('status', ''),
                message=log_data.original_log.get('message', ''),
                error_code=log_data.original_log.get('error_code'),
                atm_id=log_data.original_log.get('atm_id'),
                amount=log_data.original_log.get('amount'),
                extracted_text=log_data.extracted_text,
                metadata={
                    **log_data.metadata,
                    'text_length': log_data.metadata.get('text_length', len(log_data.extracted_text)),
                    'is_error': log_data.metadata.get('is_error', False)
                }
            )
            atm_logs.append(atm_log)

            # Prepare for FAISS (will be updated with actual FAISS indices later)
            embeddings_array.append(log_data.embedding)
            log_ids.append(log_data.log_id)

        # Convert to numpy array for FAISS
        embeddings_array = np.array(embeddings_array)

        # Insert into MongoDB
        logger.info(f"Inserting {len(atm_logs)} logs into MongoDB...")
        mongo_result = self.mongo_store.insert_atm_logs_batch(atm_logs)

        # Insert into FAISS
        logger.info(f"Inserting {len(embeddings_array)} vectors into FAISS...")
        faiss_result = self.faiss_manager.add_vectors(embeddings_array, log_ids, batch_size)

        # Create vector metadata entries with FAISS indices
        current_faiss_base = self.faiss_manager.total_vectors - faiss_result['added']

        for i, log_data in enumerate(log_embeddings):
            if log_data.log_id in self.faiss_manager.reverse_mapping:
                faiss_idx = self.faiss_manager.reverse_mapping[log_data.log_id]

                vector_meta = VectorMetadataSchema(
                    log_id=log_data.log_id,
                    faiss_index=faiss_idx,
                    text_content=log_data.extracted_text,
                    vector_norm=float(np.linalg.norm(log_data.embedding)),
                    confidence_score=log_data.metadata.get('confidence_score', 1.0)
                )
                vector_metadata.append(vector_meta)

        # Insert vector metadata
        if vector_metadata:
            logger.info(f"Inserting {len(vector_metadata)} vector metadata entries...")
            meta_result = self.mongo_store.insert_vector_metadata_batch(vector_metadata)
        else:
            meta_result = {"inserted": 0, "errors": 0}

        # Auto-save if needed
        self.operations_since_save += len(log_embeddings)
        if self.operations_since_save >= self.auto_save_interval:
            self.save_faiss_index()

        total_time = time.time() - start_time

        result = {
            "total_processed": len(log_embeddings),
            "mongo_logs": mongo_result,
            "faiss_vectors": faiss_result,
            "vector_metadata": meta_result,
            "processing_time_seconds": total_time,
            "logs_per_second": len(log_embeddings) / total_time if total_time > 0 else 0
        }

        logger.info(f"Hybrid insertion completed in {total_time:.2f}s: "
                   f"{mongo_result['inserted']} MongoDB logs, "
                   f"{faiss_result['added']} FAISS vectors")

        return result

    def search_similar_logs(self,
                           query_vector: np.ndarray,
                           top_k: int = 10,
                           min_similarity: float = 0.0,
                           filters: Optional[Dict[str, Any]] = None,
                           include_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        Search for similar ATM logs using hybrid approach.

        Args:
            query_vector (np.ndarray): Query embedding vector
            top_k (int): Number of results to return
            min_similarity (float): Minimum similarity threshold
            filters (Optional[Dict[str, Any]]): MongoDB filters to apply
            include_metadata (bool): Include full log metadata

        Returns:
            List[Dict[str, Any]]: Search results with similarity scores
        """
        start_time = time.time()

        # Step 1: Fast vector search with FAISS
        similar_log_ids, similarity_scores = self.faiss_manager.search_similar(
            query_vector, top_k * 2, min_similarity  # Get extra results for filtering
        )

        if not similar_log_ids:
            return []

        # Step 2: Rich metadata filtering with MongoDB
        if filters:
            # Add log_id filter to only query relevant logs
            mongo_filters = {**filters, "log_id": {"$in": similar_log_ids}}

            # Query MongoDB for filtered results
            filtered_logs = self.mongo_store.query_atm_logs(
                filters=mongo_filters,
                sort_by="log_id",  # Maintain order for score mapping
                limit=top_k * 2
            )

            # Create mapping of filtered log IDs
            filtered_log_ids = {log.log_id for log in filtered_logs}

            # Filter similarity results to match MongoDB filters
            filtered_results = []
            for log_id, score in zip(similar_log_ids, similarity_scores):
                if log_id in filtered_log_ids:
                    filtered_results.append((log_id, score))

            # Take top_k after filtering
            filtered_results = filtered_results[:top_k]
            similar_log_ids = [result[0] for result in filtered_results]
            similarity_scores = [result[1] for result in filtered_results]

        # Step 3: Enrich results with metadata if requested
        results = []

        if include_metadata:
            # Batch query for all metadata
            log_details = {log.log_id: log for log in
                          self.mongo_store.query_atm_logs(
                              filters={"log_id": {"$in": similar_log_ids}},
                              limit=len(similar_log_ids)
                          )}

            for log_id, score in zip(similar_log_ids, similarity_scores):
                log_detail = log_details.get(log_id)
                if log_detail:
                    results.append({
                        "log_id": log_id,
                        "similarity_score": score,
                        "timestamp": log_detail.timestamp,
                        "operation": log_detail.operation,
                        "status": log_detail.status,
                        "message": log_detail.message,
                        "error_code": log_detail.error_code,
                        "atm_id": log_detail.atm_id,
                        "amount": log_detail.amount,
                        "extracted_text": log_detail.extracted_text,
                        "metadata": log_detail.metadata
                    })
        else:
            # Return minimal results
            for log_id, score in zip(similar_log_ids, similarity_scores):
                results.append({
                    "log_id": log_id,
                    "similarity_score": score
                })

        search_time = time.time() - start_time
        logger.debug(f"Hybrid search completed in {search_time*1000:.1f}ms, "
                    f"found {len(results)} results")

        return results

    def query_logs_with_vector_search(self,
                                     text_query: str,
                                     embedding_generator,
                                     operation: Optional[str] = None,
                                     status: Optional[str] = None,
                                     atm_id: Optional[str] = None,
                                     error_only: bool = False,
                                     date_range: Optional[Tuple[datetime, datetime]] = None,
                                     top_k: int = 10) -> List[Dict[str, Any]]:
        """
        High-level query interface combining text search with metadata filtering.

        Args:
            text_query (str): Natural language query
            embedding_generator: EmbeddingGenerator instance
            operation (Optional[str]): Filter by operation type
            status (Optional[str]): Filter by status
            atm_id (Optional[str]): Filter by ATM ID
            error_only (bool): Only return error logs
            date_range (Optional[Tuple[datetime, datetime]]): Date range filter
            top_k (int): Number of results

        Returns:
            List[Dict[str, Any]]: Query results
        """
        # Generate query embedding
        query_vector = embedding_generator.generate_embedding(text_query)

        # Build MongoDB filters
        filters = {}

        if operation:
            filters["operation"] = operation

        if status:
            filters["status"] = status

        if atm_id:
            filters["atm_id"] = atm_id

        if error_only:
            filters["$or"] = [
                {"status": {"$in": ["denied", "failed", "error", "timeout"]}},
                {"error_code": {"$exists": True, "$ne": ""}},
                {"metadata.is_error": True}
            ]

        if date_range:
            filters["timestamp"] = {
                "$gte": date_range[0],
                "$lte": date_range[1]
            }

        # Perform hybrid search
        return self.search_similar_logs(
            query_vector=query_vector,
            top_k=top_k,
            filters=filters if filters else None,
            include_metadata=True
        )

    def get_similar_to_log(self,
                          reference_log_id: str,
                          top_k: int = 5,
                          exclude_same_session: bool = True) -> List[Dict[str, Any]]:
        """
        Find logs similar to a specific reference log.

        Args:
            reference_log_id (str): Reference log ID
            top_k (int): Number of similar logs to return
            exclude_same_session (bool): Exclude logs from same session

        Returns:
            List[Dict[str, Any]]: Similar logs
        """
        # Get reference log details
        reference_log = self.mongo_store.get_atm_log(reference_log_id)
        if not reference_log:
            logger.warning(f"Reference log {reference_log_id} not found")
            return []

        # Get reference vector from FAISS
        if reference_log_id not in self.faiss_manager.reverse_mapping:
            logger.warning(f"Vector for log {reference_log_id} not found in FAISS")
            return []

        faiss_idx = self.faiss_manager.reverse_mapping[reference_log_id]
        try:
            reference_vector = self.faiss_manager.index.reconstruct(faiss_idx)
        except Exception as e:
            logger.error(f"Could not reconstruct vector for {reference_log_id}: {e}")
            return []

        # Build filters
        filters = {}
        if exclude_same_session:
            filters["session_id"] = {"$ne": reference_log.session_id}

        # Search for similar logs
        similar_logs = self.search_similar_logs(
            query_vector=reference_vector,
            top_k=top_k + 1,  # +1 to account for the reference log itself
            filters=filters,
            include_metadata=True
        )

        # Remove the reference log from results
        return [log for log in similar_logs if log["log_id"] != reference_log_id][:top_k]

    def analyze_error_patterns(self,
                              error_code: Optional[str] = None,
                              days_back: int = 30,
                              min_occurrences: int = 3) -> Dict[str, Any]:
        """
        Analyze error patterns using vector clustering and metadata analysis.

        Args:
            error_code (Optional[str]): Specific error code to analyze
            days_back (int): Number of days to look back
            min_occurrences (int): Minimum occurrences to consider a pattern

        Returns:
            Dict[str, Any]: Error pattern analysis
        """
        # Date range for analysis
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)

        # Build filters
        filters = {
            "timestamp": {"$gte": start_date, "$lte": end_date},
            "$or": [
                {"status": {"$in": ["denied", "failed", "error", "timeout"]}},
                {"error_code": {"$exists": True, "$ne": ""}},
                {"metadata.is_error": True}
            ]
        }

        if error_code:
            filters["error_code"] = error_code

        # Get error logs from MongoDB
        error_logs = self.mongo_store.query_atm_logs(
            filters=filters,
            limit=10000,  # Large limit for pattern analysis
            sort_by="timestamp",
            sort_order=1
        )

        if len(error_logs) < min_occurrences:
            return {"message": "Insufficient data for pattern analysis"}

        # Aggregate statistics
        pattern_analysis = {
            "total_errors": len(error_logs),
            "date_range": {"start": start_date, "end": end_date},
            "error_codes": {},
            "operations": {},
            "atms": {},
            "time_patterns": {},
            "similar_error_clusters": []
        }

        # Count patterns
        for log in error_logs:
            # Error code distribution
            if log.error_code:
                pattern_analysis["error_codes"][log.error_code] = \
                    pattern_analysis["error_codes"].get(log.error_code, 0) + 1

            # Operation distribution
            pattern_analysis["operations"][log.operation] = \
                pattern_analysis["operations"].get(log.operation, 0) + 1

            # ATM distribution
            if log.atm_id:
                pattern_analysis["atms"][log.atm_id] = \
                    pattern_analysis["atms"].get(log.atm_id, 0) + 1

            # Time pattern (hour of day)
            hour = log.timestamp.hour
            pattern_analysis["time_patterns"][hour] = \
                pattern_analysis["time_patterns"].get(hour, 0) + 1

        # Find similar error clusters using vector search
        if len(error_logs) >= 10:
            try:
                # Sample some error logs for clustering
                sample_logs = error_logs[:min(100, len(error_logs))]
                clusters = self._find_error_clusters(sample_logs, min_cluster_size=min_occurrences)
                pattern_analysis["similar_error_clusters"] = clusters
            except Exception as e:
                logger.warning(f"Error clustering failed: {e}")

        return pattern_analysis

    def _find_error_clusters(self, error_logs: List[ATMLogSchema], min_cluster_size: int = 3) -> List[Dict[str, Any]]:
        """Find clusters of similar errors using vector similarity."""
        if len(error_logs) < min_cluster_size:
            return []

        # Get vectors for error logs
        log_vectors = []
        log_details = []

        for log in error_logs:
            if log.log_id in self.faiss_manager.reverse_mapping:
                faiss_idx = self.faiss_manager.reverse_mapping[log.log_id]
                try:
                    vector = self.faiss_manager.index.reconstruct(faiss_idx)
                    log_vectors.append(vector)
                    log_details.append(log)
                except:
                    continue

        if len(log_vectors) < min_cluster_size:
            return []

        # Simple clustering using pairwise similarity
        vectors = np.array(log_vectors)
        similarities = np.dot(vectors, vectors.T)

        # Find clusters (simplified approach)
        clusters = []
        used_indices = set()

        for i in range(len(similarities)):
            if i in used_indices:
                continue

            # Find similar logs to this one
            similar_indices = []
            for j in range(len(similarities)):
                if j != i and similarities[i][j] > 0.8:  # High similarity threshold
                    similar_indices.append(j)

            if len(similar_indices) >= min_cluster_size - 1:  # -1 because we don't count the reference
                cluster_logs = [log_details[i]] + [log_details[idx] for idx in similar_indices]

                # Extract common patterns
                common_error_codes = {}
                common_operations = {}
                common_atms = {}

                for log in cluster_logs:
                    if log.error_code:
                        common_error_codes[log.error_code] = common_error_codes.get(log.error_code, 0) + 1
                    common_operations[log.operation] = common_operations.get(log.operation, 0) + 1
                    if log.atm_id:
                        common_atms[log.atm_id] = common_atms.get(log.atm_id, 0) + 1

                clusters.append({
                    "cluster_size": len(cluster_logs),
                    "sample_messages": [log.message for log in cluster_logs[:3]],
                    "common_error_codes": common_error_codes,
                    "common_operations": common_operations,
                    "common_atms": common_atms,
                    "time_range": {
                        "start": min(log.timestamp for log in cluster_logs),
                        "end": max(log.timestamp for log in cluster_logs)
                    }
                })

                # Mark indices as used
                used_indices.add(i)
                used_indices.update(similar_indices)

        return clusters[:5]  # Return top 5 clusters

    def _verify_synchronization(self) -> Dict[str, Any]:
        """Verify that MongoDB and FAISS are synchronized."""
        try:
            # Get counts
            mongo_log_count = self.mongo_store.atm_logs.count_documents({})
            mongo_vector_count = self.mongo_store.vector_metadata.count_documents({})
            faiss_vector_count = self.faiss_manager.total_vectors

            sync_status = {
                "mongo_logs": mongo_log_count,
                "mongo_vectors": mongo_vector_count,
                "faiss_vectors": faiss_vector_count,
                "synchronized": mongo_vector_count == faiss_vector_count
            }

            if not sync_status["synchronized"]:
                logger.warning(f"Synchronization mismatch: MongoDB vectors={mongo_vector_count}, FAISS vectors={faiss_vector_count}")
            else:
                logger.info(f"Stores synchronized: {faiss_vector_count} vectors")

            return sync_status

        except Exception as e:
            logger.error(f"Error verifying synchronization: {e}")
            return {"error": str(e)}

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the hybrid store."""
        try:
            # Get individual store statistics
            mongo_log_stats = self.mongo_store.get_log_statistics()
            mongo_vector_stats = self.mongo_store.get_vector_statistics()
            faiss_stats = self.faiss_manager.get_statistics()

            # Synchronization status
            sync_status = self._verify_synchronization()

            return {
                "mongodb": {
                    "logs": mongo_log_stats,
                    "vectors": mongo_vector_stats
                },
                "faiss": faiss_stats,
                "synchronization": sync_status,
                "hybrid_capabilities": {
                    "rich_metadata_queries": True,
                    "fast_vector_search": True,
                    "combined_filtering": True,
                    "error_pattern_analysis": True
                }
            }

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}

    def save_faiss_index(self) -> bool:
        """Save FAISS index to disk."""
        try:
            result = self.faiss_manager.save_index()
            if result:
                self.operations_since_save = 0
            return result
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of the hybrid store."""
        try:
            mongo_health = self.mongo_store.health_check()
            faiss_health = {"status": "healthy", "total_vectors": self.faiss_manager.total_vectors}

            return {
                "overall_status": "healthy" if mongo_health["status"] == "healthy" else "degraded",
                "mongodb": mongo_health,
                "faiss": faiss_health,
                "synchronization": self._verify_synchronization(),
                "last_check": datetime.utcnow()
            }

        except Exception as e:
            return {
                "overall_status": "unhealthy",
                "error": str(e),
                "last_check": datetime.utcnow()
            }

    def close(self):
        """Close connections and save state."""
        logger.info("Closing hybrid vector store...")

        # Save FAISS index
        self.save_faiss_index()

        # Close MongoDB connection
        if hasattr(self, 'mongo_store'):
            self.mongo_store.close()

        logger.info("Hybrid vector store closed successfully")