"""
Similarity Search Module

Advanced similarity search functionality for finding relevant ATM log entries
based on vector embeddings and semantic similarity.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from sklearn.metrics.pairwise import cosine_similarity
from .batch_processor import LogEmbeddingData
import time

logger = logging.getLogger(__name__)


class SimilaritySearch:
    """
    Advanced similarity search for ATM log embeddings.
    """

    def __init__(self, log_embeddings: List[LogEmbeddingData]):
        """
        Initialize similarity search with processed log embeddings.

        Args:
            log_embeddings (List[LogEmbeddingData]): List of logs with embeddings
        """
        self.log_embeddings = log_embeddings
        self.embedding_matrix = None
        self.metadata_index = None
        self._build_search_index()

    def _build_search_index(self) -> None:
        """Build search index for efficient similarity search."""
        if not self.log_embeddings:
            logger.warning("No log embeddings provided for search index")
            return

        # Build embedding matrix
        embeddings = [log.embedding for log in self.log_embeddings]
        self.embedding_matrix = np.array(embeddings, dtype=np.float32)

        # Build metadata index for filtering
        self.metadata_index = []
        for i, log in enumerate(self.log_embeddings):
            self.metadata_index.append({
                'index': i,
                'log_id': log.log_id,
                'operation': log.metadata.get('operation', '').lower(),
                'status': log.metadata.get('status', '').lower(),
                'is_error': log.metadata.get('is_error', False),
                'text_length': len(log.extracted_text)
            })

        logger.info(f"Built search index with {len(self.log_embeddings)} entries")

    def search(self, query_embedding: np.ndarray,
               top_k: int = 10,
               min_similarity: float = 0.0,
               operation_filter: Optional[str] = None,
               status_filter: Optional[str] = None,
               error_only: bool = False) -> List[Dict[str, Any]]:
        """
        Search for similar log entries with optional filtering.

        Args:
            query_embedding (np.ndarray): Query embedding vector
            top_k (int): Maximum number of results to return
            min_similarity (float): Minimum similarity threshold
            operation_filter (Optional[str]): Filter by operation type
            status_filter (Optional[str]): Filter by status
            error_only (bool): Return only error logs

        Returns:
            List[Dict[str, Any]]: Search results with similarity scores
        """
        if self.embedding_matrix is None:
            logger.error("Search index not built")
            return []

        start_time = time.time()

        # Calculate similarities
        similarities = cosine_similarity([query_embedding], self.embedding_matrix)[0]

        # Create results with metadata
        results = []
        for i, similarity in enumerate(similarities):
            if similarity < min_similarity:
                continue

            metadata = self.metadata_index[i]

            # Apply filters
            if operation_filter and metadata['operation'] != operation_filter.lower():
                continue
            if status_filter and metadata['status'] != status_filter.lower():
                continue
            if error_only and not metadata['is_error']:
                continue

            log_data = self.log_embeddings[i]
            results.append({
                'log_id': log_data.log_id,
                'similarity': float(similarity),
                'extracted_text': log_data.extracted_text,
                'original_log': log_data.original_log,
                'metadata': log_data.metadata,
                'index': i
            })

        # Sort by similarity (descending) and limit results
        results.sort(key=lambda x: x['similarity'], reverse=True)
        results = results[:top_k]

        search_time = time.time() - start_time
        logger.info(f"Search completed in {search_time*1000:.1f}ms, found {len(results)} results")

        return results

    def search_by_text(self, query_text: str,
                      embedding_generator,
                      top_k: int = 10,
                      **kwargs) -> List[Dict[str, Any]]:
        """
        Search using a text query (generates embedding first).

        Args:
            query_text (str): Text query
            embedding_generator: EmbeddingGenerator instance
            top_k (int): Maximum number of results
            **kwargs: Additional arguments for search()

        Returns:
            List[Dict[str, Any]]: Search results
        """
        query_embedding = embedding_generator.generate_embedding(query_text)
        return self.search(query_embedding, top_k, **kwargs)

    def find_similar_errors(self, error_log_id: str,
                           top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar error logs to a given error log.

        Args:
            error_log_id (str): ID of the error log to find similar ones for
            top_k (int): Number of similar errors to return

        Returns:
            List[Dict[str, Any]]: Similar error logs
        """
        # Find the target log
        target_log = None
        target_index = None

        for i, log in enumerate(self.log_embeddings):
            if log.log_id == error_log_id:
                target_log = log
                target_index = i
                break

        if target_log is None:
            logger.error(f"Error log with ID {error_log_id} not found")
            return []

        if not target_log.metadata.get('is_error', False):
            logger.warning(f"Log {error_log_id} is not marked as an error")

        # Search for similar errors
        results = self.search(
            target_log.embedding,
            top_k=top_k + 1,  # +1 to account for the target log itself
            error_only=True
        )

        # Remove the target log from results
        filtered_results = [r for r in results if r['index'] != target_index]
        return filtered_results[:top_k]

    def cluster_analysis(self, operation: Optional[str] = None,
                        n_clusters: int = 5) -> Dict[str, Any]:
        """
        Perform clustering analysis on log embeddings.

        Args:
            operation (Optional[str]): Focus on specific operation type
            n_clusters (int): Number of clusters to create

        Returns:
            Dict[str, Any]: Clustering analysis results
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
        except ImportError:
            logger.error("scikit-learn required for clustering analysis")
            return {}

        # Filter embeddings if operation specified
        if operation:
            indices = [i for i, meta in enumerate(self.metadata_index)
                      if meta['operation'] == operation.lower()]
            embeddings = self.embedding_matrix[indices]
            filtered_logs = [self.log_embeddings[i] for i in indices]
        else:
            embeddings = self.embedding_matrix
            filtered_logs = self.log_embeddings
            indices = list(range(len(self.log_embeddings)))

        if len(embeddings) < n_clusters:
            logger.warning(f"Not enough samples ({len(embeddings)}) for {n_clusters} clusters")
            return {}

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Analyze clusters
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append({
                'log_id': filtered_logs[i].log_id,
                'text': filtered_logs[i].extracted_text[:100],
                'operation': filtered_logs[i].metadata.get('operation'),
                'status': filtered_logs[i].metadata.get('status'),
                'is_error': filtered_logs[i].metadata.get('is_error')
            })

        # Calculate cluster characteristics
        cluster_stats = {}
        for label, items in clusters.items():
            operations = {}
            statuses = {}
            error_count = 0

            for item in items:
                op = item['operation']
                st = item['status']
                operations[op] = operations.get(op, 0) + 1
                statuses[st] = statuses.get(st, 0) + 1
                if item['is_error']:
                    error_count += 1

            cluster_stats[label] = {
                'size': len(items),
                'error_rate': error_count / len(items),
                'top_operations': sorted(operations.items(), key=lambda x: x[1], reverse=True),
                'top_statuses': sorted(statuses.items(), key=lambda x: x[1], reverse=True),
                'sample_texts': [item['text'] for item in items[:3]]
            }

        return {
            'n_clusters': n_clusters,
            'total_samples': len(embeddings),
            'operation_filter': operation,
            'clusters': cluster_stats,
            'cluster_centers': kmeans.cluster_centers_.tolist()
        }

    def get_embedding_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the embedding space.

        Returns:
            Dict[str, Any]: Embedding statistics
        """
        if self.embedding_matrix is None:
            return {}

        # Calculate basic statistics
        mean_embedding = np.mean(self.embedding_matrix, axis=0)
        std_embedding = np.std(self.embedding_matrix, axis=0)

        # Calculate pairwise similarities
        similarity_matrix = cosine_similarity(self.embedding_matrix)
        # Remove diagonal (self-similarities)
        mask = np.ones(similarity_matrix.shape, dtype=bool)
        np.fill_diagonal(mask, False)
        similarities = similarity_matrix[mask]

        # Operation-wise statistics
        operation_stats = {}
        for meta in self.metadata_index:
            op = meta['operation']
            if op not in operation_stats:
                operation_stats[op] = {'count': 0, 'error_count': 0}
            operation_stats[op]['count'] += 1
            if meta['is_error']:
                operation_stats[op]['error_count'] += 1

        # Calculate error rates
        for op_stats in operation_stats.values():
            op_stats['error_rate'] = op_stats['error_count'] / op_stats['count']

        return {
            'total_embeddings': len(self.log_embeddings),
            'embedding_dimensions': self.embedding_matrix.shape[1],
            'mean_similarity': float(np.mean(similarities)),
            'std_similarity': float(np.std(similarities)),
            'min_similarity': float(np.min(similarities)),
            'max_similarity': float(np.max(similarities)),
            'operation_statistics': operation_stats,
            'embedding_mean_norm': float(np.mean(np.linalg.norm(self.embedding_matrix, axis=1))),
            'embedding_std_norm': float(np.std(np.linalg.norm(self.embedding_matrix, axis=1)))
        }

    def explain_similarity(self, log_id1: str, log_id2: str) -> Dict[str, Any]:
        """
        Explain why two logs are similar or different.

        Args:
            log_id1 (str): First log ID
            log_id2 (str): Second log ID

        Returns:
            Dict[str, Any]: Similarity explanation
        """
        # Find the logs
        log1 = log2 = None
        for log in self.log_embeddings:
            if log.log_id == log_id1:
                log1 = log
            elif log.log_id == log_id2:
                log2 = log

        if not log1 or not log2:
            return {'error': 'One or both log IDs not found'}

        # Calculate similarity
        similarity = cosine_similarity([log1.embedding], [log2.embedding])[0][0]

        # Compare metadata
        metadata_comparison = {
            'operation_match': log1.metadata.get('operation') == log2.metadata.get('operation'),
            'status_match': log1.metadata.get('status') == log2.metadata.get('status'),
            'both_errors': log1.metadata.get('is_error') and log2.metadata.get('is_error'),
            'text_length_diff': abs(len(log1.extracted_text) - len(log2.extracted_text))
        }

        # Find common words (simple approach)
        words1 = set(log1.extracted_text.lower().split())
        words2 = set(log2.extracted_text.lower().split())
        common_words = words1.intersection(words2)
        unique_words1 = words1 - words2
        unique_words2 = words2 - words1

        return {
            'similarity_score': float(similarity),
            'log1_summary': {
                'id': log1.log_id,
                'operation': log1.metadata.get('operation'),
                'status': log1.metadata.get('status'),
                'is_error': log1.metadata.get('is_error'),
                'text_preview': log1.extracted_text[:100]
            },
            'log2_summary': {
                'id': log2.log_id,
                'operation': log2.metadata.get('operation'),
                'status': log2.metadata.get('status'),
                'is_error': log2.metadata.get('is_error'),
                'text_preview': log2.extracted_text[:100]
            },
            'metadata_comparison': metadata_comparison,
            'text_analysis': {
                'common_words': list(common_words)[:10],  # Top 10 common words
                'unique_to_log1': list(unique_words1)[:5],
                'unique_to_log2': list(unique_words2)[:5]
            }
        }