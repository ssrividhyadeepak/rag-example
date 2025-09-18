"""
FAISS Index Manager

Manages FAISS vector indexes for ultra-fast similarity search.
Provides persistent storage, efficient batch operations, and
optimized search capabilities for ATM log embeddings.
"""

import faiss
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import logging
import time

logger = logging.getLogger(__name__)


class FAISSIndexManager:
    """
    FAISS vector index management for ATM log embeddings.

    Provides efficient vector storage, similarity search, and persistence
    for 384-dimensional sentence-transformer embeddings.
    """

    def __init__(self,
                 dimensions: int = 384,
                 index_type: str = "IndexFlatIP",
                 storage_path: str = "data/vector_store"):
        """
        Initialize FAISS index manager.

        Args:
            dimensions (int): Vector dimensions (384 for all-MiniLM-L6-v2)
            index_type (str): FAISS index type ('IndexFlatIP', 'IndexIVFFlat')
            storage_path (str): Directory to store index files
        """
        self.dimensions = dimensions
        self.index_type = index_type
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize index
        self.index = self._create_index()
        self.id_mapping: Dict[int, str] = {}  # FAISS index -> log_id
        self.reverse_mapping: Dict[str, int] = {}  # log_id -> FAISS index

        # Index statistics
        self.total_vectors = 0
        self.last_save_time = None

        # File paths
        self.index_file = self.storage_path / "faiss_index.bin"
        self.mapping_file = self.storage_path / "id_mapping.json"
        self.metadata_file = self.storage_path / "index_metadata.json"

        # Load existing index if available
        self._load_index()

        logger.info(f"FAISS index initialized: {index_type} with {dimensions} dimensions")

    def _create_index(self) -> faiss.Index:
        """Create appropriate FAISS index based on type."""
        if self.index_type == "IndexFlatIP":
            # Inner Product index for cosine similarity
            # Best for small to medium datasets (<100K vectors)
            return faiss.IndexFlatIP(self.dimensions)

        elif self.index_type == "IndexIVFFlat":
            # Inverted File index for larger datasets
            # More memory efficient but approximate search
            quantizer = faiss.IndexFlatIP(self.dimensions)
            n_clusters = 100  # Number of clusters
            index = faiss.IndexIVFFlat(quantizer, self.dimensions, n_clusters)
            return index

        elif self.index_type == "IndexPQ":
            # Product Quantization for very large datasets
            # Highly compressed but with accuracy trade-off
            m = 64  # Number of subquantizers
            bits = 8  # Bits per code
            return faiss.IndexPQ(self.dimensions, m, bits)

        else:
            logger.warning(f"Unknown index type {self.index_type}, using IndexFlatIP")
            return faiss.IndexFlatIP(self.dimensions)

    def add_vectors(self,
                   embeddings: np.ndarray,
                   log_ids: List[str],
                   batch_size: int = 1000) -> Dict[str, Any]:
        """
        Add vectors to the FAISS index with batch processing.

        Args:
            embeddings (np.ndarray): Array of embeddings (n_vectors, dimensions)
            log_ids (List[str]): Corresponding log IDs
            batch_size (int): Batch size for processing

        Returns:
            Dict[str, Any]: Addition results
        """
        if len(embeddings) != len(log_ids):
            raise ValueError("Number of embeddings must match number of log IDs")

        if embeddings.shape[1] != self.dimensions:
            raise ValueError(f"Embedding dimensions {embeddings.shape[1]} != {self.dimensions}")

        # Ensure float32 for FAISS
        embeddings = embeddings.astype('float32')

        # Normalize vectors for cosine similarity (if using IndexFlatIP)
        if self.index_type == "IndexFlatIP":
            faiss.normalize_L2(embeddings)

        start_time = time.time()
        total_added = 0
        skipped = 0

        # Process in batches
        for i in range(0, len(embeddings), batch_size):
            batch_end = min(i + batch_size, len(embeddings))
            batch_embeddings = embeddings[i:batch_end]
            batch_log_ids = log_ids[i:batch_end]

            # Check for duplicates
            batch_new_embeddings = []
            batch_new_log_ids = []
            batch_start_idx = self.index.ntotal

            for j, log_id in enumerate(batch_log_ids):
                if log_id not in self.reverse_mapping:
                    batch_new_embeddings.append(batch_embeddings[j])
                    batch_new_log_ids.append(log_id)
                else:
                    skipped += 1

            if batch_new_embeddings:
                batch_new_embeddings = np.array(batch_new_embeddings)

                # Add to FAISS index
                self.index.add(batch_new_embeddings)

                # Update mappings
                for j, log_id in enumerate(batch_new_log_ids):
                    faiss_idx = batch_start_idx + j
                    self.id_mapping[faiss_idx] = log_id
                    self.reverse_mapping[log_id] = faiss_idx

                total_added += len(batch_new_log_ids)

        self.total_vectors = self.index.ntotal
        add_time = time.time() - start_time

        logger.info(f"Added {total_added} vectors, skipped {skipped} duplicates in {add_time:.2f}s")

        return {
            "added": total_added,
            "skipped": skipped,
            "total_vectors": self.total_vectors,
            "add_time_seconds": add_time
        }

    def search_similar(self,
                      query_vector: np.ndarray,
                      top_k: int = 10,
                      min_score: float = 0.0) -> Tuple[List[str], List[float]]:
        """
        Search for similar vectors in the index.

        Args:
            query_vector (np.ndarray): Query vector (1D array)
            top_k (int): Number of results to return
            min_score (float): Minimum similarity score threshold

        Returns:
            Tuple[List[str], List[float]]: (log_ids, similarity_scores)
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty, no vectors to search")
            return [], []

        # Ensure correct shape and type
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        query_vector = query_vector.astype('float32')

        # Normalize for cosine similarity (if using IndexFlatIP)
        if self.index_type == "IndexFlatIP":
            faiss.normalize_L2(query_vector)

        start_time = time.time()

        try:
            # Perform search
            scores, indices = self.index.search(query_vector, top_k)

            # Convert to lists and filter by score
            scores = scores[0].tolist()
            indices = indices[0].tolist()

            # Filter results and map to log IDs
            result_log_ids = []
            result_scores = []

            for score, idx in zip(scores, indices):
                if score >= min_score and idx in self.id_mapping:
                    result_log_ids.append(self.id_mapping[idx])
                    result_scores.append(float(score))

            search_time = time.time() - start_time
            logger.debug(f"Search completed in {search_time*1000:.1f}ms, found {len(result_log_ids)} results")

            return result_log_ids, result_scores

        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            return [], []

    def batch_search(self,
                    query_vectors: np.ndarray,
                    top_k: int = 10) -> List[Tuple[List[str], List[float]]]:
        """
        Search multiple query vectors efficiently.

        Args:
            query_vectors (np.ndarray): Multiple query vectors (n_queries, dimensions)
            top_k (int): Number of results per query

        Returns:
            List[Tuple[List[str], List[float]]]: Results for each query
        """
        if self.index.ntotal == 0:
            return [[], []] * len(query_vectors)

        query_vectors = query_vectors.astype('float32')

        # Normalize for cosine similarity
        if self.index_type == "IndexFlatIP":
            faiss.normalize_L2(query_vectors)

        try:
            scores, indices = self.index.search(query_vectors, top_k)

            results = []
            for i in range(len(query_vectors)):
                query_log_ids = []
                query_scores = []

                for score, idx in zip(scores[i], indices[i]):
                    if idx in self.id_mapping:
                        query_log_ids.append(self.id_mapping[idx])
                        query_scores.append(float(score))

                results.append((query_log_ids, query_scores))

            return results

        except Exception as e:
            logger.error(f"Error during batch search: {e}")
            return [[], []] * len(query_vectors)

    def get_vector_by_log_id(self, log_id: str) -> Optional[np.ndarray]:
        """
        Retrieve vector by log ID.

        Args:
            log_id (str): Log ID to retrieve

        Returns:
            Optional[np.ndarray]: Vector or None if not found
        """
        if log_id not in self.reverse_mapping:
            return None

        faiss_idx = self.reverse_mapping[log_id]

        try:
            # FAISS doesn't have direct vector retrieval, so we reconstruct
            # This is mainly for debugging/validation purposes
            vector = self.index.reconstruct(faiss_idx)
            return vector
        except Exception as e:
            logger.error(f"Error retrieving vector for {log_id}: {e}")
            return None

    def remove_vector(self, log_id: str) -> bool:
        """
        Remove vector from index by log ID.
        Note: FAISS doesn't support direct removal, so we use remove_ids.

        Args:
            log_id (str): Log ID to remove

        Returns:
            bool: True if removed successfully
        """
        if log_id not in self.reverse_mapping:
            logger.warning(f"Log ID {log_id} not found in index")
            return False

        faiss_idx = self.reverse_mapping[log_id]

        try:
            # Remove from FAISS index
            self.index.remove_ids(np.array([faiss_idx]))

            # Clean up mappings
            del self.id_mapping[faiss_idx]
            del self.reverse_mapping[log_id]

            self.total_vectors = self.index.ntotal
            logger.debug(f"Removed vector for log ID: {log_id}")
            return True

        except Exception as e:
            logger.error(f"Error removing vector {log_id}: {e}")
            return False

    def save_index(self) -> bool:
        """
        Save FAISS index and mappings to disk.

        Returns:
            bool: True if saved successfully
        """
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_file))

            # Save ID mappings
            mapping_data = {
                "id_mapping": {str(k): v for k, v in self.id_mapping.items()},
                "reverse_mapping": self.reverse_mapping
            }

            with open(self.mapping_file, 'w') as f:
                json.dump(mapping_data, f, indent=2)

            # Save metadata
            metadata = {
                "index_type": self.index_type,
                "dimensions": self.dimensions,
                "total_vectors": self.total_vectors,
                "saved_at": time.time(),
                "version": "1.0"
            }

            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            self.last_save_time = time.time()
            logger.info(f"FAISS index saved to {self.storage_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
            return False

    def _load_index(self) -> bool:
        """
        Load FAISS index and mappings from disk.

        Returns:
            bool: True if loaded successfully
        """
        if not self.index_file.exists():
            logger.info("No existing FAISS index found, starting fresh")
            return False

        try:
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_file))

            # Load mappings
            if self.mapping_file.exists():
                with open(self.mapping_file, 'r') as f:
                    mapping_data = json.load(f)

                # Convert string keys back to integers for id_mapping
                self.id_mapping = {int(k): v for k, v in mapping_data["id_mapping"].items()}
                self.reverse_mapping = mapping_data["reverse_mapping"]

            # Load metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.total_vectors = metadata.get("total_vectors", self.index.ntotal)

            self.total_vectors = self.index.ntotal
            logger.info(f"Loaded FAISS index with {self.total_vectors} vectors")
            return True

        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            # Reset to empty index on load failure
            self.index = self._create_index()
            self.id_mapping = {}
            self.reverse_mapping = {}
            self.total_vectors = 0
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get FAISS index statistics.

        Returns:
            Dict[str, Any]: Index statistics
        """
        return {
            "index_type": self.index_type,
            "dimensions": self.dimensions,
            "total_vectors": self.total_vectors,
            "storage_path": str(self.storage_path),
            "index_file_size_mb": self._get_file_size_mb(self.index_file),
            "mapping_file_size_mb": self._get_file_size_mb(self.mapping_file),
            "last_save_time": self.last_save_time,
            "memory_usage_estimate_mb": (self.total_vectors * self.dimensions * 4) / (1024 * 1024)
        }

    def _get_file_size_mb(self, file_path: Path) -> float:
        """Get file size in MB."""
        try:
            if file_path.exists():
                return file_path.stat().st_size / (1024 * 1024)
            return 0.0
        except Exception:
            return 0.0

    def rebuild_index(self, embeddings: np.ndarray, log_ids: List[str]) -> bool:
        """
        Rebuild the entire index from scratch.

        Args:
            embeddings (np.ndarray): All embeddings
            log_ids (List[str]): Corresponding log IDs

        Returns:
            bool: True if rebuilt successfully
        """
        try:
            # Create new index
            self.index = self._create_index()
            self.id_mapping = {}
            self.reverse_mapping = {}

            # Add all vectors
            result = self.add_vectors(embeddings, log_ids)

            logger.info(f"Index rebuilt with {result['added']} vectors")
            return True

        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
            return False

    def clear_index(self) -> bool:
        """
        Clear all vectors from the index.

        Returns:
            bool: True if cleared successfully
        """
        try:
            self.index = self._create_index()
            self.id_mapping = {}
            self.reverse_mapping = {}
            self.total_vectors = 0

            logger.info("FAISS index cleared")
            return True

        except Exception as e:
            logger.error(f"Error clearing index: {e}")
            return False

    def optimize_index(self) -> bool:
        """
        Optimize index for better performance (train IVF if needed).

        Returns:
            bool: True if optimized successfully
        """
        try:
            if hasattr(self.index, 'train') and not self.index.is_trained:
                # For IVF indexes, we need training data
                if self.total_vectors > 0:
                    # Get some vectors for training
                    training_vectors = []
                    for i in range(min(1000, self.total_vectors)):
                        try:
                            vector = self.index.reconstruct(i)
                            training_vectors.append(vector)
                        except:
                            continue

                    if training_vectors:
                        training_data = np.array(training_vectors)
                        self.index.train(training_data)
                        logger.info("Index trained successfully")

            return True

        except Exception as e:
            logger.error(f"Error optimizing index: {e}")
            return False