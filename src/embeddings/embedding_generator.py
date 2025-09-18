"""
Embedding Generator Module

Generates vector embeddings from text using local sentence-transformers models.
No external API calls required - completely offline after initial model download.
"""

import numpy as np
from typing import List, Union, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import logging
import time
import os

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generate embeddings from text using local sentence-transformers models.

    Supports multiple models with different trade-offs:
    - all-MiniLM-L6-v2: Fast, 384 dimensions, 22MB (default)
    - all-mpnet-base-v2: Higher quality, 768 dimensions, 420MB
    """

    # Supported models with their characteristics
    MODELS = {
        'fast': {
            'name': 'all-MiniLM-L6-v2',
            'dimensions': 384,
            'size_mb': 22,
            'description': 'Fast inference, good for most use cases'
        },
        'quality': {
            'name': 'all-mpnet-base-v2',
            'dimensions': 768,
            'size_mb': 420,
            'description': 'Higher quality embeddings, slower inference'
        }
    }

    def __init__(self, model_type: str = 'fast', cache_dir: Optional[str] = None):
        """
        Initialize the embedding generator.

        Args:
            model_type (str): 'fast' or 'quality' - determines which model to use
            cache_dir (Optional[str]): Directory to cache model files
        """
        self.model_type = model_type
        self.cache_dir = cache_dir
        self.model = None
        self.model_info = self.MODELS.get(model_type)

        if not self.model_info:
            raise ValueError(f"Invalid model_type: {model_type}. Choose from: {list(self.MODELS.keys())}")

        self.model_name = self.model_info['name']
        self.dimensions = self.model_info['dimensions']

        logger.info(f"Initialized EmbeddingGenerator with model: {self.model_name}")
        logger.info(f"Model info: {self.model_info['description']}")

    def load_model(self) -> None:
        """
        Load the sentence transformer model.
        Downloads model on first use (~22MB for fast model).
        """
        if self.model is not None:
            logger.info("Model already loaded")
            return

        logger.info(f"Loading model: {self.model_name}")
        start_time = time.time()

        try:
            # Set cache directory if specified
            if self.cache_dir:
                os.environ['SENTENCE_TRANSFORMERS_HOME'] = self.cache_dir

            self.model = SentenceTransformer(self.model_name)
            load_time = time.time() - start_time

            logger.info(f"✅ Model loaded successfully in {load_time:.2f}s")
            logger.info(f"Model dimensions: {self.dimensions}")

        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text (str): Input text to embed

        Returns:
            np.ndarray: Embedding vector
        """
        if self.model is None:
            self.load_model()

        if not text or not text.strip():
            logger.warning("Empty text provided, returning zero vector")
            return np.zeros(self.dimensions)

        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)  # Use float32 to save memory

        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {text[:50]}... Error: {e}")
            return np.zeros(self.dimensions, dtype=np.float32)

    def generate_embeddings(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts (List[str]): List of texts to embed
            show_progress (bool): Whether to show progress bar

        Returns:
            np.ndarray: Array of embeddings with shape (n_texts, dimensions)
        """
        if self.model is None:
            self.load_model()

        if not texts:
            logger.warning("Empty text list provided")
            return np.empty((0, self.dimensions), dtype=np.float32)

        # Filter out empty texts and keep track of indices
        valid_texts = []
        valid_indices = []

        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)

        if not valid_texts:
            logger.warning("All texts are empty, returning zero vectors")
            return np.zeros((len(texts), self.dimensions), dtype=np.float32)

        try:
            start_time = time.time()

            # Generate embeddings for valid texts
            valid_embeddings = self.model.encode(
                valid_texts,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
                batch_size=32  # Process in batches for memory efficiency
            )

            # Create result array and fill in valid embeddings
            all_embeddings = np.zeros((len(texts), self.dimensions), dtype=np.float32)
            all_embeddings[valid_indices] = valid_embeddings.astype(np.float32)

            generation_time = time.time() - start_time
            logger.info(f"Generated {len(texts)} embeddings in {generation_time:.2f}s")
            logger.info(f"Average time per embedding: {generation_time/len(texts)*1000:.1f}ms")

            return all_embeddings

        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            return np.zeros((len(texts), self.dimensions), dtype=np.float32)

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1 (np.ndarray): First embedding
            embedding2 (np.ndarray): Second embedding

        Returns:
            float: Cosine similarity score (-1 to 1)
        """
        try:
            # Normalize vectors
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0

    def find_most_similar(self, query_embedding: np.ndarray,
                         candidate_embeddings: np.ndarray,
                         top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find most similar embeddings to a query embedding.

        Args:
            query_embedding (np.ndarray): Query embedding
            candidate_embeddings (np.ndarray): Array of candidate embeddings
            top_k (int): Number of top results to return

        Returns:
            List[Dict[str, Any]]: List of similarity results with indices and scores
        """
        if candidate_embeddings.shape[0] == 0:
            return []

        try:
            # Calculate similarities to all candidates
            similarities = []
            for i, candidate in enumerate(candidate_embeddings):
                sim = self.similarity(query_embedding, candidate)
                similarities.append({
                    'index': i,
                    'similarity': sim
                })

            # Sort by similarity (descending) and return top_k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:top_k]

        except Exception as e:
            logger.error(f"Failed to find similar embeddings: {e}")
            return []

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dict[str, Any]: Model information
        """
        return {
            'model_type': self.model_type,
            'model_name': self.model_name,
            'dimensions': self.dimensions,
            'size_mb': self.model_info['size_mb'],
            'description': self.model_info['description'],
            'is_loaded': self.model is not None
        }

    def benchmark(self, sample_texts: List[str]) -> Dict[str, Any]:
        """
        Benchmark the embedding generation performance.

        Args:
            sample_texts (List[str]): Sample texts for benchmarking

        Returns:
            Dict[str, Any]: Benchmark results
        """
        if not sample_texts:
            return {}

        logger.info(f"Benchmarking with {len(sample_texts)} texts...")

        # Single embedding benchmark
        start_time = time.time()
        single_embedding = self.generate_embedding(sample_texts[0])
        single_time = time.time() - start_time

        # Batch embedding benchmark
        start_time = time.time()
        batch_embeddings = self.generate_embeddings(sample_texts, show_progress=False)
        batch_time = time.time() - start_time

        # Similarity benchmark
        if len(sample_texts) >= 2:
            start_time = time.time()
            sim_score = self.similarity(batch_embeddings[0], batch_embeddings[1])
            similarity_time = time.time() - start_time
        else:
            sim_score = 0.0
            similarity_time = 0.0

        results = {
            'model_info': self.get_model_info(),
            'single_embedding_time_ms': single_time * 1000,
            'batch_embedding_time_ms': batch_time * 1000,
            'avg_embedding_time_ms': (batch_time / len(sample_texts)) * 1000,
            'similarity_time_ms': similarity_time * 1000,
            'embeddings_per_second': len(sample_texts) / batch_time,
            'sample_similarity': sim_score,
            'embedding_shape': single_embedding.shape
        }

        logger.info(f"Benchmark results: {results['avg_embedding_time_ms']:.1f}ms per embedding")
        return results