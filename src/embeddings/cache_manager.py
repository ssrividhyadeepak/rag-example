"""
Cache Manager Module

Manages caching and persistence of embeddings to avoid regenerating them
and improve performance for repeated operations.
"""

import os
import json
import pickle
import hashlib
import time
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manages caching of embeddings and processed data for performance optimization.
    """

    def __init__(self, cache_dir: str = "cache", enable_disk_cache: bool = True):
        """
        Initialize the cache manager.

        Args:
            cache_dir (str): Directory to store cache files
            enable_disk_cache (bool): Whether to enable disk-based caching
        """
        self.cache_dir = Path(cache_dir)
        self.enable_disk_cache = enable_disk_cache
        self.memory_cache: Dict[str, Any] = {}

        if self.enable_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.embeddings_dir = self.cache_dir / "embeddings"
            self.metadata_dir = self.cache_dir / "metadata"

            self.embeddings_dir.mkdir(exist_ok=True)
            self.metadata_dir.mkdir(exist_ok=True)

        logger.info(f"CacheManager initialized with cache_dir: {self.cache_dir}")

    def _generate_cache_key(self, data: Union[str, Dict, List]) -> str:
        """
        Generate a unique cache key for the given data.

        Args:
            data: Data to generate key for

        Returns:
            str: SHA256 hash as cache key
        """
        if isinstance(data, str):
            content = data
        else:
            content = json.dumps(data, sort_keys=True)

        return hashlib.sha256(content.encode()).hexdigest()

    def _get_cache_filepath(self, cache_key: str, file_type: str = "embeddings") -> Path:
        """
        Get the filepath for a cache key.

        Args:
            cache_key (str): Cache key
            file_type (str): Type of cache file ('embeddings' or 'metadata')

        Returns:
            Path: Cache file path
        """
        if file_type == "embeddings":
            return self.embeddings_dir / f"{cache_key}.npy"
        elif file_type == "metadata":
            return self.metadata_dir / f"{cache_key}.json"
        else:
            return self.cache_dir / f"{cache_key}.pkl"

    def cache_embedding(self, text: str, embedding: np.ndarray,
                       model_info: Dict[str, Any]) -> str:
        """
        Cache an embedding for a text with model information.

        Args:
            text (str): Original text
            embedding (np.ndarray): Generated embedding
            model_info (Dict[str, Any]): Information about the model used

        Returns:
            str: Cache key for the stored embedding
        """
        # Create cache key from text and model info
        cache_data = {
            "text": text,
            "model_name": model_info.get("model_name"),
            "model_type": model_info.get("model_type"),
            "dimensions": model_info.get("dimensions")
        }
        cache_key = self._generate_cache_key(cache_data)

        # Store in memory cache
        self.memory_cache[cache_key] = {
            "embedding": embedding,
            "text": text,
            "model_info": model_info,
            "cached_at": time.time()
        }

        # Store on disk if enabled
        if self.enable_disk_cache:
            try:
                # Save embedding as numpy file
                embedding_path = self._get_cache_filepath(cache_key, "embeddings")
                np.save(embedding_path, embedding)

                # Save metadata as JSON
                metadata = {
                    "text": text,
                    "model_info": model_info,
                    "embedding_shape": embedding.shape,
                    "cached_at": time.time(),
                    "cache_key": cache_key
                }
                metadata_path = self._get_cache_filepath(cache_key, "metadata")
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)

                logger.debug(f"Cached embedding for text: {text[:50]}...")

            except Exception as e:
                logger.error(f"Failed to cache embedding to disk: {e}")

        return cache_key

    def get_cached_embedding(self, text: str, model_info: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Retrieve cached embedding for text and model.

        Args:
            text (str): Original text
            model_info (Dict[str, Any]): Model information

        Returns:
            Optional[np.ndarray]: Cached embedding or None if not found
        """
        # Generate cache key
        cache_data = {
            "text": text,
            "model_name": model_info.get("model_name"),
            "model_type": model_info.get("model_type"),
            "dimensions": model_info.get("dimensions")
        }
        cache_key = self._generate_cache_key(cache_data)

        # Check memory cache first
        if cache_key in self.memory_cache:
            logger.debug(f"Found embedding in memory cache for: {text[:50]}...")
            return self.memory_cache[cache_key]["embedding"]

        # Check disk cache
        if self.enable_disk_cache:
            try:
                embedding_path = self._get_cache_filepath(cache_key, "embeddings")
                metadata_path = self._get_cache_filepath(cache_key, "metadata")

                if embedding_path.exists() and metadata_path.exists():
                    # Load embedding
                    embedding = np.load(embedding_path)

                    # Load and verify metadata
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)

                    # Verify the cached data matches request
                    if (metadata["text"] == text and
                        metadata["model_info"]["model_name"] == model_info.get("model_name")):

                        # Add to memory cache for faster future access
                        self.memory_cache[cache_key] = {
                            "embedding": embedding,
                            "text": text,
                            "model_info": model_info,
                            "cached_at": metadata["cached_at"]
                        }

                        logger.debug(f"Found embedding in disk cache for: {text[:50]}...")
                        return embedding

            except Exception as e:
                logger.error(f"Failed to load cached embedding: {e}")

        return None

    def cache_batch_embeddings(self, texts: List[str], embeddings: np.ndarray,
                              model_info: Dict[str, Any]) -> List[str]:
        """
        Cache multiple embeddings at once.

        Args:
            texts (List[str]): List of texts
            embeddings (np.ndarray): Array of embeddings
            model_info (Dict[str, Any]): Model information

        Returns:
            List[str]: List of cache keys
        """
        cache_keys = []

        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            try:
                cache_key = self.cache_embedding(text, embedding, model_info)
                cache_keys.append(cache_key)
            except Exception as e:
                logger.error(f"Failed to cache embedding for text {i}: {e}")
                cache_keys.append(None)

        logger.info(f"Cached {len([k for k in cache_keys if k])} out of {len(texts)} embeddings")
        return cache_keys

    def get_batch_cached_embeddings(self, texts: List[str],
                                   model_info: Dict[str, Any]) -> Dict[int, np.ndarray]:
        """
        Get cached embeddings for multiple texts.

        Args:
            texts (List[str]): List of texts to check
            model_info (Dict[str, Any]): Model information

        Returns:
            Dict[int, np.ndarray]: Mapping of text indices to cached embeddings
        """
        cached_embeddings = {}

        for i, text in enumerate(texts):
            embedding = self.get_cached_embedding(text, model_info)
            if embedding is not None:
                cached_embeddings[i] = embedding

        logger.info(f"Found {len(cached_embeddings)} cached embeddings out of {len(texts)} requested")
        return cached_embeddings

    def clear_memory_cache(self) -> None:
        """Clear the in-memory cache."""
        self.memory_cache.clear()
        logger.info("Memory cache cleared")

    def clear_disk_cache(self) -> None:
        """Clear the disk cache."""
        if not self.enable_disk_cache:
            logger.warning("Disk cache is disabled")
            return

        try:
            # Remove all embedding files
            for file_path in self.embeddings_dir.glob("*.npy"):
                file_path.unlink()

            # Remove all metadata files
            for file_path in self.metadata_dir.glob("*.json"):
                file_path.unlink()

            logger.info("Disk cache cleared")

        except Exception as e:
            logger.error(f"Failed to clear disk cache: {e}")

    def clear_all_cache(self) -> None:
        """Clear both memory and disk cache."""
        self.clear_memory_cache()
        self.clear_disk_cache()

    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the cache usage.

        Returns:
            Dict[str, Any]: Cache statistics
        """
        stats = {
            "memory_cache_size": len(self.memory_cache),
            "disk_cache_enabled": self.enable_disk_cache,
            "cache_directory": str(self.cache_dir)
        }

        if self.enable_disk_cache and self.cache_dir.exists():
            # Count disk cache files
            embedding_files = list(self.embeddings_dir.glob("*.npy"))
            metadata_files = list(self.metadata_dir.glob("*.json"))

            stats.update({
                "disk_embedding_files": len(embedding_files),
                "disk_metadata_files": len(metadata_files),
                "disk_cache_size_mb": sum(f.stat().st_size for f in embedding_files + metadata_files) / (1024 * 1024)
            })

        # Memory cache details
        if self.memory_cache:
            cache_ages = [time.time() - item["cached_at"] for item in self.memory_cache.values()]
            stats.update({
                "avg_cache_age_seconds": sum(cache_ages) / len(cache_ages),
                "oldest_cache_age_seconds": max(cache_ages),
                "newest_cache_age_seconds": min(cache_ages)
            })

        return stats

    def cleanup_old_cache(self, max_age_hours: int = 24) -> int:
        """
        Clean up cache entries older than specified age.

        Args:
            max_age_hours (int): Maximum age in hours

        Returns:
            int: Number of entries cleaned up
        """
        max_age_seconds = max_age_hours * 3600
        current_time = time.time()
        cleanup_count = 0

        # Clean memory cache
        expired_keys = []
        for key, item in self.memory_cache.items():
            if current_time - item["cached_at"] > max_age_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            del self.memory_cache[key]
            cleanup_count += 1

        # Clean disk cache
        if self.enable_disk_cache:
            try:
                for metadata_file in self.metadata_dir.glob("*.json"):
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)

                        if current_time - metadata["cached_at"] > max_age_seconds:
                            # Remove both metadata and embedding files
                            cache_key = metadata["cache_key"]
                            embedding_file = self._get_cache_filepath(cache_key, "embeddings")

                            metadata_file.unlink()
                            if embedding_file.exists():
                                embedding_file.unlink()
                            cleanup_count += 1

                    except Exception as e:
                        logger.error(f"Error cleaning up cache file {metadata_file}: {e}")

            except Exception as e:
                logger.error(f"Error during disk cache cleanup: {e}")

        logger.info(f"Cleaned up {cleanup_count} old cache entries (older than {max_age_hours}h)")
        return cleanup_count

    def save_cache_index(self) -> None:
        """Save an index of all cached items for faster lookup."""
        if not self.enable_disk_cache:
            return

        try:
            index_data = {
                "created_at": time.time(),
                "memory_cache_keys": list(self.memory_cache.keys()),
                "disk_cache_files": []
            }

            # Add disk cache file information
            for metadata_file in self.metadata_dir.glob("*.json"):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)

                    index_data["disk_cache_files"].append({
                        "cache_key": metadata["cache_key"],
                        "text_preview": metadata["text"][:100],
                        "model_name": metadata["model_info"]["model_name"],
                        "cached_at": metadata["cached_at"]
                    })
                except Exception as e:
                    logger.warning(f"Could not read metadata file {metadata_file}: {e}")

            # Save index
            index_file = self.cache_dir / "cache_index.json"
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2)

            logger.info(f"Saved cache index with {len(index_data['disk_cache_files'])} entries")

        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")

    def load_cache_index(self) -> Optional[Dict[str, Any]]:
        """
        Load the cache index for inspection.

        Returns:
            Optional[Dict[str, Any]]: Cache index data or None if not found
        """
        if not self.enable_disk_cache:
            return None

        try:
            index_file = self.cache_dir / "cache_index.json"
            if index_file.exists():
                with open(index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load cache index: {e}")

        return None