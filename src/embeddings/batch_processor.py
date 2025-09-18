"""
Batch Processor Module

Efficiently processes multiple ATM log entries and generates embeddings for them.
Integrates with the log_processor component to create a complete pipeline.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging
import time
import json
from pathlib import Path

# Import log processor components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from log_processor import LogReader, LogParser, TextExtractor, LogValidator
from .embedding_generator import EmbeddingGenerator

logger = logging.getLogger(__name__)


class LogEmbeddingData:
    """
    Container for log entry with its embedding and metadata.
    """

    def __init__(self, log_id: str, original_log: Dict[str, Any],
                 extracted_text: str, embedding: np.ndarray,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize log embedding data.

        Args:
            log_id (str): Unique identifier for the log entry
            original_log (Dict[str, Any]): Original log entry data
            extracted_text (str): Text extracted for embedding
            embedding (np.ndarray): Generated embedding vector
            metadata (Optional[Dict[str, Any]]): Additional metadata
        """
        self.log_id = log_id
        self.original_log = original_log
        self.extracted_text = extracted_text
        self.embedding = embedding
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for serialization."""
        return {
            'log_id': self.log_id,
            'original_log': self.original_log,
            'extracted_text': self.extracted_text,
            'embedding': self.embedding.tolist() if isinstance(self.embedding, np.ndarray) else self.embedding,
            'metadata': self.metadata
        }


class BatchProcessor:
    """
    Processes ATM logs in batches to generate embeddings efficiently.
    """

    def __init__(self, embedding_model: str = 'fast',
                 include_timestamp: bool = True,
                 include_session_info: bool = True,
                 include_metadata: bool = True):
        """
        Initialize the batch processor.

        Args:
            embedding_model (str): Model type for embeddings ('fast' or 'quality')
            include_timestamp (bool): Include timestamp in text extraction
            include_session_info (bool): Include session info in text extraction
            include_metadata (bool): Include metadata in text extraction
        """
        self.embedding_generator = EmbeddingGenerator(model_type=embedding_model)
        self.log_reader = LogReader()
        self.log_validator = LogValidator(strict_mode=False)
        self.log_parser = LogParser()
        self.text_extractor = TextExtractor(
            include_timestamp=include_timestamp,
            include_session_info=include_session_info,
            include_metadata=include_metadata
        )

        self.processed_logs: List[LogEmbeddingData] = []

    def process_log_files(self, log_directory: str = "data/logs",
                         file_pattern: str = "*.json") -> List[LogEmbeddingData]:
        """
        Process all log files in a directory and generate embeddings.

        Args:
            log_directory (str): Directory containing log files
            file_pattern (str): Pattern to match log files

        Returns:
            List[LogEmbeddingData]: List of processed log entries with embeddings
        """
        logger.info(f"Processing log files from: {log_directory}")

        # Step 1: Read all log files
        self.log_reader.log_directory = log_directory
        raw_logs = self.log_reader.read_all_logs(file_pattern)
        logger.info(f"Read {len(raw_logs)} log entries")

        if not raw_logs:
            logger.warning("No log entries found")
            return []

        # Step 2: Validate logs
        validation_results = self.log_validator.validate_logs(raw_logs)
        valid_logs = self.log_validator.get_valid_logs(raw_logs)

        logger.info(f"Validation: {validation_results['valid_logs']}/{validation_results['total_logs']} valid")

        if not valid_logs:
            logger.error("No valid log entries found")
            return []

        # Step 3: Parse logs
        parsed_logs = self.log_parser.parse_logs(valid_logs)
        logger.info(f"Parsed {len(parsed_logs)} log entries")

        # Step 4: Extract text
        extracted_data = self.text_extractor.extract_batch(parsed_logs)
        logger.info(f"Extracted text from {len(extracted_data)} entries")

        # Step 5: Generate embeddings
        texts = [item['extracted_text'] for item in extracted_data]
        logger.info("Generating embeddings...")

        embeddings = self.embedding_generator.generate_embeddings(texts, show_progress=True)

        # Step 6: Create LogEmbeddingData objects
        processed_logs = []
        for i, (item, embedding) in enumerate(zip(extracted_data, embeddings)):
            log_embedding = LogEmbeddingData(
                log_id=item['log_id'],
                original_log=item['original_log'],
                extracted_text=item['extracted_text'],
                embedding=embedding,
                metadata={
                    'text_length': item['text_length'],
                    'is_error': item['is_error'],
                    'operation': item['operation'],
                    'status': item['status'],
                    'embedding_model': self.embedding_generator.model_type,
                    'embedding_dimensions': len(embedding)
                }
            )
            processed_logs.append(log_embedding)

        self.processed_logs = processed_logs
        logger.info(f"✅ Successfully processed {len(processed_logs)} log entries with embeddings")

        return processed_logs

    def process_single_log(self, log_entry: Dict[str, Any]) -> Optional[LogEmbeddingData]:
        """
        Process a single log entry and generate its embedding.

        Args:
            log_entry (Dict[str, Any]): Single log entry

        Returns:
            Optional[LogEmbeddingData]: Processed log with embedding or None if failed
        """
        try:
            # Validate
            is_valid, errors = self.log_validator.validate_single_log(log_entry)
            if not is_valid:
                logger.warning(f"Invalid log entry: {errors}")
                return None

            # Parse
            parsed_log = self.log_parser.parse_log_entry(log_entry)

            # Extract text
            extracted_text = self.text_extractor.extract_text(parsed_log)

            # Generate embedding
            embedding = self.embedding_generator.generate_embedding(extracted_text)

            # Create result
            log_id = f"single_{int(time.time() * 1000)}"
            return LogEmbeddingData(
                log_id=log_id,
                original_log=parsed_log.to_dict(),
                extracted_text=extracted_text,
                embedding=embedding,
                metadata={
                    'is_error': parsed_log.is_error,
                    'operation': parsed_log.operation,
                    'status': parsed_log.status
                }
            )

        except Exception as e:
            logger.error(f"Failed to process single log: {e}")
            return None

    def filter_by_operation(self, operation: str) -> List[LogEmbeddingData]:
        """
        Filter processed logs by operation type.

        Args:
            operation (str): Operation type to filter by

        Returns:
            List[LogEmbeddingData]: Filtered logs
        """
        return [log for log in self.processed_logs
                if log.metadata.get('operation', '').lower() == operation.lower()]

    def filter_by_status(self, status: str) -> List[LogEmbeddingData]:
        """
        Filter processed logs by status.

        Args:
            status (str): Status to filter by

        Returns:
            List[LogEmbeddingData]: Filtered logs
        """
        return [log for log in self.processed_logs
                if log.metadata.get('status', '').lower() == status.lower()]

    def filter_errors_only(self) -> List[LogEmbeddingData]:
        """
        Get only error logs with embeddings.

        Returns:
            List[LogEmbeddingData]: Error logs only
        """
        return [log for log in self.processed_logs
                if log.metadata.get('is_error', False)]

    def find_similar_logs(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find logs similar to a query text.

        Args:
            query_text (str): Query text to search for
            top_k (int): Number of results to return

        Returns:
            List[Dict[str, Any]]: Similar logs with similarity scores
        """
        if not self.processed_logs:
            logger.warning("No processed logs available for similarity search")
            return []

        # Generate embedding for query
        query_embedding = self.embedding_generator.generate_embedding(query_text)

        # Get all embeddings
        all_embeddings = np.array([log.embedding for log in self.processed_logs])

        # Find similar embeddings
        similar_results = self.embedding_generator.find_most_similar(
            query_embedding, all_embeddings, top_k
        )

        # Add log data to results
        results = []
        for result in similar_results:
            idx = result['index']
            log_data = self.processed_logs[idx]

            results.append({
                'log_id': log_data.log_id,
                'similarity': result['similarity'],
                'extracted_text': log_data.extracted_text,
                'operation': log_data.metadata.get('operation'),
                'status': log_data.metadata.get('status'),
                'is_error': log_data.metadata.get('is_error'),
                'original_log': log_data.original_log
            })

        return results

    def save_embeddings(self, filepath: str) -> None:
        """
        Save processed logs with embeddings to file.

        Args:
            filepath (str): Path to save the data
        """
        if not self.processed_logs:
            logger.warning("No processed logs to save")
            return

        try:
            save_data = {
                'metadata': {
                    'total_logs': len(self.processed_logs),
                    'embedding_model': self.embedding_generator.model_type,
                    'embedding_dimensions': self.embedding_generator.dimensions,
                    'created_at': time.time()
                },
                'logs': [log.to_dict() for log in self.processed_logs]
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(self.processed_logs)} processed logs to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")

    def load_embeddings(self, filepath: str) -> bool:
        """
        Load processed logs with embeddings from file.

        Args:
            filepath (str): Path to load the data from

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                save_data = json.load(f)

            loaded_logs = []
            for log_data in save_data['logs']:
                # Convert embedding back to numpy array
                embedding = np.array(log_data['embedding'], dtype=np.float32)

                log_embedding = LogEmbeddingData(
                    log_id=log_data['log_id'],
                    original_log=log_data['original_log'],
                    extracted_text=log_data['extracted_text'],
                    embedding=embedding,
                    metadata=log_data['metadata']
                )
                loaded_logs.append(log_embedding)

            self.processed_logs = loaded_logs
            metadata = save_data['metadata']

            logger.info(f"Loaded {len(loaded_logs)} processed logs from {filepath}")
            logger.info(f"Model used: {metadata.get('embedding_model')}")
            logger.info(f"Dimensions: {metadata.get('embedding_dimensions')}")

            return True

        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about processed logs.

        Returns:
            Dict[str, Any]: Statistics about the processed data
        """
        if not self.processed_logs:
            return {}

        operations = {}
        statuses = {}
        error_count = 0
        text_lengths = []

        for log in self.processed_logs:
            operation = log.metadata.get('operation', 'unknown')
            status = log.metadata.get('status', 'unknown')
            is_error = log.metadata.get('is_error', False)

            operations[operation] = operations.get(operation, 0) + 1
            statuses[status] = statuses.get(status, 0) + 1

            if is_error:
                error_count += 1

            text_lengths.append(len(log.extracted_text))

        return {
            'total_logs': len(self.processed_logs),
            'error_logs': error_count,
            'error_rate': error_count / len(self.processed_logs),
            'operations': operations,
            'statuses': statuses,
            'avg_text_length': sum(text_lengths) / len(text_lengths),
            'embedding_model': self.embedding_generator.model_type,
            'embedding_dimensions': self.embedding_generator.dimensions
        }