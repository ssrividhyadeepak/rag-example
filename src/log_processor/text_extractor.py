"""
Text Extractor Module

Converts structured ATM log data into meaningful text strings
suitable for embedding generation and semantic search.
"""

from typing import List, Dict, Any, Optional
from .log_parser import ParsedLogEntry
import logging

logger = logging.getLogger(__name__)


class TextExtractor:
    """
    Extracts and formats text from parsed ATM log entries for embedding generation.
    """

    def __init__(self, include_timestamp: bool = True,
                 include_session_info: bool = True,
                 include_metadata: bool = True):
        """
        Initialize the TextExtractor.

        Args:
            include_timestamp (bool): Whether to include timestamp in text
            include_session_info (bool): Whether to include session IDs
            include_metadata (bool): Whether to include metadata fields
        """
        self.include_timestamp = include_timestamp
        self.include_session_info = include_session_info
        self.include_metadata = include_metadata

    def extract_text(self, parsed_log: ParsedLogEntry) -> str:
        """
        Extract meaningful text from a parsed log entry.

        Args:
            parsed_log (ParsedLogEntry): Parsed log entry

        Returns:
            str: Formatted text suitable for embeddings
        """
        text_parts = []

        # Add timestamp if requested
        if self.include_timestamp and parsed_log.timestamp:
            text_parts.append(f"Time: {parsed_log.timestamp}")

        # Add session information if requested
        if self.include_session_info:
            if parsed_log.session_id:
                text_parts.append(f"Session: {parsed_log.session_id}")
            if parsed_log.customer_session_id:
                text_parts.append(f"Customer Session: {parsed_log.customer_session_id}")

        # Add core operation information
        if parsed_log.operation and parsed_log.status:
            text_parts.append(f"{parsed_log.operation} {parsed_log.status}")

        # Add main message
        if parsed_log.message:
            text_parts.append(parsed_log.message)

        # Add error code if present
        if parsed_log.error_code:
            text_parts.append(f"Error Code: {parsed_log.error_code}")

        # Add ATM ID
        if parsed_log.atm_id:
            text_parts.append(f"ATM: {parsed_log.atm_id}")

        # Add amount if present
        if parsed_log.amount is not None:
            text_parts.append(f"Amount: {parsed_log.amount}")

        # Add metadata if requested
        if self.include_metadata:
            if parsed_log.location:
                text_parts.append(f"Location: {parsed_log.location}")
            if parsed_log.card_number:
                text_parts.append(f"Card: {parsed_log.card_number}")

        # Add event type for additional context
        if parsed_log.event_type:
            text_parts.append(f"Event: {parsed_log.event_type}")

        # Join all parts with periods and spaces
        text = ". ".join(filter(None, text_parts))
        return text.strip()

    def extract_batch(self, parsed_logs: List[ParsedLogEntry]) -> List[Dict[str, Any]]:
        """
        Extract text from multiple parsed log entries.

        Args:
            parsed_logs (List[ParsedLogEntry]): List of parsed log entries

        Returns:
            List[Dict[str, Any]]: List of dictionaries with original log and extracted text
        """
        results = []

        for i, parsed_log in enumerate(parsed_logs):
            try:
                text = self.extract_text(parsed_log)
                result = {
                    'log_id': f"log_{i:06d}",
                    'original_log': parsed_log.to_dict(),
                    'extracted_text': text,
                    'text_length': len(text),
                    'is_error': parsed_log.is_error,
                    'operation': parsed_log.operation,
                    'status': parsed_log.status
                }
                results.append(result)

            except Exception as e:
                logger.error(f"Failed to extract text from log {i}: {e}")
                continue

        logger.info(f"Extracted text from {len(results)} log entries")
        return results

    def extract_summary_text(self, parsed_log: ParsedLogEntry) -> str:
        """
        Extract a shorter summary text for quick search/display.

        Args:
            parsed_log (ParsedLogEntry): Parsed log entry

        Returns:
            str: Short summary text
        """
        parts = []

        if parsed_log.operation and parsed_log.status:
            parts.append(f"{parsed_log.operation} {parsed_log.status}")

        if parsed_log.error_code:
            parts.append(f"({parsed_log.error_code})")

        if parsed_log.amount is not None:
            parts.append(f"${parsed_log.amount}")

        if parsed_log.atm_id:
            parts.append(f"@ {parsed_log.atm_id}")

        return " ".join(parts)

    def extract_contextual_text(self, parsed_log: ParsedLogEntry) -> str:
        """
        Extract text optimized for RAG context understanding.
        Focuses on problem description and actionable information.

        Args:
            parsed_log (ParsedLogEntry): Parsed log entry

        Returns:
            str: Context-optimized text
        """
        context_parts = []

        # Lead with the problem/situation
        if parsed_log.is_error:
            context_parts.append(f"ATM Error: {parsed_log.operation} operation failed")
        else:
            context_parts.append(f"ATM Operation: {parsed_log.operation} {parsed_log.status}")

        # Add specific details
        if parsed_log.message:
            context_parts.append(f"Details: {parsed_log.message}")

        if parsed_log.error_code:
            context_parts.append(f"Error Code: {parsed_log.error_code}")

        # Add context information
        context_info = []
        if parsed_log.amount is not None:
            context_info.append(f"amount ${parsed_log.amount}")
        if parsed_log.atm_id:
            context_info.append(f"ATM {parsed_log.atm_id}")
        if parsed_log.location:
            context_info.append(f"location {parsed_log.location}")

        if context_info:
            context_parts.append(f"Context: {', '.join(context_info)}")

        return ". ".join(context_parts)

    def get_extraction_statistics(self, extracted_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about extracted text data.

        Args:
            extracted_data (List[Dict[str, Any]]): Extracted text data

        Returns:
            Dict[str, Any]: Statistics about the extracted text
        """
        if not extracted_data:
            return {}

        text_lengths = [item['text_length'] for item in extracted_data]
        error_count = sum(1 for item in extracted_data if item['is_error'])

        operations = {}
        for item in extracted_data:
            op = item['operation']
            operations[op] = operations.get(op, 0) + 1

        return {
            'total_entries': len(extracted_data),
            'error_entries': error_count,
            'avg_text_length': sum(text_lengths) / len(text_lengths),
            'min_text_length': min(text_lengths),
            'max_text_length': max(text_lengths),
            'operations': operations
        }