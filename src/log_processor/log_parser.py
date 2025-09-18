"""
Log Parser Module

Parses ATM log entries and extracts explicit fields including timestamp,
session IDs, operation details, and metadata.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ParsedLogEntry:
    """
    Represents a parsed ATM log entry with structured access to fields.
    """

    def __init__(self, raw_log: Dict[str, Any]):
        """
        Initialize a parsed log entry.

        Args:
            raw_log (Dict[str, Any]): Raw log entry from JSON
        """
        self.raw_log = raw_log
        self._parse_fields()

    def _parse_fields(self) -> None:
        """Parse and extract fields from the raw log entry."""
        # Core required fields
        self.timestamp = self.raw_log.get('timestamp', '')
        self.session_id = self.raw_log.get('session_id', '')
        self.customer_session_id = self.raw_log.get('customer_session_id', '')
        self.event_type = self.raw_log.get('event_type', '')
        self.operation = self.raw_log.get('operation', '')
        self.status = self.raw_log.get('status', '')
        self.message = self.raw_log.get('message', '')

        # Optional fields
        self.error_code = self.raw_log.get('error_code', '')
        self.atm_id = self.raw_log.get('atm_id', '')
        self.amount = self.raw_log.get('amount')

        # Metadata
        self.metadata = self.raw_log.get('metadata', {})

        # Derived fields
        self.card_number = self.metadata.get('card_number', '') if self.metadata else ''
        self.location = self.metadata.get('location', '') if self.metadata else ''

    @property
    def is_valid(self) -> bool:
        """
        Check if the log entry has required fields.

        Returns:
            bool: True if log entry has minimum required fields
        """
        required_fields = [self.timestamp, self.session_id, self.operation, self.status]
        return all(field for field in required_fields)

    @property
    def is_error(self) -> bool:
        """
        Check if this log entry represents an error condition.

        Returns:
            bool: True if this is an error log
        """
        error_statuses = ['denied', 'failed', 'error', 'timeout', 'rejected']
        return (self.status.lower() in error_statuses or
                bool(self.error_code) or
                'error' in self.message.lower())

    @property
    def parsed_timestamp(self) -> Optional[datetime]:
        """
        Parse timestamp into datetime object.

        Returns:
            Optional[datetime]: Parsed timestamp or None if invalid
        """
        if not self.timestamp:
            return None

        try:
            # Try ISO format first
            if 'T' in self.timestamp:
                return datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
            else:
                # Try other common formats
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f']:
                    try:
                        return datetime.strptime(self.timestamp, fmt)
                    except ValueError:
                        continue
        except Exception as e:
            logger.warning(f"Could not parse timestamp '{self.timestamp}': {e}")

        return None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert parsed entry back to dictionary format.

        Returns:
            Dict[str, Any]: Dictionary representation of parsed entry
        """
        return {
            'timestamp': self.timestamp,
            'session_id': self.session_id,
            'customer_session_id': self.customer_session_id,
            'event_type': self.event_type,
            'operation': self.operation,
            'status': self.status,
            'message': self.message,
            'error_code': self.error_code,
            'atm_id': self.atm_id,
            'amount': self.amount,
            'metadata': self.metadata,
            'is_error': self.is_error,
            'parsed_timestamp': self.parsed_timestamp.isoformat() if self.parsed_timestamp else None
        }


class LogParser:
    """
    Parses ATM log entries and provides structured access to fields.
    """

    def __init__(self):
        """Initialize the LogParser."""
        self.parsed_logs: List[ParsedLogEntry] = []

    def parse_log_entry(self, log_entry: Dict[str, Any]) -> ParsedLogEntry:
        """
        Parse a single log entry.

        Args:
            log_entry (Dict[str, Any]): Raw log entry

        Returns:
            ParsedLogEntry: Parsed log entry object
        """
        return ParsedLogEntry(log_entry)

    def parse_logs(self, log_entries: List[Dict[str, Any]]) -> List[ParsedLogEntry]:
        """
        Parse multiple log entries.

        Args:
            log_entries (List[Dict[str, Any]]): List of raw log entries

        Returns:
            List[ParsedLogEntry]: List of parsed log entries
        """
        parsed_entries = []
        invalid_count = 0

        for i, log_entry in enumerate(log_entries):
            try:
                parsed = self.parse_log_entry(log_entry)
                parsed_entries.append(parsed)

                if not parsed.is_valid:
                    invalid_count += 1
                    logger.warning(f"Log entry {i} is missing required fields")

            except Exception as e:
                logger.error(f"Failed to parse log entry {i}: {e}")
                continue

        logger.info(f"Parsed {len(parsed_entries)} log entries, {invalid_count} with missing fields")
        self.parsed_logs = parsed_entries
        return parsed_entries

    def filter_by_operation(self, operation: str) -> List[ParsedLogEntry]:
        """
        Filter parsed logs by operation type.

        Args:
            operation (str): Operation type to filter by

        Returns:
            List[ParsedLogEntry]: Filtered log entries
        """
        return [log for log in self.parsed_logs
                if log.operation.lower() == operation.lower()]

    def filter_by_status(self, status: str) -> List[ParsedLogEntry]:
        """
        Filter parsed logs by status.

        Args:
            status (str): Status to filter by

        Returns:
            List[ParsedLogEntry]: Filtered log entries
        """
        return [log for log in self.parsed_logs
                if log.status.lower() == status.lower()]

    def filter_errors_only(self) -> List[ParsedLogEntry]:
        """
        Get only error log entries.

        Returns:
            List[ParsedLogEntry]: Error log entries only
        """
        return [log for log in self.parsed_logs if log.is_error]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about parsed logs.

        Returns:
            Dict[str, Any]: Statistics about the logs
        """
        if not self.parsed_logs:
            return {}

        total_logs = len(self.parsed_logs)
        error_logs = len(self.filter_errors_only())

        operations = {}
        statuses = {}

        for log in self.parsed_logs:
            operations[log.operation] = operations.get(log.operation, 0) + 1
            statuses[log.status] = statuses.get(log.status, 0) + 1

        return {
            'total_logs': total_logs,
            'error_logs': error_logs,
            'error_rate': error_logs / total_logs if total_logs > 0 else 0,
            'operations': operations,
            'statuses': statuses
        }