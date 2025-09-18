"""
Log Validator Module

Validates ATM log entries to ensure they contain required fields
and meet quality standards for processing.
"""

from typing import List, Dict, Any, Set, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class LogValidator:
    """
    Validates ATM log entries for completeness and correctness.
    """

    # Required fields that must be present in every log entry
    REQUIRED_FIELDS = {
        'timestamp',
        'session_id',
        'operation',
        'status',
        'message'
    }

    # Optional but recommended fields
    RECOMMENDED_FIELDS = {
        'customer_session_id',
        'event_type',
        'atm_id'
    }

    # Valid operation types
    VALID_OPERATIONS = {
        'withdrawal',
        'deposit',
        'balance_inquiry',
        'transfer',
        'pin_change',
        'card_operation',
        'system_operation'
    }

    # Valid status types
    VALID_STATUSES = {
        'success',
        'completed',
        'denied',
        'failed',
        'error',
        'timeout',
        'cancelled',
        'pending'
    }

    def __init__(self, strict_mode: bool = False):
        """
        Initialize the LogValidator.

        Args:
            strict_mode (bool): If True, enforce strict validation rules
        """
        self.strict_mode = strict_mode
        self.validation_errors: List[Dict[str, Any]] = []

    def validate_single_log(self, log_entry: Dict[str, Any], log_index: int = 0) -> Tuple[bool, List[str]]:
        """
        Validate a single log entry.

        Args:
            log_entry (Dict[str, Any]): Log entry to validate
            log_index (int): Index of the log entry for error reporting

        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []

        # Check required fields
        missing_required = self.REQUIRED_FIELDS - set(log_entry.keys())
        if missing_required:
            errors.append(f"Missing required fields: {', '.join(missing_required)}")

        # Check empty required fields
        for field in self.REQUIRED_FIELDS:
            if field in log_entry and not str(log_entry[field]).strip():
                errors.append(f"Required field '{field}' is empty")

        # Validate timestamp format
        if 'timestamp' in log_entry:
            timestamp_error = self._validate_timestamp(log_entry['timestamp'])
            if timestamp_error:
                errors.append(timestamp_error)

        # Validate operation
        if 'operation' in log_entry:
            operation_error = self._validate_operation(log_entry['operation'])
            if operation_error:
                errors.append(operation_error)

        # Validate status
        if 'status' in log_entry:
            status_error = self._validate_status(log_entry['status'])
            if status_error:
                errors.append(status_error)

        # Validate session ID format
        if 'session_id' in log_entry:
            session_error = self._validate_session_id(log_entry['session_id'])
            if session_error:
                errors.append(session_error)

        # Validate amount if present
        if 'amount' in log_entry and log_entry['amount'] is not None:
            amount_error = self._validate_amount(log_entry['amount'])
            if amount_error:
                errors.append(amount_error)

        # Check for recommended fields in strict mode
        if self.strict_mode:
            missing_recommended = self.RECOMMENDED_FIELDS - set(log_entry.keys())
            if missing_recommended:
                errors.append(f"Missing recommended fields: {', '.join(missing_recommended)}")

        # Log validation results
        if errors:
            self.validation_errors.append({
                'log_index': log_index,
                'errors': errors,
                'log_entry': log_entry
            })

        return len(errors) == 0, errors

    def validate_logs(self, log_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate multiple log entries.

        Args:
            log_entries (List[Dict[str, Any]]): List of log entries to validate

        Returns:
            Dict[str, Any]: Validation results summary
        """
        self.validation_errors = []
        valid_logs = []
        invalid_logs = []

        for i, log_entry in enumerate(log_entries):
            is_valid, errors = self.validate_single_log(log_entry, i)

            if is_valid:
                valid_logs.append(i)
            else:
                invalid_logs.append({
                    'index': i,
                    'errors': errors,
                    'log_entry': log_entry
                })

        total_logs = len(log_entries)
        valid_count = len(valid_logs)
        invalid_count = len(invalid_logs)

        logger.info(f"Validated {total_logs} logs: {valid_count} valid, {invalid_count} invalid")

        return {
            'total_logs': total_logs,
            'valid_logs': valid_count,
            'invalid_logs': invalid_count,
            'validation_rate': valid_count / total_logs if total_logs > 0 else 0,
            'valid_log_indices': valid_logs,
            'invalid_log_details': invalid_logs,
            'errors_summary': self._get_error_summary()
        }

    def _validate_timestamp(self, timestamp: str) -> Optional[str]:
        """Validate timestamp format."""
        if not timestamp:
            return "Timestamp is empty"

        try:
            # Try ISO format first
            if 'T' in timestamp:
                datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                # Try other common formats
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f']:
                    try:
                        datetime.strptime(timestamp, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    return f"Invalid timestamp format: {timestamp}"
        except Exception:
            return f"Invalid timestamp format: {timestamp}"

        return None

    def _validate_operation(self, operation: str) -> Optional[str]:
        """Validate operation type."""
        if not operation:
            return "Operation is empty"

        if operation.lower() not in self.VALID_OPERATIONS:
            return f"Invalid operation: {operation}. Valid operations: {', '.join(self.VALID_OPERATIONS)}"

        return None

    def _validate_status(self, status: str) -> Optional[str]:
        """Validate status type."""
        if not status:
            return "Status is empty"

        if status.lower() not in self.VALID_STATUSES:
            return f"Invalid status: {status}. Valid statuses: {', '.join(self.VALID_STATUSES)}"

        return None

    def _validate_session_id(self, session_id: str) -> Optional[str]:
        """Validate session ID format."""
        if not session_id:
            return "Session ID is empty"

        # Basic format validation - should be non-empty string
        if len(session_id.strip()) < 3:
            return "Session ID too short (minimum 3 characters)"

        return None

    def _validate_amount(self, amount: Any) -> Optional[str]:
        """Validate amount field."""
        try:
            amount_val = float(amount)
            if amount_val < 0:
                return "Amount cannot be negative"
            if amount_val > 1000000:  # Reasonable upper limit
                return "Amount exceeds reasonable limit (1,000,000)"
        except (ValueError, TypeError):
            return f"Invalid amount format: {amount}"

        return None

    def _get_error_summary(self) -> Dict[str, int]:
        """Get summary of error types."""
        error_counts = {}

        for validation_error in self.validation_errors:
            for error in validation_error['errors']:
                # Extract error type from error message
                if 'Missing required fields' in error:
                    error_type = 'missing_required_fields'
                elif 'empty' in error.lower():
                    error_type = 'empty_fields'
                elif 'Invalid timestamp' in error:
                    error_type = 'invalid_timestamp'
                elif 'Invalid operation' in error:
                    error_type = 'invalid_operation'
                elif 'Invalid status' in error:
                    error_type = 'invalid_status'
                elif 'Invalid amount' in error:
                    error_type = 'invalid_amount'
                else:
                    error_type = 'other'

                error_counts[error_type] = error_counts.get(error_type, 0) + 1

        return error_counts

    def get_valid_logs(self, log_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get only valid log entries from a list.

        Args:
            log_entries (List[Dict[str, Any]]): List of log entries

        Returns:
            List[Dict[str, Any]]: List of valid log entries only
        """
        valid_logs = []

        for log_entry in log_entries:
            is_valid, _ = self.validate_single_log(log_entry)
            if is_valid:
                valid_logs.append(log_entry)

        logger.info(f"Filtered to {len(valid_logs)} valid logs from {len(log_entries)} total")
        return valid_logs

    def get_validation_report(self) -> str:
        """
        Generate a human-readable validation report.

        Returns:
            str: Formatted validation report
        """
        if not self.validation_errors:
            return "All logs passed validation!"

        report_lines = [
            f"Validation Report - {len(self.validation_errors)} logs with errors:",
            "=" * 50
        ]

        error_summary = self._get_error_summary()
        for error_type, count in error_summary.items():
            report_lines.append(f"- {error_type.replace('_', ' ').title()}: {count}")

        return "\n".join(report_lines)