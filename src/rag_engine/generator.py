"""
Response Generator for ATM RAG System

Generates intelligent, contextual responses to ATM-related queries
using retrieved context and specialized prompt templates. Handles
different response types and formats responses appropriately.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from .prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """
    Generates contextual responses for ATM queries using retrieved context.

    Combines retrieved ATM logs with specialized templates to produce
    helpful, accurate responses for troubleshooting and analysis.
    """

    def __init__(self):
        """Initialize response generator with prompt templates."""
        self.prompt_templates = PromptTemplates()

        # Response configuration
        self.max_response_length = 2000
        self.max_examples_per_response = 5
        self.include_log_details = True

        logger.info("Response generator initialized")

    def generate_response(self,
                         query: str,
                         context: Dict[str, Any],
                         response_type: str = "auto") -> Dict[str, Any]:
        """
        Generate a response based on query and retrieved context.

        Args:
            query: User's original query
            context: Retrieved context from ContextRetriever
            response_type: Type of response ("auto", "troubleshooting", "error", "analysis", "info")

        Returns:
            Dict containing generated response and metadata
        """
        try:
            # Auto-detect response type if needed
            if response_type == "auto":
                response_type = self._detect_response_type(query, context)

            # Generate response based on type
            response_text = self._generate_typed_response(
                query, context, response_type
            )

            # Post-process response
            final_response = self._post_process_response(response_text)

            # Build response metadata
            metadata = self._build_response_metadata(query, context, response_type)

            return {
                "response": final_response,
                "response_type": response_type,
                "query": query,
                "confidence": self._calculate_confidence(context),
                "sources_count": len(context.get("relevant_logs", [])),
                "metadata": metadata,
                "generated_at": datetime.utcnow()
            }

        except Exception as e:
            logger.error(f"Error generating response for query '{query}': {e}")
            return self._generate_error_response(query, str(e))

    def generate_troubleshooting_response(self,
                                        query: str,
                                        context: Dict[str, Any]) -> str:
        """
        Generate troubleshooting-focused response.

        Args:
            query: User's troubleshooting query
            context: Retrieved context with relevant logs

        Returns:
            Troubleshooting response string
        """
        relevant_logs = context.get("relevant_logs", [])

        if not relevant_logs:
            return self.prompt_templates.generate_troubleshooting_response(
                query, [], "No relevant ATM logs found for this issue."
            )

        # Extract key information
        error_logs = [log for log in relevant_logs if self._is_error_log(log)]
        similar_issues = relevant_logs[:self.max_examples_per_response]

        # Build context summary
        context_summary = self._build_troubleshooting_context(similar_issues)

        return self.prompt_templates.generate_troubleshooting_response(
            query, similar_issues, context_summary
        )

    def generate_error_code_response(self,
                                   error_code: str,
                                   context: Dict[str, Any]) -> str:
        """
        Generate error code explanation response.

        Args:
            error_code: Specific error code
            context: Retrieved context with relevant logs

        Returns:
            Error code explanation response
        """
        relevant_logs = context.get("relevant_logs", [])
        examples = [log for log in relevant_logs if log.get("error_code") == error_code]

        return self.prompt_templates.generate_error_code_response(
            error_code, examples[:self.max_examples_per_response]
        )

    def generate_analysis_response(self,
                                  query: str,
                                  context: Dict[str, Any]) -> str:
        """
        Generate analytical response with trends and patterns.

        Args:
            query: Analysis query
            context: Retrieved context with relevant logs

        Returns:
            Analysis response string
        """
        relevant_logs = context.get("relevant_logs", [])

        if not relevant_logs:
            return "No data available for analysis."

        # Analyze patterns
        analysis_data = self._analyze_log_patterns(relevant_logs)

        return self.prompt_templates.generate_analysis_response(
            query, analysis_data
        )

    def generate_informational_response(self,
                                      query: str,
                                      context: Dict[str, Any]) -> str:
        """
        Generate informational response.

        Args:
            query: Information query
            context: Retrieved context

        Returns:
            Informational response string
        """
        context_summary = context.get("context_summary", "")
        relevant_logs = context.get("relevant_logs", [])

        return self.prompt_templates.generate_informational_response(
            query, context_summary, relevant_logs[:3]
        )

    def _detect_response_type(self, query: str, context: Dict[str, Any]) -> str:
        """
        Auto-detect the appropriate response type based on query and context.

        Args:
            query: User's query
            context: Retrieved context

        Returns:
            Detected response type
        """
        query_lower = query.lower()

        # Error code detection
        if any(keyword in query_lower for keyword in ["error", "code", "failed", "denied"]):
            return "error"

        # Troubleshooting detection
        if any(keyword in query_lower for keyword in ["fix", "solve", "troubleshoot", "problem", "issue", "why"]):
            return "troubleshooting"

        # Analysis detection
        if any(keyword in query_lower for keyword in ["analyze", "trend", "pattern", "statistics", "how many", "count"]):
            return "analysis"

        # Check context for error patterns
        relevant_logs = context.get("relevant_logs", [])
        if relevant_logs and any(self._is_error_log(log) for log in relevant_logs):
            return "troubleshooting"

        # Default to informational
        return "info"

    def _generate_typed_response(self,
                               query: str,
                               context: Dict[str, Any],
                               response_type: str) -> str:
        """
        Generate response for specific type.

        Args:
            query: User's query
            context: Retrieved context
            response_type: Type of response to generate

        Returns:
            Generated response text
        """
        if response_type == "troubleshooting":
            return self.generate_troubleshooting_response(query, context)

        elif response_type == "error":
            # Try to extract error code from query
            error_code = self._extract_error_code(query)
            if error_code:
                return self.generate_error_code_response(error_code, context)
            else:
                return self.generate_troubleshooting_response(query, context)

        elif response_type == "analysis":
            return self.generate_analysis_response(query, context)

        elif response_type == "info":
            return self.generate_informational_response(query, context)

        else:
            # Fallback to informational
            return self.generate_informational_response(query, context)

    def _post_process_response(self, response: str) -> str:
        """
        Post-process generated response for quality and length.

        Args:
            response: Raw generated response

        Returns:
            Post-processed response
        """
        # Trim to max length
        if len(response) > self.max_response_length:
            # Find last complete sentence within limit
            truncated = response[:self.max_response_length]
            last_period = truncated.rfind('.')

            if last_period > self.max_response_length * 0.8:
                response = truncated[:last_period + 1] + "\n\n[Response truncated for brevity]"
            else:
                response = truncated + "..."

        # Clean up extra whitespace
        response = ' '.join(response.split())

        # Ensure proper formatting
        if not response.endswith('.'):
            response += '.'

        return response

    def _build_response_metadata(self,
                               query: str,
                               context: Dict[str, Any],
                               response_type: str) -> Dict[str, Any]:
        """
        Build metadata for the generated response.

        Args:
            query: User's query
            context: Retrieved context
            response_type: Response type used

        Returns:
            Response metadata dictionary
        """
        relevant_logs = context.get("relevant_logs", [])

        metadata = {
            "response_type": response_type,
            "sources_used": len(relevant_logs),
            "time_range_covered": self._get_time_range(relevant_logs),
            "operations_covered": self._get_unique_operations(relevant_logs),
            "atms_covered": self._get_unique_atms(relevant_logs),
            "error_logs_found": sum(1 for log in relevant_logs if self._is_error_log(log))
        }

        return metadata

    def _calculate_confidence(self, context: Dict[str, Any]) -> float:
        """
        Calculate confidence score for the response.

        Args:
            context: Retrieved context

        Returns:
            Confidence score between 0 and 1
        """
        relevant_logs = context.get("relevant_logs", [])

        if not relevant_logs:
            return 0.0

        # Base confidence on number of relevant logs and their scores
        log_count = len(relevant_logs)
        avg_score = sum(log.get("retrieval_score", 0) for log in relevant_logs) / log_count

        # Confidence factors
        count_factor = min(log_count / 5.0, 1.0)  # More logs = higher confidence
        score_factor = avg_score  # Higher similarity scores = higher confidence

        # Recent logs boost confidence
        recent_factor = sum(1 for log in relevant_logs if self._is_recent_log(log)) / log_count

        confidence = (count_factor * 0.4 + score_factor * 0.4 + recent_factor * 0.2)
        return min(confidence, 1.0)

    def _analyze_log_patterns(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze patterns in retrieved logs for analysis responses.

        Args:
            logs: List of relevant log entries

        Returns:
            Analysis data dictionary
        """
        if not logs:
            return {}

        # Count patterns
        operations = {}
        statuses = {}
        error_codes = {}
        atms = {}

        for log in logs:
            # Count operations
            operation = log.get("operation", "unknown")
            operations[operation] = operations.get(operation, 0) + 1

            # Count statuses
            status = log.get("status", "unknown")
            statuses[status] = statuses.get(status, 0) + 1

            # Count error codes
            error_code = log.get("error_code")
            if error_code:
                error_codes[error_code] = error_codes.get(error_code, 0) + 1

            # Count ATMs
            atm_id = log.get("atm_id")
            if atm_id:
                atms[atm_id] = atms.get(atm_id, 0) + 1

        return {
            "total_logs": len(logs),
            "operations": operations,
            "statuses": statuses,
            "error_codes": error_codes,
            "atms": atms,
            "time_span": self._get_time_range(logs)
        }

    def _build_troubleshooting_context(self, logs: List[Dict[str, Any]]) -> str:
        """
        Build context summary for troubleshooting responses.

        Args:
            logs: Relevant log entries

        Returns:
            Context summary string
        """
        if not logs:
            return "No similar issues found in recent logs."

        # Analyze common patterns
        error_codes = set()
        operations = set()
        atms = set()

        for log in logs:
            if log.get("error_code"):
                error_codes.add(log["error_code"])
            if log.get("operation"):
                operations.add(log["operation"])
            if log.get("atm_id"):
                atms.add(log["atm_id"])

        context_parts = [f"Found {len(logs)} similar cases."]

        if error_codes:
            context_parts.append(f"Common error codes: {', '.join(error_codes)}.")

        if operations:
            context_parts.append(f"Affected operations: {', '.join(operations)}.")

        if atms:
            atm_list = list(atms)[:3]
            if len(atms) > 3:
                context_parts.append(f"Affected ATMs: {', '.join(atm_list)} and {len(atms) - 3} others.")
            else:
                context_parts.append(f"Affected ATMs: {', '.join(atm_list)}.")

        return " ".join(context_parts)

    def _is_error_log(self, log: Dict[str, Any]) -> bool:
        """Check if a log entry represents an error."""
        return (
            log.get("status") in ["denied", "failed", "error", "timeout"] or
            bool(log.get("error_code")) or
            log.get("metadata", {}).get("is_error", False)
        )

    def _is_recent_log(self, log: Dict[str, Any]) -> bool:
        """Check if a log entry is recent (within 7 days)."""
        if "timestamp" not in log:
            return False

        log_time = log["timestamp"]
        if isinstance(log_time, str):
            try:
                log_time = datetime.fromisoformat(log_time.replace('Z', '+00:00'))
            except:
                return False

        age = datetime.utcnow() - log_time.replace(tzinfo=None)
        return age.days <= 7

    def _extract_error_code(self, query: str) -> Optional[str]:
        """Extract error code from query if present."""
        import re
        # Look for common error code patterns
        patterns = [
            r'\b([A-Z]{2,}_[A-Z_]+)\b',  # DDL_EXCEEDED pattern
            r'\berror\s+code\s*:?\s*([A-Z0-9_]+)\b',
            r'\bcode\s*:?\s*([A-Z0-9_]+)\b'
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        return None

    def _get_time_range(self, logs: List[Dict[str, Any]]) -> Optional[str]:
        """Get time range covered by logs."""
        if not logs:
            return None

        timestamps = []
        for log in logs:
            if "timestamp" in log:
                timestamps.append(log["timestamp"])

        if not timestamps:
            return None

        min_time = min(timestamps)
        max_time = max(timestamps)

        if min_time == max_time:
            return f"Single timestamp: {min_time}"
        else:
            return f"From {min_time} to {max_time}"

    def _get_unique_operations(self, logs: List[Dict[str, Any]]) -> List[str]:
        """Get unique operations from logs."""
        operations = set()
        for log in logs:
            if "operation" in log:
                operations.add(log["operation"])
        return list(operations)

    def _get_unique_atms(self, logs: List[Dict[str, Any]]) -> List[str]:
        """Get unique ATM IDs from logs."""
        atms = set()
        for log in logs:
            if "atm_id" in log:
                atms.add(log["atm_id"])
        return list(atms)

    def _generate_error_response(self, query: str, error: str) -> Dict[str, Any]:
        """Generate error response when generation fails."""
        return {
            "response": f"I apologize, but I encountered an error while processing your query: '{query}'. Please try rephrasing your question or contact support if the issue persists.",
            "response_type": "error",
            "query": query,
            "confidence": 0.0,
            "sources_count": 0,
            "metadata": {"error": error},
            "generated_at": datetime.utcnow()
        }