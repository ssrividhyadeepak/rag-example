"""
Entity Extractor for ATM Query Processing

Extracts relevant entities from ATM-related queries including
error codes, ATM IDs, operations, amounts, dates, and other
structured information for enhanced retrieval and filtering.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """Represents an extracted entity from a query."""
    entity_type: str
    value: str
    confidence: float
    start_pos: int
    end_pos: int
    normalized_value: Optional[Any] = None


@dataclass
class EntityExtractionResult:
    """Result of entity extraction from a query."""
    entities: List[ExtractedEntity] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    temporal_context: Optional[Dict[str, Any]] = None
    extraction_metadata: Dict[str, Any] = field(default_factory=dict)


class EntityExtractor:
    """
    Extracts structured entities from ATM-related queries.

    Identifies and normalizes entities like error codes, ATM IDs,
    operations, amounts, dates, and other relevant information
    for improved query processing.
    """

    def __init__(self):
        """Initialize entity extractor with patterns and rules."""
        self.entity_patterns = self._build_entity_patterns()
        self.normalization_rules = self._build_normalization_rules()
        self.temporal_expressions = self._build_temporal_patterns()

        logger.info("Entity extractor initialized")

    def extract_entities(self, query: str) -> EntityExtractionResult:
        """
        Extract all entities from a query.

        Args:
            query: User's query text

        Returns:
            EntityExtractionResult with extracted entities and filters
        """
        entities = []
        query_lower = query.lower()

        # Extract each entity type
        for entity_type, pattern_info in self.entity_patterns.items():
            found_entities = self._extract_entity_type(
                query, query_lower, entity_type, pattern_info
            )
            entities.extend(found_entities)

        # Build MongoDB filters from entities
        filters = self._build_filters_from_entities(entities)

        # Extract temporal context
        temporal_context = self._extract_temporal_context(query_lower, entities)

        # Build extraction metadata
        metadata = {
            "total_entities": len(entities),
            "entity_types_found": list(set(e.entity_type for e in entities)),
            "query_length": len(query),
            "extraction_confidence": self._calculate_overall_confidence(entities)
        }

        return EntityExtractionResult(
            entities=entities,
            filters=filters,
            temporal_context=temporal_context,
            extraction_metadata=metadata
        )

    def _build_entity_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Build regex patterns for different entity types."""
        return {
            "error_code": {
                "patterns": [
                    r"\b([A-Z]{2,}_[A-Z_]+)\b",  # DDL_EXCEEDED, CARD_ERROR
                    r"\berror\s+code\s*:?\s*([A-Z0-9_]+)\b",
                    r"\bcode\s*:?\s*([A-Z0-9_]+)\b",
                    r"\b(TIMEOUT|NETWORK_ERROR|CARD_ERROR|PIN_ERROR)\b"
                ],
                "confidence_base": 0.9,
                "normalize": True
            },

            "atm_id": {
                "patterns": [
                    r"\b(ATM[0-9]+)\b",
                    r"\batm\s*([0-9]+)\b",
                    r"\b(ATM-[0-9]+)\b",
                    r"\bmachine\s*([0-9]+)\b"
                ],
                "confidence_base": 0.8,
                "normalize": True
            },

            "operation": {
                "patterns": [
                    r"\b(withdrawal|withdrawals)\b",
                    r"\b(deposit|deposits)\b",
                    r"\b(balance|balance inquiry)\b",
                    r"\b(transfer|transfers)\b",
                    r"\b(pin change|pin_change)\b",
                    r"\b(cash advance)\b"
                ],
                "confidence_base": 0.85,
                "normalize": True
            },

            "status": {
                "patterns": [
                    r"\b(successful|success)\b",
                    r"\b(failed|failure)\b",
                    r"\b(denied|denial)\b",
                    r"\b(timeout|timed out)\b",
                    r"\b(error|errors)\b",
                    r"\b(completed|complete)\b",
                    r"\b(pending|processing)\b"
                ],
                "confidence_base": 0.7,
                "normalize": True
            },

            "amount": {
                "patterns": [
                    r"\$\s*([0-9]+(?:\.[0-9]{2})?)\b",
                    r"\b([0-9]+(?:\.[0-9]{2})?)\s*dollars?\b",
                    r"\bamount\s*:?\s*\$?\s*([0-9]+(?:\.[0-9]{2})?)\b"
                ],
                "confidence_base": 0.8,
                "normalize": True
            },

            "session_id": {
                "patterns": [
                    r"\b(SES_[A-Z0-9]+)\b",
                    r"\bsession\s*:?\s*([A-Z0-9_]+)\b",
                    r"\bsession\s+id\s*:?\s*([A-Z0-9_]+)\b"
                ],
                "confidence_base": 0.9,
                "normalize": False
            },

            "customer_id": {
                "patterns": [
                    r"\bcustomer\s*:?\s*([A-Z0-9_]+)\b",
                    r"\buser\s*:?\s*([A-Z0-9_]+)\b",
                    r"\bcust\s*:?\s*([A-Z0-9_]+)\b"
                ],
                "confidence_base": 0.7,
                "normalize": False
            },

            "date": {
                "patterns": [
                    r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b",
                    r"\b(\d{4}-\d{2}-\d{2})\b",
                    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b",
                    r"\b\d{1,2}\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b"
                ],
                "confidence_base": 0.8,
                "normalize": True
            },

            "time_range": {
                "patterns": [
                    r"\b(today|yesterday)\b",
                    r"\blast\s+(hour|day|week|month|year)\b",
                    r"\bpast\s+(\d+)\s+(hours?|days?|weeks?|months?)\b",
                    r"\bin\s+the\s+last\s+(\d+)\s+(hours?|days?|weeks?|months?)\b",
                    r"\bthis\s+(morning|afternoon|evening|week|month)\b"
                ],
                "confidence_base": 0.7,
                "normalize": True
            }
        }

    def _build_normalization_rules(self) -> Dict[str, callable]:
        """Build normalization functions for different entity types."""
        return {
            "error_code": lambda x: x.upper().strip(),
            "atm_id": lambda x: f"ATM{x}" if x.isdigit() else x.upper(),
            "operation": self._normalize_operation,
            "status": self._normalize_status,
            "amount": lambda x: float(x.replace("$", "").replace(",", "")),
            "date": self._normalize_date,
            "time_range": self._normalize_time_range
        }

    def _build_temporal_patterns(self) -> Dict[str, str]:
        """Build patterns for temporal expression extraction."""
        return {
            "relative_time": r"\b(today|yesterday|last\s+(?:hour|day|week|month)|past\s+\d+\s+(?:hours?|days?))\b",
            "specific_date": r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
            "time_period": r"\b(?:from|between|since)\s+.*?(?:to|until|and)\s+.*?\b"
        }

    def _extract_entity_type(self,
                           original_query: str,
                           query_lower: str,
                           entity_type: str,
                           pattern_info: Dict[str, Any]) -> List[ExtractedEntity]:
        """
        Extract entities of a specific type from query.

        Args:
            original_query: Original query text
            query_lower: Lowercase query text
            entity_type: Type of entity to extract
            pattern_info: Pattern configuration

        Returns:
            List of extracted entities
        """
        entities = []
        patterns = pattern_info["patterns"]
        confidence_base = pattern_info["confidence_base"]
        normalize = pattern_info.get("normalize", False)

        for pattern in patterns:
            for match in re.finditer(pattern, query_lower, re.IGNORECASE):
                # Get the matched text
                matched_text = match.group(1) if match.groups() else match.group(0)

                # Calculate confidence based on pattern specificity
                confidence = confidence_base
                if len(matched_text) > 10:  # Longer matches get higher confidence
                    confidence = min(confidence + 0.1, 1.0)

                # Normalize if needed
                normalized_value = None
                if normalize and entity_type in self.normalization_rules:
                    try:
                        normalized_value = self.normalization_rules[entity_type](matched_text)
                    except Exception as e:
                        logger.warning(f"Error normalizing {entity_type} '{matched_text}': {e}")
                        normalized_value = matched_text

                entity = ExtractedEntity(
                    entity_type=entity_type,
                    value=matched_text,
                    confidence=confidence,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    normalized_value=normalized_value
                )

                entities.append(entity)

        return entities

    def _build_filters_from_entities(self, entities: List[ExtractedEntity]) -> Dict[str, Any]:
        """
        Build MongoDB filters from extracted entities.

        Args:
            entities: List of extracted entities

        Returns:
            MongoDB filter dictionary
        """
        filters = {}

        for entity in entities:
            if entity.confidence < 0.5:  # Skip low-confidence entities
                continue

            value = entity.normalized_value if entity.normalized_value is not None else entity.value

            # Map entity types to MongoDB fields
            if entity.entity_type == "error_code":
                filters["error_code"] = value
            elif entity.entity_type == "atm_id":
                filters["atm_id"] = value
            elif entity.entity_type == "operation":
                filters["operation"] = value
            elif entity.entity_type == "status":
                filters["status"] = value
            elif entity.entity_type == "amount":
                # Handle amount ranges or exact matches
                filters["amount"] = value
            elif entity.entity_type == "session_id":
                filters["session_id"] = value
            elif entity.entity_type == "customer_id":
                filters["customer_session_id"] = value

        return filters

    def _extract_temporal_context(self,
                                 query_lower: str,
                                 entities: List[ExtractedEntity]) -> Optional[Dict[str, Any]]:
        """
        Extract temporal context from query and entities.

        Args:
            query_lower: Lowercase query text
            entities: Extracted entities

        Returns:
            Temporal context dictionary or None
        """
        temporal_context = {}

        # Look for time range entities
        time_entities = [e for e in entities if e.entity_type in ["date", "time_range"]]

        if time_entities:
            for entity in time_entities:
                if entity.normalized_value:
                    if isinstance(entity.normalized_value, dict):
                        temporal_context.update(entity.normalized_value)
                    else:
                        temporal_context["timestamp"] = entity.normalized_value

        # Handle relative time expressions
        if "today" in query_lower:
            today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            tomorrow = today + timedelta(days=1)
            temporal_context["timestamp"] = {"$gte": today, "$lt": tomorrow}

        elif "yesterday" in query_lower:
            today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            yesterday = today - timedelta(days=1)
            temporal_context["timestamp"] = {"$gte": yesterday, "$lt": today}

        elif "last week" in query_lower:
            now = datetime.utcnow()
            week_ago = now - timedelta(days=7)
            temporal_context["timestamp"] = {"$gte": week_ago, "$lte": now}

        return temporal_context if temporal_context else None

    def _normalize_operation(self, operation: str) -> str:
        """Normalize operation names."""
        operation_map = {
            "withdrawal": "withdrawal",
            "withdrawals": "withdrawal",
            "deposit": "deposit",
            "deposits": "deposit",
            "balance": "balance_inquiry",
            "balance inquiry": "balance_inquiry",
            "transfer": "transfer",
            "transfers": "transfer",
            "pin change": "pin_change",
            "pin_change": "pin_change",
            "cash advance": "cash_advance"
        }
        return operation_map.get(operation.lower(), operation.lower())

    def _normalize_status(self, status: str) -> str:
        """Normalize status values."""
        status_map = {
            "successful": "success",
            "success": "success",
            "failed": "failed",
            "failure": "failed",
            "denied": "denied",
            "denial": "denied",
            "timeout": "timeout",
            "timed out": "timeout",
            "error": "error",
            "errors": "error",
            "completed": "completed",
            "complete": "completed",
            "pending": "pending"
        }
        return status_map.get(status.lower(), status.lower())

    def _normalize_date(self, date_str: str) -> datetime:
        """Normalize date strings to datetime objects."""
        try:
            # Try different date formats
            for fmt in ["%m/%d/%Y", "%m-%d-%Y", "%Y-%m-%d", "%d/%m/%Y"]:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue

            # If no format works, return current date
            logger.warning(f"Could not parse date: {date_str}")
            return datetime.utcnow()

        except Exception as e:
            logger.error(f"Error normalizing date '{date_str}': {e}")
            return datetime.utcnow()

    def _normalize_time_range(self, time_range: str) -> Dict[str, Any]:
        """Normalize time range expressions to MongoDB queries."""
        now = datetime.utcnow()

        if "today" in time_range:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
            return {"$gte": start, "$lt": end}

        elif "yesterday" in time_range:
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            yesterday_start = today_start - timedelta(days=1)
            return {"$gte": yesterday_start, "$lt": today_start}

        elif "last hour" in time_range:
            hour_ago = now - timedelta(hours=1)
            return {"$gte": hour_ago, "$lte": now}

        elif "last day" in time_range:
            day_ago = now - timedelta(days=1)
            return {"$gte": day_ago, "$lte": now}

        elif "last week" in time_range:
            week_ago = now - timedelta(days=7)
            return {"$gte": week_ago, "$lte": now}

        elif "last month" in time_range:
            month_ago = now - timedelta(days=30)
            return {"$gte": month_ago, "$lte": now}

        # Handle "past X hours/days" pattern
        past_match = re.search(r"past\s+(\d+)\s+(hours?|days?|weeks?|months?)", time_range)
        if past_match:
            amount = int(past_match.group(1))
            unit = past_match.group(2)

            if "hour" in unit:
                start_time = now - timedelta(hours=amount)
            elif "day" in unit:
                start_time = now - timedelta(days=amount)
            elif "week" in unit:
                start_time = now - timedelta(weeks=amount)
            elif "month" in unit:
                start_time = now - timedelta(days=amount * 30)
            else:
                start_time = now - timedelta(days=1)

            return {"$gte": start_time, "$lte": now}

        # Default to last 24 hours
        return {"$gte": now - timedelta(days=1), "$lte": now}

    def _calculate_overall_confidence(self, entities: List[ExtractedEntity]) -> float:
        """Calculate overall confidence score for entity extraction."""
        if not entities:
            return 0.0

        # Average confidence weighted by entity importance
        weights = {
            "error_code": 1.0,
            "atm_id": 0.9,
            "operation": 0.8,
            "amount": 0.7,
            "status": 0.6,
            "date": 0.5,
            "time_range": 0.5,
            "session_id": 0.4,
            "customer_id": 0.3
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for entity in entities:
            weight = weights.get(entity.entity_type, 0.5)
            weighted_sum += entity.confidence * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def get_entity_statistics(self) -> Dict[str, Any]:
        """Get statistics about the entity extractor."""
        total_patterns = sum(len(info["patterns"]) for info in self.entity_patterns.values())

        return {
            "entity_types": len(self.entity_patterns),
            "total_patterns": total_patterns,
            "normalization_rules": len(self.normalization_rules),
            "temporal_patterns": len(self.temporal_expressions),
            "supported_entities": list(self.entity_patterns.keys())
        }