"""
Main Query Processor for ATM RAG System

Orchestrates query understanding by combining intent classification
and entity extraction to optimize retrieval parameters and enhance
the overall RAG pipeline performance.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

from .intent_classifier import IntentClassifier, ATMIntent, IntentResult
from .entity_extractor import EntityExtractor, EntityExtractionResult

logger = logging.getLogger(__name__)


@dataclass
class ProcessedQuery:
    """Complete processed query with intent and entities."""
    original_query: str
    intent_result: IntentResult
    entity_result: EntityExtractionResult
    processing_metadata: Dict[str, Any]

    # Derived fields for RAG pipeline optimization
    optimized_filters: Dict[str, Any]
    suggested_top_k: int
    response_type: str
    query_complexity: str


class QueryProcessor:
    """
    Main query processing orchestrator for ATM RAG system.

    Combines intent classification and entity extraction to provide
    comprehensive query understanding for optimized retrieval.
    """

    def __init__(self):
        """Initialize query processor with intent classifier and entity extractor."""
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()

        # Processing configuration
        self.default_top_k = 10
        self.top_k_by_intent = {
            ATMIntent.TROUBLESHOOTING: 8,
            ATMIntent.ERROR_EXPLANATION: 5,
            ATMIntent.OPERATION_INQUIRY: 12,
            ATMIntent.PERFORMANCE_ANALYSIS: 20,
            ATMIntent.STATUS_CHECK: 6,
            ATMIntent.HISTORICAL_SEARCH: 15,
            ATMIntent.GENERAL_INFO: 8,
            ATMIntent.UNKNOWN: 10
        }

        # Response type mapping
        self.intent_to_response_type = {
            ATMIntent.TROUBLESHOOTING: "troubleshooting",
            ATMIntent.ERROR_EXPLANATION: "error",
            ATMIntent.OPERATION_INQUIRY: "info",
            ATMIntent.PERFORMANCE_ANALYSIS: "analysis",
            ATMIntent.STATUS_CHECK: "info",
            ATMIntent.HISTORICAL_SEARCH: "info",
            ATMIntent.GENERAL_INFO: "info",
            ATMIntent.UNKNOWN: "auto"
        }

        logger.info("Query processor initialized")

    def process_query(self, query: str) -> ProcessedQuery:
        """
        Process a user query with complete understanding.

        Args:
            query: User's original query

        Returns:
            ProcessedQuery with intent, entities, and optimization parameters
        """
        start_time = datetime.utcnow()

        try:
            # Step 1: Intent Classification
            intent_result = self.intent_classifier.classify_intent(query)
            logger.debug(f"Classified intent: {intent_result.intent.value} (confidence: {intent_result.confidence:.2f})")

            # Step 2: Entity Extraction
            entity_result = self.entity_extractor.extract_entities(query)
            logger.debug(f"Extracted {len(entity_result.entities)} entities")

            # Step 3: Combine and Optimize
            optimized_filters = self._merge_filters(intent_result, entity_result)
            suggested_top_k = self._determine_top_k(intent_result, entity_result)
            response_type = self._determine_response_type(intent_result, entity_result)
            query_complexity = self._assess_complexity(query, intent_result, entity_result)

            # Build processing metadata
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            metadata = {
                "processing_time_ms": round(processing_time * 1000, 2),
                "intent_confidence": intent_result.confidence,
                "entity_confidence": entity_result.extraction_metadata.get("extraction_confidence", 0.0),
                "patterns_matched": intent_result.matched_patterns,
                "entities_found": len(entity_result.entities),
                "filters_applied": len(optimized_filters),
                "temporal_context": entity_result.temporal_context is not None
            }

            return ProcessedQuery(
                original_query=query,
                intent_result=intent_result,
                entity_result=entity_result,
                processing_metadata=metadata,
                optimized_filters=optimized_filters,
                suggested_top_k=suggested_top_k,
                response_type=response_type,
                query_complexity=query_complexity
            )

        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            # Return a basic processed query on error
            return self._create_fallback_processed_query(query, str(e))

    def process_batch_queries(self, queries: List[str]) -> List[ProcessedQuery]:
        """
        Process multiple queries efficiently.

        Args:
            queries: List of query strings

        Returns:
            List of ProcessedQuery objects
        """
        processed_queries = []

        for query in queries:
            try:
                processed = self.process_query(query)
                processed_queries.append(processed)
            except Exception as e:
                logger.error(f"Error in batch processing query '{query}': {e}")
                fallback = self._create_fallback_processed_query(query, str(e))
                processed_queries.append(fallback)

        return processed_queries

    def _merge_filters(self,
                      intent_result: IntentResult,
                      entity_result: EntityExtractionResult) -> Dict[str, Any]:
        """
        Merge filters from intent classification and entity extraction.

        Args:
            intent_result: Intent classification result
            entity_result: Entity extraction result

        Returns:
            Combined MongoDB filters
        """
        filters = {}

        # Start with entity-based filters
        filters.update(entity_result.filters)

        # Add temporal context if available
        if entity_result.temporal_context:
            filters.update(entity_result.temporal_context)

        # Add intent-specific filters
        intent = intent_result.intent

        if intent == ATMIntent.TROUBLESHOOTING:
            # Focus on error logs for troubleshooting
            if "status" not in filters:
                filters["$or"] = [
                    {"status": {"$in": ["denied", "failed", "error", "timeout"]}},
                    {"error_code": {"$exists": True, "$ne": ""}},
                    {"metadata.is_error": True}
                ]

        elif intent == ATMIntent.ERROR_EXPLANATION:
            # Extract error code from intent entities if not already found
            if "error_code" not in filters:
                for entity in intent_result.extracted_entities:
                    if "error" in entity.lower():
                        error_code = self._extract_error_code_from_text(entity)
                        if error_code:
                            filters["error_code"] = error_code
                            break

        elif intent == ATMIntent.PERFORMANCE_ANALYSIS:
            # For analysis, we might want broader results
            # Remove very specific filters to get better patterns
            if len(filters) > 2:  # Keep only most important filters
                important_filters = {}
                for key in ["operation", "atm_id", "timestamp"]:
                    if key in filters:
                        important_filters[key] = filters[key]
                filters = important_filters

        return filters

    def _determine_top_k(self,
                        intent_result: IntentResult,
                        entity_result: EntityExtractionResult) -> int:
        """
        Determine optimal top_k based on intent and entities.

        Args:
            intent_result: Intent classification result
            entity_result: Entity extraction result

        Returns:
            Suggested top_k value
        """
        # Base top_k from intent
        base_top_k = self.top_k_by_intent.get(intent_result.intent, self.default_top_k)

        # Adjust based on specificity of filters
        filter_count = len(entity_result.filters)

        if filter_count == 0:
            # No specific filters, might need more results
            return min(base_top_k + 5, 25)
        elif filter_count >= 3:
            # Very specific query, fewer results needed
            return max(base_top_k - 3, 3)
        else:
            return base_top_k

    def _determine_response_type(self,
                               intent_result: IntentResult,
                               entity_result: EntityExtractionResult) -> str:
        """
        Determine response type for the RAG pipeline.

        Args:
            intent_result: Intent classification result
            entity_result: Entity extraction result

        Returns:
            Response type string
        """
        # Base response type from intent
        response_type = self.intent_to_response_type.get(
            intent_result.intent, "auto"
        )

        # Override based on entities
        if any(e.entity_type == "error_code" for e in entity_result.entities):
            response_type = "error"
        elif intent_result.confidence < 0.5:
            response_type = "auto"  # Let the RAG pipeline decide

        return response_type

    def _assess_complexity(self,
                          query: str,
                          intent_result: IntentResult,
                          entity_result: EntityExtractionResult) -> str:
        """
        Assess query complexity for processing optimization.

        Args:
            query: Original query
            intent_result: Intent classification result
            entity_result: Entity extraction result

        Returns:
            Complexity level: "simple", "medium", "complex"
        """
        # Factor 1: Query length
        word_count = len(query.split())

        # Factor 2: Number of entities
        entity_count = len(entity_result.entities)

        # Factor 3: Intent confidence
        intent_confidence = intent_result.confidence

        # Factor 4: Multiple intents possible
        intent_suggestions = self.intent_classifier.get_intent_suggestions(query, top_k=3)
        multiple_intents = len([s for s in intent_suggestions if s[1] > 0.3]) > 1

        # Complexity scoring
        complexity_score = 0

        if word_count > 15:
            complexity_score += 2
        elif word_count > 8:
            complexity_score += 1

        if entity_count > 3:
            complexity_score += 2
        elif entity_count > 1:
            complexity_score += 1

        if intent_confidence < 0.5:
            complexity_score += 1

        if multiple_intents:
            complexity_score += 1

        # Determine complexity level
        if complexity_score <= 1:
            return "simple"
        elif complexity_score <= 3:
            return "medium"
        else:
            return "complex"

    def _extract_error_code_from_text(self, text: str) -> Optional[str]:
        """Extract error code from text string."""
        import re
        patterns = [
            r'\b([A-Z]{2,}_[A-Z_]+)\b',
            r'\berror\s+code\s*:?\s*([A-Z0-9_]+)\b'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        return None

    def _create_fallback_processed_query(self, query: str, error: str) -> ProcessedQuery:
        """Create a fallback ProcessedQuery when processing fails."""
        from .intent_classifier import IntentResult, ATMIntent
        from .entity_extractor import EntityExtractionResult

        # Create minimal intent result
        intent_result = IntentResult(
            intent=ATMIntent.UNKNOWN,
            confidence=0.0,
            matched_patterns=[],
            extracted_entities={}
        )

        # Create minimal entity result
        entity_result = EntityExtractionResult(
            entities=[],
            filters={},
            temporal_context=None,
            extraction_metadata={"error": error}
        )

        return ProcessedQuery(
            original_query=query,
            intent_result=intent_result,
            entity_result=entity_result,
            processing_metadata={"error": error, "fallback": True},
            optimized_filters={},
            suggested_top_k=self.default_top_k,
            response_type="auto",
            query_complexity="unknown"
        )

    def analyze_query_patterns(self, queries: List[str]) -> Dict[str, Any]:
        """
        Analyze patterns across multiple queries.

        Args:
            queries: List of queries to analyze

        Returns:
            Analysis results
        """
        if not queries:
            return {"error": "No queries provided"}

        processed_queries = self.process_batch_queries(queries)

        # Analyze intent distribution
        intent_counts = {}
        for pq in processed_queries:
            intent = pq.intent_result.intent.value
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

        # Analyze entity types
        entity_type_counts = {}
        for pq in processed_queries:
            for entity in pq.entity_result.entities:
                entity_type = entity.entity_type
                entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1

        # Analyze complexity distribution
        complexity_counts = {}
        for pq in processed_queries:
            complexity = pq.query_complexity
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1

        # Calculate average processing time
        processing_times = [
            pq.processing_metadata.get("processing_time_ms", 0)
            for pq in processed_queries
        ]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0

        return {
            "total_queries": len(queries),
            "intent_distribution": intent_counts,
            "entity_distribution": entity_type_counts,
            "complexity_distribution": complexity_counts,
            "average_processing_time_ms": round(avg_processing_time, 2),
            "successful_processing": len([pq for pq in processed_queries if not pq.processing_metadata.get("error")])
        }

    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive processing statistics.

        Returns:
            Statistics dictionary
        """
        intent_stats = self.intent_classifier.get_classification_stats()
        entity_stats = self.entity_extractor.get_entity_statistics()

        return {
            "intent_classifier": intent_stats,
            "entity_extractor": entity_stats,
            "configuration": {
                "default_top_k": self.default_top_k,
                "top_k_by_intent": {intent.value: k for intent, k in self.top_k_by_intent.items()},
                "intent_to_response_type": {intent.value: resp for intent, resp in self.intent_to_response_type.items()}
            },
            "supported_features": [
                "intent_classification",
                "entity_extraction",
                "filter_optimization",
                "top_k_adjustment",
                "response_type_mapping",
                "complexity_assessment",
                "temporal_context",
                "batch_processing"
            ]
        }

    def validate_query(self, query: str) -> Dict[str, Any]:
        """
        Validate if a query is suitable for ATM RAG processing.

        Args:
            query: Query to validate

        Returns:
            Validation result
        """
        validation_result = {
            "is_valid": True,
            "is_atm_related": True,
            "confidence": 1.0,
            "suggestions": [],
            "warnings": []
        }

        # Check if query is empty or too short
        if not query or len(query.strip()) < 3:
            validation_result["is_valid"] = False
            validation_result["suggestions"].append("Query is too short. Please provide more details.")
            return validation_result

        # Check if query is ATM-related
        is_atm_related = self.intent_classifier.is_atm_related(query)
        validation_result["is_atm_related"] = is_atm_related

        if not is_atm_related:
            validation_result["warnings"].append("Query doesn't appear to be ATM-related.")
            validation_result["confidence"] = 0.3

        # Check query complexity
        complexity = self.intent_classifier.get_query_complexity(query)
        if complexity == "complex":
            validation_result["suggestions"].append("Consider breaking down complex queries into simpler parts.")

        # Check for common issues
        query_lower = query.lower()
        if len(query.split()) > 50:
            validation_result["warnings"].append("Very long queries may not process optimally.")

        if any(char in query for char in ['<', '>', '{', '}', '[', ']']):
            validation_result["warnings"].append("Query contains special characters that might affect processing.")

        return validation_result