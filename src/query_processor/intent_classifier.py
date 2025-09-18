"""
Intent Classifier for ATM Query Processing

Classifies user queries into specific intent categories to optimize
retrieval and response generation for ATM-related assistance.
"""

import logging
import re
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ATMIntent(Enum):
    """ATM query intent categories."""
    TROUBLESHOOTING = "troubleshooting"
    ERROR_EXPLANATION = "error_explanation"
    OPERATION_INQUIRY = "operation_inquiry"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    STATUS_CHECK = "status_check"
    HISTORICAL_SEARCH = "historical_search"
    GENERAL_INFO = "general_info"
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    """Result of intent classification."""
    intent: ATMIntent
    confidence: float
    matched_patterns: List[str]
    extracted_entities: Dict[str, str]


class IntentClassifier:
    """
    Rule-based intent classifier for ATM queries.

    Uses pattern matching and keyword analysis to determine
    the user's intent and extract relevant entities.
    """

    def __init__(self):
        """Initialize intent classifier with patterns and keywords."""
        self.intent_patterns = self._build_intent_patterns()
        self.entity_patterns = self._build_entity_patterns()

        # Confidence thresholds
        self.high_confidence_threshold = 0.8
        self.medium_confidence_threshold = 0.5

        logger.info("Intent classifier initialized")

    def classify_intent(self, query: str) -> IntentResult:
        """
        Classify the intent of a user query.

        Args:
            query: User's query text

        Returns:
            IntentResult with classification and confidence
        """
        query_clean = query.lower().strip()

        # Score each intent category
        intent_scores = {}
        matched_patterns = {}

        for intent, patterns in self.intent_patterns.items():
            score, matches = self._score_intent(query_clean, patterns)
            intent_scores[intent] = score
            matched_patterns[intent] = matches

        # Find best matching intent
        best_intent = max(intent_scores, key=intent_scores.get)
        best_score = intent_scores[best_intent]
        best_matches = matched_patterns[best_intent]

        # Extract entities from the query
        entities = self._extract_query_entities(query_clean)

        # Determine confidence level
        confidence = self._calculate_confidence(best_score, len(best_matches))

        # Fallback to UNKNOWN if confidence is too low
        if confidence < 0.3:
            best_intent = ATMIntent.UNKNOWN

        return IntentResult(
            intent=best_intent,
            confidence=confidence,
            matched_patterns=best_matches,
            extracted_entities=entities
        )

    def _build_intent_patterns(self) -> Dict[ATMIntent, List[Dict[str, any]]]:
        """Build pattern rules for each intent category."""
        return {
            ATMIntent.TROUBLESHOOTING: [
                {"pattern": r"\b(fix|solve|resolve|help|repair)\b", "weight": 0.8},
                {"pattern": r"\b(problem|issue|trouble|error|fail)\b", "weight": 0.7},
                {"pattern": r"\b(not working|broken|stuck|jammed)\b", "weight": 0.9},
                {"pattern": r"\b(why.*(?:fail|error|denied))\b", "weight": 0.8},
                {"pattern": r"\b(how to fix|troubleshoot)\b", "weight": 0.9},
                {"pattern": r"\b(won't|can't|doesn't)\b", "weight": 0.6}
            ],

            ATMIntent.ERROR_EXPLANATION: [
                {"pattern": r"\b(error code|error)\s+([A-Z0-9_]+)\b", "weight": 0.9},
                {"pattern": r"\b(what does.*mean|explain.*error)\b", "weight": 0.8},
                {"pattern": r"\b(DDL_EXCEEDED|TIMEOUT|CARD_ERROR|NETWORK_ERROR)\b", "weight": 0.9},
                {"pattern": r"\b(denied|failed|timeout)\b", "weight": 0.6},
                {"pattern": r"\b(what is|what does|meaning of)\b", "weight": 0.5}
            ],

            ATMIntent.OPERATION_INQUIRY: [
                {"pattern": r"\b(withdrawal|deposit|balance|transfer)\b", "weight": 0.8},
                {"pattern": r"\b(transaction|operation|activity)\b", "weight": 0.6},
                {"pattern": r"\b(how many|count|frequency)\b", "weight": 0.7},
                {"pattern": r"\b(today|yesterday|last|recent)\b", "weight": 0.5},
                {"pattern": r"\b(successful|completed|processed)\b", "weight": 0.6}
            ],

            ATMIntent.PERFORMANCE_ANALYSIS: [
                {"pattern": r"\b(analyze|analysis|performance|trends)\b", "weight": 0.8},
                {"pattern": r"\b(statistics|stats|metrics|report)\b", "weight": 0.8},
                {"pattern": r"\b(pattern|frequency|distribution)\b", "weight": 0.7},
                {"pattern": r"\b(compare|comparison|vs|versus)\b", "weight": 0.6},
                {"pattern": r"\b(summary|overview|insights)\b", "weight": 0.6}
            ],

            ATMIntent.STATUS_CHECK: [
                {"pattern": r"\b(status|state|condition|health)\b", "weight": 0.8},
                {"pattern": r"\b(is.*working|is.*online|is.*available)\b", "weight": 0.8},
                {"pattern": r"\b(uptime|downtime|availability)\b", "weight": 0.9},
                {"pattern": r"\b(current|now|currently|present)\b", "weight": 0.5},
                {"pattern": r"\b(check|verify|confirm)\b", "weight": 0.6}
            ],

            ATMIntent.HISTORICAL_SEARCH: [
                {"pattern": r"\b(search|find|look for|locate)\b", "weight": 0.7},
                {"pattern": r"\b(history|historical|past|previous)\b", "weight": 0.8},
                {"pattern": r"\b(logs|records|entries|data)\b", "weight": 0.7},
                {"pattern": r"\b(when|date|time|period)\b", "weight": 0.5},
                {"pattern": r"\b(show me|get me|retrieve)\b", "weight": 0.6}
            ],

            ATMIntent.GENERAL_INFO: [
                {"pattern": r"\b(what|how|who|where|which)\b", "weight": 0.4},
                {"pattern": r"\b(information|info|details|about)\b", "weight": 0.6},
                {"pattern": r"\b(tell me|explain|describe)\b", "weight": 0.5},
                {"pattern": r"\b(list|show|display)\b", "weight": 0.5}
            ]
        }

    def _build_entity_patterns(self) -> Dict[str, str]:
        """Build regex patterns for entity extraction."""
        return {
            "error_code": r"\b([A-Z]{2,}_[A-Z_]+|[A-Z]{3,}[0-9]+)\b",
            "atm_id": r"\b(ATM[0-9]+|atm[0-9]+|ATM-[0-9]+)\b",
            "operation": r"\b(withdrawal|deposit|balance|transfer|inquiry|pin_change)\b",
            "status": r"\b(success|successful|failed|denied|error|timeout|completed)\b",
            "amount": r"\$?([0-9]+(?:\.[0-9]{2})?)",
            "date": r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}-\d{2}-\d{2})\b",
            "time_period": r"\b(today|yesterday|last week|last month|past.*(?:hour|day|week|month))\b",
            "session_id": r"\b(SES_[A-Z0-9]+|session[_\s]*[0-9]+)\b"
        }

    def _score_intent(self, query: str, patterns: List[Dict[str, any]]) -> Tuple[float, List[str]]:
        """
        Score a query against intent patterns.

        Args:
            query: Cleaned query text
            patterns: List of pattern dictionaries

        Returns:
            Tuple of (score, matched_patterns)
        """
        total_score = 0.0
        matched_patterns = []

        for pattern_info in patterns:
            pattern = pattern_info["pattern"]
            weight = pattern_info["weight"]

            if re.search(pattern, query, re.IGNORECASE):
                total_score += weight
                matched_patterns.append(pattern)

        # Normalize score (don't divide by all patterns, just cap at 1.0)
        normalized_score = min(total_score, 1.0)

        return normalized_score, matched_patterns

    def _extract_query_entities(self, query: str) -> Dict[str, str]:
        """
        Extract entities from the query text.

        Args:
            query: Query text to analyze

        Returns:
            Dictionary of extracted entities
        """
        entities = {}

        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                if isinstance(matches[0], tuple):
                    # Handle capturing groups
                    entities[entity_type] = matches[0][0] if matches[0][0] else matches[0]
                else:
                    entities[entity_type] = matches[0]

        return entities

    def _calculate_confidence(self, score: float, num_matches: int) -> float:
        """
        Calculate confidence based on score and number of matches.

        Args:
            score: Intent score
            num_matches: Number of pattern matches

        Returns:
            Confidence value between 0 and 1
        """
        # Base confidence from score
        base_confidence = score

        # Boost confidence with more matches (but with diminishing returns)
        match_boost = min(num_matches * 0.1, 0.3)

        # Combine and cap at 1.0
        confidence = min(base_confidence + match_boost, 1.0)

        return confidence

    def get_intent_suggestions(self, query: str, top_k: int = 3) -> List[Tuple[ATMIntent, float]]:
        """
        Get top-k intent suggestions for a query.

        Args:
            query: User's query
            top_k: Number of suggestions to return

        Returns:
            List of (intent, confidence) tuples
        """
        query_clean = query.lower().strip()

        # Score all intents
        intent_scores = []

        for intent, patterns in self.intent_patterns.items():
            score, matches = self._score_intent(query_clean, patterns)
            confidence = self._calculate_confidence(score, len(matches))

            if confidence > 0.1:  # Only include reasonable suggestions
                intent_scores.append((intent, confidence))

        # Sort by confidence and return top-k
        intent_scores.sort(key=lambda x: x[1], reverse=True)
        return intent_scores[:top_k]

    def is_atm_related(self, query: str) -> bool:
        """
        Check if a query is ATM-related.

        Args:
            query: Query to check

        Returns:
            True if query appears to be ATM-related
        """
        atm_keywords = [
            "atm", "cash", "withdrawal", "deposit", "balance", "card",
            "transaction", "error", "denied", "failed", "timeout",
            "money", "bank", "account", "pin", "receipt"
        ]

        query_lower = query.lower()
        return any(keyword in query_lower for keyword in atm_keywords)

    def get_query_complexity(self, query: str) -> str:
        """
        Assess query complexity.

        Args:
            query: Query to assess

        Returns:
            Complexity level: "simple", "medium", "complex"
        """
        word_count = len(query.split())
        entity_count = len(self._extract_query_entities(query.lower()))

        if word_count <= 5 and entity_count <= 1:
            return "simple"
        elif word_count <= 15 and entity_count <= 3:
            return "medium"
        else:
            return "complex"

    def get_classification_stats(self) -> Dict[str, any]:
        """
        Get statistics about the intent classifier.

        Returns:
            Statistics dictionary
        """
        total_patterns = sum(len(patterns) for patterns in self.intent_patterns.values())

        return {
            "total_intent_categories": len(self.intent_patterns),
            "total_patterns": total_patterns,
            "entity_types": len(self.entity_patterns),
            "confidence_thresholds": {
                "high": self.high_confidence_threshold,
                "medium": self.medium_confidence_threshold
            },
            "supported_intents": [intent.value for intent in ATMIntent]
        }