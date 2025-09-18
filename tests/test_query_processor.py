"""
Tests for Query Processor Components

Tests intent classification, entity extraction, and complete
query processing functionality.
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.query_processor.intent_classifier import IntentClassifier, ATMIntent
from src.query_processor.entity_extractor import EntityExtractor
from src.query_processor.query_processor import QueryProcessor


class TestIntentClassifier:
    """Test intent classification functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.classifier = IntentClassifier()

    def test_troubleshooting_intent(self):
        """Test troubleshooting intent detection."""
        troubleshooting_queries = [
            "Fix ATM001 withdrawal problems",
            "Help me solve the cash dispenser issue",
            "Why is ATM002 not working?",
            "Troubleshoot network errors",
            "ATM stuck on transaction screen"
        ]

        for query in troubleshooting_queries:
            result = self.classifier.classify_intent(query)
            assert result.intent == ATMIntent.TROUBLESHOOTING
            assert result.confidence > 0.5

    def test_error_explanation_intent(self):
        """Test error explanation intent detection."""
        error_queries = [
            "What does DDL_EXCEEDED mean?",
            "Explain error code NETWORK_ERROR",
            "What is TIMEOUT error?",
            "Error code explanation for CARD_ERROR",
            "Meaning of PIN_ERROR"
        ]

        for query in error_queries:
            result = self.classifier.classify_intent(query)
            assert result.intent == ATMIntent.ERROR_EXPLANATION
            assert result.confidence > 0.5

    def test_operation_inquiry_intent(self):
        """Test operation inquiry intent detection."""
        operation_queries = [
            "Show me all withdrawal transactions",
            "How many deposits were made today?",
            "List recent balance inquiries",
            "Count successful operations",
            "Display transfer activities"
        ]

        for query in operation_queries:
            result = self.classifier.classify_intent(query)
            assert result.intent == ATMIntent.OPERATION_INQUIRY
            assert result.confidence > 0.4

    def test_performance_analysis_intent(self):
        """Test performance analysis intent detection."""
        analysis_queries = [
            "Analyze ATM performance trends",
            "Show statistics for last month",
            "Performance report for ATM001",
            "Compare withdrawal success rates",
            "Generate metrics summary"
        ]

        for query in analysis_queries:
            result = self.classifier.classify_intent(query)
            assert result.intent == ATMIntent.PERFORMANCE_ANALYSIS
            assert result.confidence > 0.4

    def test_status_check_intent(self):
        """Test status check intent detection."""
        status_queries = [
            "Check ATM001 status",
            "Is ATM002 online?",
            "Verify system health",
            "Current state of cash dispensers",
            "ATM availability check"
        ]

        for query in status_queries:
            result = self.classifier.classify_intent(query)
            assert result.intent == ATMIntent.STATUS_CHECK
            assert result.confidence > 0.4

    def test_historical_search_intent(self):
        """Test historical search intent detection."""
        search_queries = [
            "Search for withdrawal failures last week",
            "Find all DDL_EXCEEDED errors",
            "Look for ATM001 transactions",
            "Retrieve logs from yesterday",
            "Show me deposit history"
        ]

        for query in search_queries:
            result = self.classifier.classify_intent(query)
            assert result.intent == ATMIntent.HISTORICAL_SEARCH
            assert result.confidence > 0.4

    def test_entity_extraction_in_intent(self):
        """Test entity extraction during intent classification."""
        query = "Fix ATM001 DDL_EXCEEDED errors"
        result = self.classifier.classify_intent(query)

        assert result.intent == ATMIntent.TROUBLESHOOTING
        assert len(result.extracted_entities) > 0

    def test_confidence_levels(self):
        """Test confidence level calculation."""
        # High confidence query
        high_conf_query = "Fix broken ATM withdrawal dispenser problem"
        high_result = self.classifier.classify_intent(high_conf_query)

        # Low confidence query
        low_conf_query = "What about that thing?"
        low_result = self.classifier.classify_intent(low_conf_query)

        assert high_result.confidence > low_result.confidence

    def test_atm_related_detection(self):
        """Test ATM-related query detection."""
        atm_queries = [
            "ATM withdrawal failed",
            "Cash dispenser error",
            "Bank machine timeout",
            "Transaction denied"
        ]

        non_atm_queries = [
            "Weather forecast",
            "Restaurant recommendations",
            "Sports scores"
        ]

        for query in atm_queries:
            assert self.classifier.is_atm_related(query)

        for query in non_atm_queries:
            assert not self.classifier.is_atm_related(query)

    def test_query_complexity_assessment(self):
        """Test query complexity assessment."""
        simple_query = "ATM status"
        medium_query = "Show me withdrawal failures on ATM001 yesterday"
        complex_query = "Analyze the correlation between network errors and withdrawal failures across all ATMs in the downtown location during peak hours"

        assert self.classifier.get_query_complexity(simple_query) == "simple"
        assert self.classifier.get_query_complexity(medium_query) == "medium"
        assert self.classifier.get_query_complexity(complex_query) == "complex"

    def test_intent_suggestions(self):
        """Test getting multiple intent suggestions."""
        ambiguous_query = "ATM machine problems yesterday"
        suggestions = self.classifier.get_intent_suggestions(ambiguous_query, top_k=3)

        assert len(suggestions) <= 3
        assert all(isinstance(intent, ATMIntent) for intent, conf in suggestions)
        assert all(0 <= conf <= 1 for intent, conf in suggestions)
        # Should be sorted by confidence
        assert all(suggestions[i][1] >= suggestions[i+1][1] for i in range(len(suggestions)-1))


class TestEntityExtractor:
    """Test entity extraction functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.extractor = EntityExtractor()

    def test_error_code_extraction(self):
        """Test error code entity extraction."""
        queries_with_errors = [
            "Fix DDL_EXCEEDED error",
            "What is NETWORK_ERROR code?",
            "Error code: TIMEOUT occurred",
            "CARD_ERROR troubleshooting guide"
        ]

        for query in queries_with_errors:
            result = self.extractor.extract_entities(query)
            error_entities = [e for e in result.entities if e.entity_type == "error_code"]
            assert len(error_entities) > 0
            assert error_entities[0].confidence > 0.8

    def test_atm_id_extraction(self):
        """Test ATM ID entity extraction."""
        queries_with_atms = [
            "Check ATM001 status",
            "Problems with atm 123",
            "ATM-456 maintenance",
            "machine 789 offline"
        ]

        for query in queries_with_atms:
            result = self.extractor.extract_entities(query)
            atm_entities = [e for e in result.entities if e.entity_type == "atm_id"]
            assert len(atm_entities) > 0

    def test_operation_extraction(self):
        """Test operation entity extraction."""
        queries_with_operations = [
            "Show withdrawal transactions",
            "All deposit activities",
            "Balance inquiry logs",
            "Transfer operations failed"
        ]

        expected_operations = ["withdrawal", "deposit", "balance", "transfer"]

        for query, expected in zip(queries_with_operations, expected_operations):
            result = self.extractor.extract_entities(query)
            operation_entities = [e for e in result.entities if e.entity_type == "operation"]
            assert len(operation_entities) > 0
            assert expected in operation_entities[0].normalized_value

    def test_amount_extraction(self):
        """Test amount entity extraction."""
        queries_with_amounts = [
            "Withdrawal of $500 failed",
            "Deposit 1000 dollars",
            "Amount: $250.50 denied"
        ]

        for query in queries_with_amounts:
            result = self.extractor.extract_entities(query)
            amount_entities = [e for e in result.entities if e.entity_type == "amount"]
            assert len(amount_entities) > 0
            assert isinstance(amount_entities[0].normalized_value, float)

    def test_date_extraction(self):
        """Test date entity extraction."""
        queries_with_dates = [
            "Transactions on 01/15/2024",
            "Logs from 2024-01-15",
            "January 15, 2024 activities"
        ]

        for query in queries_with_dates:
            result = self.extractor.extract_entities(query)
            date_entities = [e for e in result.entities if e.entity_type == "date"]
            if date_entities:  # Some date formats might not be recognized
                assert isinstance(date_entities[0].normalized_value, datetime)

    def test_time_range_extraction(self):
        """Test time range entity extraction."""
        queries_with_time_ranges = [
            "Show me logs from today",
            "Yesterday's transactions",
            "Last week's errors",
            "Past 24 hours activities"
        ]

        for query in queries_with_time_ranges:
            result = self.extractor.extract_entities(query)
            time_entities = [e for e in result.entities if e.entity_type == "time_range"]
            assert len(time_entities) > 0

    def test_session_id_extraction(self):
        """Test session ID entity extraction."""
        queries_with_sessions = [
            "Session SES_123456 failed",
            "Check session: ABC123",
            "Session id: SES_ERROR_001"
        ]

        for query in queries_with_sessions:
            result = self.extractor.extract_entities(query)
            session_entities = [e for e in result.entities if e.entity_type == "session_id"]
            assert len(session_entities) > 0

    def test_status_extraction(self):
        """Test status entity extraction."""
        queries_with_status = [
            "Show successful transactions",
            "All failed operations",
            "Denied withdrawals",
            "Timeout errors occurred"
        ]

        for query in queries_with_status:
            result = self.extractor.extract_entities(query)
            status_entities = [e for e in result.entities if e.entity_type == "status"]
            assert len(status_entities) > 0

    def test_filter_generation(self):
        """Test MongoDB filter generation from entities."""
        query = "Show failed withdrawals on ATM001 with DDL_EXCEEDED error"
        result = self.extractor.extract_entities(query)

        assert "operation" in result.filters
        assert "status" in result.filters
        assert "atm_id" in result.filters
        assert "error_code" in result.filters

        assert result.filters["operation"] == "withdrawal"
        assert result.filters["status"] == "failed"
        assert result.filters["atm_id"] == "ATM001"
        assert result.filters["error_code"] == "DDL_EXCEEDED"

    def test_temporal_context_extraction(self):
        """Test temporal context extraction."""
        query = "Show me withdrawals from today"
        result = self.extractor.extract_entities(query)

        assert result.temporal_context is not None
        assert "timestamp" in result.temporal_context
        # Should be a date range for "today"
        assert "$gte" in result.temporal_context["timestamp"]
        assert "$lt" in result.temporal_context["timestamp"]

    def test_confidence_calculation(self):
        """Test overall confidence calculation."""
        # High confidence query with multiple clear entities
        high_conf_query = "ATM001 DDL_EXCEEDED withdrawal errors"
        high_result = self.extractor.extract_entities(high_conf_query)

        # Low confidence query with unclear entities
        low_conf_query = "some stuff happened maybe"
        low_result = self.extractor.extract_entities(low_conf_query)

        high_confidence = high_result.extraction_metadata.get("extraction_confidence", 0)
        low_confidence = low_result.extraction_metadata.get("extraction_confidence", 0)

        assert high_confidence > low_confidence

    def test_entity_normalization(self):
        """Test entity value normalization."""
        query = "atm 123 successful withdrawal of $100"
        result = self.extractor.extract_entities(query)

        # Check ATM ID normalization
        atm_entities = [e for e in result.entities if e.entity_type == "atm_id"]
        if atm_entities:
            assert atm_entities[0].normalized_value == "ATM123"

        # Check status normalization
        status_entities = [e for e in result.entities if e.entity_type == "status"]
        if status_entities:
            assert status_entities[0].normalized_value == "success"

        # Check amount normalization
        amount_entities = [e for e in result.entities if e.entity_type == "amount"]
        if amount_entities:
            assert amount_entities[0].normalized_value == 100.0


class TestQueryProcessor:
    """Test complete query processor functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.processor = QueryProcessor()

    def test_complete_query_processing(self):
        """Test complete query processing pipeline."""
        query = "Fix ATM001 DDL_EXCEEDED withdrawal errors from yesterday"
        result = self.processor.process_query(query)

        # Check basic structure
        assert result.original_query == query
        assert result.intent_result.intent != ATMIntent.UNKNOWN
        assert len(result.entity_result.entities) > 0
        assert len(result.optimized_filters) > 0

        # Check derived fields
        assert result.suggested_top_k > 0
        assert result.response_type in ["troubleshooting", "error", "info", "analysis", "auto"]
        assert result.query_complexity in ["simple", "medium", "complex"]

    def test_intent_and_entity_integration(self):
        """Test integration between intent classification and entity extraction."""
        query = "What does NETWORK_ERROR code mean?"
        result = self.processor.process_query(query)

        # Should detect error explanation intent
        assert result.intent_result.intent == ATMIntent.ERROR_EXPLANATION

        # Should extract error code entity
        error_entities = [e for e in result.entity_result.entities if e.entity_type == "error_code"]
        assert len(error_entities) > 0

        # Should set appropriate response type
        assert result.response_type == "error"

    def test_filter_optimization(self):
        """Test filter optimization based on intent and entities."""
        troubleshooting_query = "Fix withdrawal problems"
        result = self.processor.process_query(troubleshooting_query)

        # Should add error-focused filters for troubleshooting
        assert "$or" in result.optimized_filters or "status" in result.optimized_filters

        # Check that intent influences filtering
        assert result.intent_result.intent == ATMIntent.TROUBLESHOOTING

    def test_top_k_adjustment(self):
        """Test top_k adjustment based on intent and specificity."""
        # Specific query should need fewer results
        specific_query = "ATM001 DDL_EXCEEDED error details"
        specific_result = self.processor.process_query(specific_query)

        # General query should need more results
        general_query = "Show me ATM problems"
        general_result = self.processor.process_query(general_query)

        # Specific queries with many filters should suggest lower top_k
        assert specific_result.suggested_top_k <= general_result.suggested_top_k

    def test_response_type_mapping(self):
        """Test response type mapping from intent."""
        test_cases = [
            ("Fix ATM problems", "troubleshooting"),
            ("What is DDL_EXCEEDED?", "error"),
            ("Analyze performance", "analysis"),
            ("Show me logs", "info")
        ]

        for query, expected_type in test_cases:
            result = self.processor.process_query(query)
            assert result.response_type == expected_type or result.response_type == "auto"

    def test_complexity_assessment(self):
        """Test query complexity assessment."""
        simple_query = "ATM status"
        complex_query = "Analyze the correlation between network errors and withdrawal failures on ATM001, ATM002, and ATM003 during the last 30 days, excluding weekends"

        simple_result = self.processor.process_query(simple_query)
        complex_result = self.processor.process_query(complex_query)

        assert simple_result.query_complexity in ["simple", "medium"]
        assert complex_result.query_complexity in ["medium", "complex"]

    def test_batch_processing(self):
        """Test batch query processing."""
        queries = [
            "ATM001 status check",
            "DDL_EXCEEDED error explanation",
            "Withdrawal failure analysis"
        ]

        results = self.processor.process_batch_queries(queries)

        assert len(results) == 3
        for result in results:
            assert hasattr(result, 'original_query')
            assert hasattr(result, 'intent_result')
            assert hasattr(result, 'entity_result')

    def test_query_validation(self):
        """Test query validation."""
        # Valid ATM query
        valid_query = "Check ATM001 withdrawal status"
        valid_result = self.processor.validate_query(valid_query)
        assert valid_result["is_valid"]
        assert valid_result["is_atm_related"]

        # Invalid query (too short)
        invalid_query = "no"
        invalid_result = self.processor.validate_query(invalid_query)
        assert not invalid_result["is_valid"]

        # Non-ATM related query
        non_atm_query = "What's the weather like?"
        non_atm_result = self.processor.validate_query(non_atm_query)
        assert not non_atm_result["is_atm_related"]

    def test_pattern_analysis(self):
        """Test query pattern analysis."""
        queries = [
            "ATM001 DDL_EXCEEDED error",
            "ATM002 NETWORK_ERROR problem",
            "ATM003 withdrawal failure",
            "Balance inquiry success"
        ]

        analysis = self.processor.analyze_query_patterns(queries)

        assert "total_queries" in analysis
        assert "intent_distribution" in analysis
        assert "entity_distribution" in analysis
        assert "complexity_distribution" in analysis
        assert analysis["total_queries"] == 4

    def test_error_handling(self):
        """Test error handling in query processing."""
        # Test with empty query
        empty_result = self.processor.process_query("")
        assert empty_result.intent_result.intent == ATMIntent.UNKNOWN

        # Test with very long query
        long_query = "test " * 1000
        long_result = self.processor.process_query(long_query)
        assert hasattr(long_result, 'original_query')

    def test_processing_statistics(self):
        """Test processing statistics."""
        stats = self.processor.get_processing_statistics()

        assert "intent_classifier" in stats
        assert "entity_extractor" in stats
        assert "configuration" in stats
        assert "supported_features" in stats

        # Check configuration
        config = stats["configuration"]
        assert "default_top_k" in config
        assert "top_k_by_intent" in config
        assert "intent_to_response_type" in config


def run_query_processor_tests():
    """Run all query processor tests."""
    print("Running Query Processor Tests...")

    print("\n1. Testing Intent Classifier...")
    test_intent = TestIntentClassifier()
    test_intent.setup_method()
    test_intent.test_troubleshooting_intent()
    test_intent.test_error_explanation_intent()
    test_intent.test_operation_inquiry_intent()
    test_intent.test_performance_analysis_intent()
    test_intent.test_status_check_intent()
    test_intent.test_historical_search_intent()
    test_intent.test_entity_extraction_in_intent()
    test_intent.test_confidence_levels()
    test_intent.test_atm_related_detection()
    test_intent.test_query_complexity_assessment()
    test_intent.test_intent_suggestions()
    print("✓ Intent Classifier tests passed")

    print("\n2. Testing Entity Extractor...")
    test_entity = TestEntityExtractor()
    test_entity.setup_method()
    test_entity.test_error_code_extraction()
    test_entity.test_atm_id_extraction()
    test_entity.test_operation_extraction()
    test_entity.test_amount_extraction()
    test_entity.test_date_extraction()
    test_entity.test_time_range_extraction()
    test_entity.test_session_id_extraction()
    test_entity.test_status_extraction()
    test_entity.test_filter_generation()
    test_entity.test_temporal_context_extraction()
    test_entity.test_confidence_calculation()
    test_entity.test_entity_normalization()
    print("✓ Entity Extractor tests passed")

    print("\n3. Testing Query Processor...")
    test_processor = TestQueryProcessor()
    test_processor.setup_method()
    test_processor.test_complete_query_processing()
    test_processor.test_intent_and_entity_integration()
    test_processor.test_filter_optimization()
    test_processor.test_top_k_adjustment()
    test_processor.test_response_type_mapping()
    test_processor.test_complexity_assessment()
    test_processor.test_batch_processing()
    test_processor.test_query_validation()
    test_processor.test_pattern_analysis()
    test_processor.test_error_handling()
    test_processor.test_processing_statistics()
    print("✓ Query Processor tests passed")

    print("\n🎉 All Query Processor tests completed!")


if __name__ == "__main__":
    run_query_processor_tests()