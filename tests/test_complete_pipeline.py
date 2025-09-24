"""
Complete End-to-End Pipeline Tests

Tests the complete ATM RAG system from query input to final response,
including all components working together in realistic scenarios.
"""

import pytest
import numpy as np
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.log_processor import LogReader, LogParser, TextExtractor
from src.embeddings import EmbeddingGenerator
from src.vector_store import HybridVectorStore
from src.vector_store.schema import ATMLogSchema
from src.rag_engine import ATMRagPipeline
from src.query_processor import QueryProcessor


class TestCompleteATMRagPipeline:
    """Test complete ATM RAG pipeline with realistic data."""

    def setup_method(self):
        """Set up complete test environment."""
        self.temp_dir = tempfile.mkdtemp()

        # Create test log data
        self.test_logs_dir = Path(self.temp_dir) / "logs"
        self.test_logs_dir.mkdir()
        self._create_test_log_files()

        # Initialize all components
        self.log_reader = LogReader(str(self.test_logs_dir))
        self.log_parser = LogParser()
        self.text_extractor = TextExtractor()

        # Use fast embedding for testing
        self.embedding_generator = EmbeddingGenerator(
            model_type="fast",
            cache_dir=str(Path(self.temp_dir) / "embeddings")
        )

        # Initialize vector store (skip MongoDB tests if not available)
        self.skip_mongo = os.getenv("SKIP_MONGO_TESTS") == "1"
        if not self.skip_mongo:
            try:
                self.vector_store = HybridVectorStore(
                    mongodb_uri="mongodb://localhost:27017",
                    database_name="atm_rag_test_complete",
                    storage_path=str(Path(self.temp_dir) / "vector_store")
                )
                # Clear any existing test data
                self.vector_store.mongo_store.clear_all_data("CONFIRM_DELETE_ALL_DATA")
                self.vector_store.faiss_manager.clear_index()

                # Initialize RAG pipeline
                self.rag_pipeline = ATMRagPipeline(
                    vector_store=self.vector_store,
                    embedding_generator=self.embedding_generator
                )
            except Exception as e:
                print(f"MongoDB not available, skipping integration tests: {e}")
                self.skip_mongo = True

        # Initialize query processor
        self.query_processor = QueryProcessor()

        # Load and process test data
        if not self.skip_mongo:
            self._load_test_data_into_pipeline()

    def teardown_method(self):
        """Clean up test environment."""
        try:
            if not self.skip_mongo and hasattr(self, 'vector_store'):
                self.vector_store.mongo_store.clear_all_data("CONFIRM_DELETE_ALL_DATA")
                self.vector_store.close()
        except:
            pass

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_log_files(self):
        """Create realistic test log files."""
        # Test scenarios: DDL exceeded, network errors, successful transactions
        test_logs = [
            # DDL exceeded errors on ATM001
            {
                "timestamp": "2024-01-15T10:30:00Z",
                "session_id": "SES_001",
                "customer_session_id": "CUST_001",
                "operation": "withdrawal",
                "status": "denied",
                "message": "Withdrawal denied DDL exceeded",
                "error_code": "DDL_EXCEEDED",
                "atm_id": "ATM001",
                "amount": 500.0,
                "metadata": {"location": "Downtown", "card_type": "debit"}
            },
            {
                "timestamp": "2024-01-15T11:15:00Z",
                "session_id": "SES_002",
                "customer_session_id": "CUST_002",
                "operation": "withdrawal",
                "status": "denied",
                "message": "Daily withdrawal limit exceeded",
                "error_code": "DDL_EXCEEDED",
                "atm_id": "ATM001",
                "amount": 1000.0,
                "metadata": {"location": "Downtown", "card_type": "credit"}
            },
            # Network errors on ATM002
            {
                "timestamp": "2024-01-15T12:00:00Z",
                "session_id": "SES_003",
                "customer_session_id": "CUST_003",
                "operation": "withdrawal",
                "status": "failed",
                "message": "Network connection timeout",
                "error_code": "NETWORK_ERROR",
                "atm_id": "ATM002",
                "amount": 200.0,
                "metadata": {"location": "Mall", "network_status": "unstable"}
            },
            {
                "timestamp": "2024-01-15T12:30:00Z",
                "session_id": "SES_004",
                "customer_session_id": "CUST_004",
                "operation": "balance_inquiry",
                "status": "failed",
                "message": "Network communication error",
                "error_code": "NETWORK_ERROR",
                "atm_id": "ATM002",
                "metadata": {"location": "Mall", "network_status": "down"}
            },
            # Successful transactions on ATM003
            {
                "timestamp": "2024-01-15T13:00:00Z",
                "session_id": "SES_005",
                "customer_session_id": "CUST_005",
                "operation": "withdrawal",
                "status": "success",
                "message": "Cash dispensed successfully",
                "atm_id": "ATM003",
                "amount": 100.0,
                "metadata": {"location": "Branch", "bills_dispensed": 5}
            },
            {
                "timestamp": "2024-01-15T13:15:00Z",
                "session_id": "SES_006",
                "customer_session_id": "CUST_006",
                "operation": "deposit",
                "status": "success",
                "message": "Deposit processed successfully",
                "atm_id": "ATM003",
                "amount": 250.0,
                "metadata": {"location": "Branch", "checks_deposited": 1}
            },
            # Card errors on ATM001
            {
                "timestamp": "2024-01-15T14:00:00Z",
                "session_id": "SES_007",
                "customer_session_id": "CUST_007",
                "operation": "withdrawal",
                "status": "denied",
                "message": "Card read error",
                "error_code": "CARD_ERROR",
                "atm_id": "ATM001",
                "metadata": {"location": "Downtown", "card_attempts": 3}
            },
            # Recent timeout on ATM002
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "session_id": "SES_008",
                "customer_session_id": "CUST_008",
                "operation": "withdrawal",
                "status": "timeout",
                "message": "Transaction timeout",
                "error_code": "TIMEOUT",
                "atm_id": "ATM002",
                "amount": 300.0,
                "metadata": {"location": "Mall", "timeout_seconds": 60}
            }
        ]

        # Write test logs to file
        log_file = self.test_logs_dir / "atm_logs_test.json"
        with open(log_file, 'w') as f:
            json.dump(test_logs, f, indent=2)

    def _load_test_data_into_pipeline(self):
        """Load test data into the RAG pipeline."""
        # Read and parse logs
        log_files = self.log_reader.get_available_files()
        all_logs = []

        for log_file in log_files:
            log_entries = self.log_reader.read_log_file(log_file)
            for entry in log_entries:
                parsed_log = self.log_parser.parse_log_entry(entry)
                extracted_text = self.text_extractor.extract_searchable_text(parsed_log)

                atm_log = ATMLogSchema(
                    log_id=parsed_log["log_id"],
                    timestamp=parsed_log["timestamp"],
                    session_id=parsed_log["session_id"],
                    customer_session_id=parsed_log.get("customer_session_id"),
                    operation=parsed_log["operation"],
                    status=parsed_log["status"],
                    message=parsed_log["message"],
                    error_code=parsed_log.get("error_code"),
                    atm_id=parsed_log.get("atm_id"),
                    amount=parsed_log.get("amount"),
                    extracted_text=extracted_text,
                    metadata=parsed_log.get("metadata", {})
                )
                all_logs.append(atm_log)

        # Generate embeddings
        texts = [log.extracted_text for log in all_logs]
        embeddings = self.embedding_generator.generate_embeddings(texts)

        # Insert into vector store
        result = self.vector_store.insert_log_embeddings(all_logs, embeddings)
        print(f"Loaded {result['mongodb_inserted']} logs and {result['faiss_added']} embeddings for testing")

    @pytest.mark.skipif(
        os.getenv("SKIP_MONGO_TESTS") == "1",
        reason="MongoDB not available"
    )
    def test_complete_troubleshooting_workflow(self):
        """Test complete troubleshooting workflow."""
        if self.skip_mongo:
            pytest.skip("MongoDB not available")

        # User asks about DDL exceeded errors
        query = "Why is ATM001 showing DDL_EXCEEDED errors?"

        # Step 1: Process query
        processed_query = self.query_processor.process_query(query)
        assert processed_query.intent_result.intent.value == "troubleshooting"
        assert "error_code" in processed_query.optimized_filters
        assert processed_query.optimized_filters["error_code"] == "DDL_EXCEEDED"

        # Step 2: Get response from RAG pipeline
        response = self.rag_pipeline.process_query_sync(
            query=query,
            filters=processed_query.optimized_filters,
            top_k=processed_query.suggested_top_k,
            response_type=processed_query.response_type
        )

        # Verify response quality
        assert response["response_type"] in ["troubleshooting", "error"]
        assert response["confidence"] > 0.5
        assert response["sources_count"] > 0
        assert "DDL_EXCEEDED" in response["response"]
        assert "ATM001" in response["response"]

    @pytest.mark.skipif(
        os.getenv("SKIP_MONGO_TESTS") == "1",
        reason="MongoDB not available"
    )
    def test_error_code_explanation_workflow(self):
        """Test error code explanation workflow."""
        if self.skip_mongo:
            pytest.skip("MongoDB not available")

        query = "What does NETWORK_ERROR mean?"

        # Process and get response
        processed_query = self.query_processor.process_query(query)
        response = self.rag_pipeline.process_query_sync(
            query=query,
            filters=processed_query.optimized_filters,
            response_type=processed_query.response_type
        )

        assert response["response_type"] == "error"
        assert "NETWORK_ERROR" in response["response"]
        assert "network" in response["response"].lower()

    @pytest.mark.skipif(
        os.getenv("SKIP_MONGO_TESTS") == "1",
        reason="MongoDB not available"
    )
    def test_performance_analysis_workflow(self):
        """Test performance analysis workflow."""
        if self.skip_mongo:
            pytest.skip("MongoDB not available")

        query = "Analyze ATM performance for today"

        processed_query = self.query_processor.process_query(query)
        response = self.rag_pipeline.process_query_sync(
            query=query,
            filters=processed_query.optimized_filters,
            response_type=processed_query.response_type
        )

        assert response["response_type"] in ["analysis", "info"]
        assert response["sources_count"] > 0

    @pytest.mark.skipif(
        os.getenv("SKIP_MONGO_TESTS") == "1",
        reason="MongoDB not available"
    )
    def test_specific_atm_troubleshooting(self):
        """Test troubleshooting specific to an ATM."""
        if self.skip_mongo:
            pytest.skip("MongoDB not available")

        # Use specialized troubleshooting method
        response = self.rag_pipeline.troubleshoot_error(
            error_code="NETWORK_ERROR",
            atm_id="ATM002"
        )

        assert response["response_type"] == "troubleshooting"
        assert response["sources_count"] > 0
        assert "NETWORK_ERROR" in response["response"]
        assert "ATM002" in response["response"]

    @pytest.mark.skipif(
        os.getenv("SKIP_MONGO_TESTS") == "1",
        reason="MongoDB not available"
    )
    def test_recent_issues_detection(self):
        """Test recent issues detection."""
        if self.skip_mongo:
            pytest.skip("MongoDB not available")

        response = self.rag_pipeline.get_recent_issues(hours_back=24)

        assert isinstance(response, dict)
        assert "response" in response
        # Should find our recent timeout issue
        assert response["sources_count"] > 0

    @pytest.mark.skipif(
        os.getenv("SKIP_MONGO_TESTS") == "1",
        reason="MongoDB not available"
    )
    def test_semantic_search_accuracy(self):
        """Test semantic search accuracy."""
        if self.skip_mongo:
            pytest.skip("MongoDB not available")

        # Search for withdrawal failures
        query = "withdrawal transaction failures"
        response = self.rag_pipeline.search_logs(query, top_k=5)

        # Should find logs related to withdrawal failures
        assert response["sources_count"] > 0

        # Search for network problems
        query = "network connection issues"
        response = self.rag_pipeline.search_logs(query, top_k=5)

        # Should find network-related errors
        assert response["sources_count"] > 0

    @pytest.mark.skipif(
        os.getenv("SKIP_MONGO_TESTS") == "1",
        reason="MongoDB not available"
    )
    def test_multi_query_batch_processing(self):
        """Test processing multiple queries in batch."""
        if self.skip_mongo:
            pytest.skip("MongoDB not available")

        queries = [
            "What is DDL_EXCEEDED error?",
            "ATM002 network problems",
            "Show successful transactions",
            "Analyze withdrawal patterns"
        ]

        responses = self.rag_pipeline.batch_process_queries(queries)

        assert len(responses) == 4
        for response in responses:
            assert "response" in response
            assert "confidence" in response
            assert response["confidence"] >= 0

    @pytest.mark.skipif(
        os.getenv("SKIP_MONGO_TESTS") == "1",
        reason="MongoDB not available"
    )
    def test_query_processing_optimization(self):
        """Test that query processing optimizes retrieval."""
        if self.skip_mongo:
            pytest.skip("MongoDB not available")

        # Complex query with multiple entities
        query = "Fix DDL_EXCEEDED errors on ATM001 from yesterday's withdrawal attempts"

        processed_query = self.query_processor.process_query(query)

        # Should extract multiple entities
        assert len(processed_query.entity_result.entities) >= 3

        # Should have optimized filters
        assert len(processed_query.optimized_filters) >= 2

        # Should suggest appropriate response type
        assert processed_query.response_type in ["troubleshooting", "error"]

        # Use optimized parameters in RAG pipeline
        response = self.rag_pipeline.process_query_sync(
            query=query,
            filters=processed_query.optimized_filters,
            top_k=processed_query.suggested_top_k,
            response_type=processed_query.response_type
        )

        # Should get relevant results
        assert response["sources_count"] > 0
        assert response["confidence"] > 0.3

    @pytest.mark.skipif(
        os.getenv("SKIP_MONGO_TESTS") == "1",
        reason="MongoDB not available"
    )
    def test_pipeline_health_and_statistics(self):
        """Test pipeline health monitoring and statistics."""
        if self.skip_mongo:
            pytest.skip("MongoDB not available")

        # Health check
        health = self.rag_pipeline.health_check()
        assert health["overall_status"] in ["healthy", "degraded"]

        # Statistics
        stats = self.rag_pipeline.get_pipeline_statistics()
        assert "pipeline_stats" in stats
        assert "vector_store_stats" in stats

        # Vector store should have our test data
        vector_stats = stats["vector_store_stats"]
        assert vector_stats["mongodb_docs"] >= 8  # Our test logs
        assert vector_stats["faiss_vectors"] >= 8

    def test_query_processing_without_mongo(self):
        """Test query processing components without MongoDB."""
        # This test runs even without MongoDB

        # Test intent classification
        troubleshooting_query = "Fix ATM withdrawal problems"
        processed = self.query_processor.process_query(troubleshooting_query)
        assert processed.intent_result.intent.value == "troubleshooting"

        # Test entity extraction
        entity_query = "ATM001 DDL_EXCEEDED error yesterday"
        processed = self.query_processor.process_query(entity_query)

        error_entities = [e for e in processed.entity_result.entities if e.entity_type == "error_code"]
        atm_entities = [e for e in processed.entity_result.entities if e.entity_type == "atm_id"]

        assert len(error_entities) > 0
        assert len(atm_entities) > 0
        assert error_entities[0].normalized_value == "DDL_EXCEEDED"

        # Test filter generation
        assert "error_code" in processed.optimized_filters
        assert "atm_id" in processed.optimized_filters

    def test_component_integration_validation(self):
        """Test that all components integrate correctly."""
        # Test log processing components
        log_files = self.log_reader.get_available_files()
        assert len(log_files) > 0

        log_entries = self.log_reader.read_log_file(log_files[0])
        assert len(log_entries) > 0

        parsed_log = self.log_parser.parse_log_entry(log_entries[0])
        assert "log_id" in parsed_log
        assert "operation" in parsed_log

        extracted_text = self.text_extractor.extract_searchable_text(parsed_log)
        assert len(extracted_text) > 0

        # Test embedding generation
        test_embedding = self.embedding_generator.generate_embedding(extracted_text)
        assert test_embedding.shape == (384,)

        # Test query processing
        test_query = "Test ATM query"
        processed = self.query_processor.process_query(test_query)
        assert processed.original_query == test_query

    @pytest.mark.skipif(
        os.getenv("SKIP_MONGO_TESTS") == "1",
        reason="MongoDB not available"
    )
    def test_real_world_scenarios(self):
        """Test realistic ATM support scenarios."""
        if self.skip_mongo:
            pytest.skip("MongoDB not available")

        # Scenario 1: Support agent investigating customer complaint
        complaint_query = "Customer says ATM001 keeps denying withdrawals"
        response = self.rag_pipeline.process_query_sync(complaint_query)

        assert response["sources_count"] > 0
        assert "ATM001" in response["response"]
        assert response["confidence"] > 0

        # Scenario 2: Technician needs error code explanation
        tech_query = "NETWORK_ERROR troubleshooting steps"
        response = self.rag_pipeline.process_query_sync(tech_query)

        assert "NETWORK_ERROR" in response["response"]
        assert response["response_type"] in ["troubleshooting", "error"]

        # Scenario 3: Manager wants performance overview
        manager_query = "ATM performance summary for today"
        response = self.rag_pipeline.process_query_sync(manager_query)

        assert response["response_type"] in ["analysis", "info"]

        # Scenario 4: Operations team monitoring recent issues
        ops_query = "Any new problems in the last hour?"
        response = self.rag_pipeline.get_recent_issues(hours_back=1)

        assert isinstance(response, dict)

    def test_error_handling_and_resilience(self):
        """Test system resilience and error handling."""
        # Test with empty query
        empty_response = self.query_processor.process_query("")
        assert empty_response.intent_result.intent.value == "unknown"

        # Test with non-ATM query
        non_atm_query = "What's the weather like?"
        validation = self.query_processor.validate_query(non_atm_query)
        assert not validation["is_atm_related"]

        # Test with very long query
        long_query = "ATM problem " * 100
        processed = self.query_processor.process_query(long_query)
        assert processed.query_complexity == "complex"

        # Test embedding with empty text
        empty_embedding = self.embedding_generator.generate_embedding("")
        assert empty_embedding.shape == (384,)


def run_complete_pipeline_tests():
    """Run complete pipeline tests."""
    print("Running Complete Pipeline Tests...")

    test_pipeline = TestCompleteATMRagPipeline()

    print("\n1. Setting up test environment...")
    test_pipeline.setup_method()

    try:
        print("\n2. Testing component integration...")
        test_pipeline.test_component_integration_validation()
        print("✓ Component integration tests passed")

        print("\n3. Testing query processing without database...")
        test_pipeline.test_query_processing_without_mongo()
        print("✓ Query processing tests passed")

        print("\n4. Testing error handling and resilience...")
        test_pipeline.test_error_handling_and_resilience()
        print("✓ Error handling tests passed")

        if os.getenv("SKIP_MONGO_TESTS") != "1" and not test_pipeline.skip_mongo:
            print("\n5. Testing complete troubleshooting workflow...")
            test_pipeline.test_complete_troubleshooting_workflow()
            print("✓ Troubleshooting workflow tests passed")

            print("\n6. Testing error code explanation workflow...")
            test_pipeline.test_error_code_explanation_workflow()
            print("✓ Error explanation workflow tests passed")

            print("\n7. Testing performance analysis workflow...")
            test_pipeline.test_performance_analysis_workflow()
            print("✓ Performance analysis workflow tests passed")

            print("\n8. Testing specific ATM troubleshooting...")
            test_pipeline.test_specific_atm_troubleshooting()
            print("✓ Specific ATM troubleshooting tests passed")

            print("\n9. Testing recent issues detection...")
            test_pipeline.test_recent_issues_detection()
            print("✓ Recent issues detection tests passed")

            print("\n10. Testing semantic search accuracy...")
            test_pipeline.test_semantic_search_accuracy()
            print("✓ Semantic search accuracy tests passed")

            print("\n11. Testing batch processing...")
            test_pipeline.test_multi_query_batch_processing()
            print("✓ Batch processing tests passed")

            print("\n12. Testing query optimization...")
            test_pipeline.test_query_processing_optimization()
            print("✓ Query optimization tests passed")

            print("\n13. Testing health monitoring...")
            test_pipeline.test_pipeline_health_and_statistics()
            print("✓ Health monitoring tests passed")

            print("\n14. Testing real-world scenarios...")
            test_pipeline.test_real_world_scenarios()
            print("✓ Real-world scenario tests passed")

            print("\n🎉 All Complete Pipeline tests passed!")
        else:
            print("\n5-14. MongoDB integration tests skipped")
            print("    To run full tests: brew services start mongodb-community")
            print("\n✅ Available tests completed successfully!")

    finally:
        print("\nCleaning up test environment...")
        test_pipeline.teardown_method()


if __name__ == "__main__":
    run_complete_pipeline_tests()