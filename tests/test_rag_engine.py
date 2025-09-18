"""
Tests for RAG Engine Components

Tests prompt templates, context retriever, response generator,
and the complete RAG pipeline functionality.
"""

import pytest
import numpy as np
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag_engine.prompt_templates import PromptTemplates
from src.rag_engine.generator import ResponseGenerator
from src.vector_store.schema import ATMLogSchema
from src.embeddings import EmbeddingGenerator


class TestPromptTemplates:
    """Test prompt template functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.templates = PromptTemplates()

    def test_troubleshooting_response(self):
        """Test troubleshooting response generation."""
        query = "Why is ATM001 showing DDL_EXCEEDED errors?"
        similar_logs = [
            {
                "log_id": "log_001",
                "operation": "withdrawal",
                "status": "denied",
                "error_code": "DDL_EXCEEDED",
                "atm_id": "ATM001",
                "message": "Withdrawal denied DDL exceeded",
                "timestamp": datetime.utcnow()
            }
        ]
        context = "Found 1 similar case with DDL_EXCEEDED error on ATM001."

        response = self.templates.generate_troubleshooting_response(
            query, similar_logs, context
        )

        assert isinstance(response, str)
        assert len(response) > 0
        assert "DDL_EXCEEDED" in response
        assert "ATM001" in response
        assert "troubleshoot" in response.lower() or "resolve" in response.lower()

    def test_error_code_response(self):
        """Test error code explanation response."""
        error_code = "DDL_EXCEEDED"
        examples = [
            {
                "log_id": "log_001",
                "operation": "withdrawal",
                "status": "denied",
                "error_code": "DDL_EXCEEDED",
                "message": "Withdrawal denied DDL exceeded"
            },
            {
                "log_id": "log_002",
                "operation": "withdrawal",
                "status": "denied",
                "error_code": "DDL_EXCEEDED",
                "message": "Daily limit exceeded"
            }
        ]

        response = self.templates.generate_error_code_response(error_code, examples)

        assert isinstance(response, str)
        assert len(response) > 0
        assert "DDL_EXCEEDED" in response
        assert "daily" in response.lower() or "limit" in response.lower()

    def test_analysis_response(self):
        """Test analysis response generation."""
        query = "Analyze withdrawal failures in the last 24 hours"
        analysis_data = {
            "total_logs": 10,
            "operations": {"withdrawal": 8, "deposit": 2},
            "statuses": {"denied": 6, "success": 4},
            "error_codes": {"DDL_EXCEEDED": 4, "NETWORK_ERROR": 2},
            "atms": {"ATM001": 5, "ATM002": 3, "ATM003": 2}
        }

        response = self.templates.generate_analysis_response(query, analysis_data)

        assert isinstance(response, str)
        assert len(response) > 0
        assert "10" in response  # total_logs
        assert "withdrawal" in response.lower()
        assert "DDL_EXCEEDED" in response

    def test_informational_response(self):
        """Test informational response generation."""
        query = "What are the recent ATM operations?"
        context_summary = "Found 5 recent ATM operations across 3 machines."
        relevant_logs = [
            {
                "log_id": "log_001",
                "operation": "withdrawal",
                "status": "success",
                "atm_id": "ATM001"
            }
        ]

        response = self.templates.generate_informational_response(
            query, context_summary, relevant_logs
        )

        assert isinstance(response, str)
        assert len(response) > 0
        assert "ATM" in response

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        # Test with empty logs
        response = self.templates.generate_troubleshooting_response(
            "Test query", [], "No relevant logs found."
        )

        assert isinstance(response, str)
        assert len(response) > 0
        assert "no" in response.lower() or "not found" in response.lower()

        # Test with empty error examples
        response = self.templates.generate_error_code_response("UNKNOWN_ERROR", [])

        assert isinstance(response, str)
        assert len(response) > 0
        assert "UNKNOWN_ERROR" in response


class TestResponseGenerator:
    """Test response generator functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.generator = ResponseGenerator()

    def test_generate_response_auto_detection(self):
        """Test automatic response type detection."""
        # Troubleshooting query
        query = "Fix ATM001 withdrawal problems"
        context = {
            "relevant_logs": [
                {
                    "log_id": "log_001",
                    "operation": "withdrawal",
                    "status": "denied",
                    "error_code": "DDL_EXCEEDED",
                    "atm_id": "ATM001",
                    "retrieval_score": 0.9
                }
            ],
            "context_summary": "Found 1 relevant log with DDL_EXCEEDED error."
        }

        response = self.generator.generate_response(query, context, "auto")

        assert response["response_type"] in ["troubleshooting", "error"]
        assert isinstance(response["response"], str)
        assert len(response["response"]) > 0
        assert response["confidence"] > 0
        assert response["sources_count"] == 1

    def test_generate_troubleshooting_response(self):
        """Test troubleshooting response generation."""
        query = "ATM machine not dispensing cash"
        context = {
            "relevant_logs": [
                {
                    "log_id": "log_001",
                    "operation": "withdrawal",
                    "status": "failed",
                    "error_code": "DISPENSER_ERROR",
                    "message": "Cash dispenser malfunction",
                    "retrieval_score": 0.8
                }
            ]
        }

        response = self.generator.generate_troubleshooting_response(query, context)

        assert isinstance(response, str)
        assert len(response) > 0
        assert "dispenser" in response.lower() or "cash" in response.lower()

    def test_generate_error_code_response(self):
        """Test error code response generation."""
        error_code = "NETWORK_ERROR"
        context = {
            "relevant_logs": [
                {
                    "log_id": "log_001",
                    "error_code": "NETWORK_ERROR",
                    "operation": "withdrawal",
                    "status": "failed",
                    "message": "Network connection failed"
                }
            ]
        }

        response = self.generator.generate_error_code_response(error_code, context)

        assert isinstance(response, str)
        assert len(response) > 0
        assert "NETWORK_ERROR" in response
        assert "network" in response.lower()

    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        # High confidence context
        high_conf_context = {
            "relevant_logs": [
                {"retrieval_score": 0.9, "timestamp": datetime.utcnow()},
                {"retrieval_score": 0.8, "timestamp": datetime.utcnow()},
                {"retrieval_score": 0.85, "timestamp": datetime.utcnow()}
            ]
        }

        high_confidence = self.generator._calculate_confidence(high_conf_context)

        # Low confidence context
        low_conf_context = {
            "relevant_logs": [
                {"retrieval_score": 0.3, "timestamp": datetime.utcnow() - timedelta(days=10)}
            ]
        }

        low_confidence = self.generator._calculate_confidence(low_conf_context)

        assert high_confidence > low_confidence
        assert 0 <= high_confidence <= 1
        assert 0 <= low_confidence <= 1

    def test_response_post_processing(self):
        """Test response post-processing."""
        # Test length limiting
        long_response = "This is a very long response. " * 100
        processed = self.generator._post_process_response(long_response)

        assert len(processed) <= self.generator.max_response_length + 50  # Allow some buffer

        # Test proper sentence ending
        short_response = "This is a short response"
        processed = self.generator._post_process_response(short_response)

        assert processed.endswith('.')

    def test_error_handling(self):
        """Test error handling in response generation."""
        # Test with invalid context
        query = "Test query"
        invalid_context = {"invalid": "data"}

        response = self.generator.generate_response(query, invalid_context)

        assert "error" in response.get("metadata", {}) or response.get("response_type") == "error"
        assert isinstance(response["response"], str)


class MockVectorStore:
    """Mock vector store for testing without database."""

    def __init__(self):
        self.logs = []
        self.embeddings = []

    def search_similar_logs(self, query_vector, top_k=10, min_score=0.0, filters=None):
        """Mock search that returns test data."""
        # Return some mock results
        mock_results = []
        for i, log in enumerate(self.logs[:top_k]):
            score = 0.8 - (i * 0.1)  # Decreasing scores
            if score >= min_score:
                mock_results.append((log, score))
        return mock_results

    def get_statistics(self):
        """Mock statistics."""
        return {
            "mongodb_docs": len(self.logs),
            "faiss_vectors": len(self.embeddings)
        }

    def health_check(self):
        """Mock health check."""
        return {"status": "healthy"}

    def add_mock_data(self, logs, embeddings):
        """Add mock data for testing."""
        self.logs.extend(logs)
        self.embeddings.extend(embeddings)


class MockEmbeddingGenerator:
    """Mock embedding generator for testing."""

    def generate_embedding(self, text):
        """Generate a mock embedding."""
        # Return a deterministic embedding based on text hash
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        seed = int(hash_obj.hexdigest()[:8], 16)
        np.random.seed(seed % (2**32))
        return np.random.rand(384).astype('float32')


@pytest.mark.skipif(
    os.getenv("SKIP_INTEGRATION_TESTS") == "1",
    reason="Integration tests disabled"
)
class TestRAGPipelineIntegration:
    """Test RAG pipeline integration (without requiring external services)."""

    def setup_method(self):
        """Set up test environment with mocks."""
        from src.rag_engine.retriever import ContextRetriever
        from src.rag_engine.pipeline import ATMRagPipeline

        # Create mock components
        self.mock_vector_store = MockVectorStore()
        self.mock_embedding_generator = MockEmbeddingGenerator()

        # Add some mock data
        mock_logs = [
            ATMLogSchema(
                log_id="mock_001",
                timestamp=datetime.utcnow(),
                session_id="SES_001",
                operation="withdrawal",
                status="denied",
                message="Withdrawal denied DDL exceeded",
                error_code="DDL_EXCEEDED",
                atm_id="ATM001",
                extracted_text="Withdrawal denied DDL exceeded ATM001"
            ),
            ATMLogSchema(
                log_id="mock_002",
                timestamp=datetime.utcnow(),
                session_id="SES_002",
                operation="withdrawal",
                status="success",
                message="Withdrawal successful",
                atm_id="ATM002",
                extracted_text="Withdrawal successful ATM002"
            )
        ]

        mock_embeddings = [np.random.rand(384) for _ in mock_logs]
        self.mock_vector_store.add_mock_data(mock_logs, mock_embeddings)

        # Create RAG pipeline with mocks
        self.rag_pipeline = ATMRagPipeline(
            vector_store=self.mock_vector_store,
            embedding_generator=self.mock_embedding_generator
        )

    def test_pipeline_query_processing(self):
        """Test complete pipeline query processing."""
        query = "Why are withdrawals being denied on ATM001?"

        # Process query synchronously
        result = self.rag_pipeline.process_query_sync(query)

        assert isinstance(result, dict)
        assert "response" in result
        assert "response_type" in result
        assert "confidence" in result
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0

    def test_pipeline_troubleshooting(self):
        """Test specialized troubleshooting method."""
        result = self.rag_pipeline.troubleshoot_error(
            error_code="DDL_EXCEEDED",
            operation="withdrawal",
            atm_id="ATM001"
        )

        assert isinstance(result, dict)
        assert "response" in result
        assert result["response_type"] in ["troubleshooting", "error"]
        assert "DDL_EXCEEDED" in result["response"]

    def test_pipeline_analysis(self):
        """Test performance analysis method."""
        result = self.rag_pipeline.analyze_atm_performance(
            atm_id="ATM001"
        )

        assert isinstance(result, dict)
        assert "response" in result
        assert result["response_type"] in ["analysis", "info"]

    def test_pipeline_recent_issues(self):
        """Test recent issues method."""
        result = self.rag_pipeline.get_recent_issues(hours_back=24)

        assert isinstance(result, dict)
        assert "response" in result

    def test_pipeline_statistics(self):
        """Test pipeline statistics."""
        stats = self.rag_pipeline.get_pipeline_statistics()

        assert isinstance(stats, dict)
        assert "pipeline_stats" in stats
        assert "retriever_stats" in stats
        assert "vector_store_stats" in stats

    def test_pipeline_health_check(self):
        """Test pipeline health check."""
        health = self.rag_pipeline.health_check()

        assert isinstance(health, dict)
        assert "overall_status" in health
        assert "components" in health
        assert health["overall_status"] in ["healthy", "degraded", "unhealthy"]

    def test_batch_query_processing(self):
        """Test batch query processing."""
        queries = [
            "What are DDL_EXCEEDED errors?",
            "Show me recent withdrawal failures",
            "ATM001 status check"
        ]

        results = self.rag_pipeline.batch_process_queries(queries)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, dict)
            assert "response" in result
            assert "query" in result


def run_rag_engine_tests():
    """Run all RAG engine tests."""
    print("Running RAG Engine Tests...")

    print("\n1. Testing Prompt Templates...")
    test_templates = TestPromptTemplates()
    test_templates.setup_method()
    test_templates.test_troubleshooting_response()
    test_templates.test_error_code_response()
    test_templates.test_analysis_response()
    test_templates.test_informational_response()
    test_templates.test_empty_data_handling()
    print("✓ Prompt Templates tests passed")

    print("\n2. Testing Response Generator...")
    test_generator = TestResponseGenerator()
    test_generator.setup_method()
    test_generator.test_generate_response_auto_detection()
    test_generator.test_generate_troubleshooting_response()
    test_generator.test_generate_error_code_response()
    test_generator.test_confidence_calculation()
    test_generator.test_response_post_processing()
    test_generator.test_error_handling()
    print("✓ Response Generator tests passed")

    if os.getenv("SKIP_INTEGRATION_TESTS") != "1":
        print("\n3. Testing RAG Pipeline Integration...")
        test_pipeline = TestRAGPipelineIntegration()
        test_pipeline.setup_method()
        test_pipeline.test_pipeline_query_processing()
        test_pipeline.test_pipeline_troubleshooting()
        test_pipeline.test_pipeline_analysis()
        test_pipeline.test_pipeline_recent_issues()
        test_pipeline.test_pipeline_statistics()
        test_pipeline.test_pipeline_health_check()
        test_pipeline.test_batch_query_processing()
        print("✓ RAG Pipeline Integration tests passed")
    else:
        print("\n3. RAG Pipeline Integration tests skipped (SKIP_INTEGRATION_TESTS=1)")

    print("\n🎉 All RAG Engine tests completed!")


if __name__ == "__main__":
    run_rag_engine_tests()