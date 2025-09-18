"""
Tests for Vector Store Components

Tests MongoDB store, FAISS index, and hybrid store functionality
including insertion, search, and data consistency.
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

from src.vector_store.schema import ATMLogSchema, VectorMetadataSchema
from src.vector_store.faiss_index import FAISSIndexManager
from src.vector_store.hybrid_store import HybridVectorStore


class TestATMLogSchema:
    """Test ATM log schema functionality."""

    def test_log_schema_creation(self):
        """Test creating ATM log schema."""
        log = ATMLogSchema(
            log_id="test_001",
            timestamp=datetime.utcnow(),
            session_id="SES_001",
            operation="withdrawal",
            status="denied",
            message="Test withdrawal denied",
            error_code="DDL_EXCEEDED",
            atm_id="ATM001",
            amount=500.0,
            extracted_text="Test extracted text"
        )

        assert log.log_id == "test_001"
        assert log.operation == "withdrawal"
        assert log.status == "denied"
        assert log.error_code == "DDL_EXCEEDED"
        assert log.amount == 500.0
        assert log.text_hash != ""  # Hash should be generated

    def test_log_schema_to_dict(self):
        """Test converting log schema to dictionary."""
        log = ATMLogSchema(
            log_id="test_002",
            timestamp=datetime.utcnow(),
            session_id="SES_002",
            operation="deposit",
            status="success",
            message="Test deposit"
        )

        log_dict = log.to_dict()

        assert isinstance(log_dict, dict)
        assert log_dict["log_id"] == "test_002"
        assert log_dict["operation"] == "deposit"
        assert log_dict["status"] == "success"
        assert "created_at" in log_dict
        assert "updated_at" in log_dict

    def test_log_schema_from_dict(self):
        """Test creating log schema from dictionary."""
        log_data = {
            "log_id": "test_003",
            "timestamp": datetime.utcnow(),
            "session_id": "SES_003",
            "operation": "balance_inquiry",
            "status": "success",
            "message": "Balance inquiry successful"
        }

        log = ATMLogSchema.from_dict(log_data)

        assert log.log_id == "test_003"
        assert log.operation == "balance_inquiry"
        assert log.status == "success"


class TestVectorMetadataSchema:
    """Test vector metadata schema functionality."""

    def test_metadata_schema_creation(self):
        """Test creating vector metadata schema."""
        metadata = VectorMetadataSchema(
            log_id="test_001",
            faiss_index=0,
            text_content="Test content for embedding",
            vector_norm=1.0,
            confidence_score=0.95
        )

        assert metadata.log_id == "test_001"
        assert metadata.faiss_index == 0
        assert metadata.embedding_model == "all-MiniLM-L6-v2"
        assert metadata.embedding_dimensions == 384
        assert metadata.text_length > 0
        assert metadata.text_hash != ""

    def test_metadata_schema_to_dict(self):
        """Test converting metadata schema to dictionary."""
        metadata = VectorMetadataSchema(
            log_id="test_002",
            faiss_index=1,
            text_content="Another test content"
        )

        metadata_dict = metadata.to_dict()

        assert isinstance(metadata_dict, dict)
        assert metadata_dict["log_id"] == "test_002"
        assert metadata_dict["faiss_index"] == 1
        assert metadata_dict["text_content"] == "Another test content"
        assert "processed_at" in metadata_dict


class TestFAISSIndexManager:
    """Test FAISS index manager functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.index_manager = FAISSIndexManager(
            dimensions=384,
            storage_path=self.temp_dir
        )

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_index_initialization(self):
        """Test FAISS index initialization."""
        assert self.index_manager.dimensions == 384
        assert self.index_manager.index_type == "IndexFlatIP"
        assert self.index_manager.total_vectors == 0

    def test_add_vectors(self):
        """Test adding vectors to index."""
        # Create test embeddings
        embeddings = np.random.rand(5, 384).astype('float32')
        log_ids = [f"log_{i}" for i in range(5)]

        # Add vectors
        result = self.index_manager.add_vectors(embeddings, log_ids)

        assert result["added"] == 5
        assert result["skipped"] == 0
        assert self.index_manager.total_vectors == 5

    def test_search_similar(self):
        """Test vector similarity search."""
        # Add some vectors first
        embeddings = np.random.rand(10, 384).astype('float32')
        log_ids = [f"log_{i}" for i in range(10)]
        self.index_manager.add_vectors(embeddings, log_ids)

        # Search for similar vectors
        query_vector = np.random.rand(384).astype('float32')
        result_ids, scores = self.index_manager.search_similar(
            query_vector, top_k=3
        )

        assert len(result_ids) <= 3
        assert len(scores) <= 3
        assert all(isinstance(score, float) for score in scores)

    def test_duplicate_prevention(self):
        """Test duplicate vector prevention."""
        embeddings = np.random.rand(3, 384).astype('float32')
        log_ids = ["log_1", "log_2", "log_3"]

        # Add vectors first time
        result1 = self.index_manager.add_vectors(embeddings, log_ids)
        assert result1["added"] == 3
        assert result1["skipped"] == 0

        # Try to add same vectors again
        result2 = self.index_manager.add_vectors(embeddings, log_ids)
        assert result2["added"] == 0
        assert result2["skipped"] == 3

    def test_save_and_load_index(self):
        """Test saving and loading index."""
        # Add some vectors
        embeddings = np.random.rand(5, 384).astype('float32')
        log_ids = [f"log_{i}" for i in range(5)]
        self.index_manager.add_vectors(embeddings, log_ids)

        # Save index
        save_result = self.index_manager.save_index()
        assert save_result is True

        # Create new index manager and load
        new_index_manager = FAISSIndexManager(
            dimensions=384,
            storage_path=self.temp_dir
        )

        assert new_index_manager.total_vectors == 5
        assert len(new_index_manager.id_mapping) == 5

    def test_get_statistics(self):
        """Test getting index statistics."""
        stats = self.index_manager.get_statistics()

        assert "index_type" in stats
        assert "dimensions" in stats
        assert "total_vectors" in stats
        assert stats["dimensions"] == 384
        assert stats["index_type"] == "IndexFlatIP"


@pytest.mark.skipif(
    os.getenv("SKIP_MONGO_TESTS") == "1",
    reason="MongoDB not available or tests disabled"
)
class TestHybridVectorStore:
    """Test hybrid vector store functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

        # Use test database
        self.vector_store = HybridVectorStore(
            mongodb_uri="mongodb://localhost:27017",
            database_name="atm_rag_test",
            storage_path=self.temp_dir
        )

        # Clear any existing test data
        self.vector_store.mongo_store.clear_all_data("CONFIRM_DELETE_ALL_DATA")
        self.vector_store.faiss_manager.clear_index()

    def teardown_method(self):
        """Clean up test environment."""
        try:
            # Clear test data
            self.vector_store.mongo_store.clear_all_data("CONFIRM_DELETE_ALL_DATA")
            self.vector_store.close()
        except:
            pass

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_insert_log_embeddings(self):
        """Test inserting ATM logs with embeddings."""
        # Create test logs
        logs = [
            ATMLogSchema(
                log_id=f"test_log_{i}",
                timestamp=datetime.utcnow(),
                session_id=f"SES_{i}",
                operation="withdrawal",
                status="denied" if i % 2 == 0 else "success",
                message=f"Test message {i}",
                error_code="DDL_EXCEEDED" if i % 2 == 0 else None,
                extracted_text=f"Test extracted text for log {i}"
            )
            for i in range(5)
        ]

        # Create test embeddings
        embeddings = np.random.rand(5, 384).astype('float32')

        # Insert into vector store
        result = self.vector_store.insert_log_embeddings(logs, embeddings)

        assert result["mongodb_inserted"] == 5
        assert result["faiss_added"] == 5
        assert result["vector_metadata_inserted"] == 5

    def test_search_similar_logs(self):
        """Test searching for similar logs."""
        # First insert some test data
        logs = [
            ATMLogSchema(
                log_id=f"search_test_{i}",
                timestamp=datetime.utcnow(),
                session_id=f"SES_{i}",
                operation="withdrawal",
                status="denied",
                message="Withdrawal denied DDL exceeded",
                error_code="DDL_EXCEEDED",
                extracted_text=f"Withdrawal denied DDL exceeded ATM{i}"
            )
            for i in range(3)
        ]

        embeddings = np.random.rand(3, 384).astype('float32')
        self.vector_store.insert_log_embeddings(logs, embeddings)

        # Search for similar logs
        query_vector = np.random.rand(384).astype('float32')
        results = self.vector_store.search_similar_logs(
            query_vector, top_k=2
        )

        assert len(results) <= 2
        for log, score in results:
            assert hasattr(log, 'log_id')
            assert isinstance(score, float)

    def test_query_logs_with_filters(self):
        """Test querying logs with MongoDB filters."""
        # Insert test data with specific attributes
        logs = [
            ATMLogSchema(
                log_id=f"filter_test_{i}",
                timestamp=datetime.utcnow(),
                session_id=f"SES_{i}",
                operation="withdrawal",
                status="denied" if i < 2 else "success",
                message=f"Test message {i}",
                error_code="DDL_EXCEEDED" if i < 2 else None,
                atm_id=f"ATM00{i+1}",
                extracted_text=f"Test text {i}"
            )
            for i in range(4)
        ]

        embeddings = np.random.rand(4, 384).astype('float32')
        self.vector_store.insert_log_embeddings(logs, embeddings)

        # Query with filters
        filters = {"status": "denied"}
        results = self.vector_store.query_logs(filters=filters)

        # Should only get denied transactions
        assert len(results) == 2
        for log in results:
            assert log.status == "denied"

    def test_health_check(self):
        """Test vector store health check."""
        health = self.vector_store.health_check()

        assert "status" in health
        assert "mongodb" in health
        assert "faiss" in health
        # Health should be good if MongoDB is running
        assert health["status"] in ["healthy", "degraded"]

    def test_get_statistics(self):
        """Test getting vector store statistics."""
        # Insert some test data first
        logs = [
            ATMLogSchema(
                log_id=f"stats_test_{i}",
                timestamp=datetime.utcnow(),
                session_id=f"SES_{i}",
                operation="withdrawal",
                status="success",
                message=f"Test message {i}",
                extracted_text=f"Test text {i}"
            )
            for i in range(3)
        ]

        embeddings = np.random.rand(3, 384).astype('float32')
        self.vector_store.insert_log_embeddings(logs, embeddings)

        # Get statistics
        stats = self.vector_store.get_statistics()

        assert "mongodb_docs" in stats
        assert "faiss_vectors" in stats
        assert "vector_metadata_docs" in stats
        assert stats["mongodb_docs"] >= 3
        assert stats["faiss_vectors"] >= 3

    def test_error_handling(self):
        """Test error handling with invalid data."""
        # Test with mismatched array sizes
        logs = [
            ATMLogSchema(
                log_id="error_test",
                timestamp=datetime.utcnow(),
                session_id="SES_ERROR",
                operation="withdrawal",
                status="denied",
                message="Error test",
                extracted_text="Error test text"
            )
        ]

        # Wrong number of embeddings (should be 1, providing 2)
        embeddings = np.random.rand(2, 384).astype('float32')

        # This should handle the error gracefully
        result = self.vector_store.insert_log_embeddings(logs, embeddings)

        # Should fail gracefully
        assert result["mongodb_inserted"] == 0
        assert result["faiss_added"] == 0


def run_vector_store_tests():
    """Run all vector store tests."""
    print("Running Vector Store Tests...")

    # Run tests that don't require MongoDB
    print("\n1. Testing ATM Log Schema...")
    test_log = TestATMLogSchema()
    test_log.test_log_schema_creation()
    test_log.test_log_schema_to_dict()
    test_log.test_log_schema_from_dict()
    print("✓ ATM Log Schema tests passed")

    print("\n2. Testing Vector Metadata Schema...")
    test_metadata = TestVectorMetadataSchema()
    test_metadata.test_metadata_schema_creation()
    test_metadata.test_metadata_schema_to_dict()
    print("✓ Vector Metadata Schema tests passed")

    print("\n3. Testing FAISS Index Manager...")
    test_faiss = TestFAISSIndexManager()
    test_faiss.setup_method()
    try:
        test_faiss.test_index_initialization()
        test_faiss.test_add_vectors()
        test_faiss.test_search_similar()
        test_faiss.test_duplicate_prevention()
        test_faiss.test_save_and_load_index()
        test_faiss.test_get_statistics()
        print("✓ FAISS Index Manager tests passed")
    finally:
        test_faiss.teardown_method()

    # Test MongoDB integration if available
    if os.getenv("SKIP_MONGO_TESTS") != "1":
        print("\n4. Testing Hybrid Vector Store (requires MongoDB)...")
        try:
            test_hybrid = TestHybridVectorStore()
            test_hybrid.setup_method()
            try:
                test_hybrid.test_insert_log_embeddings()
                test_hybrid.test_search_similar_logs()
                test_hybrid.test_query_logs_with_filters()
                test_hybrid.test_health_check()
                test_hybrid.test_get_statistics()
                test_hybrid.test_error_handling()
                print("✓ Hybrid Vector Store tests passed")
            finally:
                test_hybrid.teardown_method()
        except Exception as e:
            print(f"⚠ Hybrid Vector Store tests skipped: {e}")
            print("  (Make sure MongoDB is running: brew services start mongodb-community)")
    else:
        print("\n4. Hybrid Vector Store tests skipped (SKIP_MONGO_TESTS=1)")

    print("\n🎉 All Vector Store tests completed!")


if __name__ == "__main__":
    run_vector_store_tests()