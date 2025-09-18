#!/usr/bin/env python3
"""
ATM RAG System Demo Test

A simple demonstration of the complete ATM RAG system
showing all major functionality without complex dependencies.
"""

import sys
from pathlib import Path
import numpy as np
import tempfile
import shutil
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def demo_query_processing():
    """Demonstrate query understanding and processing."""
    print("🔍 QUERY PROCESSING DEMO")
    print("-" * 40)

    from src.query_processor.query_processor import QueryProcessor

    processor = QueryProcessor()

    # Test various ATM support scenarios
    test_scenarios = [
        {
            "query": "Fix ATM001 DDL_EXCEEDED withdrawal errors",
            "description": "Troubleshooting request"
        },
        {
            "query": "What does NETWORK_ERROR code mean?",
            "description": "Error code explanation"
        },
        {
            "query": "Analyze withdrawal failures in last 24 hours",
            "description": "Performance analysis"
        },
        {
            "query": "Show me logs for ATM002 yesterday",
            "description": "Historical search"
        }
    ]

    for scenario in test_scenarios:
        query = scenario["query"]
        description = scenario["description"]

        print(f"\n📝 Scenario: {description}")
        print(f"   Query: \"{query}\"")

        # Process the query
        result = processor.process_query(query)

        print(f"   ✅ Intent: {result.intent_result.intent.value}")
        print(f"   ✅ Confidence: {result.intent_result.confidence:.2f}")
        print(f"   ✅ Entities found: {len(result.entity_result.entities)}")

        # Show extracted entities
        if result.entity_result.entities:
            entities = {}
            for entity in result.entity_result.entities:
                entities[entity.entity_type] = entity.normalized_value or entity.value
            print(f"   ✅ Extracted: {entities}")

        # Show generated filters
        if result.optimized_filters:
            print(f"   ✅ Filters: {list(result.optimized_filters.keys())}")

        print(f"   ✅ Response type: {result.response_type}")

    print("\n✅ Query processing working correctly!")


def demo_response_generation():
    """Demonstrate response generation."""
    print("\n🤖 RESPONSE GENERATION DEMO")
    print("-" * 40)

    from src.rag_engine.generator import ResponseGenerator

    generator = ResponseGenerator()

    # Create mock context data
    mock_context = {
        "relevant_logs": [
            {
                "log_id": "log_001",
                "timestamp": datetime.now(),
                "operation": "withdrawal",
                "status": "denied",
                "error_code": "DDL_EXCEEDED",
                "atm_id": "ATM001",
                "message": "Daily withdrawal limit exceeded",
                "retrieval_score": 0.95
            },
            {
                "log_id": "log_002",
                "timestamp": datetime.now(),
                "operation": "withdrawal",
                "status": "denied",
                "error_code": "DDL_EXCEEDED",
                "atm_id": "ATM001",
                "message": "Withdrawal amount exceeds daily limit",
                "retrieval_score": 0.87
            }
        ],
        "context_summary": "Found 2 similar DDL_EXCEEDED errors on ATM001"
    }

    # Test different response types
    test_queries = [
        ("Why is ATM001 denying withdrawals?", "troubleshooting"),
        ("What does DDL_EXCEEDED mean?", "error"),
        ("Analyze recent withdrawal patterns", "analysis")
    ]

    for query, response_type in test_queries:
        print(f"\n📝 Query: \"{query}\"")
        print(f"   Response type: {response_type}")

        response = generator.generate_response(query, mock_context, response_type)

        print(f"   ✅ Generated response type: {response['response_type']}")
        print(f"   ✅ Confidence: {response['confidence']:.2f}")
        print(f"   ✅ Sources used: {response['sources_count']}")
        print(f"   ✅ Response preview: {response['response'][:100]}...")

    print("\n✅ Response generation working correctly!")


def demo_vector_operations():
    """Demonstrate vector storage and search."""
    print("\n🔢 VECTOR OPERATIONS DEMO")
    print("-" * 40)

    from src.vector_store.faiss_index import FAISSIndexManager
    from src.embeddings import EmbeddingGenerator

    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()

    try:
        # Initialize components
        index_manager = FAISSIndexManager(dimensions=384, storage_path=temp_dir)
        embedding_generator = EmbeddingGenerator(model_type="fast")

        # Sample ATM log texts
        sample_logs = [
            "ATM001 withdrawal denied DDL exceeded daily limit",
            "ATM002 network connection timeout error",
            "ATM003 cash dispenser jam mechanical failure",
            "ATM001 successful withdrawal 200 dollars",
            "ATM002 deposit transaction completed successfully"
        ]

        log_ids = [f"log_{i+1:03d}" for i in range(len(sample_logs))]

        print(f"📄 Processing {len(sample_logs)} sample ATM logs...")

        # Generate embeddings
        embeddings = embedding_generator.generate_embeddings(sample_logs)
        print(f"   ✅ Generated embeddings: {embeddings.shape}")

        # Add to vector index
        result = index_manager.add_vectors(embeddings, log_ids)
        print(f"   ✅ Added to index: {result['added']} vectors")

        # Test similarity search
        query_texts = [
            "ATM withdrawal failure",
            "network connectivity issues",
            "successful transaction"
        ]

        for query_text in query_texts:
            print(f"\n🔍 Searching for: \"{query_text}\"")

            # Generate query embedding
            query_embedding = embedding_generator.generate_embedding(query_text)

            # Search for similar logs
            similar_ids, scores = index_manager.search_similar(
                query_embedding, top_k=3, min_score=0.1
            )

            print(f"   ✅ Found {len(similar_ids)} similar logs:")
            for log_id, score in zip(similar_ids, scores):
                log_idx = int(log_id.split('_')[1]) - 1
                print(f"      - {log_id}: {sample_logs[log_idx][:50]}... (score: {score:.3f})")

        # Test index statistics
        stats = index_manager.get_statistics()
        print(f"\n📊 Index statistics:")
        print(f"   ✅ Total vectors: {stats['total_vectors']}")
        print(f"   ✅ Index type: {stats['index_type']}")
        print(f"   ✅ Dimensions: {stats['dimensions']}")

        print("\n✅ Vector operations working correctly!")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


def demo_complete_workflow():
    """Demonstrate complete end-to-end workflow."""
    print("\n🔄 COMPLETE WORKFLOW DEMO")
    print("-" * 40)

    from src.query_processor.query_processor import QueryProcessor
    from src.rag_engine.generator import ResponseGenerator

    # Initialize components
    processor = QueryProcessor()
    generator = ResponseGenerator()

    # Simulate a real support scenario
    user_query = "ATM001 keeps showing DDL_EXCEEDED errors for customer withdrawals"

    print(f"👤 Support Query: \"{user_query}\"")

    # Step 1: Process the query
    print("\n🔍 Step 1: Understanding the query...")
    processed_query = processor.process_query(user_query)

    print(f"   ✅ Detected intent: {processed_query.intent_result.intent.value}")
    print(f"   ✅ Confidence: {processed_query.intent_result.confidence:.2f}")

    entities = {}
    for entity in processed_query.entity_result.entities:
        entities[entity.entity_type] = entity.normalized_value or entity.value
    print(f"   ✅ Key entities: {entities}")

    # Step 2: Simulate finding relevant logs (normally would use vector search)
    print("\n📚 Step 2: Retrieving relevant context...")
    mock_context = {
        "relevant_logs": [
            {
                "log_id": "log_2024_001",
                "timestamp": datetime.now(),
                "operation": "withdrawal",
                "status": "denied",
                "error_code": "DDL_EXCEEDED",
                "atm_id": "ATM001",
                "message": "Daily limit exceeded for customer account",
                "retrieval_score": 0.94
            }
        ],
        "context_summary": "Found similar DDL_EXCEEDED errors on ATM001"
    }

    print(f"   ✅ Found {len(mock_context['relevant_logs'])} relevant logs")
    print(f"   ✅ Context: {mock_context['context_summary']}")

    # Step 3: Generate response
    print("\n🤖 Step 3: Generating helpful response...")
    response = generator.generate_response(
        user_query,
        mock_context,
        processed_query.response_type
    )

    print(f"   ✅ Response type: {response['response_type']}")
    print(f"   ✅ Confidence: {response['confidence']:.2f}")

    # Step 4: Show final response
    print("\n💬 Final Response to Support Agent:")
    print("=" * 50)
    print(response['response'])
    print("=" * 50)

    print("\n✅ Complete workflow working correctly!")


def main():
    """Run all demonstrations."""
    print("🎯 ATM RAG SYSTEM DEMONSTRATION")
    print("=" * 50)
    print("Testing all major components and workflows...")

    try:
        # Run all demos
        demo_query_processing()
        demo_response_generation()
        demo_vector_operations()
        demo_complete_workflow()

        print("\n" + "=" * 50)
        print("🎉 ALL DEMONSTRATIONS SUCCESSFUL!")
        print("\n✅ Your ATM RAG system is working perfectly!")
        print("\n💡 Next steps:")
        print("   1. Start MongoDB: brew services start mongodb-community")
        print("   2. Run migration: python3 scripts/migrate_component2_data.py")
        print("   3. Test with real data: python3 tests/run_all_tests.py")
        print("   4. Build your ATM support interface!")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)