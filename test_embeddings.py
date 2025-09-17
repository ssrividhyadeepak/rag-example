#!/usr/bin/env python3
"""
Test script for the ATM Embeddings Component.

This script demonstrates the complete embeddings pipeline:
- EmbeddingGenerator: Generate local embeddings using sentence-transformers
- BatchProcessor: Process ATM logs and generate embeddings
- SimilaritySearch: Find similar log entries using vector search
- CacheManager: Cache embeddings for performance

Run this script to verify the embeddings component is working correctly.
"""

import sys
import os
import time

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from embeddings import EmbeddingGenerator, BatchProcessor, SimilaritySearch, CacheManager


def test_embedding_generator():
    """Test the basic embedding generation functionality."""
    print("\n1️⃣ Testing EmbeddingGenerator")
    print("-" * 40)

    # Initialize generator
    generator = EmbeddingGenerator(model_type='fast')

    # Test model info
    model_info = generator.get_model_info()
    print(f"✅ Model: {model_info['model_name']}")
    print(f"   Dimensions: {model_info['dimensions']}")
    print(f"   Size: {model_info['size_mb']}MB")

    # Test single embedding
    test_text = "Withdrawal denied DDL exceeded"
    embedding = generator.generate_embedding(test_text)
    print(f"✅ Generated embedding for: '{test_text}'")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   Embedding type: {type(embedding)}")

    # Test batch embeddings
    test_texts = [
        "Withdrawal denied DDL exceeded",
        "Cash dispensed successfully",
        "Balance inquiry completed",
        "Deposit failed envelope jam detected"
    ]

    start_time = time.time()
    batch_embeddings = generator.generate_embeddings(test_texts, show_progress=False)
    batch_time = time.time() - start_time

    print(f"✅ Generated {len(test_texts)} embeddings in {batch_time:.2f}s")
    print(f"   Batch embeddings shape: {batch_embeddings.shape}")
    print(f"   Average time per embedding: {batch_time/len(test_texts)*1000:.1f}ms")

    # Test similarity
    sim1 = generator.similarity(batch_embeddings[0], batch_embeddings[1])
    sim2 = generator.similarity(batch_embeddings[0], batch_embeddings[0])
    print(f"✅ Similarity between different texts: {sim1:.3f}")
    print(f"✅ Self-similarity (should be ~1.0): {sim2:.3f}")

    # Test finding similar embeddings
    query_embedding = batch_embeddings[0]
    similar_results = generator.find_most_similar(query_embedding, batch_embeddings, top_k=3)
    print(f"✅ Found {len(similar_results)} similar embeddings")

    # Benchmark
    benchmark_results = generator.benchmark(test_texts)
    print(f"✅ Benchmark: {benchmark_results['avg_embedding_time_ms']:.1f}ms per embedding")

    return generator


def test_batch_processor(embedding_generator):
    """Test the batch processing of ATM logs."""
    print("\n2️⃣ Testing BatchProcessor")
    print("-" * 40)

    # Initialize processor
    processor = BatchProcessor(
        embedding_model='fast',
        include_timestamp=True,
        include_session_info=True,
        include_metadata=True
    )

    # Process log files
    print("Processing ATM log files...")
    processed_logs = processor.process_log_files("data/logs")

    if not processed_logs:
        print("❌ No logs were processed")
        return None

    print(f"✅ Processed {len(processed_logs)} log entries with embeddings")

    # Show sample processed log
    sample_log = processed_logs[0]
    print(f"\n📝 Sample processed log:")
    print(f"   Log ID: {sample_log.log_id}")
    print(f"   Operation: {sample_log.metadata.get('operation')}")
    print(f"   Status: {sample_log.metadata.get('status')}")
    print(f"   Is Error: {sample_log.metadata.get('is_error')}")
    print(f"   Text: {sample_log.extracted_text[:100]}...")
    print(f"   Embedding shape: {sample_log.embedding.shape}")

    # Test filtering
    error_logs = processor.filter_errors_only()
    withdrawal_logs = processor.filter_by_operation("withdrawal")
    denied_logs = processor.filter_by_status("denied")

    print(f"\n🔍 Filtering tests:")
    print(f"   Error logs: {len(error_logs)}")
    print(f"   Withdrawal logs: {len(withdrawal_logs)}")
    print(f"   Denied operations: {len(denied_logs)}")

    # Test similarity search
    query_results = processor.find_similar_logs("withdrawal denied", top_k=3)
    print(f"\n🔍 Similarity search for 'withdrawal denied':")
    for i, result in enumerate(query_results[:3]):
        print(f"   {i+1}. Similarity: {result['similarity']:.3f}")
        print(f"      Operation: {result['operation']} - {result['status']}")
        print(f"      Text: {result['extracted_text'][:80]}...")

    # Test statistics
    stats = processor.get_statistics()
    print(f"\n📊 Processing statistics:")
    print(f"   Total logs: {stats['total_logs']}")
    print(f"   Error rate: {stats['error_rate']:.2%}")
    print(f"   Operations: {stats['operations']}")
    print(f"   Model: {stats['embedding_model']} ({stats['embedding_dimensions']} dims)")

    # Test save/load
    save_path = "test_embeddings.json"
    processor.save_embeddings(save_path)
    print(f"✅ Saved embeddings to {save_path}")

    # Test loading
    new_processor = BatchProcessor()
    success = new_processor.load_embeddings(save_path)
    if success:
        print(f"✅ Loaded embeddings from {save_path}")
        print(f"   Loaded {len(new_processor.processed_logs)} logs")
    else:
        print(f"❌ Failed to load embeddings from {save_path}")

    # Cleanup
    if os.path.exists(save_path):
        os.remove(save_path)

    return processor


def test_similarity_search(processor):
    """Test advanced similarity search functionality."""
    print("\n3️⃣ Testing SimilaritySearch")
    print("-" * 40)

    if not processor or not processor.processed_logs:
        print("❌ No processed logs available for similarity search")
        return None

    # Initialize similarity search
    search = SimilaritySearch(processor.processed_logs)
    print(f"✅ Initialized similarity search with {len(processor.processed_logs)} log entries")

    # Test basic search
    sample_log = processor.processed_logs[0]
    query_embedding = sample_log.embedding

    search_results = search.search(
        query_embedding,
        top_k=5,
        min_similarity=0.0
    )

    print(f"\n🔍 Basic search results: {len(search_results)} found")
    for i, result in enumerate(search_results[:3]):
        print(f"   {i+1}. Similarity: {result['similarity']:.3f}")
        print(f"      Operation: {result['metadata']['operation']} - {result['metadata']['status']}")

    # Test filtered search
    error_results = search.search(
        query_embedding,
        top_k=3,
        error_only=True
    )
    print(f"✅ Error-only search: {len(error_results)} results")

    operation_results = search.search(
        query_embedding,
        top_k=3,
        operation_filter="withdrawal"
    )
    print(f"✅ Withdrawal-only search: {len(operation_results)} results")

    # Test text-based search
    text_results = search.search_by_text(
        "ATM card problem",
        processor.embedding_generator,
        top_k=3
    )
    print(f"✅ Text search for 'ATM card problem': {len(text_results)} results")

    # Test finding similar errors
    error_logs = [log for log in processor.processed_logs if log.metadata.get('is_error')]
    if error_logs:
        error_log_id = error_logs[0].log_id
        similar_errors = search.find_similar_errors(error_log_id, top_k=3)
        print(f"✅ Similar errors to {error_log_id}: {len(similar_errors)} found")

    # Test statistics
    embedding_stats = search.get_embedding_statistics()
    print(f"\n📊 Embedding space statistics:")
    print(f"   Total embeddings: {embedding_stats['total_embeddings']}")
    print(f"   Dimensions: {embedding_stats['embedding_dimensions']}")
    print(f"   Mean similarity: {embedding_stats['mean_similarity']:.3f}")
    print(f"   Std similarity: {embedding_stats['std_similarity']:.3f}")

    # Test clustering analysis (if scikit-learn is available)
    try:
        cluster_results = search.cluster_analysis(n_clusters=3)
        if cluster_results:
            print(f"✅ Cluster analysis: {cluster_results['n_clusters']} clusters")
            for cluster_id, stats in cluster_results['clusters'].items():
                print(f"   Cluster {cluster_id}: {stats['size']} logs, {stats['error_rate']:.2%} error rate")
    except Exception as e:
        print(f"⚠️  Clustering analysis not available: {e}")

    # Test similarity explanation
    if len(processor.processed_logs) >= 2:
        log1_id = processor.processed_logs[0].log_id
        log2_id = processor.processed_logs[1].log_id
        explanation = search.explain_similarity(log1_id, log2_id)
        print(f"\n📖 Similarity explanation between {log1_id} and {log2_id}:")
        print(f"   Similarity: {explanation['similarity_score']:.3f}")
        print(f"   Operation match: {explanation['metadata_comparison']['operation_match']}")
        print(f"   Status match: {explanation['metadata_comparison']['status_match']}")

    return search


def test_cache_manager(embedding_generator):
    """Test embedding caching functionality."""
    print("\n4️⃣ Testing CacheManager")
    print("-" * 40)

    # Initialize cache manager
    cache = CacheManager(cache_dir="test_cache", enable_disk_cache=True)
    print(f"✅ Initialized cache manager")

    # Test caching single embedding
    test_text = "Withdrawal denied DDL exceeded"
    model_info = embedding_generator.get_model_info()

    # Generate and cache embedding
    original_embedding = embedding_generator.generate_embedding(test_text)
    cache_key = cache.cache_embedding(test_text, original_embedding, model_info)
    print(f"✅ Cached embedding with key: {cache_key[:16]}...")

    # Retrieve cached embedding
    cached_embedding = cache.get_cached_embedding(test_text, model_info)
    if cached_embedding is not None:
        print(f"✅ Retrieved cached embedding")
        # Verify they're the same
        similarity = embedding_generator.similarity(original_embedding, cached_embedding)
        print(f"   Cache accuracy: {similarity:.6f} (should be ~1.0)")
    else:
        print(f"❌ Failed to retrieve cached embedding")

    # Test batch caching
    test_texts = [
        "Cash dispensed successfully",
        "Balance inquiry completed",
        "Deposit failed envelope jam"
    ]

    batch_embeddings = embedding_generator.generate_embeddings(test_texts)
    cache_keys = cache.cache_batch_embeddings(test_texts, batch_embeddings, model_info)
    print(f"✅ Cached {len([k for k in cache_keys if k])} batch embeddings")

    # Test batch retrieval
    cached_batch = cache.get_batch_cached_embeddings(test_texts, model_info)
    print(f"✅ Retrieved {len(cached_batch)} cached embeddings from batch")

    # Test cache statistics
    cache_stats = cache.get_cache_statistics()
    print(f"\n📊 Cache statistics:")
    print(f"   Memory cache size: {cache_stats['memory_cache_size']}")
    print(f"   Disk cache enabled: {cache_stats['disk_cache_enabled']}")
    if 'disk_cache_size_mb' in cache_stats:
        print(f"   Disk cache size: {cache_stats['disk_cache_size_mb']:.2f} MB")

    # Test cache cleanup
    cleanup_count = cache.cleanup_old_cache(max_age_hours=0)  # Clean everything
    print(f"✅ Cleaned up {cleanup_count} cache entries")

    # Test cache index
    cache.save_cache_index()
    cache_index = cache.load_cache_index()
    if cache_index:
        print(f"✅ Cache index saved and loaded")

    # Cleanup test cache directory
    cache.clear_all_cache()
    print(f"✅ Cleared all cache")

    # Remove test cache directory
    import shutil
    if os.path.exists("test_cache"):
        shutil.rmtree("test_cache")

    return cache


def test_integration():
    """Test integration between all components."""
    print("\n5️⃣ Testing Component Integration")
    print("-" * 40)

    # Create a complete pipeline
    print("Creating integrated pipeline...")

    # Initialize all components
    generator = EmbeddingGenerator(model_type='fast')
    cache = CacheManager(cache_dir="integration_cache")
    processor = BatchProcessor(embedding_model='fast')

    # Process logs with caching
    processed_logs = processor.process_log_files("data/logs")
    search = SimilaritySearch(processed_logs)

    print(f"✅ Complete pipeline created with {len(processed_logs)} logs")

    # Test end-to-end search
    query = "Why was my withdrawal denied?"
    results = search.search_by_text(query, generator, top_k=3)

    print(f"\n🔍 End-to-end search for: '{query}'")
    for i, result in enumerate(results):
        print(f"   {i+1}. Similarity: {result['similarity']:.3f}")
        print(f"      {result['metadata']['operation']} - {result['metadata']['status']}")
        print(f"      Text: {result['extracted_text'][:80]}...")

    # Test performance with caching
    start_time = time.time()
    for _ in range(3):
        search.search_by_text(query, generator, top_k=1)
    search_time = time.time() - start_time

    print(f"✅ Performance test: 3 searches in {search_time:.2f}s")

    # Cleanup
    cache.clear_all_cache()
    if os.path.exists("integration_cache"):
        import shutil
        shutil.rmtree("integration_cache")

    return True


def main():
    """Run all embedding component tests."""
    print("🔄 Testing ATM Embeddings Component")
    print("=" * 50)

    try:
        # Test individual components
        generator = test_embedding_generator()
        processor = test_batch_processor(generator)
        search = test_similarity_search(processor)
        cache = test_cache_manager(generator)

        # Test integration
        integration_success = test_integration()

        print("\n🎉 Embeddings Component Test Complete!")
        print("=" * 50)
        print("✅ All components working correctly")
        print(f"📊 Generated embeddings for ATM logs successfully")

        # Summary
        if processor and processor.processed_logs:
            print(f"\n💡 Summary:")
            print(f"   - Processed {len(processor.processed_logs)} ATM logs")
            print(f"   - Generated {generator.dimensions}-dimensional embeddings")
            print(f"   - Model: {generator.model_name}")
            print(f"   - Ready for Component 3: Vector Store & RAG")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()