#!/usr/bin/env python3
"""
Test Runner for ATM RAG System

Runs all test suites and provides comprehensive test results.
Includes options for running specific test categories and
handling different test environments (with/without MongoDB).
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import test modules
from test_vector_store import run_vector_store_tests
from test_rag_engine import run_rag_engine_tests
from test_query_processor import run_query_processor_tests
from test_complete_pipeline import run_complete_pipeline_tests


def check_mongodb_availability():
    """Check if MongoDB is available for testing."""
    try:
        import pymongo
        client = pymongo.MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=1000)
        client.server_info()
        client.close()
        return True
    except Exception:
        return False


def run_specific_tests(test_category):
    """Run specific test category."""
    test_functions = {
        "vector_store": run_vector_store_tests,
        "rag_engine": run_rag_engine_tests,
        "query_processor": run_query_processor_tests,
        "complete_pipeline": run_complete_pipeline_tests
    }

    if test_category in test_functions:
        print(f"\n{'='*60}")
        print(f"Running {test_category.replace('_', ' ').title()} Tests")
        print(f"{'='*60}")
        test_functions[test_category]()
    else:
        print(f"Unknown test category: {test_category}")
        print("Available categories:", list(test_functions.keys()))


def run_all_tests(skip_mongo=False):
    """Run all test suites."""
    start_time = time.time()

    print("🧪 ATM RAG System - Comprehensive Test Suite")
    print("=" * 60)

    # Environment info
    print("\n📋 Test Environment:")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   Platform: {sys.platform}")
    print(f"   Working Directory: {os.getcwd()}")

    # MongoDB availability
    mongo_available = check_mongodb_availability()
    print(f"   MongoDB Available: {'✅ Yes' if mongo_available else '❌ No'}")

    if skip_mongo or not mongo_available:
        os.environ["SKIP_MONGO_TESTS"] = "1"
        print("   📝 MongoDB tests will be skipped")

    print("\n" + "=" * 60)

    test_results = {}

    # Test Suite 1: Vector Store
    try:
        print("\n🗃️  TEST SUITE 1: Vector Store Components")
        print("-" * 60)
        run_vector_store_tests()
        test_results["vector_store"] = "✅ PASSED"
    except Exception as e:
        print(f"❌ Vector Store tests failed: {e}")
        test_results["vector_store"] = "❌ FAILED"

    # Test Suite 2: RAG Engine
    try:
        print("\n🤖 TEST SUITE 2: RAG Engine Components")
        print("-" * 60)
        run_rag_engine_tests()
        test_results["rag_engine"] = "✅ PASSED"
    except Exception as e:
        print(f"❌ RAG Engine tests failed: {e}")
        test_results["rag_engine"] = "❌ FAILED"

    # Test Suite 3: Query Processor
    try:
        print("\n🔍 TEST SUITE 3: Query Processor Components")
        print("-" * 60)
        run_query_processor_tests()
        test_results["query_processor"] = "✅ PASSED"
    except Exception as e:
        print(f"❌ Query Processor tests failed: {e}")
        test_results["query_processor"] = "❌ FAILED"

    # Test Suite 4: Complete Pipeline
    try:
        print("\n🏗️  TEST SUITE 4: Complete Pipeline Integration")
        print("-" * 60)
        run_complete_pipeline_tests()
        test_results["complete_pipeline"] = "✅ PASSED"
    except Exception as e:
        print(f"❌ Complete Pipeline tests failed: {e}")
        test_results["complete_pipeline"] = "❌ FAILED"

    # Final Results
    end_time = time.time()
    total_time = end_time - start_time

    print("\n" + "=" * 60)
    print("📊 FINAL TEST RESULTS")
    print("=" * 60)

    all_passed = True
    for test_suite, result in test_results.items():
        print(f"   {test_suite.replace('_', ' ').title():<25} {result}")
        if "FAILED" in result:
            all_passed = False

    print("\n" + "-" * 60)
    print(f"⏱️  Total Test Time: {total_time:.2f} seconds")

    if all_passed:
        print("🎉 ALL TESTS PASSED! 🎉")
        print("\nYour ATM RAG system is working correctly!")

        if skip_mongo or not mongo_available:
            print("\n📝 Note: Some integration tests were skipped due to MongoDB unavailability.")
            print("   To run complete tests: brew services start mongodb-community")

    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease check the error messages above and fix any issues.")

    print("\n" + "=" * 60)

    return all_passed


def print_test_help():
    """Print help information about running tests."""
    help_text = """
🧪 ATM RAG System Test Suite

This test suite validates all components of the ATM RAG system:

📋 Test Categories:
   vector_store     - MongoDB, FAISS, and hybrid store tests
   rag_engine       - Prompt templates, retrieval, generation, pipeline tests
   query_processor  - Intent classification, entity extraction tests
   complete_pipeline - End-to-end integration tests

🛠️  Prerequisites:
   • Python 3.8+ with required packages installed
   • MongoDB running locally (for full integration tests)
     Start MongoDB: brew services start mongodb-community

🏃 Running Tests:
   python run_all_tests.py                    # Run all tests
   python run_all_tests.py --skip-mongo       # Skip MongoDB tests
   python run_all_tests.py --category vector_store  # Run specific category
   python run_all_tests.py --help             # Show this help

🔧 Environment Variables:
   SKIP_MONGO_TESTS=1        # Skip MongoDB integration tests
   SKIP_INTEGRATION_TESTS=1  # Skip integration tests

💡 Tips:
   • If MongoDB is not available, tests will automatically skip database operations
   • Vector and query processing tests work without external dependencies
   • Use --category to run specific test suites during development
   • Check logs for detailed error information if tests fail

🎯 Expected Results:
   ✅ All tests should pass if components are correctly implemented
   ⚠️  Some tests may be skipped if dependencies are unavailable
   ❌ Failed tests indicate bugs that need to be fixed
"""
    print(help_text)


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(
        description="Run ATM RAG system tests",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--category",
        choices=["vector_store", "rag_engine", "query_processor", "complete_pipeline"],
        help="Run specific test category only"
    )
    parser.add_argument(
        "--skip-mongo",
        action="store_true",
        help="Skip MongoDB integration tests"
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List available test categories"
    )

    args = parser.parse_args()

    if args.list_categories:
        print("Available test categories:")
        print("  vector_store     - Vector storage and search components")
        print("  rag_engine       - RAG pipeline and response generation")
        print("  query_processor  - Query understanding and processing")
        print("  complete_pipeline - End-to-end integration tests")
        return

    try:
        if args.category:
            run_specific_tests(args.category)
        else:
            success = run_all_tests(skip_mongo=args.skip_mongo)
            sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()