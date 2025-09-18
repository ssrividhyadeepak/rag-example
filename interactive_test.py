#!/usr/bin/env python3
"""
Interactive ATM RAG Test

Test the system interactively with your own queries.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.query_processor.query_processor import QueryProcessor

def main():
    print("🎯 Interactive ATM RAG System Test")
    print("=" * 40)
    print("Enter ATM support queries to test the system.")
    print("Examples:")
    print("  - Fix ATM001 withdrawal problems")
    print("  - What does DDL_EXCEEDED mean?")
    print("  - Show me network errors")
    print("  - Analyze recent failures")
    print("\nType 'quit' to exit.\n")

    processor = QueryProcessor()

    while True:
        try:
            query = input("🔍 Your query: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                break

            if not query:
                continue

            print(f"\n📝 Processing: \"{query}\"")
            result = processor.process_query(query)

            print(f"✅ Intent: {result.intent_result.intent.value}")
            print(f"✅ Confidence: {result.intent_result.confidence:.2f}")

            if result.entity_result.entities:
                entities = {}
                for e in result.entity_result.entities:
                    entities[e.entity_type] = e.normalized_value or e.value
                print(f"✅ Entities: {entities}")

            if result.optimized_filters:
                print(f"✅ Filters: {list(result.optimized_filters.keys())}")

            print(f"✅ Response type: {result.response_type}")
            print(f"✅ Suggested top_k: {result.suggested_top_k}")
            print("-" * 40)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n👋 Thanks for testing the ATM RAG system!")

if __name__ == "__main__":
    main()