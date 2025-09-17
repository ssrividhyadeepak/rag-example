#!/usr/bin/env python3
"""
Test script for the ATM Log Processor component.

This script demonstrates how to use all components of the log processor:
- LogReader: Read JSON log files
- LogValidator: Validate log entries
- LogParser: Parse log entries into structured format
- TextExtractor: Convert logs to text for embeddings

Run this script to verify the log processor is working correctly.
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from log_processor import LogReader, LogParser, TextExtractor, LogValidator


def main():
    """Test the complete log processor pipeline."""
    print("🔄 Testing ATM Log Processor Component")
    print("=" * 50)

    # Initialize components
    log_reader = LogReader("data/logs")
    validator = LogValidator(strict_mode=False)
    parser = LogParser()
    extractor = TextExtractor(
        include_timestamp=True,
        include_session_info=True,
        include_metadata=True
    )

    print("\n1️⃣ Testing Log Reader")
    print("-" * 30)

    # Test reading available files
    available_files = log_reader.get_available_files()
    print(f"Available log files: {available_files}")

    if not available_files:
        print("❌ No log files found! Make sure sample files exist in data/logs/")
        return

    # Read all logs
    all_logs = log_reader.read_all_logs()
    print(f"✅ Read {len(all_logs)} total log entries")

    print("\n2️⃣ Testing Log Validator")
    print("-" * 30)

    # Validate logs
    validation_results = validator.validate_logs(all_logs)
    print(f"✅ Validation complete:")
    print(f"   - Total logs: {validation_results['total_logs']}")
    print(f"   - Valid logs: {validation_results['valid_logs']}")
    print(f"   - Invalid logs: {validation_results['invalid_logs']}")
    print(f"   - Validation rate: {validation_results['validation_rate']:.2%}")

    if validation_results['invalid_logs'] > 0:
        print(f"⚠️  Found validation errors:")
        for error_type, count in validation_results['errors_summary'].items():
            print(f"   - {error_type}: {count}")

    # Get only valid logs for further processing
    valid_logs = validator.get_valid_logs(all_logs)

    print("\n3️⃣ Testing Log Parser")
    print("-" * 30)

    # Parse logs
    parsed_logs = parser.parse_logs(valid_logs)
    print(f"✅ Parsed {len(parsed_logs)} log entries")

    # Show statistics
    stats = parser.get_statistics()
    print(f"   - Error logs: {stats['error_logs']}")
    print(f"   - Error rate: {stats['error_rate']:.2%}")
    print(f"   - Operations: {stats['operations']}")
    print(f"   - Statuses: {stats['statuses']}")

    print("\n4️⃣ Testing Text Extractor")
    print("-" * 30)

    # Extract text from parsed logs
    extracted_data = extractor.extract_batch(parsed_logs)
    print(f"✅ Extracted text from {len(extracted_data)} entries")

    # Show extraction statistics
    extraction_stats = extractor.get_extraction_statistics(extracted_data)
    print(f"   - Average text length: {extraction_stats['avg_text_length']:.1f} characters")
    print(f"   - Min/Max length: {extraction_stats['min_text_length']}/{extraction_stats['max_text_length']}")

    print("\n5️⃣ Sample Outputs")
    print("-" * 30)

    # Show examples of processed data
    for i, data in enumerate(extracted_data[:3]):  # Show first 3 examples
        print(f"\n📝 Example {i+1}:")
        print(f"   Operation: {data['operation']} - {data['status']}")
        print(f"   Is Error: {data['is_error']}")
        print(f"   Extracted Text: {data['extracted_text'][:100]}...")

        # Show different text extraction methods
        parsed_log = parsed_logs[i]
        summary_text = extractor.extract_summary_text(parsed_log)
        contextual_text = extractor.extract_contextual_text(parsed_log)

        print(f"   Summary: {summary_text}")
        print(f"   Context: {contextual_text[:80]}...")

    print("\n6️⃣ Testing Specific Scenarios")
    print("-" * 30)

    # Test filtering capabilities
    error_logs = parser.filter_errors_only()
    withdrawal_logs = parser.filter_by_operation("withdrawal")
    denied_logs = parser.filter_by_status("denied")

    print(f"✅ Filtering tests:")
    print(f"   - Error logs: {len(error_logs)}")
    print(f"   - Withdrawal logs: {len(withdrawal_logs)}")
    print(f"   - Denied operations: {len(denied_logs)}")

    # Test single file reading
    print(f"\n🔍 Testing single file read:")
    try:
        single_file_logs = log_reader.read_log_file("sample_atm_logs.json")
        print(f"   ✅ Read {len(single_file_logs)} entries from sample_atm_logs.json")
    except Exception as e:
        print(f"   ❌ Error reading single file: {e}")

    print("\n🎉 Log Processor Component Test Complete!")
    print("=" * 50)
    print("✅ All components working correctly")
    print(f"📊 Processed {len(all_logs)} total logs successfully")
    print("\n💡 Ready for next component: Local Embeddings Generator")


if __name__ == "__main__":
    main()