# ATM Assist RAG Application

A local Retrieval Augmented Generation (RAG) system for intelligent ATM operations assistance, built with Python and designed to analyze ATM logs and provide contextual help for troubleshooting and operational insights.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   JSON Logs     │───▶│  Log Processor  │───▶│   Embeddings    │
│   (Splunk)      │    │    Component    │    │   Generator     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Query API     │◀───│   RAG Engine    │◀───│  Vector Store   │
│   (FastAPI)     │    │                 │    │   (MongoDB)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🎯 Purpose

Transform ATM operational logs into an intelligent assistance system that can:
- **Answer questions** about ATM errors and operations
- **Provide contextual help** for troubleshooting specific issues
- **Analyze patterns** in ATM transaction failures
- **Offer insights** based on historical operational data

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **MongoDB**: Local instance (no Atlas required)
- **Memory**: 4GB RAM minimum (for local embeddings)
- **Storage**: 1GB for models and data

### Python Dependencies
```bash
pip install -r requirements.txt
```

Key packages:
- `sentence-transformers` - Local embeddings generation
- `pymongo` - MongoDB operations
- `fastapi` - REST API framework
- `torch` - Machine learning backend

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Clone repository
git clone <repository-url>
cd atm-rag

# Install Python dependencies
pip install -r requirements.txt

# Install MongoDB (macOS)
brew install mongodb-community
brew services start mongodb-community
```

### 2. Test Log Processor (Component 1)
```bash
# Run the log processor test
python3 test_log_processor.py
```

Expected output:
```
✅ All components working correctly
📊 Processed 13 total logs successfully
💡 Ready for next component: Local Embeddings Generator
```

### 3. Project Structure
```
atm-rag/
├── src/
│   └── log_processor/          # ✅ Component 1: JSON Log Processing
│       ├── log_reader.py       # Read JSON log files
│       ├── log_parser.py       # Parse structured log entries
│       ├── text_extractor.py   # Convert logs to text for embeddings
│       └── validator.py        # Validate log data quality
├── data/
│   └── logs/                   # Sample ATM log files
│       ├── sample_atm_logs.json
│       └── error_logs.json
├── test_log_processor.py       # Test script for Component 1
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔄 Functional Flow

### Phase 1: Log Processing (✅ Complete)
```
JSON Logs → LogReader → LogValidator → LogParser → TextExtractor → Structured Data
```

**Input**: JSON files with ATM transaction logs
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "session_id": "SES_20240115_103000_ATM001",
  "customer_session_id": "CUST_789ABC123",
  "operation": "withdrawal",
  "status": "denied",
  "message": "Withdrawal denied DDL exceeded",
  "error_code": "DDL_EXCEEDED",
  "atm_id": "ATM001",
  "amount": 500
}
```

**Output**: Processed text ready for embeddings
```
"Time: 2024-01-15T10:30:00Z. Session: SES_20240115_103000_ATM001.
withdrawal denied. Withdrawal denied DDL exceeded. Error Code: DDL_EXCEEDED.
ATM: ATM001. Amount: 500. Event: withdrawal_denied"
```

### Phase 2: Embeddings Generation (🔄 Next)
```
Processed Text → sentence-transformers → Vector Embeddings → MongoDB Storage
```

### Phase 3: Vector Search & RAG (🔄 Planned)
```
User Query → Generate Embedding → Similarity Search → Context Retrieval → AI Response
```

### Phase 4: API Interface (🔄 Planned)
```
FastAPI Endpoints → Query Processing → RAG Engine → Contextual Responses
```

## 📊 Component 1 Status: Log Processor

### ✅ Completed Features
- **LogReader**: Reads JSON files with error handling
- **LogValidator**: Validates required fields (100% validation rate)
- **LogParser**: Structured parsing with filtering capabilities
- **TextExtractor**: Multiple text formats for embedding optimization

### 📈 Test Results
- **13 log entries** processed successfully
- **69% error detection** rate working correctly
- **Average 269 characters** per extracted text
- **6 operation types** categorized (withdrawal, deposit, etc.)

### 🎯 Log Categories Supported
- **Operations**: withdrawal, deposit, balance_inquiry, transfer, card_operation
- **Statuses**: success, denied, failed, error, timeout, cancelled
- **Error Detection**: Automatic identification of failure conditions
- **Filtering**: By operation type, status, or error conditions

## 🔧 Usage Examples

### Basic Log Processing
```python
from log_processor import LogReader, LogParser, TextExtractor

# Read logs
reader = LogReader("data/logs")
logs = reader.read_all_logs()

# Parse and extract
parser = LogParser()
parsed_logs = parser.parse_logs(logs)

extractor = TextExtractor()
extracted_data = extractor.extract_batch(parsed_logs)

print(f"Processed {len(extracted_data)} log entries")
```

### Filter Error Logs Only
```python
error_logs = parser.filter_errors_only()
withdrawal_errors = parser.filter_by_operation("withdrawal")
denied_operations = parser.filter_by_status("denied")
```

### Multiple Text Extraction Formats
```python
# Standard format (full context)
standard_text = extractor.extract_text(parsed_log)

# Summary format (quick overview)
summary_text = extractor.extract_summary_text(parsed_log)

# Contextual format (RAG optimized)
context_text = extractor.extract_contextual_text(parsed_log)
```

## 🗺️ Roadmap

### ✅ Phase 1: Log Processing (Complete)
- JSON log reading and validation
- Structured parsing with explicit fields
- Text extraction for embeddings
- Comprehensive testing

### 🔄 Phase 2: Local Embeddings (Next)
- sentence-transformers integration
- Vector generation for log entries
- MongoDB vector storage
- Similarity search capabilities

### 🔄 Phase 3: RAG Engine (Planned)
- Context retrieval system
- Query processing pipeline
- Response generation
- Knowledge base integration

### 🔄 Phase 4: API Interface (Planned)
- FastAPI REST endpoints
- Query interface for ATM assistance
- Real-time log processing
- Web-based demo interface

## 🤝 Contributing

This is a component-based development approach. Each component is built and tested independently:

1. **Component 1**: Log Processing ✅
2. **Component 2**: Local Embeddings 🔄
3. **Component 3**: Vector Search & RAG 🔄
4. **Component 4**: API Interface 🔄

## 📝 Sample Log Data

The project includes sample ATM logs demonstrating various scenarios:
- Successful transactions (withdrawals, deposits, transfers)
- Error conditions (DDL exceeded, insufficient funds, system errors)
- Operational events (maintenance, card retention, timeouts)

## 🔒 Privacy & Security

- **Local Processing**: All data processing happens locally
- **No External APIs**: Uses local sentence-transformers models
- **No Cloud Dependencies**: Self-contained system
- **Data Privacy**: ATM logs never leave your environment

## 📞 Support

For questions or issues:
1. Check the test script output: `python3 test_log_processor.py`
2. Review component documentation in individual modules
3. Validate your JSON log format matches the expected structure

---

**🎯 Current Status**: Component 2 (Local Embeddings) complete and tested
**🔄 Next Step**: Build Component 3 (Vector Store & RAG Engine)
