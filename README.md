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

# Install Node.js dependencies for React UI
cd web-ui
npm install
cd ..
```

### 2. Start the Complete System
```bash
# Start both API server and React UI
./start_dev.sh
```

### 3. Access the Chatbot
- **React Chatbot UI**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

### 4. Try Sample Queries
```
"Why did ATM 1123 fail today at 10 AM?"
"What does DDL_EXCEEDED error mean?"
"Show me network timeout issues"
"Analyze withdrawal patterns"
```

## 📁 Project Structure
```
atm-rag/
├── src/
│   ├── log_processor/          # ✅ Component 1: JSON Log Processing
│   ├── embeddings/             # ✅ Component 2: Embeddings Generation
│   ├── vector_store/           # ✅ Component 3: Vector Storage (MongoDB + FAISS)
│   ├── rag_engine/             # ✅ Component 4: RAG Pipeline & Response Generation
│   ├── query_processor/        # ✅ Query Processing & Intent Classification
│   └── api/                    # ✅ FastAPI REST Endpoints
├── web-ui/                     # ✅ React Chatbot Interface
│   ├── src/components/         # React components (ChatInterface, etc.)
│   ├── src/services/           # API integration layer
│   ├── package.json            # Node.js dependencies
│   └── start_dev.sh            # Development startup script
├── data/
│   └── logs/                   # Sample ATM log files
├── simple_demo.py              # Working demo API (fallback)
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

## 🗺️ Implementation Status

### ✅ Phase 1: Log Processing (Complete)
- JSON log reading and validation
- Structured parsing with explicit fields
- Text extraction for embeddings
- Comprehensive testing

### ✅ Phase 2: Local Embeddings (Complete)
- sentence-transformers integration
- Vector generation for log entries
- MongoDB vector storage with FAISS fallback
- Similarity search capabilities

### ✅ Phase 3: RAG Engine (Complete)
- Context retrieval system
- Query processing pipeline
- Response generation with intelligent templates
- Knowledge base integration

### ✅ Phase 4: API Interface (Complete)
- FastAPI REST endpoints
- **React Chatbot UI**: Modern web interface at http://localhost:3000
- Real-time log processing
- Complete demo system with intelligent responses

## 🤝 Contributing

This is a component-based development approach. All components are now complete and integrated:

1. **Component 1**: Log Processing ✅
2. **Component 2**: Local Embeddings ✅
3. **Component 3**: Vector Search & RAG ✅
4. **Component 4**: API Interface + React UI ✅

## 🌐 Live Demo Features

### Intelligent Responses
The chatbot provides contextual, intelligent responses for:
- **ATM Error Analysis**: Detailed explanations of error codes like DDL_EXCEEDED
- **Troubleshooting Guidance**: Step-by-step resolution instructions
- **Pattern Recognition**: Identification of recurring issues and trends
- **Operational Insights**: Performance analysis and recommendations

### Real-time Processing
- **Query Processing**: Natural language understanding with intent classification
- **Context Retrieval**: Semantic search through ATM log data
- **Response Generation**: AI-powered responses using retrieved context
- **System Monitoring**: Live health checks of all components

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

**🎯 Current Status**: ✅ ALL COMPONENTS COMPLETE! Full RAG system with React UI working
**🚀 Ready to Use**: Run `./start_dev.sh` then visit http://localhost:3000 for the chatbot interface
