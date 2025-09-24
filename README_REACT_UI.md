# ATM RAG React UI

A modern React-based chatbot interface for the ATM RAG system, integrating all 4 components into a conversational support assistant.

## 🎯 Features

### 💬 Chatbot Interface
- **Natural Language Queries**: Ask questions in plain English like "Why did ATM 1123 fail today at 10 AM?"
- **Real-time Responses**: Get intelligent answers powered by RAG (Retrieval Augmented Generation)
- **Message History**: Full conversation history with timestamps
- **Typing Indicators**: Visual feedback while processing queries

### 🔍 Query Types Supported
- **Troubleshooting**: "Why is ATM001 showing DDL_EXCEEDED errors?"
- **Error Explanations**: "What does NETWORK_ERROR mean?"
- **Log Analysis**: "Show me all timeout issues from yesterday"
- **Performance Analysis**: "Analyze withdrawal patterns for the last 24 hours"

### 🖥️ Modern UI Features
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Gradient Background**: Beautiful visual design
- **System Status**: Real-time component health monitoring
- **Example Queries**: Quick-start buttons for common questions
- **Input Suggestions**: Helper buttons for common terms

### 🔧 Technical Integration
- **4-Component Integration**:
  - Log Processor ✅
  - Embeddings Generator ✅
  - Vector Store ✅
  - RAG Engine ✅
- **FastAPI Backend**: Full REST API integration
- **Real-time Health Checks**: Monitor system components
- **Error Handling**: Graceful error messages and recovery

## 🚀 Quick Start

### Prerequisites
- Node.js 16+ installed
- Python 3.8+ with ATM RAG backend running
- All backend dependencies installed

### 1. Install Dependencies
```bash
cd web-ui
npm install
```

### 2. Start Development Environment
```bash
# Option 1: Use the startup script (recommended)
./start_dev.sh

# Option 2: Manual startup
# Terminal 1 - Start API server
cd ..
python3 -m uvicorn src.api.main:app --reload --port 8000

# Terminal 2 - Start React app
cd web-ui
npm start
```

### 3. Access the Application
- **React App**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

## 💻 Usage Examples

### Basic Queries
```
"Why did ATM 1123 fail today at 10 AM?"
"Show me DDL_EXCEEDED errors from yesterday"
"What does NETWORK_ERROR mean?"
"Analyze withdrawal patterns for the last 24 hours"
```

### Advanced Queries
```
"Show me all timeout issues from ATM 5567"
"What's causing high failure rates today?"
"Compare ATM001 vs ATM002 performance"
"Find patterns in failed transactions"
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React UI      │────│   FastAPI       │────│   RAG Engine    │
│   (Port 3000)   │    │   (Port 8000)   │    │   Component     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │              ┌─────────────────┐
         │                       │──────────────│  Vector Store   │
         │                       │              │   Component     │
         │                       │              └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐    ┌─────────────────┐
         │──────────────│   Embeddings    │────│  Log Processor  │
                        │   Component     │    │   Component     │
                        └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
web-ui/
├── public/
│   └── index.html              # HTML template
├── src/
│   ├── components/
│   │   ├── ChatInterface.js    # Main chat component
│   │   ├── MessageBubble.js    # Individual message display
│   │   ├── SystemStatus.js     # System health panel
│   │   └── *.css               # Component styles
│   ├── services/
│   │   └── api.js              # API integration layer
│   ├── App.js                  # Root component
│   ├── index.js                # Entry point
│   └── *.css                   # Global styles
├── package.json                # Dependencies
└── start_dev.sh               # Development startup script
```

## 🎨 UI Components

### ChatInterface
- Main conversation area
- Message history management
- Input handling and validation
- System status integration

### MessageBubble
- Individual message rendering
- User vs Assistant styling
- Metadata display (confidence, response time, sources)
- Intent classification badges

### SystemStatus
- Component health monitoring
- System statistics
- Performance metrics
- API endpoint status

## 🔌 API Integration

The React app integrates with all FastAPI endpoints:

- `POST /api/v1/query` - General queries
- `POST /api/v1/search` - Log search
- `POST /api/v1/troubleshoot` - Troubleshooting help
- `POST /api/v1/analyze` - Analysis requests
- `GET /api/v1/health` - Health checks
- `GET /api/v1/stats` - System statistics

## 🎯 Testing the Integration

1. **Start the services**:
   ```bash
   ./start_dev.sh
   ```

2. **Test basic functionality**:
   - Open http://localhost:3000
   - Try the example queries
   - Check system status panel

3. **Test all 4 components**:
   - **Log Processing**: Ask about specific ATM logs
   - **Embeddings**: Search for similar issues
   - **Vector Store**: Query historical data
   - **RAG Engine**: Get intelligent responses

4. **Verify API integration**:
   - Check http://localhost:8000/docs
   - Monitor network requests in browser dev tools
   - Verify responses include all metadata

## 🛠️ Development

### Adding New Features
1. Create components in `src/components/`
2. Add API calls in `src/services/api.js`
3. Update routing and state management
4. Add corresponding backend endpoints

### Styling
- Uses CSS modules for component styling
- Gradient backgrounds and modern design
- Responsive breakpoints for mobile
- Smooth animations and transitions

### State Management
- React hooks for local state
- API service layer for backend calls
- Message history in component state
- Real-time updates and error handling

## 🚨 Troubleshooting

### Common Issues

**React app won't start**:
```bash
rm -rf node_modules package-lock.json
npm install
npm start
```

**API connection failed**:
- Ensure FastAPI server is running on port 8000
- Check proxy setting in package.json
- Verify backend dependencies are installed

**Components showing as unhealthy**:
- Start MongoDB: `brew services start mongodb-community`
- Run data migration: `python3 scripts/migrate_component2_data.py`
- Check backend logs for specific errors

### Getting Help
- Check browser console for errors
- Review API server logs
- Use `/health` endpoint to check component status
- Verify all backend components are running

---

**🎉 Ready to chat with your ATM support assistant!**

Try asking: *"Why did ATM 1123 fail today at 10 AM?"*