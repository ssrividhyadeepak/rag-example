#!/bin/bash

# ATM RAG Development Startup Script

echo "🚀 Starting ATM RAG Development Environment"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: Please run this script from the web-ui directory"
    exit 1
fi

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing React dependencies..."
    npm install
fi

echo ""
echo "🔧 Starting services..."
echo ""

# Function to start API server
start_api() {
    echo "🖥️  Starting FastAPI server on http://localhost:8000"
    cd ../
    python3 -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000 &
    API_PID=$!
    echo "   API Server PID: $API_PID"
    cd web-ui/
}

# Function to start React app
start_react() {
    echo "⚛️  Starting React app on http://localhost:3000"
    npm start &
    REACT_PID=$!
    echo "   React App PID: $REACT_PID"
}

# Start API server
start_api

# Wait a moment for API to start
sleep 3

# Start React app
start_react

# Show status
echo ""
echo "✅ Services started successfully!"
echo ""
echo "🌐 Open your browser and go to:"
echo "   React App:    http://localhost:3000"
echo "   API Docs:     http://localhost:8000/docs"
echo "   API Health:   http://localhost:8000/health"
echo ""
echo "💡 Try asking the chatbot:"
echo "   'Why did ATM 1123 fail today at 10 AM?'"
echo "   'Show me DDL_EXCEEDED errors from yesterday'"
echo "   'What does NETWORK_ERROR mean?'"
echo ""
echo "🛑 Press Ctrl+C to stop all services"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    if [ ! -z "$API_PID" ]; then
        kill $API_PID 2>/dev/null
        echo "   Stopped API server"
    fi
    if [ ! -z "$REACT_PID" ]; then
        kill $REACT_PID 2>/dev/null
        echo "   Stopped React app"
    fi
    echo "✅ All services stopped"
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Wait for services to keep running
wait