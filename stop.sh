#!/bin/bash

echo "Stopping ATM RAG application components..."

# Stop React development server
echo "Stopping React web-ui..."
pkill -f "react-scripts" 2>/dev/null

# Stop Python backend/demo processes
echo "Stopping Python processes..."
pkill -f "simple_demo.py" 2>/dev/null
pkill -f "demo_test.py" 2>/dev/null
pkill -f "interactive_test.py" 2>/dev/null

# Stop MongoDB if running locally
echo "Stopping MongoDB..."
pkill -f "mongod" 2>/dev/null

# Stop any other Node.js processes related to this project
echo "Stopping other Node.js processes..."
pkill -f "node.*atm-rag" 2>/dev/null

echo "All ATM RAG components stopped!"
echo ""
echo "Verify with: ps aux | grep -E '(react-scripts|simple_demo|mongod)' | grep -v grep"