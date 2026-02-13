#!/bin/bash

# Startup script for Experimentation Suite
# Runs both Streamlit UI and FastAPI server

echo "🔬 Starting Experimentation & Causal Analysis Suite..."
echo ""

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo "Running in Docker container"
    PYTHON_CMD="python"
else
    echo "Running locally"
    PYTHON_CMD="python3"
fi

# Check dependencies
echo "Checking dependencies..."
$PYTHON_CMD -c "import streamlit, fastapi, scipy, pandas" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Installing..."
    pip install -r requirements.txt
fi

echo "✅ Dependencies OK"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down services..."
    kill $API_PID $UI_PID 2>/dev/null
    exit 0
}

trap cleanup INT TERM

# Start API server in background
echo "🚀 Starting API server on http://localhost:8000"
$PYTHON_CMD api_server.py &
API_PID=$!

# Wait for API to start
sleep 2

# Start Streamlit UI in background
echo "🎨 Starting Streamlit UI on http://localhost:8501"
streamlit run app.py --server.port=8501 --server.address=0.0.0.0 &
UI_PID=$!

echo ""
echo "✅ Services started successfully!"
echo ""
echo "  📊 Streamlit UI:  http://localhost:8501"
echo "  🔌 API Server:    http://localhost:8000"
echo "  📚 API Docs:      http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for both processes
wait $API_PID $UI_PID
