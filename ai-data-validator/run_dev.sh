#!/bin/bash
# AI Data Adequacy Agent - Development Startup Script (Linux/Mac)
# This script starts both the FastAPI backend and Streamlit frontend

echo "========================================"
echo "AI Data Adequacy Agent - Development Mode"
echo "========================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.11+ and ensure it's available as 'python3'"
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "ERROR: pip3 is not installed or not in PATH"
    echo "Please install pip3 for Python package management"
    exit 1
fi

# Check if .env file exists
if [ ! -f "backend/.env" ]; then
    echo "WARNING: .env file not found in backend directory"
    echo "Please copy .env.template to .env and configure your API keys"
    echo
    echo "Example:"
    echo "  cd backend"
    echo "  cp .env.template .env"
    echo "  nano .env"
    echo
    read -p "Press Enter to continue anyway or Ctrl+C to exit..."
fi

# Check if requirements are installed
echo "Checking Python dependencies..."
python3 -c "import fastapi, streamlit, openai, pinecone" 2>/dev/null
if [ $? -ne 0 ]; then
    echo
    echo "Installing required Python packages..."
    echo "This may take a few minutes..."
    pip3 install -r backend/requirements.txt
    pip3 install streamlit
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies"
        echo "You may need to run: sudo pip3 install -r backend/requirements.txt"
        exit 1
    fi
fi

echo
echo "Starting AI Data Adequacy Agent..."
echo
echo "Backend API will be available at: http://localhost:8000"
echo "Frontend UI will be available at: http://localhost:8501"
echo "API Documentation will be available at: http://localhost:8000/docs"
echo

# Function to cleanup background processes
cleanup() {
    echo
    echo "Shutting down services..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start backend server in background
echo "[1/2] Starting FastAPI backend server..."
cd backend
python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Check if backend started successfully
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "ERROR: Backend server failed to start"
    exit 1
fi

# Start frontend
echo "[2/2] Starting Streamlit frontend..."
cd ../frontend
echo
echo "================================================="
echo "Frontend starting... Browser should open automatically"
echo "================================================="

# Start Streamlit (this will block)
streamlit run streamlit_app.py --server.port 8501 --server.address localhost

# If we get here, Streamlit has exited
cleanup
