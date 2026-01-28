#!/bin/bash
echo "Starting Backend Server..."

if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating new one..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Upgrading pip..."
    python -m pip install --upgrade pip
    echo "Installing dependencies..."
    pip install -r backend/requirements.txt
else
    source venv/bin/activate
fi

cd backend
echo "Starting FastAPI server on http://127.0.0.1:8000"
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
