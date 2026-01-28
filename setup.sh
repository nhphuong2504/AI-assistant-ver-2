#!/bin/bash
echo "Setting up Backend..."

echo "Removing old virtual environment if it exists..."
if [ -d "venv" ]; then
    echo "Deactivating and removing old venv..."
    deactivate 2>/dev/null || true
    rm -rf venv
fi

echo "Creating new virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing dependencies..."
pip install -r backend/requirements.txt

echo ""
echo "Backend setup complete!"
echo ""
echo "To start the server, run:"
echo "  source venv/bin/activate"
echo "  cd backend"
echo "  uvicorn app.main:app --reload --host 127.0.0.1 --port 8000"
echo ""
echo "Or use: ./start-backend.sh"
echo ""
