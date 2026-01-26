@echo off
echo Starting Backend Server...

if not exist venv (
    echo Virtual environment not found. Creating new one...
    python -m venv venv
    call venv\Scripts\activate
    echo Upgrading pip...
    python -m pip install --upgrade pip
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate
)

cd backend
echo Starting FastAPI server on http://127.0.0.1:8000
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
