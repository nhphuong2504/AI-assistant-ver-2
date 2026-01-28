@echo off
echo Setting up Backend...

echo Removing old virtual environment if it exists...
if exist venv (
    echo Deactivating and removing old venv...
    call venv\Scripts\deactivate.bat 2>nul
    rmdir /s /q venv
)

echo Creating new virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing dependencies...
pip install -r backend/requirements.txt

echo.
echo Backend setup complete!
echo.
echo To start the server, run:
echo   venv\Scripts\activate
echo   cd backend
echo   uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
echo.
echo Or use: start-backend.bat
echo.
pause
