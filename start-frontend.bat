@echo off
echo Starting Frontend Server...
cd frontend
if not exist node_modules (
    echo Installing dependencies...
    call npm install
)
echo Starting Vite dev server on http://localhost:8080
call npm run dev
