#!/bin/bash
echo "Starting Frontend Server..."
cd frontend
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi
echo "Starting Vite dev server on http://localhost:8080"
npm run dev
