#!/bin/bash

# Start FastAPI backend in the background
echo "Starting FastAPI Backend on port 8000..."
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 &

# Wait for backend to initialize
sleep 3

# Start Streamlit frontend in the foreground
echo "Starting Streamlit Frontend on port 8501..."
python -m streamlit run frontend/app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --server.fileWatcherType poll

