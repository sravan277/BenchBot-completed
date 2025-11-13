@echo off
echo Starting RAG Benchmark Backend...
cd backend
python -m uvicorn app.main:app --reload --port 8000

