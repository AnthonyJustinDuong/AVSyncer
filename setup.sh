#!/bin/bash
set -e

echo "=== AV Syncer Setup ==="

# System deps
echo "[1/3] Installing system dependencies (ffmpeg)..."
apt-get install -y ffmpeg 2>/dev/null || echo "Note: ffmpeg install requires sudo/root. Run: sudo apt-get install -y ffmpeg"

# Backend
echo "[2/3] Setting up Python backend..."
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p uploads
cd ..

# Frontend
echo "[3/3] Installing frontend dependencies..."
cd frontend
npm install
cd ..

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To start the app, open two terminals:"
echo ""
echo "  Terminal 1 (backend):"
echo "    cd backend && source .venv/bin/activate && python -m uvicorn main:app --reload --port 8143"
echo ""
echo "  Terminal 2 (frontend):"
echo "    cd frontend && npm run dev"
echo ""
echo "Then open http://localhost:5173"
