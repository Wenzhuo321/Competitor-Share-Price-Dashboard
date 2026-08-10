#!/bin/bash
# Share Price Performance Dashboard — LAN startup
# Colleagues access: http://<your-IP>:5173

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Kill any existing processes on our ports ──────────────────────────────────
echo "Freeing ports 8000 and 5173..."
lsof -ti:8000 | xargs kill -9 2>/dev/null; true
lsof -ti:5173 | xargs kill -9 2>/dev/null; true
sleep 1

# ── Install dependencies ──────────────────────────────────────────────────────
echo "Installing/checking Python dependencies..."
pip3 install yfinance matplotlib plotly fastapi uvicorn --quiet --break-system-packages 2>/dev/null || \
pip3 install yfinance matplotlib plotly fastapi uvicorn --quiet

echo "Installing/checking Node dependencies..."
cd "$SCRIPT_DIR/frontend" && npm install --silent

# ── Start servers ─────────────────────────────────────────────────────────────
LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}' || echo "<your-IP>")

echo ""
echo "Starting FastAPI backend on port 8000..."
cd "$SCRIPT_DIR" && uvicorn app:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

echo "Starting Vite frontend on port 5173..."
cd "$SCRIPT_DIR/frontend" && npm run dev -- --port 5173 --strictPort &
FRONTEND_PID=$!

echo ""
echo "Dashboard ready at:"
echo "  Local:    http://localhost:5173"
echo "  Network:  http://${LAN_IP}:5173"
echo ""
echo "Press Ctrl+C to stop."

# ── Stop both on exit ─────────────────────────────────────────────────────────
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
