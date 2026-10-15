#!/usr/bin/env bash
# start.sh — Launch the MedGemma Report API
# ==========================================
# Usage:
#   chmod +x start.sh
#   ./start.sh                          # Ollama backend (default)
#   MEDGEMMA_BACKEND=mock ./start.sh    # Mock backend for UI testing
#   MEDGEMMA_BACKEND=transformers ./start.sh   # HuggingFace backend (needs GPU)

set -e

# ── Defaults ──────────────────────────────────────────────────────────────────
export MEDGEMMA_BACKEND="${MEDGEMMA_BACKEND:-ollama}"
export MEDGEMMA_MODEL="${MEDGEMMA_MODEL:-medgemma3:4b}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
export PORT="${PORT:-8001}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  NeuroDash MedGemma Report API"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Backend : $MEDGEMMA_BACKEND"
echo "  Model   : $MEDGEMMA_MODEL"
if [ "$MEDGEMMA_BACKEND" = "ollama" ]; then
    echo "  Ollama  : $OLLAMA_BASE_URL"
    echo ""
    echo "  ⚠  Make sure Ollama is running and the model is pulled:"
    echo "     ollama serve"
    echo "     ollama pull $MEDGEMMA_MODEL"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Install dependencies if needed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing Python dependencies…"
    pip install -r requirements.txt
fi

# Launch
uvicorn main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --reload \
    --log-level info
