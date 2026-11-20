#!/usr/bin/env bash
# ===============================================================================
# MaxEnt Studio Pro: 1-Click Launcher for macOS (MacBook Air/Pro) and Linux Laptops
# ===============================================================================

set -e

cd "$(dirname "$0")"

echo "==============================================================================="
echo "                MaxEnt Studio Pro - Next-Generation SDM Suite"
echo "        Concept & Design by K S Chandra 2026 (Source: MaxEnt Java App)"
echo "==============================================================================="
echo ""
echo "Launching Interactive Studio in your default web browser (http://localhost:8501)..."
echo ""

# Find Python
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

# Run Streamlit with 2 GB upload allocation
$PYTHON_CMD -m streamlit run app.py \
    --server.maxUploadSize 2048 \
    --server.port 8501 \
    --browser.serverAddress localhost \
    --server.headless false
