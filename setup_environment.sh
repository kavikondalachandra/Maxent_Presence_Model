#!/usr/bin/env bash
# ===============================================================================
# MaxEnt Studio Pro: Environment Setup Script for macOS and Linux Computers/Laptops
# ===============================================================================

set -e

echo "==============================================================================="
echo "       MaxEnt Studio Pro: Automated Setup for macOS and Linux Computers"
echo "          Concept & Design by K S Chandra 2026 (Source: MaxEnt Java App)"
echo "==============================================================================="
echo ""

# Change to script directory
cd "$(dirname "$0")"

# 1. Check Python
echo "[Step 1/4] Checking Python installation..."
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "[ERROR] Python 3 was not found on your system."
    echo "Please install Python 3.9+ via Homebrew ('brew install python') or your package manager ('apt install python3 python3-pip')."
    exit 1
fi
$PYTHON_CMD --version

# 2. Install dependencies
echo ""
echo "[Step 2/4] Installing Python requirements..."
$PYTHON_CMD -m pip install --upgrade pip
$PYTHON_CMD -m pip install -r requirements.txt

# 3. Check Java
echo ""
echo "[Step 3/4] Checking Java Runtime Environment..."
if command -v java &>/dev/null; then
    echo "[OK] Java runtime detected: $(java -version 2>&1 | head -n 1)"
else
    echo "[NOTE] Java is not installed in system PATH."
    echo "The Native Python MaxEnt Engine will function 100% offline without Java."
fi

# 4. Probe hardware and generate app_config.json
echo ""
echo "[Step 4/4] Detecting laptop/computer hardware profile..."
$PYTHON_CMD system_config.py

echo ""
echo "==============================================================================="
echo "[SUCCESS] Environment is fully configured!"
echo "Launch MaxEnt Studio Pro anytime by running: ./run_studio.sh"
echo "==============================================================================="
