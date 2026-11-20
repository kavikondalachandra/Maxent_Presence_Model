@echo off
setlocal
title MaxEnt Studio Pro - Species Distribution & Archaeological Modeling Suite

echo ===============================================================================
echo                MaxEnt Studio Pro - Next-Generation SDM Suite
echo        Concept & Design by K S Chandra 2026 (Source: MaxEnt Java App)
echo ===============================================================================
echo.
echo Initializing hardware-optimized configuration...

cd /d "%~dp0"

REM If app_config.json does not exist, generate it
if not exist "app_config.json" (
    echo Probing computer hardware specifications...
    python system_config.py
)

echo.
echo Launching Interactive Studio in your default web browser (http://localhost:8501)...
echo.

python -m streamlit run app.py --server.maxUploadSize 2048 --server.port 8501 --browser.serverAddress localhost --server.headless false

if %ERRORLEVEL% neq 0 (
    echo.
    echo [Fallback] Streamlit CLI returned non-zero code. Relaunching via python bootstrap...
    python app.py
    pause
)
