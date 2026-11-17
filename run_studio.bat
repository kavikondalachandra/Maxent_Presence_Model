@echo off
setlocal
title MaxEnt Studio Pro - Species Distribution Modeling Suite

echo ===============================================================================
echo                MaxEnt Studio Pro - Next-Generation SDM Suite
echo        Concept & Design by K S Chandra 2026 (Source: MaxEnt Java App)
echo ===============================================================================
echo.
echo Launching Interactive Studio in your default web browser...
echo.

cd /d "%~dp0"
python -m streamlit run app.py --server.maxUploadSize 2048 --browser.serverAddress localhost --server.headless false

if %ERRORLEVEL% neq 0 (
    echo.
    echo [Fallback] Streamlit CLI returned non-zero code. Relaunching via python script...
    python app.py
    pause
)
