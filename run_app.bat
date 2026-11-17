@echo off
title MaxEnt Presence Prediction Studio
echo ======================================================================
echo          Starting MaxEnt Presence Prediction Studio...
echo ======================================================================
cd /d "%~dp0"
python -m streamlit run app.py --server.maxUploadSize 2048
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [Fallback] Trying direct python launch...
    python app.py
)
pause
