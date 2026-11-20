@echo off
setlocal enabledelayedexpansion
title MaxEnt Studio Pro - Environment Setup & Diagnostic Check

echo ===============================================================================
echo        MaxEnt Studio Pro: Automated Setup for Computers and Laptops
echo           Concept & Design by K S Chandra 2026 (Source: MaxEnt Java App)
echo ===============================================================================
echo.

cd /d "%~dp0"

echo [Step 1/4] Checking Python installation...
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not found in your system PATH!
    echo Please install Python 3.9+ from https://www.python.org/downloads/
    echo Ensure you check the box: "Add Python to PATH" during installation.
    pause
    exit /b 1
)
python --version

echo.
echo [Step 2/4] Installing / Updating Python dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo [Step 3/4] Checking Java Runtime Environment (for Official MaxEnt Engine)...
java -version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [NOTE] Java was not found in PATH.
    echo The Native Python MaxEnt Engine will run 100%% offline without Java.
    echo If you need the Official AMNH Java binary, install 64-bit Java (JRE/JDK 8+).
) else (
    echo [OK] 64-bit Java Runtime is available.
)

echo.
echo [Step 4/4] Probing hardware specifications and generating optimal configuration...
python system_config.py

echo.
echo ===============================================================================
echo [SUCCESS] Environment setup is complete!
echo You can now launch the studio anytime by running: run_studio.bat
echo ===============================================================================
echo.
pause
