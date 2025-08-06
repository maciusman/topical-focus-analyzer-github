@echo off
REM ============================================================================
REM Topical Focus Analyzer - Windows Startup Script
REM ============================================================================
REM This script automatically starts the Topical Focus Analyzer application
REM Double-click this file to launch the application in your web browser
REM ============================================================================

title Topical Focus Analyzer - Starting...

echo.
echo ============================================================================
echo   🔍 TOPICAL FOCUS ANALYZER - STARTUP SCRIPT
echo ============================================================================
echo.
echo   Starting the application...
echo   Please wait while the server initializes...
echo.

REM Change to the application directory
cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: Python is not installed or not in PATH
    echo.
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

REM Check if required dependencies are installed
set NEED_INSTALL=0
python -c "import streamlit" >nul 2>&1
if errorlevel 1 set NEED_INSTALL=1

python -c "import selenium" >nul 2>&1
if errorlevel 1 set NEED_INSTALL=1

python -c "import webdriver_manager" >nul 2>&1
if errorlevel 1 set NEED_INSTALL=1

if %NEED_INSTALL%==1 (
    echo ⚠️  WARNING: Missing dependencies detected. Installing all requirements...
    echo.
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ ERROR: Failed to install dependencies
        echo.
        echo Please run the following command manually:
        echo   pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
)

REM Check if Google Chrome is installed (for Advanced Extraction Mode)
set CHROME_FOUND=0
if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" set CHROME_FOUND=1
if exist "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" set CHROME_FOUND=1
if %CHROME_FOUND%==0 (
    echo ⚠️  WARNING: Google Chrome not detected
    echo   • Standard content extraction will work normally
    echo   • Advanced Extraction Mode requires Google Chrome
    echo   • Download Chrome from: https://www.google.com/chrome/
    echo.
)

REM Display startup information
echo ✅ Python environment ready
echo ✅ Starting Streamlit server...
echo.
echo 📋 APPLICATION INFO:
echo   • Application will open in your default web browser
echo   • Server will run on: http://localhost:8501
echo   • To stop the server: Close this window or press Ctrl+C
echo.
echo 🔑 REQUIRED API KEYS:
echo   • Jina API Key (for embeddings): https://jina.ai/
echo   • OpenRouter API Key (for AI summaries): https://openrouter.ai/
echo   • Keys can be entered in the app or saved in .env file
echo.
echo 🚀 EXTRACTION MODES:
echo   • Standard Mode: Fast, works with all websites
echo   • Advanced Mode: Slower, handles JavaScript-heavy sites
echo   • Advanced Mode requires Google Chrome browser
echo.
echo ============================================================================
echo   Application is starting... Please wait for browser to open
echo ============================================================================
echo.

REM Start the Streamlit application
REM The --server.headless false ensures the browser opens automatically
python -m streamlit run multi_sitemap_app.py --server.headless false --server.port 8501

REM If we reach here, the application has stopped
echo.
echo ============================================================================
echo   Application has stopped
echo ============================================================================
echo.
pause