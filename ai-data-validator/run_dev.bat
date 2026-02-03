@echo off
REM AI Data Adequacy Agent - Development Startup Script (Windows)
REM One command to start FastAPI backend and Streamlit frontend using a local venv

setlocal ENABLEDELAYEDEXPANSION

echo ========================================
echo AI Data Adequacy Agent - Development Mode
echo ========================================
echo.

REM Resolve repo paths
set SCRIPT_DIR=%~dp0
set ROOT=%SCRIPT_DIR%
set BACKEND=%ROOT%backend
set FRONTEND=%ROOT%frontend
set VENV_DIR=%ROOT%.venv
set ACTIVATE=%VENV_DIR%\Scripts\activate.bat

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11+ and add it to your PATH
    pause
    exit /b 1
)

REM Create venv if missing
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating virtual environment at %VENV_DIR% ...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Check if .env file exists
if not exist "%BACKEND%\.env" (
    echo WARNING: .env file not found in backend directory
    echo Please copy .env.template to .env and configure your API keys
    echo.
    echo Example:
    echo   copy backend\.env.template backend\.env
    echo   notepad backend\.env
    echo.
    pause
    exit /b 1
)

REM Install/verify dependencies inside venv
echo Checking Python dependencies...
call "%ACTIVATE%" >nul 2>&1
python -c "import fastapi, openai, pkgutil; import importlib; \
assert importlib.util.find_spec('pinecone') or importlib.util.find_spec('pinecone-client'), 'pinecone sdk missing'" 2>nul
if errorlevel 1 (
    echo.
    echo Installing required Python packages...
    echo This may take a few minutes...
    pip install -r "%BACKEND%\requirements.txt"
    pip install streamlit
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

echo.
echo Starting AI Data Adequacy Agent...
echo.
echo Backend API:   http://localhost:8000
echo API Docs:      http://localhost:8000/docs
echo Frontend UI:   http://localhost:8501
echo.

REM Start backend server in background (with venv)
echo [1/2] Starting FastAPI backend server...
start "AI Data Agent - Backend" cmd /k "call \"%ACTIVATE%\" && cd /d \"%BACKEND%\" && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

REM Wait a moment for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend in background (with venv)
echo [2/2] Starting Streamlit frontend...
start "AI Data Agent - Frontend" cmd /k "call \"%ACTIVATE%\" && cd /d \"%FRONTEND%\" && streamlit run streamlit_app.py --server.port 8501 --server.address localhost"

REM Open browser to frontend
start "" http://localhost:8501/

echo.
echo Launched. Use the opened terminals to stop processes (Ctrl+C) when done.
echo.
endlocal
