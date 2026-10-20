@echo off
setlocal
set "TWIN_PYTHON=%~dp0..\.venv\Scripts\python.exe"
if not exist "%TWIN_PYTHON%" set "TWIN_PYTHON=python"
"%TWIN_PYTHON%" -m streamlit run "%~dp0dashboard.py" --server.address 127.0.0.1 --browser.gatherUsageStats false
pause
