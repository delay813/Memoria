@echo off
title Memoria
cd /d "%~dp0agent"

echo ============================================
echo   ÐÄÒä ¡¤ Memoria
echo   URL: http://127.0.0.1:8080/
echo   Press Ctrl+C or close this window to stop
echo ============================================
echo.

rem Prefer the project virtualenv; fall back to system python
set "PYTHON=..\.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

"%PYTHON%" -X utf8 -m uvicorn server:app --host 127.0.0.1 --port 8080

pause
