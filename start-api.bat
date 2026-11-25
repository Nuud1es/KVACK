@echo off
rem Starts the KVACK API and keeps it running. Double-click it, or let
rem install-autostart.bat run it for you each time you log in.
rem
rem Leave the window open: closing it stops the API. Output goes to logs\api.log.

setlocal
title KVACK API
cd /d "%~dp0"

if not exist ".env" (
    echo No .env file in %CD%. Copy .env.example to .env and fill it in first.
    goto :stop
)
if not exist ".venv\Scripts\python.exe" (
    echo No virtual environment in %CD%\.venv. Set it up first, see README "Production".
    goto :stop
)
if not exist "web\dist\index.html" (
    echo Note: web\dist is missing, so the web page will not load. Build it once with:
    echo     cd web ^&^& npm ci ^&^& npm run build
    echo The API itself will still start.
    echo.
)

rem The venv's own Python on purpose: another Python on this PC lacks pydantic.
set "PY=%CD%\.venv\Scripts\python.exe"
if not exist "logs" mkdir "logs"
set /a restarts=0

:run
echo [%date% %time%] Starting the API on port 8000 ...
echo [%date% %time%] Starting >> "logs\api.log"
rem 0.0.0.0 lets the Linux machine (Prometheus) reach /metrics; keep port 8000 firewalled to it.
"%PY%" -m uvicorn qa_api.main:app --app-dir services\api\src --host 0.0.0.0 --port 8000 >> "logs\api.log" 2>&1
echo [%date% %time%] The API stopped (exit code %errorlevel%).
echo Last lines of logs\api.log:
powershell -NoProfile -Command "Get-Content -Path 'logs\api.log' -Tail 12"
set /a restarts+=1
if %restarts% GEQ 10 (
    echo.
    echo It has stopped 10 times, so something is wrong. Read logs\api.log, fix it, then run this again.
    goto :stop
)
echo.
echo Restarting in 10 seconds. Close this window to stop it.
timeout /t 10 /nobreak >nul
goto :run

:stop
echo.
pause
exit /b 1
