@echo off
rem Makes the API start by itself every time you log in to Windows (30 seconds after logon).
rem Run once. It starts start-api.bat in your own logged-in session, which is what Outlook needs.
rem To undo it, run remove-autostart.bat.

setlocal
cd /d "%~dp0"

if not exist "start-api.bat" (
    echo start-api.bat is not next to this file.
    pause
    exit /b 1
)

schtasks /create /tn "KVACK API" /tr "\"%~dp0start-api.bat\"" /sc onlogon /delay 0000:30 /f
if errorlevel 1 (
    echo.
    echo Could not create the task. Try right-click, Run as administrator.
    pause
    exit /b 1
)

echo.
echo Done. The API will start 30 seconds after you log in. To start it now: double-click start-api.bat
echo (or run: schtasks /run /tn "KVACK API").
echo.
echo Reminder: allow inbound port 8000 from the Linux machine only (Windows Defender Firewall),
echo if you want Prometheus and Grafana to see this API.
pause
