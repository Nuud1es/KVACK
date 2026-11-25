@echo off
rem Stops the API from starting at logon (it does not stop a copy that is already running).
schtasks /delete /tn "KVACK API" /f
pause
