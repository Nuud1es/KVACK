@echo off
rem Apply the KVACK database migrations (alembic upgrade head).
rem Safe to run again: it only applies migrations that are missing.
rem Needs QACOPILOT_DATABASE_URL in qa-copilot\.env (the Postgres on the Linux laptop).

setlocal
cd /d "%~dp0"

if not exist ".env" (
    echo.
    echo No .env file in %CD%
    echo Copy .env.example to .env and set QACOPILOT_DATABASE_URL, then run this again.
    goto :fail
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating the virtual environment...
    python -m venv .venv
    if errorlevel 1 goto :fail
)

rem The venv's own Python is used on purpose: another Python on this PC lacks pydantic, and a
rem bare "alembic" is only found when the venv is activated.
set "PY=%CD%\.venv\Scripts\python.exe"

echo Installing the database packages (alembic, SQLAlchemy, Postgres driver)...
"%PY%" -m pip install --quiet -e "packages/qacopilot[db]"
if errorlevel 1 goto :fail

echo.
echo Applying migrations...
"%PY%" -m alembic -c packages/qacopilot/alembic.ini upgrade head
if errorlevel 1 goto :fail

echo.
echo Database is now at:
"%PY%" -m alembic -c packages/qacopilot/alembic.ini current
if errorlevel 1 goto :fail

echo.
echo Done.
pause
exit /b 0

:fail
echo.
echo Migration did not finish. Common causes:
echo   - QACOPILOT_DATABASE_URL is missing from .env, or still says localhost
echo     (it must be the laptop's address, with the password from infra\.env on the laptop)
echo   - the laptop is off, or Postgres is not reachable from this PC
echo   - git pull has not brought in the newest migration files
pause
exit /b 1
