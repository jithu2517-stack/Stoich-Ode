@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    py -3 -m venv .venv
    if errorlevel 1 goto :error
)

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

if not exist "models\stoich_ode_models.joblib" (
    ".venv\Scripts\python.exe" train_stoich_ode.py
    if errorlevel 1 goto :error
)

echo.
echo Opening the local STOICH-ODE console at http://127.0.0.1:8001
echo Press Ctrl+C to stop the FastAPI server.
".venv\Scripts\python.exe" -m uvicorn api:app --host 127.0.0.1 --port 8001
goto :eof

:error
echo.
echo Setup did not complete. Check the Python installation and the error above.
pause
exit /b 1
