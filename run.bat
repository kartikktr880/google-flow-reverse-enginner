@echo off
setlocal enabledelayedexpansion

echo ==========================================================
echo [1/4] Cleaning Stale Engine Processes on Port 8080...
echo ==========================================================
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8080 ^| findstr LISTENING') do (
    taskkill /f /pid %%a >nul 2>&1
)

if not exist "output\scenes" mkdir "output\scenes"
if not exist "output\manifests" mkdir "output\manifests"

echo ==========================================================
echo [2/4] Verifying Environment...
echo ==========================================================
python -m pip install -r requirements.txt --quiet
python -m pip install -e ./gflow-cli --quiet

echo ==========================================================
echo [3/4] Launching Studio Engine Daemon...
echo ==========================================================
start "Studio-Bridge-Server" /b python server.py

:WAIT_HEALTH
timeout /t 2 /nobreak >nul
powershell -Command "try { (Invoke-WebRequest -Uri 'http://127.0.0.1:8080/health' -UseBasicParsing).StatusCode } catch { exit 1 }" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Waiting for bridge to respond...
    goto WAIT_HEALTH
)
echo Engine bridge is fully online and ready!

echo ==========================================================
echo [4/4] Starting Nursery Rhyme Dynamic Orchestrator...
echo ==========================================================
python dynamic_rhyme_pipeline.py --preset hindi

pause
