@echo off
setlocal enabledelayedexpansion

:: Add user script directories to PATH dynamically
set "PATH=%APPDATA%\Python\Python314\Scripts;%LOCALAPPDATA%\Programs\Python\Python314\Scripts;%PATH%"

echo [1/4] Ensuring Directories...
if not exist "output\scenes" mkdir "output\scenes"
if not exist "output\manifests" mkdir "output\manifests"

echo [2/4] Installing Required Dependencies...
python -m pip install -r requirements.txt --quiet
python -m pip install -e ./gflow-cli --quiet
python -m playwright install chromium

echo [3/4] Starting Engine Daemon on Port 8080...
start "Studio-Bridge-Server" /b python server.py

echo Waiting for Daemon startup...
timeout /t 4 /nobreak >nul

echo [4/4] Triggering Dynamic Rhyme Pipeline...
python dynamic_rhyme_pipeline.py --preset hindi

pause
