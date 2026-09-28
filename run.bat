@echo off
echo [1/3] Installing Dependencies...
pip install -r requirements.txt
pip install -e ./gflow-cli
echo [2/3] Starting Local Studio Engine Bridge...
start /b python server.py
timeout /t 3 /nobreak >nul
echo [3/3] Launching Dynamic Rhyme Pipeline...
python dynamic_rhyme_pipeline.py --preset hindi
pause
