@echo off
REM 1. Run Carla
echo Launching the Carla program...
start "" ".\AAAMain.py"

REM Optional delay to ensure the EXE starts first
timeout /t 10 /nobreak >nul

REM 2. Run multiple Python scripts in new terminal windows

echo Launching z3_solver...
start cmd /k python ".\z3_solver.py"

echo Launching app.py...
start cmd /k python ".\app.py"

echo Launching VLM...
start cmd /k python ".\VLM.py"

REM 3. Open the Z3 monitor website
echo Opening the website...
start "" "https://localhost:9000"

echo All tasks have been launched.
pause
