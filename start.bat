@echo off
REM 1. Run Carla
echo Launching the Carla program...
start "" "C:\path\to\your\program.exe"

REM Optional delay to ensure the EXE starts first
timeout /t 10 /nobreak >nul

REM 2. Run multiple Python scripts in new terminal windows
echo Launching Python script 1...
start cmd /k python "C:\path\to\your\script1.py"

echo Launching Python script 2...
start cmd /k python "C:\path\to\your\script2.py"

echo Launching Python script 3...
start cmd /k python "C:\path\to\your\script3.py"

REM 3. Open the Z3 monitor website
echo Opening the website...
start "" "https://localhost:9000"

echo All tasks have been launched.
pause
