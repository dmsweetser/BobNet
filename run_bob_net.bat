@echo off
setlocal enabledelayedexpansion

REM Set the path to the virtual environment activation script
set "activate_script=venv\Scripts\activate"

REM Display diagnostic information
echo --- Diagnostic Information ---
echo Virtual Environment Activation Script: %activate_script%
echo.

REM Check if the virtual environment activation script exists
if not exist "!activate_script!" (
    echo Error: Virtual environment activation script not found. Please check your virtual environment path.
    exit /b 1
)

REM Activate the virtual environment
call "!activate_script!"

REM Check if the virtual environment is activated
if not defined VIRTUAL_ENV (
    echo Error: Virtual environment is not activated. Please activate it before running this script.
    exit /b 1
)

echo Virtual environment activated successfully.
echo Running your script

REM Run your Python script within the virtual environment
python bob_net.py %1%

REM Check the exit code of the script
if %errorlevel% neq 0 (
    echo Error: The Python script encountered an error.
    exit /b 1
)

echo Script executed successfully.

REM Deactivate the virtual environment
call deactivate
if %errorlevel% neq 0 (
    echo Error: Unable to deactivate virtual environment.
    exit /b 1
)

echo Virtual environment deactivated successfully.

echo.
echo Script execution complete.
