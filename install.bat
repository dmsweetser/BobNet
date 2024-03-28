@echo off
setlocal enabledelayedexpansion

echo Setting up Python virtual environment...

REM Check if Python is installed
where py -V:3.9 > nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed. Please install Python before running this script.
    exit /b 1
)

REM Create a virtual environment
py -V:3.9 -m venv venv
if %errorlevel% neq 0 (
    echo Error: Unable to create virtual environment.
    exit /b 1
)

echo Virtual environment created successfully.

REM Activate the virtual environment
call venv\Scripts\python.exe -m pip install --upgrade pip
call venv\Scripts\Activate.bat
if %errorlevel% neq 0 (
    echo Error: Unable to activate virtual environment.
    exit /b 1
)

echo Virtual environment activated successfully.

echo Installing required packages...

REM Install required packages
pip install --upgrade --force-reinstall --no-cache-dir -r requirements.txt
if %errorlevel% neq 0 (
    echo Error: Unable to install required packages.
    exit /b 1
)

echo Required packages installed successfully.

echo.
echo Environment setup complete.
echo To activate the virtual environment, run: venv\Scripts\Activate
