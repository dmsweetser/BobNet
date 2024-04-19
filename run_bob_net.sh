#!/bin/bash

# Set the path to the virtual environment activation script
activate_script="venv/Scripts/activate"

# Display diagnostic information
echo "--- Diagnostic Information ---"
echo "Virtual Environment Activation Script: $activate_script"
echo

# Check if the virtual environment activation script exists
if [ ! -f "$activate_script" ]; then
    echo "Error: Virtual environment activation script not found. Please check your virtual environment path."
    exit 1
fi

# Activate the virtual environment
source "$activate_script"

# Check if the virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: Virtual environment is not activated. Please activate it before running this script."
    exit 1
fi

echo "Virtual environment activated successfully."
echo "Running your script"

# Run your Python script within the virtual environment
python bob_net.py "$1"

# Check the exit code of the script
if [ $? -ne 0 ]; then
    echo "Error: The Python script encountered an error."
    exit 1
fi

echo "Script executed successfully."

# Deactivate the virtual environment
deactivate
if [ $? -ne 0 ]; then
    echo "Error: Unable to deactivate virtual environment."
    exit 1
fi

echo "Virtual environment deactivated successfully."

echo
echo "Script execution complete."
