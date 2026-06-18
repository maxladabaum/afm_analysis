#!/bin/bash
set -u

cd "$(dirname "$0")" || exit 1

PYTHON_CMD=""
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo "Python was not found."
    echo "Install Python 3 for macOS from https://www.python.org/downloads/macos/ and run this file again."
    read -r -p "Press Return to close this window..."
    exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
    echo "Creating local Python environment..."
    "$PYTHON_CMD" -m venv .venv
    if [ $? -ne 0 ]; then
        echo "Failed to create the Python environment."
        read -r -p "Press Return to close this window..."
        exit 1
    fi
fi

echo "Checking macOS Tkinter support..."
".venv/bin/python" - <<'PY'
import tkinter
PY
if [ $? -ne 0 ]; then
    echo "Tkinter is not available in this Python install."
    echo "Install the standard macOS Python package from https://www.python.org/downloads/macos/ and run this file again."
    read -r -p "Press Return to close this window..."
    exit 1
fi

echo "Installing or checking dependencies..."
".venv/bin/python" -m pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo "Failed to update pip."
    read -r -p "Press Return to close this window..."
    exit 1
fi

".venv/bin/python" -m pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to install dependencies."
    read -r -p "Press Return to close this window..."
    exit 1
fi

echo "Starting DNA Origami AFM Counter..."
".venv/bin/python" origami_counter_app.py
APP_STATUS=$?

if [ $APP_STATUS -ne 0 ]; then
    echo "The app exited with an error."
    read -r -p "Press Return to close this window..."
fi

exit $APP_STATUS
