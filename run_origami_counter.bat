@echo off
setlocal

cd /d "%~dp0"

if not exist "requirements.txt" (
    echo requirements.txt was not found next to this launcher.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating local Python environment...
    where py >nul 2>&1
    if not errorlevel 1 (
        py -3 -m venv .venv
    ) else (
        where python >nul 2>&1
        if errorlevel 1 (
            echo Python 3 was not found.
            echo Install Python 3 from https://www.python.org/downloads/windows/
            echo Be sure to select "Add Python to PATH", then run this file again.
            pause
            exit /b 1
        )
        python -m venv .venv
    )
    if errorlevel 1 (
        echo Failed to create the Python environment.
        pause
        exit /b 1
    )

    echo Preparing pip...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    if errorlevel 1 (
        echo Failed to prepare pip in the Python environment.
        pause
        exit /b 1
    )
)

echo Checking Tkinter support...
".venv\Scripts\python.exe" -c "import tkinter"
if errorlevel 1 (
    echo Tkinter is not available in this Python installation.
    echo Reinstall Python 3 from https://www.python.org/downloads/windows/
    echo and include the Tcl/Tk and IDLE feature.
    pause
    exit /b 1
)

echo Installing or checking dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install dependencies.
    pause
    exit /b 1
)

echo Starting DNA Origami AFM Counter...
".venv\Scripts\python.exe" origami_counter_app.py
set "APP_STATUS=%ERRORLEVEL%"

if not "%APP_STATUS%"=="0" (
    echo The app exited with an error.
    pause
)

endlocal
exit /b %APP_STATUS%
