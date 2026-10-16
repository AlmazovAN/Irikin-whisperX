@echo off
setlocal EnableExtensions DisableDelayedExpansion
set "ROOT=%~dp0"
cd /d "%ROOT%"
if errorlevel 1 (
    echo ERROR: Cannot open the application directory.
    pause
    exit /b 2
)

rem Do not install dependencies or delete files during startup.
if not exist "%ROOT%venv\Scripts\python.exe" (
    if not exist "%ROOT%logs" mkdir "%ROOT%logs"
    echo ERROR: Missing venv Python. Run Install.bat --profile cpu first.
    >>"%ROOT%logs\start-bootstrap-error.log" echo Missing venv Python. Run Install.bat --profile cpu first.
    pause
    exit /b 2
)

"%ROOT%venv\Scripts\python.exe" -I "%ROOT%scripts\launch_windows.py" %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Launcher failed with exit code %EXIT_CODE%.
    echo See the latest file in "%ROOT%logs" for details.
) else (
    echo.
    echo WhisperX server is stopped.
)
echo.
pause
exit /b %EXIT_CODE%
