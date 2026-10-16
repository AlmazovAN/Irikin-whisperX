@echo off
setlocal
set "ROOT=%~dp0"
cd /d "%ROOT%"
set "PYTHON="

rem Prefer 64-bit Python 3.11; do not silently use an unsupported Python version.
where py >nul 2>&1
if not errorlevel 1 (
    py -3.11 -c "import sys; assert sys.maxsize > 2**32" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON=py -3.11"
        goto run_installer
    )
    py -3.12 -c "import sys; assert sys.maxsize > 2**32" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON=py -3.12"
        goto run_installer
    )
    py -3.10 -c "import sys; assert sys.maxsize > 2**32" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON=py -3.10"
        goto run_installer
    )
)

where python >nul 2>&1
if errorlevel 1 (
    echo Python 3.10-3.12 x64 was not found.
    echo Install Python 3.11 x64 and retry.
    set "INSTALL_RESULT=2"
    goto finished
)
set "PYTHON=python"

:run_installer
%PYTHON% "%ROOT%scripts\install_windows.py" %*
set "INSTALL_RESULT=%ERRORLEVEL%"

:finished
echo.
if "%INSTALL_RESULT%"=="0" (
    echo Installer completed. See the messages above for the remaining smoke tests.
) else (
    echo Installer stopped with exit code %INSTALL_RESULT%.
    echo Existing virtual environments have not been deleted.
)
pause
exit /b %INSTALL_RESULT%
