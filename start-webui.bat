@echo off
setlocal
call "%~dp0Start.bat" %*
exit /b %ERRORLEVEL%
