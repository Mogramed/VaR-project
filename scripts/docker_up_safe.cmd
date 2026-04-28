@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%docker_up_safe.ps1" %*
exit /b %ERRORLEVEL%
