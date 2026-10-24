@echo off
setlocal
cd /d "%~dp0"
python -X utf8 -u reproduce.py %*
exit /b %errorlevel%
