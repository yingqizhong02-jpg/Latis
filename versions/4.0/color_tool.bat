@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
set "SCRIPT=%~dp0colorlab.py"

rem 版本号从 colorlab.py 的 APP_VERSION 行读取（回归测试第 12 组保证它与 CHANGELOG.md 一致）
set "CLVER=?"
for /f "tokens=3" %%a in ('findstr /b APP_VERSION "%~dp0colorlab.py"') do set "CLVER=%%a"
set "CLVER=%CLVER:"=%"
title colorlab v%CLVER%
echo [%TIME%] colorlab %CLVER% starting...

rem 优先用项目内 .venv（由 setup.bat 创建），否则退回系统 python
set "PY=python"
if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"

"%PY%" --version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python not found: %PY%
  pause
  exit /b 1
)

echo [%TIME%] Launching %PY% ...
"%PY%" "%SCRIPT%"
if errorlevel 1 (
  echo [%TIME%] ERROR
  pause
)
pause
