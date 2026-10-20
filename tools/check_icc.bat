@echo off
rem ============================================================
rem  Color-Lab ICC / EXIF checker
rem
rem  IMPORTANT: this file is intentionally PURE ASCII.
rem  cmd.exe reads .bat files using the OEM codepage (GBK on a
rem  Chinese Windows), NOT UTF-8. Non-ASCII bytes here get mangled
rem  and can even swallow the following newline, merging the next
rem  line into this command and breaking parsing.
rem  ASCII is byte-identical in both UTF-8 and GBK, so it is safe.
rem  All Chinese messages come from the Python script instead.
rem ============================================================
chcp 65001 >nul 2>&1
setlocal
title Color-Lab ICC / EXIF checker
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

set "PY=%~dp0..\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" "%~dp0check_icc.py" %*
echo.
pause
