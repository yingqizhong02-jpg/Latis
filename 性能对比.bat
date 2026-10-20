@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
title Latis performance A/B

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
set "OUTDIR=%~dp0tests\perf"

set "OLD=%~dp0..\Color-Lab3.3"
if not exist "%OLD%\colorlab.py" set "OLD="

echo ============================================================
echo  colorlab 4.0 vs 3.3  --  same photos, same measuring script
echo ============================================================
echo  measure script : tests\perf_baseline.py
echo  it calls the REAL App._export_one path (not a re-implementation)
echo.

if not "%OLD%"=="" (
  echo ---- BEFORE : %OLD%
  "%PY%" "%~dp0tests\perf_baseline.py" --root "%OLD%" --outdir "%OUTDIR%" --tag before --files 3 --fmt png
  echo.
) else (
  echo [SKIP] Color-Lab3.3 not found, only measuring 4.0
  echo.
)

echo ---- AFTER  : %~dp0
"%PY%" "%~dp0tests\perf_baseline.py" --outdir "%OUTDIR%" --tag after --files 3 --fmt png

echo.
echo ============================================================
echo  PNG size note: level 3 makes files bigger. Measured on real
echo  33MP photos: +3%% to +11%% depending on the picture.
echo  Want 3.3's size back?  set COLORLAB_PNG_LEVEL=6  (or add
echo  "pnglvl": 6 to .colorlab_config) -- but you lose ~1.1s/photo.
echo ============================================================
echo.
echo JSON results: %OUTDIR%
pause
