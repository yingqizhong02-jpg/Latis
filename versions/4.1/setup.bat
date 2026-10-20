@echo off
rem ============================================================
rem  colorlab 4.0 一键安装（虚拟环境自包含版）
rem  目标电脑：先装 Python 3.13 (勾选 Add to PATH)，然后双击本脚本
rem  效果：在项目内创建 .venv 虚拟环境并安装全部依赖，不污染系统
rem
rem  注意：本文件是 UTF-8。中文只出现在 rem 注释与 chcp 之后的 echo 里，
rem  且所有命令本身都是纯 ASCII —— cmd 按 GBK 读 .bat，命令含多字节中文时
rem  可能吞掉换行把下一条命令并进来（踩过一次坑）。
rem ============================================================
chcp 65001 >nul 2>&1
cd /d "%~dp0"
title colorlab setup

echo [1/5] 检查 Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
  echo   [ERROR] 未找到 Python。请先安装 Python 3.13，安装时勾选 "Add Python to PATH"
  pause & exit /b 1
)
python -c "import sys; assert sys.version_info[:2]==(3,13), 'need 3.13'" 2>nul
if %errorlevel% neq 0 (
  echo   [ERROR] 需要 Python 3.13，当前是其他版本。请安装 Python 3.13
  pause & exit /b 1
)
echo   OK

echo [2/5] 创建虚拟环境 (.venv)...
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
  if %errorlevel% neq 0 (echo   [ERROR] 创建虚拟环境失败 & pause & exit /b 1)
)
echo   OK

echo [3/5] 安装依赖（约 200MB，首次需几分钟）...
".venv\Scripts\python.exe" -m pip install --upgrade pip -q
".venv\Scripts\python.exe" -m pip install -q numpy customtkinter Pillow tifffile pillow-heif rawpy numba
if %errorlevel% neq 0 (
  echo   [ERROR] 依赖安装失败，请检查网络后重试
  pause & exit /b 1
)
echo   OK

echo [4/5] 尝试安装 CUDA 后端（预览加速，约 90MB）...
echo       不需要装 CUDA Toolkit；装不上就自动用 CPU，功能完全一样，只是预览慢些。
".venv\Scripts\python.exe" -m pip install -q cupy-cuda13x "cuda-toolkit[nvrtc,cudart]"
if %errorlevel% neq 0 (
  ".venv\Scripts\python.exe" -m pip install -q cupy-cuda12x
)
".venv\Scripts\python.exe" -c "import colorlab_gpu as g; print('  CUDA 后端可用' if g.available() else '  未启用')" 2>nul
if %errorlevel% neq 0 echo   [SKIP] CUDA 后端未装上，将使用 CPU（不影响功能）

echo [5/5] 完成
echo   启动方式：双击 color_tool.bat
echo   想强制用 CPU：set COLORLAB_BACKEND=numba  然后再启动
echo   查看导出文件的 ICC/EXIF：把图片拖到 tools\check_icc.bat 上
pause
