@echo off
REM LiveSwapping Installation Script for Windows
REM 
REM Usage:
REM   install.bat                                    - Use system Python
REM   install.bat "path\to\python.exe"               - Use specific Python
REM   install.bat "..\python311\python.exe"         - Use relative path
REM   install.bat "C:\Python311\python.exe"         - Use absolute path
REM
REM This script automatically:
REM   - Detects your GPU type (NVIDIA/AMD/Intel)
REM   - Suggests optimal configuration
REM   - Installs uv package manager if needed
REM   - Handles PyTorch CUDA version selection
REM
setlocal enabledelayedexpansion

echo ====================================================
echo           LiveSwapping - Installation
echo ====================================================

REM Handle Python path argument
if "%1"=="" (
    set "PYTHON_PATH=python"
    echo Using system Python...
) else (
    REM Convert relative path to absolute if needed
    for %%i in ("%~1") do set "PYTHON_PATH=%%~fi"
    echo Using Python from: !PYTHON_PATH!
)

call :check_python
if !errorlevel! neq 0 exit /b 1

cd /d "%~dp0"

call :detect_gpu
set gpu_type=!errorlevel!

echo.
echo Configuration options:

if !gpu_type! equ 0 (
    echo 1. CUDA ^(NVIDIA GPU^) - recommended
    echo 2. DirectML ^(AMD GPU^)
    echo 3. OpenVINO ^(Intel optimization^)
    echo 4. CPU only
    echo.
    set "choice="
    set /p choice="Select mode (1-4, default 1): "
    if "!choice!"=="" set choice=1
    
    echo Debug: GPU type=!gpu_type!, Selected choice=!choice!
    
    if "!choice!"=="1" (
        echo Launching CUDA installer...
        call "installers\install_cuda.bat" "!PYTHON_PATH!"
    ) else if "!choice!"=="2" (
        echo Launching DirectML installer...
        call "installers\install_dml.bat" "!PYTHON_PATH!"
    ) else if "!choice!"=="3" (
        echo Launching OpenVINO installer...
        call "installers\install_openvino.bat" "!PYTHON_PATH!"
    ) else if "!choice!"=="4" (
        echo Launching CPU installer...
        call "installers\install_cpu.bat" "!PYTHON_PATH!"
    ) else (
        echo ERROR: Invalid choice: !choice!
        exit /b 1
    )
) else if !gpu_type! equ 1 (
    echo 1. DirectML ^(AMD GPU^) - recommended
    echo 2. OpenVINO ^(Intel optimization^)
    echo 3. CPU only
    echo.
    set "choice="
    set /p choice="Select mode (1-3, default 1): "
    if "!choice!"=="" set choice=1
    
    echo Debug: GPU type=!gpu_type!, Selected choice=!choice!
    
    if "!choice!"=="1" (
        echo Launching DirectML installer...
        call "installers\install_dml.bat" "!PYTHON_PATH!"
    ) else if "!choice!"=="2" (
        echo Launching OpenVINO installer...
        call "installers\install_openvino.bat" "!PYTHON_PATH!"
    ) else if "!choice!"=="3" (
        echo Launching CPU installer...
        call "installers\install_cpu.bat" "!PYTHON_PATH!"
    ) else (
        echo ERROR: Invalid choice: !choice!
        exit /b 1
    )
) else if !gpu_type! equ 2 (
    echo 1. OpenVINO ^(Intel GPU^) - recommended
    echo 2. CPU only
    echo.
    set "choice="
    set /p choice="Select mode (1-2, default 1): "
    if "!choice!"=="" set choice=1
    
    echo Debug: GPU type=!gpu_type!, Selected choice=!choice!
    
    if "!choice!"=="1" (
        echo Launching OpenVINO installer...
        call "installers\install_openvino.bat" "!PYTHON_PATH!"
    ) else if "!choice!"=="2" (
        echo Launching CPU installer...
        call "installers\install_cpu.bat" "!PYTHON_PATH!"
    ) else (
        echo ERROR: Invalid choice: !choice!
        exit /b 1
    )
) else (
    echo Launching CPU installer...
    call "installers\install_cpu.bat" "!PYTHON_PATH!"
)

echo.
echo [SUCCESS] Ready! You can now run:
if "!PYTHON_PATH!"=="python" (
    echo    python run.py
) else (
    echo    "!PYTHON_PATH!" run.py
)
echo.
echo [INFO] Installation used uv package manager for faster downloads
echo [INFO] For direct access to installers, check the 'installers' folder
echo.
exit /b 0

:check_python
echo Checking Python...
"!PYTHON_PATH!" --version >nul 2>&1
if !errorlevel! neq 0 (
    echo ERROR: Python not found at: !PYTHON_PATH!
    echo Please check the path or install Python 3.8+
    echo Download: https://www.python.org/downloads/
    exit /b 1
)

for /f "tokens=2" %%i in ('"!PYTHON_PATH!" --version 2^>^&1') do set python_version=%%i
echo Python version: !python_version!

"!PYTHON_PATH!" -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>&1
if !errorlevel! neq 0 (
    echo ERROR: Python 3.8 or newer required
    exit /b 1
)
exit /b 0

:detect_gpu
echo Detecting GPU type...

nvidia-smi --query-gpu=name --format=csv >nul 2>&1
if !errorlevel! equ 0 (
    for /f "skip=1 delims=" %%i in ('nvidia-smi --query-gpu=name --format=csv 2^>nul') do (
        echo NVIDIA GPU detected: %%i
        exit /b 0
    )
)

for /f "skip=1 delims=" %%i in ('wmic path win32_VideoController get name /format:value 2^>nul ^| findstr "="') do (
    set "line=%%i"
    if "!line:~0,5!"=="Name=" (
        set "gpu_name=!line:~5!"
        echo !gpu_name! | findstr /i "nvidia geforce gtx rtx quadro tesla" >nul
        if !errorlevel! equ 0 (
            echo NVIDIA GPU detected: !gpu_name!
            echo WARNING: nvidia-smi not found - ensure NVIDIA drivers are installed
            exit /b 0
        )
    )
)

for /f "skip=1 delims=" %%i in ('wmic path win32_VideoController get name /format:value 2^>nul ^| findstr "="') do (
    set "line=%%i"
    if "!line:~0,5!"=="Name=" (
        set "gpu_name=!line:~5!"
        echo !gpu_name! | findstr /i "amd radeon rx vega" >nul
        if !errorlevel! equ 0 (
            echo AMD GPU detected: !gpu_name!
            exit /b 1
        )
    )
)

for /f "skip=1 delims=" %%i in ('wmic path win32_VideoController get name /format:value 2^>nul ^| findstr "="') do (
    set "line=%%i"
    if "!line:~0,5!"=="Name=" (
        set "gpu_name=!line:~5!"
        echo !gpu_name! | findstr /i "intel hd iris xe arc" >nul
        if !errorlevel! equ 0 (
            echo Intel GPU detected: !gpu_name!
            exit /b 2
        )
    )
)

echo WARNING: No GPU detected or unknown type - using CPU mode
exit /b 3 