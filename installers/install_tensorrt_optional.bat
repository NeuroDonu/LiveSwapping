@echo off
setlocal

REM Get absolute path to project root (one level up from installers)
set "PROJECT_ROOT=%~dp0.."

REM Check if Python path is provided as argument
if "%1"=="" (
    set PYTHON_CMD=python
    echo Using system Python...
) else (
    set "PYTHON_CMD=%~1"
    echo Using Python from: %~1
)

echo ====================================================
echo     TensorRT Optional Installation for LiveSwapping
echo ====================================================
echo.
echo TensorRT provides additional optimizations for NVIDIA GPUs.
echo This is OPTIONAL and not required for basic functionality.
echo.
echo Prerequisites:
echo - NVIDIA GPU with CUDA support
echo - CUDA Toolkit 12.x installed
echo - NVIDIA TensorRT 10.x installed manually
echo.
echo If you don't have TensorRT installed, this will fail.
echo You can download TensorRT from: https://developer.nvidia.com/tensorrt
echo.

echo Attempting to install torch-tensorrt...
"%PYTHON_CMD%" -m pip install --quiet torch-tensorrt>=2.1.0 --index-url https://download.pytorch.org/whl/cu128

if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] TensorRT support installed successfully!
    echo You can now use TensorRT optimizations in LiveSwapping.
) else (
    echo.
    echo [WARNING] TensorRT installation failed - this is optional.
    echo LiveSwapping will work fine without TensorRT.
    echo For TensorRT support, install NVIDIA TensorRT manually from:
    echo https://developer.nvidia.com/tensorrt
)

echo. 