@echo off
setlocal

echo ====================================================
echo         LiveSwapping CUDA Installation
echo ====================================================
echo.
echo Please select your CUDA version:
echo.
echo 1. CUDA 12.1 (older, more compatible)
echo 2. CUDA 12.8 (newer, latest features)
echo.
set /p choice="Enter your choice (1 or 2): "

if "%choice%"=="1" (
    echo.
    echo Installing for CUDA 12.1...
    call "%~dp0install_cuda121.bat" %1
    goto :end
) else if "%choice%"=="2" (
    echo.
    echo Installing for CUDA 12.8...
    call "%~dp0install_cuda128.bat" %1
    goto :end
) else (
    echo.
    echo Invalid choice. Please run the script again and select 1 or 2.
    goto :end
)

:end
echo.
echo Installation script finished.
if "%choice%"=="1" echo Installed with CUDA 12.1 support.
if "%choice%"=="2" echo Installed with CUDA 12.8 support.