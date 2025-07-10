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

REM Check if uv is installed
echo Checking for uv package manager...
"%PYTHON_CMD%" -m uv --version >nul
if %errorlevel% neq 0 (
    echo uv not found. Installing uv...
    "%PYTHON_CMD%" -m pip install uv --quiet
    if %errorlevel% neq 0 (
        echo Failed to install uv. Please install it manually: pip install uv
        exit /b 1
    )
    echo uv installed successfully!
) else (
    echo uv is already installed.
)

echo.
echo Installing dependencies for CPU...
echo Project root: %PROJECT_ROOT%
echo.

echo Installing PyTorch with CPU support...
"%PYTHON_CMD%" -m pip install --quiet "torch>=2.1.0+cpu" --index-url https://download.pytorch.org/whl/cpu
if %errorlevel% neq 0 exit /b %errorlevel%

"%PYTHON_CMD%" -m pip install --quiet "torchvision>=0.16.0+cpu" --index-url https://download.pytorch.org/whl/cpu
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo Installing remaining dependencies...
"%PYTHON_CMD%" -m uv pip install --quiet -r "%PROJECT_ROOT%\requirements\requirements_cpu.txt"
if %errorlevel% neq 0 (
    echo uv failed, falling back to pip...
    "%PYTHON_CMD%" -m pip install --quiet -r "%PROJECT_ROOT%\requirements\requirements_cpu.txt"
    if %errorlevel% neq 0 exit /b %errorlevel%
)

echo.
echo Installation completed successfully! 