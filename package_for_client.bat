@echo off
setlocal

rem =============================================================
rem Assembles dist_portable\EdgeArtPicker\ - the folder you zip and
rem send to the client. Run build_portable.bat first (once, or
rem after any dependency change); run this every time you want to
rem re-package with the latest source code.
rem
rem Usage (from project root):
rem     package_for_client.bat
rem =============================================================

set OUT_DIR=dist_portable\EdgeArtPicker

if not exist "portable\python\python.exe" (
    echo ERROR: portable\python\python.exe not found - run build_portable.bat first.
    exit /b 1
)

echo === Cleaning previous package ===
if exist "%OUT_DIR%" rmdir /s /q "%OUT_DIR%"
mkdir "%OUT_DIR%"

echo === Copying portable Python ===
xcopy /E /I /Q "portable\python" "%OUT_DIR%\python"

echo === Copying project source ===
xcopy /E /I /Q app.py "%OUT_DIR%\"
xcopy /E /I /Q config.py "%OUT_DIR%\"
xcopy /E /I /Q core "%OUT_DIR%\core"
xcopy /E /I /Q gui "%OUT_DIR%\gui"
xcopy /E /I /Q tools "%OUT_DIR%\tools"
xcopy /E /I /Q EdgeTAM "%OUT_DIR%\EdgeTAM"
xcopy /E /I /Q models "%OUT_DIR%\models"

if not exist "hf_cache" (
    echo ERROR: hf_cache\ not found - run build_portable.bat first ^(it runs warm_hf_cache.py^).
    exit /b 1
)
echo === Copying pre-downloaded model weights ^(hf_cache^) for offline use ===
xcopy /E /I /Q hf_cache "%OUT_DIR%\hf_cache"

echo === Writing launcher ===
(
    echo @echo off
    echo cd /d "%%~dp0"
    echo python\pythonw.exe app.py
) > "%OUT_DIR%\EdgeArtPicker.bat"

echo.
echo === Done ===
echo Zip the folder "%OUT_DIR%" and send it - the client unzips it
echo anywhere and double-clicks EdgeArtPicker.bat inside.
