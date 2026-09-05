@echo off
setlocal enabledelayedexpansion

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
set DATA_DIR=%OUT_DIR%\.data
rem Optional: drop an .ico file at this path (project root) to brand the shortcut.
set ICON_FILE=app_icon.ico

if not exist "portable\python\python.exe" (
    echo ERROR: portable\python\python.exe not found - run build_portable.bat first.
    exit /b 1
)

echo === Cleaning previous package ===
if exist "%OUT_DIR%" rmdir /s /q "%OUT_DIR%"
mkdir "%OUT_DIR%"
mkdir "%DATA_DIR%"

echo === Copying portable Python ===
rem robocopy, not xcopy: xcopy's "Insufficient memory" error is actually
rem its misleading way of reporting a path that's too long - it then
rem silently stops copying past that point instead of failing clearly.
rem PySide6/shiboken6 bundle deeply-nested Qt files that hit this on long
rem project paths like this one. robocopy uses the Windows API that
rem natively supports long paths, so it doesn't have this problem.
robocopy "portable\python" "%DATA_DIR%\python" /E /R:2 /W:2 /NJH /NJS
if !ERRORLEVEL! GEQ 8 (
    echo ERROR: robocopy failed copying portable\python - see output above.
    exit /b 1
)
if exist "portable\python\Lib\site-packages\shiboken6" if not exist "%DATA_DIR%\python\Lib\site-packages\shiboken6" (
    echo ERROR: shiboken6 exists in portable\python but did not get copied to %DATA_DIR%\python.
    exit /b 1
)

echo === Copying project source ===
copy /Y app.py "%DATA_DIR%\" >nul
copy /Y config.py "%DATA_DIR%\" >nul
rem Patch just the COPIED config.py (never the dev source) so
rem input_folder/output_folder resolve one level up from .data, next to
rem the EdgeArtPicker shortcut - that's where the client actually drops
rem videos in and finds their output, instead of a hidden folder they
rem can't see. Model/state paths (models/, video_jobs.json) are left
rem alone since those are internal and fine staying inside .data.
powershell -Command "(Get-Content '%DATA_DIR%\config.py') -replace '\"videos\"', '\"../videos\"' -replace '\"output', '\"../output' | Set-Content '%DATA_DIR%\config.py'"
findstr /C:"../videos" "%DATA_DIR%\config.py" >nul
if !ERRORLEVEL! NEQ 0 (
    echo ERROR: failed to patch input_folder path in packaged config.py.
    exit /b 1
)
robocopy core "%DATA_DIR%\core" /E /R:2 /W:2 /NJH /NJS
if !ERRORLEVEL! GEQ 8 (echo ERROR: robocopy failed copying core - see output above.& exit /b 1)
robocopy gui "%DATA_DIR%\gui" /E /R:2 /W:2 /NJH /NJS
if !ERRORLEVEL! GEQ 8 (echo ERROR: robocopy failed copying gui - see output above.& exit /b 1)
robocopy tools "%DATA_DIR%\tools" /E /R:2 /W:2 /NJH /NJS
if !ERRORLEVEL! GEQ 8 (echo ERROR: robocopy failed copying tools - see output above.& exit /b 1)
robocopy EdgeTAM "%DATA_DIR%\EdgeTAM" /E /R:2 /W:2 /NJH /NJS
if !ERRORLEVEL! GEQ 8 (echo ERROR: robocopy failed copying EdgeTAM - see output above.& exit /b 1)
robocopy models "%DATA_DIR%\models" /E /R:2 /W:2 /NJH /NJS
if !ERRORLEVEL! GEQ 8 (echo ERROR: robocopy failed copying models - see output above.& exit /b 1)

if not exist "%DATA_DIR%\app.py" (
    echo ERROR: app.py was not copied into %DATA_DIR% - aborting before packaging a broken build.
    exit /b 1
)
if not exist "%DATA_DIR%\config.py" (
    echo ERROR: config.py was not copied into %DATA_DIR% - aborting before packaging a broken build.
    exit /b 1
)

if not exist "hf_cache" (
    echo ERROR: hf_cache\ not found - run build_portable.bat first ^(it runs warm_hf_cache.py^).
    exit /b 1
)
echo === Copying pre-downloaded model weights ^(hf_cache^) for offline use ===
robocopy hf_cache "%DATA_DIR%\hf_cache" /E /R:2 /W:2 /NJH /NJS
if !ERRORLEVEL! GEQ 8 (
    echo ERROR: robocopy failed copying hf_cache - see output above.
    exit /b 1
)

echo === Creating videos/output folders next to the shortcut ===
if not exist "%OUT_DIR%\videos" mkdir "%OUT_DIR%\videos"
if not exist "%OUT_DIR%\output" mkdir "%OUT_DIR%\output"

if exist "%ICON_FILE%" (
    echo === Copying icon ===
    copy /Y "%ICON_FILE%" "%DATA_DIR%\app_icon.ico" >nul
    set ICON_PATH=%DATA_DIR%\app_icon.ico
) else (
    set ICON_PATH=%DATA_DIR%\python\pythonw.exe
)

echo === Resolving absolute paths ===
rem %CD%\%DATA_DIR% etc. are relative and fragile inside a quoted PowerShell
rem -Command string (the earlier version double-escaped %%CD%% into the
rem literal text "%CD%", which PowerShell can't resolve to a real folder -
rem that's why the shortcut it created had no working target). Resolving
rem to absolute paths here, in batch, with %%~fI avoids that entirely.
for %%I in ("%OUT_DIR%") do set "OUT_DIR_ABS=%%~fI"
for %%I in ("%DATA_DIR%") do set "DATA_DIR_ABS=%%~fI"
rem Back to pythonw.exe (windowless) now that app.py itself redirects
rem sys.stdout/sys.stderr to app.log whenever they're None, instead of
rem crashing on the first write - that's the real fix, so we don't need
rem python.exe's console window as a workaround anymore.
for %%I in ("%DATA_DIR%\python\pythonw.exe") do set "PYTHONW_ABS=%%~fI"
for %%I in ("%ICON_PATH%") do set "ICON_PATH_ABS=%%~fI"
set "ICON_ARG_ABS=%ICON_PATH_ABS%,0"

echo === Creating top-level launcher shortcut ===
rem A real compiled .exe would mean reintroducing PyInstaller (the thing
rem build_portable.bat was written to avoid). Instead we create a .lnk
rem shortcut: double-click behavior and an icon just like an .exe, but it
rem just points at pythonw.exe inside the hidden .data folder below.
rem Writing a real .ps1 (instead of an inline -Command string) sidesteps
rem cmd's %-escaping problems and makes this easy to inspect if it ever
rem needs debugging again.
set PS1_FILE=%TEMP%\edgeartpicker_make_shortcut.ps1
(
    echo $ws = New-Object -ComObject WScript.Shell
    echo $sc = $ws.CreateShortcut^('%OUT_DIR_ABS%\EdgeArtPicker.lnk'^)
    echo $sc.TargetPath = '%PYTHONW_ABS%'
    echo $sc.Arguments = 'app.py'
    echo $sc.WorkingDirectory = '%DATA_DIR_ABS%'
    echo $sc.IconLocation = '%ICON_ARG_ABS%'
    echo $sc.Save^(^)
) > "%PS1_FILE%"

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1_FILE%"
if not exist "%OUT_DIR_ABS%\EdgeArtPicker.lnk" (
    echo ERROR: shortcut was not created - check the PowerShell output above.
    exit /b 1
)
del "%PS1_FILE%"

echo === Verifying the shortcut's target actually exists ===
if not exist "%PYTHONW_ABS%" (
    echo ERROR: %PYTHONW_ABS% does not exist - the shortcut points nowhere.
    echo This means portable\python\pythonw.exe was missing or the copy step failed.
    exit /b 1
)

echo === Hiding the data folder ===
attrib +h +s "%DATA_DIR%"

echo.
echo === Done ===
echo Zip the folder "%OUT_DIR%" and send it - the client unzips it
echo anywhere and sees just "EdgeArtPicker" to double-click (all the
echo Python/source/model files are tucked away in a hidden folder
echo next to it, since Windows Explorer hides both the .data folder
echo and the .lnk extension by default).
