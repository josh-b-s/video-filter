@echo off
setlocal enabledelayedexpansion

rem =============================================================
rem Builds a portable, no-install-required Python folder next to
rem your project source, using Python's official "embeddable"
rem distribution instead of PyInstaller. Run this ONCE per Python/
rem dependency version bump - not something the client ever runs.
rem
rem What this avoids, compared to PyInstaller: every hidden-import
rem guess-and-check cycle, the Hydra dynamic-target-string problem,
rem torch.jit.script needing real source text, and the _internal/
rem relative-path mismatch - all of that is a consequence of
rem freezing into compiled bytecode. This just runs your actual
rem .py files with a real (if minimal) Python interpreter, so it
rem behaves exactly like "run from source", which has been the one
rem mode that's worked reliably every time.
rem
rem Usage (from project root):
rem     build_portable.bat
rem
rem Prerequisite: download the embeddable zip that matches your
rem dev venv's Python version from:
rem     https://www.python.org/downloads/windows/
rem (look for "Windows embeddable package (64-bit)" under whichever
rem 3.10.x release you're using) and save it as embed_python.zip in
rem this project's root before running this script.
rem =============================================================

set PYTHON_VERSION=3.10.11
set EMBED_ZIP=embed_python.zip
set PORTABLE_DIR=portable\python
set CUDA_BUILD=cu118

if not exist "%EMBED_ZIP%" (
    echo ERROR: %EMBED_ZIP% not found in project root.
    echo Download "Windows embeddable package (64-bit)" for Python %PYTHON_VERSION% from
    echo https://www.python.org/downloads/windows/ and save it here as %EMBED_ZIP%.
    exit /b 1
)

echo === Extracting embeddable Python to %PORTABLE_DIR% ===
if exist "%PORTABLE_DIR%" rmdir /s /q "%PORTABLE_DIR%"
mkdir "%PORTABLE_DIR%"
powershell -Command "Expand-Archive -Path '%EMBED_ZIP%' -DestinationPath '%PORTABLE_DIR%' -Force"

echo === Enabling site-packages (needed for pip-installed packages to import) ===
rem The embeddable distribution ships with site-packages support
rem commented out by default (it's meant for embedding, not general
rem use) - uncomment "import site" in the ._pth file so anything we
rem pip-install becomes importable.
for %%f in ("%PORTABLE_DIR%\python*._pth") do (
    powershell -Command "(Get-Content '%%f') -replace '#import site', 'import site' | Set-Content '%%f'"
)

echo === Bootstrapping pip ===
if not exist get-pip.py (
    powershell -Command "Invoke-WebRequest -Uri https://bootstrap.pypa.io/get-pip.py -OutFile get-pip.py"
)
"%PORTABLE_DIR%\python.exe" get-pip.py --no-warn-script-location

echo === Installing dependencies ===
if exist ".venv\Lib\site-packages" (
    echo Found .venv - copying already-installed packages instead of
    echo re-downloading anything.
    robocopy ".venv\Lib\site-packages" "%PORTABLE_DIR%\Lib\site-packages" /E /NFL /NDL /NJH /NJS
    rem robocopy's "success" exit codes are 0-7, not just 0 - anything
    rem 8+ is a real failure, so check against that instead of ERRORLEVEL 1.
    if !ERRORLEVEL! GEQ 8 (
        echo ERROR: robocopy failed copying site-packages - see output above.
        exit /b 1
    )
) else (
    echo No .venv found - installing fresh via pip ^(will download everything^).
    "%PORTABLE_DIR%\python.exe" install_dependencies.py --cuda %CUDA_BUILD%
)

echo === Pre-downloading timm backbone weights ===
set HF_MODEL_FOLDER=models--timm--repvit_m1.dist_in1k
if exist "%USERPROFILE%\.cache\huggingface\hub\%HF_MODEL_FOLDER%" (
    echo Found this model already cached from earlier test runs -
    echo copying instead of re-downloading.
    robocopy "%USERPROFILE%\.cache\huggingface\hub\%HF_MODEL_FOLDER%" "hf_cache\hub\%HF_MODEL_FOLDER%" /E /NFL /NDL /NJH /NJS
    if !ERRORLEVEL! GEQ 8 (
        echo ERROR: robocopy failed copying the HF cache - see output above.
        exit /b 1
    )
) else (
    echo Not found in the local HF cache - downloading fresh ^(needs internet^).
    "%PORTABLE_DIR%\python.exe" warm_hf_cache.py
)

echo.
echo === Done ===
echo Portable Python is ready at %PORTABLE_DIR%\
echo Next: run package_for_client.bat to assemble the folder to send.