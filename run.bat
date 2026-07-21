@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo No .venv found — using system Python.
)
python tui.py %*
