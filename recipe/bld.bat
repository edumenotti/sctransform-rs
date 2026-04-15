@echo off
setlocal enabledelayedexpansion

maturin build --release --interpreter "%PYTHON%" --out dist
if errorlevel 1 exit /b 1

"%PYTHON%" -m pip install --no-deps --no-build-isolation --find-links dist sctransform-rs
if errorlevel 1 exit /b 1
