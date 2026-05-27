@echo off
REM Windows-native wrapper around the Makefile targets. Run from the project
REM root or via its full path. Usage:
REM
REM   dev.bat install           Install runtime deps into .venv
REM   dev.bat install-dev       Install dev deps (includes runtime)
REM   dev.bat data              Download DRAMP + UniProt corpora
REM   dev.bat validate          Validate the downloaded corpus
REM   dev.bat cluster           mmseqs2 / CD-HIT clustering at 40% identity
REM   dev.bat splits            Build cluster-aware 80/10/10 splits
REM   dev.bat baseline          Train + evaluate the three baselines
REM   dev.bat cnn               Train + evaluate the small CNN
REM   dev.bat test              Run pytest
REM   dev.bat app               Launch the Gradio demo on http://127.0.0.1:7860
REM
REM The Makefile is the canonical source; this script is a thin convenience
REM wrapper so users on PowerShell or cmd.exe don't need `make` installed.

setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo No .venv found. Run:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   dev.bat install-dev
    exit /b 1
)

call .venv\Scripts\activate.bat

if "%1"=="" goto :usage

if /I "%1"=="install"      ( python -m pip install -r requirements.txt          & goto :eof )
if /I "%1"=="install-dev"  ( python -m pip install -r requirements-dev.txt      & goto :eof )
if /I "%1"=="data"         ( python ml\scripts\download_data.py %2 %3 %4 %5     & goto :eof )
if /I "%1"=="validate"     ( python ml\scripts\validate_data.py                 & goto :eof )
if /I "%1"=="cluster"      ( python ml\scripts\make_clusters.py                 & goto :eof )
if /I "%1"=="splits"       ( python ml\scripts\make_splits.py                   & goto :eof )
if /I "%1"=="baseline"     ( python ml\scripts\train_baseline.py ^
                             && python ml\scripts\eval_baseline.py              & goto :eof )
if /I "%1"=="cnn"          ( python ml\scripts\train_cnn.py ^
                             && python ml\scripts\eval_cnn.py                   & goto :eof )
if /I "%1"=="calibrate"    ( python ml\scripts\calibrate_cnn.py ^
                             && python ml\scripts\eval_cnn.py                   & goto :eof )
if /I "%1"=="esm-embed"    ( python ml\scripts\extract_esm_embeddings.py        & goto :eof )
if /I "%1"=="esm-train"    ( python ml\scripts\train_esm_head.py                & goto :eof )
if /I "%1"=="esm-eval"     ( python ml\scripts\eval_esm_head.py                 & goto :eof )
if /I "%1"=="esm"          ( python ml\scripts\extract_esm_embeddings.py ^
                             && python ml\scripts\train_esm_head.py ^
                             && python ml\scripts\eval_esm_head.py              & goto :eof )
if /I "%1"=="activities"   ( python ml\scripts\extract_amp_activities.py        & goto :eof )
if /I "%1"=="multiclass"   ( python ml\scripts\train_esm_multiclass.py ^
                             && python ml\scripts\eval_esm_multiclass.py        & goto :eof )
if /I "%1"=="design"       ( python ml\scripts\design_mutants.py %2 %3 %4 %5 %6 & goto :eof )
if /I "%1"=="test"         ( python -m pytest -v                                & goto :eof )
if /I "%1"=="app"          ( python app.py                                      & goto :eof )

echo Unknown subcommand: %1
:usage
echo Usage:
echo   dev.bat install ^| install-dev ^| data ^| validate ^| cluster ^| splits
echo   dev.bat baseline ^| cnn ^| test ^| app
exit /b 1
