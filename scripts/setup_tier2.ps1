# VIX Tier-2 one-shot setup (Windows / PowerShell).
# Prerequisites: Python 3.11 and git installed; run from the repo root after `git clone`.
# Usage:  .\scripts\setup_tier2.ps1
# (ASCII-only on purpose: Windows PowerShell 5.1 mis-reads UTF-8 Chinese under cp950.)
$ErrorActionPreference = "Stop"

Write-Host "[1/4] Creating Python 3.11 venv .venv311 ..." -ForegroundColor Cyan
py -3.11 -m venv .venv311

$py = ".\.venv311\Scripts\python.exe"
Write-Host "[2/4] Installing VIX core + Tier-2 deps (FiftyOne/torch/playwright) ..." -ForegroundColor Cyan
& $py -m pip install --upgrade pip
& $py -m pip install -e .
& $py -m pip install -r requirements-tier2.txt

Write-Host "[3/4] Downloading Playwright Chromium ..." -ForegroundColor Cyan
& $py -m playwright install chromium

Write-Host "[4/4] Done." -ForegroundColor Green
Write-Host ""
Write-Host "Next - build the demo dataset and launch the App (first run downloads CIFAR-10, ~1-2 min):"
Write-Host "  $py examples\serve_animals.py" -ForegroundColor Yellow
Write-Host "Then open http://localhost:5151 and follow docs\guide\EMBEDDINGS_HOWTO.html"
