# One-shot publish for CrisisLens. Run from D:\DEvharbour in PowerShell:
#   powershell -ExecutionPolicy Bypass -File scripts\publish.ps1
# 1) GitHub repo (public) + push   2) HF model repo with the GGUF LoRA adapter   3) HF Docker Space (demo)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$py = "C:\Users\hp\anaconda3\python.exe"

Write-Host "== 1/3 GitHub" -ForegroundColor Cyan
git branch -M main
$hasOrigin = (git remote) -contains "origin"
if (-not $hasOrigin) {
  gh repo create crisislens --public --source . --remote origin --description "CrisisLens: disaster-response triage LLM fine-tuned with Adaption AutoScientist on HumAID" --push
} else { git push -u origin main }

Write-Host "== 2/3 + 3/3 Hugging Face model + Space" -ForegroundColor Cyan
& $py -I scripts\publish_hf.py
Write-Host "Done. Links are printed above and written to eval\results\links.json" -ForegroundColor Green
