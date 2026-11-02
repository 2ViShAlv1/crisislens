# CrisisLens alarm: beeps ~60s and shows a MessageBox with the reason. Usage: powershell -File scripts/alarm.ps1 "reason" [seconds]
param([string]$Reason = "Claude needs you", [int]$Seconds = 60)
$ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$blocked = Join-Path $PSScriptRoot "..\BLOCKED.md"
Add-Content -Path $blocked -Value "- [$ts] $Reason" -Encoding utf8
Add-Type -AssemblyName System.Windows.Forms
# MessageBox in a separate process so beeping continues while it is open
Start-Process powershell -WindowStyle Hidden -ArgumentList "-NoProfile","-Command","Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.MessageBox]::Show('$($Reason -replace "'","''")','CLAUDE NEEDS YOU','OK','Warning') | Out-Null"
$end = (Get-Date).AddSeconds($Seconds)
while ((Get-Date) -lt $end) { [console]::beep(1000,700); Start-Sleep -Milliseconds 300 }
