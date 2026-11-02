#!/usr/bin/env bash
# Stop llama-server if free physical RAM drops below 2 GB (avoid tripping the harness memory guard).
while true; do
  free=$(powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024)")
  if [ "$free" -lt 2048 ]; then taskkill //F //IM llama-server.exe; echo "mem_guard: stopped llama-server at ${free}MB free $(date +%T)"; fi
  sleep 20
done
