#!/usr/bin/env bash
# Low-memory, strictly sequential candidate evaluation (after the memory guard stopped the parallel queue).
#  1. let the in-flight run #7 val evaluation finish, stop its server, score it
#  2. for each remaining candidate: make sure it is downloaded + converted, then evaluate on val_eval
#     with 2 llama.cpp slots (about half the memory of 4)
cd "$(dirname "$0")/.."
PY=/c/Users/hp/anaconda3/python.exe
log() { echo "$(date +%T) $*"; }
running() { powershell -NoProfile -c "@(Get-CimInstance Win32_Process -Filter \"name='$1'\" | Where-Object { \$_.CommandLine -match '$2' }).Count"; }
stop_port() { powershell -NoProfile -c "Get-CimInstance Win32_Process -Filter \"name='llama-server.exe'\" | Where-Object { \$_.CommandLine -match '--port $1' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"; }

# 1. run #7
while [ "$(running python.exe 'model ft2b_run7')" != "0" ]; do sleep 30; done
stop_port 8090
n=$(grep -c '"raw": "' eval/preds/ft2b_run7__val_eval.jsonl)
log "run7 rows with output: $n / 574"
if [ "$n" -lt 574 ]; then  # finish any missing rows with a fresh 2-slot server
  bash scripts/eval_cand.sh ft2b_run7 tools/gguf/Qwen3.5-2B-Q4_K_M.gguf models/2b634ded-e0b7-4a21-9467-bf4025b0bcc1/adapter-f16.gguf val_eval 2 v2
else
  $PY -I eval/score.py --split val_eval ft2b_run7 | tail -1
fi

# 2. remaining candidates, in the order they can start (run #6 is already converted)
for spec in "ft4b_run6|fafc81b1-8876-43c1-80e9-75a1791d676c|Qwen/Qwen3.5-4B|tools/gguf/Qwen3.5-4B-Q4_K_M.gguf|v2" \
            "ft2b_run8|fab5f195-3c03-4f96-ac63-fb31f6693054|Qwen/Qwen3.5-2B|tools/gguf/Qwen3.5-2B-Q4_K_M.gguf|v2" \
            "ft2b_run4|25d04b86-af94-45cf-be37-5acc5822614c|Qwen/Qwen3.5-2B|tools/gguf/Qwen3.5-2B-Q4_K_M.gguf|v1"; do
  IFS='|' read -r key run hf base pv <<< "$spec"
  if [ ! -f "models/$run/adapter-f16.gguf" ]; then
    while [ "$(running python.exe "download --run $run")" != "0" ]; do sleep 30; done   # in-flight download
    for k in 1 2 3 4 5; do
      [ -f "models/$run/adapter_model.safetensors" ] && break
      log "downloading $key (attempt $k)"; $PY -I src/train/autoscientist_run.py download --run "$run" --fresh >> "logs_dl_$key.txt" 2>&1
    done
    [ -f "models/$run/adapter_model.safetensors" ] || { log "SKIP $key: download failed"; continue; }
    bash src/serve/lora_to_gguf.sh "models/$run" "$hf" >> "logs_dl_$key.txt" 2>&1 || { log "SKIP $key: conversion failed"; continue; }
  fi
  log "EVAL $key"
  bash scripts/eval_cand.sh "$key" "$base" "models/$run/adapter-f16.gguf" val_eval 2 "$pv"
done
log "ALL CANDIDATES DONE"
