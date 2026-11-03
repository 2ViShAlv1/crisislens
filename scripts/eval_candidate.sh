#!/usr/bin/env bash
# Evaluate one fine-tuned candidate on one split with a dedicated llama.cpp server.
#   bash scripts/eval_candidate.sh <name> <base.gguf> <adapter.gguf> <split> [np]
# e.g. bash scripts/eval_candidate.sh ft2b_run7 tools/gguf/Qwen3.5-2B-Q4_K_M.gguf models/<run>/adapter-f16.gguf val_eval 4
# Predictions -> eval/preds/<name>__<split>.jsonl (resumable). Port 8090 so it never collides with the demo server.
set -euo pipefail
NAME=$1; BASE=$2; ADAPTER=$3; SPLIT=$4; NP=${5:-4}; PORT=8090
PY=/c/Users/hp/anaconda3/python.exe
LOG=logs_cand_${NAME}_${SPLIT}.txt
tools/llama.cpp/llama-server.exe -m "$BASE" --lora "$ADAPTER" --port $PORT -np "$NP" -c $((1024 * NP)) -t 10 \
  --jinja --reasoning-budget 0 > "logs_server_${NAME}.txt" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
for i in $(seq 1 120); do curl -s http://127.0.0.1:$PORT/health | grep -q ok && break; sleep 2; done
$PY -I eval/run_llm.py --split "$SPLIT" --mode finetuned --prompt-version v2 --model "$NAME" \
  --base-url http://127.0.0.1:$PORT/v1 --name "$NAME" --workers "$NP" > "$LOG" 2>&1
# retry pass for any transport errors (run_llm only counts rows with real output as done)
$PY -I eval/run_llm.py --split "$SPLIT" --mode finetuned --prompt-version v2 --model "$NAME" \
  --base-url http://127.0.0.1:$PORT/v1 --name "$NAME" --workers "$NP" >> "$LOG" 2>&1
$PY -I eval/score.py --split "$SPLIT" "$NAME" | tail -1
