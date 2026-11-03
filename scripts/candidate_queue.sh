#!/usr/bin/env bash
# Download + convert several AutoScientist checkpoints in parallel (the platform streams each file at
# ~50-100 KB/s with no Range support, so parallel streams are faster), then evaluate each candidate on
# val_eval as soon as it is converted and the CPU is free (one evaluation at a time).
#   bash scripts/candidate_queue.sh
cd "$(dirname "$0")/.."
PY=/c/Users/hp/anaconda3/python.exe
# key | run id | HF base for conversion | base GGUF | prompt version
CANDS="ft2b_run8|fab5f195-3c03-4f96-ac63-fb31f6693054|Qwen/Qwen3.5-2B|tools/gguf/Qwen3.5-2B-Q4_K_M.gguf|v2
ft2b_run4|25d04b86-af94-45cf-be37-5acc5822614c|Qwen/Qwen3.5-2B|tools/gguf/Qwen3.5-2B-Q4_K_M.gguf|v1
ft4b_run6|fafc81b1-8876-43c1-80e9-75a1791d676c|Qwen/Qwen3.5-4B|tools/gguf/Qwen3.5-4B-Q4_K_M.gguf|v2"

fetch() {  # download (retrying from scratch: no Range support) + convert; marker file when done
  local key=$1 run=$2 hf=$3 log=logs_dl_${1}.txt
  [ -f models/$run/.converted ] && return
  for k in 1 2 3 4 5 6; do
    $PY -I src/train/autoscientist_run.py download --run "$run" --fresh >> "$log" 2>&1 && break
    echo "$(date +%T) download attempt $k failed" >> "$log"; sleep 10
  done
  [ -f models/$run/adapter_model.safetensors ] || { echo "$(date +%T) GAVE UP downloading" >> "$log"; return; }
  # conversion is CPU-heavy but short; serialize it so two torch processes never run at once
  while ! mkdir .convert.lock 2>/dev/null; do sleep 5; done
  bash src/serve/lora_to_gguf.sh "models/$run" "$hf" >> "$log" 2>&1 && touch "models/$run/.converted"
  rmdir .convert.lock
  echo "$(date +%T) CONVERTED $key" >> "$log"
}

while IFS='|' read -r key run hf base pv; do fetch "$key" "$run" "$hf" & sleep 2; done <<< "$CANDS"

# evaluation loop: CPU is free once run #7's val score exists
until [ -f eval/results/ft2b_run7__val_eval.json ]; do sleep 30; done
todo="$CANDS"
while [ -n "$todo" ]; do
  picked=""
  while IFS='|' read -r key run hf base pv; do
    [ -z "$key" ] && continue
    if [ -f "models/$run/.converted" ]; then picked="$key|$run|$hf|$base|$pv"; break; fi
  done <<< "$todo"
  if [ -z "$picked" ]; then
    alive=$(jobs -r | wc -l)
    [ "$alive" -eq 0 ] && { echo "$(date +%T) no more candidates can be converted; stopping"; break; }
    sleep 30; continue
  fi
  IFS='|' read -r key run hf base pv <<< "$picked"
  echo "$(date +%T) EVAL $key"
  bash scripts/eval_cand.sh "$key" "$base" "models/$run/adapter-f16.gguf" val_eval 4 "$pv"
  todo=$(grep -v "^$key|" <<< "$todo")
done
wait
echo "$(date +%T) QUEUE DONE"
