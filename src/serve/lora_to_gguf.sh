#!/usr/bin/env bash
# Convert a downloaded AutoScientist LoRA checkpoint (PEFT format) to a llama.cpp GGUF adapter.
# usage: bash src/serve/lora_to_gguf.sh models/<run_id> Qwen/Qwen3.5-2B
set -euo pipefail
RUN_DIR=$1; BASE_ID=$2
ADIR=$(dirname "$(find "$RUN_DIR" -name adapter_config.json | head -1)")
echo "adapter dir: $ADIR"
KMP_DUPLICATE_LIB_OK=TRUE /c/Users/hp/anaconda3/python.exe tools/llama-src/convert_lora_to_gguf.py "$ADIR" \
  --base-model-id "$BASE_ID" --outtype f16 --outfile "$RUN_DIR/adapter-f16.gguf"
ls -la "$RUN_DIR/adapter-f16.gguf"
