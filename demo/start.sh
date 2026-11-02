#!/usr/bin/env bash
# Download base GGUF + our LoRA adapter, start llama-server in the background, then the Gradio app.
set -e
mkdir -p /tmp/models
BASE_URL=${BASE_GGUF_URL:-https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf}
ADAPTER_URL=${ADAPTER_GGUF_URL:-}
curl -sSL -o /tmp/models/base.gguf "$BASE_URL" || echo "base download failed"
LORA_ARG=""
if [ -n "$ADAPTER_URL" ]; then curl -sSL -o /tmp/models/adapter.gguf "$ADAPTER_URL" && LORA_ARG="--lora /tmp/models/adapter.gguf"; fi
SERVER=$(find /opt/llama -name llama-server | head -1)
LD_LIBRARY_PATH=$(dirname "$SERVER"):$LD_LIBRARY_PATH nohup "$SERVER" -m /tmp/models/base.gguf $LORA_ARG --port 8080 -c 2048 -np 1 \
  --jinja --reasoning-budget 0 --alias crisislens > /tmp/llama.log 2>&1 &
python app.py
