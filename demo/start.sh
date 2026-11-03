#!/usr/bin/env bash
# HF Space entrypoint. The web console starts immediately (cached outputs); the model is fetched and
# served in the background, and /api/health flips to "live" once llama-server answers.
mkdir -p /tmp/models
BASE_URL=${BASE_GGUF_URL:-https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf}
ADAPTER_URL=${ADAPTER_GGUF_URL:-}
(
  curl -sSL --retry 3 -o /tmp/models/base.gguf "$BASE_URL" || { echo "base download failed"; exit 0; }
  LORA_ARG=""
  if [ -n "$ADAPTER_URL" ]; then
    curl -sSL --retry 3 -o /tmp/models/adapter.gguf "$ADAPTER_URL" && LORA_ARG="--lora /tmp/models/adapter.gguf"
  fi
  SERVER=$(find /opt/llama -name llama-server | head -1)
  LD_LIBRARY_PATH=$(dirname "$SERVER"):$LD_LIBRARY_PATH exec "$SERVER" -m /tmp/models/base.gguf $LORA_ARG \
    --host 127.0.0.1 --port 8080 -c 2048 -np 1 --jinja --reasoning-budget 0 --alias crisislens
) > /tmp/llama.log 2>&1 &
exec python -m uvicorn server:app --host 0.0.0.0 --port 7860
