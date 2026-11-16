#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
ADAPTER_REPO="${ADAPTER_REPO:-}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

ARGS=(
  "$MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --max-model-len "$MAX_MODEL_LEN"
)

if [[ -n "$ADAPTER_REPO" ]]; then
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ADAPTER_REPO=$ADAPTER_REPO requires HF_TOKEN (adapter repos are private)." >&2
    exit 1
  fi
  ADAPTER_DIR="/adapters/current"
  export ADAPTER_DIR
  mkdir -p "$ADAPTER_DIR"
  echo "Downloading adapter $ADAPTER_REPO to $ADAPTER_DIR ..."
  # Read ADAPTER_REPO / ADAPTER_DIR from the environment rather than
  # interpolating into a Python string literal: a repo id containing a single
  # quote would otherwise corrupt the expression (or, with a hostile value,
  # execute arbitrary Python in the container).
  python -c "import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id=os.environ['ADAPTER_REPO'], local_dir=os.environ['ADAPTER_DIR'])"
  ARGS+=(
    --enable-lora
    --lora-modules "adapter=$ADAPTER_DIR"
    --max-lora-rank "$MAX_LORA_RANK"
  )
fi

exec vllm serve "${ARGS[@]}"
