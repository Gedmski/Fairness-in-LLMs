#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-./.venv/bin/python}"
CACHE_DIR="${CACHE_DIR:-artifacts/models}"

"$PYTHON" -m fair_mia.cli cache-model --model-id Qwen/Qwen3-4B-Base --cache-dir "$CACHE_DIR"
"$PYTHON" -m fair_mia.cli cache-model --model-id allenai/OLMo-2-0425-1B --cache-dir "$CACHE_DIR"
"$PYTHON" -m fair_mia.cli cache-model --model-id FacebookAI/xlm-roberta-base --task masked --cache-dir "$CACHE_DIR"

if [ -n "${HF_TOKEN:-}" ] || [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  "$PYTHON" -m fair_mia.cli cache-model --model-id google/gemma-3-4b-pt --cache-dir "$CACHE_DIR"
else
  echo "Skipping gated Gemma cache: set HF_TOKEN after accepting the model license."
fi

echo "LoRA experiment model assets are cached under $CACHE_DIR."
