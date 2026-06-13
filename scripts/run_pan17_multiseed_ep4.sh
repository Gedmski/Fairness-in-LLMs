#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -x ".venv-cu124/bin/python" ]; then
  echo ".venv-cu124/bin/python not found. Activate the VM environment first." >&2
  exit 1
fi

PYTHON="./.venv-cu124/bin/python"
SEEDS=(29 41 53)

"$PYTHON" -m fair_mia.cli cache-model \
  --model-id EleutherAI/pythia-160m \
  --cache-dir artifacts/models

for seed in "${SEEDS[@]}"; do
  adapter_dir="artifacts/adapters/pythia_160m_lora_seed${seed}_ep4"
  config_path="configs/vm_finetune_lora_pythia_160m_seed${seed}_ep4.json"

  echo "==> Seed ${seed}: fine-tuning into ${adapter_dir}"
  "$PYTHON" -m fair_mia.cli finetune-lora \
    --base-model-id EleutherAI/pythia-160m \
    --train-jsonl data/pan_demo/members.jsonl \
    --output-dir "${adapter_dir}" \
    --cache-dir artifacts/models \
    --max-train-samples 1000 \
    --epochs 4 \
    --max-length 256 \
    --seed "${seed}"

  echo "==> Seed ${seed}: benchmark run ${config_path}"
  "$PYTHON" -m fair_mia.cli run --config "${config_path}"
done

echo "Completed PAN 2017 multi-seed 4-epoch sweep."
