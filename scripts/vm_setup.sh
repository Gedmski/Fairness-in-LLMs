#!/usr/bin/env bash
set -euo pipefail

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -e ".[research]"
./.venv/bin/python - <<'PY' || echo "GPU probe warning: PyTorch could not fully initialize CUDA; continuing with setup."
import torch
print("cuda_available=", torch.cuda.is_available())
print("gpu_count=", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print("gpu", i, torch.cuda.get_device_name(i))
PY
./.venv/bin/python -m unittest discover -s tests
./.venv/bin/python -m fair_mia.cli doctor --config configs/lora_studies.yaml

echo "VM setup complete. Review doctor output before caching models or starting studies."
