$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
  python -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[research]"
try {
  .\.venv\Scripts\python.exe -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('gpu_count=', torch.cuda.device_count()); [print('gpu', i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
} catch {
  Write-Warning "GPU probe warning: PyTorch could not fully initialize CUDA; continuing with setup."
}
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe -m fair_mia.cli doctor --config configs/lora_studies.yaml

Write-Host "VM setup complete. Review doctor output before caching models or starting studies."
