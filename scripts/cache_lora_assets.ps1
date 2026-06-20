$ErrorActionPreference = "Stop"

$Python = if ($env:PYTHON) { $env:PYTHON } else { ".\.venv\Scripts\python.exe" }
$CacheDir = if ($env:CACHE_DIR) { $env:CACHE_DIR } else { "artifacts\models" }

& $Python -m fair_mia.cli cache-model --model-id Qwen/Qwen3-4B-Base --cache-dir $CacheDir
& $Python -m fair_mia.cli cache-model --model-id allenai/OLMo-2-0425-1B --cache-dir $CacheDir
& $Python -m fair_mia.cli cache-model --model-id FacebookAI/xlm-roberta-base --task masked --cache-dir $CacheDir

if ($env:HF_TOKEN -or $env:HUGGING_FACE_HUB_TOKEN) {
  & $Python -m fair_mia.cli cache-model --model-id google/gemma-3-4b-pt --cache-dir $CacheDir
} else {
  Write-Warning "Skipping gated Gemma cache: set HF_TOKEN after accepting the model license."
}

Write-Host "LoRA experiment model assets are cached under $CacheDir."
