param(
  [string]$Study,
  [switch]$Resume,
  [switch]$RetryFailed
)

$ErrorActionPreference = "Stop"

$Python = if ($env:PYTHON) { $env:PYTHON } else { ".\.venv\Scripts\python.exe" }
$Config = if ($env:CONFIG) { $env:CONFIG } else { "configs\lora_studies.yaml" }

& $Python -m fair_mia.cli doctor --config $Config
& $Python -m fair_mia.cli plan --config $Config

$Arguments = @("-m", "fair_mia.cli", "run-sweep", "--config", $Config)
if ($Study) { $Arguments += @("--study", $Study) }
if ($Resume) { $Arguments += "--resume" }
if ($RetryFailed) { $Arguments += "--retry-failed" }
& $Python @Arguments
