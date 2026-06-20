#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-./.venv/bin/python}"
CONFIG="${CONFIG:-configs/lora_studies.yaml}"

"$PYTHON" -m fair_mia.cli doctor --config "$CONFIG"
"$PYTHON" -m fair_mia.cli plan --config "$CONFIG"
"$PYTHON" -m fair_mia.cli run-sweep --config "$CONFIG" "$@"
