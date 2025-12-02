#!/usr/bin/env bash
set -euo pipefail

# 仅在未激活时再激活
env="smart-turn"
env="vad"
if [[ -z "${CONDA_PREFIX:-}" || "$(basename "$CONDA_PREFIX")" != "$env" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "$env"
fi
echo " * Using conda env: $CONDA_PREFIX"

python server.py