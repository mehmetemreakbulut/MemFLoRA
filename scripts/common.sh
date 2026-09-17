#!/usr/bin/env bash
# Shared setup and SG settings for the numbered experiment scripts.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python3}"
RESULTS_ROOT="${RESULTS_ROOT:-runs}"

SG_OPTIONS=(
    --bnpa-sg-p-lr 1e-2 --bnpa-sg-p-weight-decay 0 --bnpa-sg-p-optimizer adam
    --bnpa-sg-p-update-every 1 --bnpa-sg-p-warmup-steps 0 --bnpa-sg-renorm-p none
)
