#!/usr/bin/env bash
# 00 - Smoke test (minutes, not days)
# Pipeline check only; its accuracy is not a paper result.
RESULTS_ROOT="${RESULTS_ROOT:-runs/smoke}"
source "$(dirname "$0")/common.sh"

"$PYTHON" experiments/minimal_methods_benchmark.py \
    --dataset opportunity --data-root data/opportunity \
    --backbone t_resnet_official --t-resnet-feature-maps 64 \
    --method zero_shot bnpa_fa_postbn_bnr_off_scaled \
    --rank 2 \
    --target-subjects 1 \
    --window-size 60 --window-stride 30 \
    --label-column ml_both_arms --window-label-rule majority \
    --pretrain-epochs 1 --steps-adapt 5 --batch-size 64 \
    --seed 1 \
    --reuse-first-source-model true \
    --adapt-eval-every-steps 5 \
    --adapter-layers all \
    --adabn-calibration-mode ema_no_reset \
    --adabn-calib-batches 1 \
    --proj-init random --bnpa-bottleneck-bn on \
    --profile-full-sram false \
    --adabn-stat-source target --train_mode_adaBN off --adapt-lr 1e-3 \
    --results-root "$RESULTS_ROOT"

echo
echo "smoke test finished; output under $RESULTS_ROOT/"
