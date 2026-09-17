#!/usr/bin/env bash
# 17 - AdaBN on/off ablation
# Reference output shipped in: results/ablations/adabn_on_off/
source "$(dirname "$0")/common.sh"

"$PYTHON" experiments/minimal_methods_benchmark.py \
    --dataset opportunity --data-root data/opportunity \
    --backbone t_resnet_official --t-resnet-feature-maps 64 \
    --method bnpa_fa_postbn_bnr_off_scaled \
    --rank 2 4 8 \
    --target-subjects all \
    --window-size 60 --window-stride 30 \
    --label-column ml_both_arms --window-label-rule majority \
    --pretrain-epochs 20 --steps-adapt 50 --batch-size 64 \
    --seed {1..10} \
    --reuse-first-source-model true \
    --adapt-eval-every-steps 50 \
    --adapter-layers all \
    --adabn-calibration-mode ema_no_reset \
    --adabn-calib-batches 0 1 \
    --proj-init random --bnpa-bottleneck-bn on \
    --profile-full-sram false \
    --adabn-stat-source target --train_mode_adaBN off --adapt-lr 1e-3 \
    --results-root "$RESULTS_ROOT"
