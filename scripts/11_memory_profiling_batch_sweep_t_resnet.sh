#!/usr/bin/env bash
# 11 - Batch-size SRAM sweep, T-ResNet
# Reference output shipped in: results/memory_profiling/batch_sweep_t_resnet/
source "$(dirname "$0")/common.sh"

"$PYTHON" experiments/minimal_methods_benchmark.py \
    --dataset opportunity --data-root data/opportunity \
    --backbone t_resnet_official --t-resnet-feature-maps 64 \
    --method zero_shot full bn_tuning bias_tuning bnpa_fa_postbn_bnr_off_scaled bnpa_q3_sg lora_c lora_edge_optimized \
    --rank 2 4 8 \
    --target-subjects 1 \
    --window-size 60 --window-stride 30 \
    --label-column ml_both_arms --window-label-rule majority \
    --pretrain-epochs 1 --steps-adapt 1 --batch-size 1 8 32 64 128 \
    --seed 1 \
    --reuse-first-source-model false \
    --adapt-eval-every-steps 50 \
    --adapter-layers all \
    --adabn-calibration-mode ema_no_reset \
    --adabn-calib-batches 1 \
    --proj-init random --bnpa-bottleneck-bn on \
    --profile-full-sram true \
    --adabn-stat-source target --train_mode_adaBN off --adapt-lr 1e-3 \
    --results-root "$RESULTS_ROOT"
