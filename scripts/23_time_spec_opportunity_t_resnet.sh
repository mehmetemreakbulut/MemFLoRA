#!/usr/bin/env bash
# 23 - Time-spec / threshold, T-ResNet
# Reference output shipped in: results/time_spec/opportunity_t_resnet/
source "$(dirname "$0")/common.sh"

"$PYTHON" experiments/minimal_methods_benchmark.py \
    --dataset opportunity --data-root data/opportunity \
    --backbone t_resnet_official --t-resnet-feature-maps 64 \
    --method zero_shot full bn_tuning bias_tuning bnpa_fa_postbn_bnr_off_scaled bnpa_q3_sg lora_c lora_edge_optimized lora_edge_optimized_v2 bnpa_fa_postbn_bnr_off_optimized \
    --rank 2 \
    --target-subjects all \
    --window-size 60 --window-stride 30 \
    --label-column ml_both_arms --window-label-rule majority \
    --pretrain-epochs 20 --steps-adapt 50 --batch-size 64 \
    --seed {1..5} \
    --reuse-first-source-model true \
    --adapt-eval-every-steps 1 \
    --adapter-layers all \
    --adabn-calibration-mode ema_no_reset \
    --adabn-calib-batches 1 \
    --proj-init random --bnpa-bottleneck-bn on \
    --profile-full-sram false \
    --adabn-stat-source target --train_mode_adaBN off --adapt-lr 1e-3 \
    --profile-time-spec true \
    --results-root "$RESULTS_ROOT"
