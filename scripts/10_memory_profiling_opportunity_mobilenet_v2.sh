#!/usr/bin/env bash
# 10 - SRAM + performance, MobileNetV2
# Reference output shipped in: results/memory_profiling/opportunity_mobilenet_v2/
source "$(dirname "$0")/common.sh"

"$PYTHON" experiments/minimal_methods_benchmark.py \
    --dataset opportunity --data-root data/opportunity \
    --backbone mobilenet_v2 --mobilenet-v2-pretrained true --width-mult 1.0 \
    --method zero_shot full bnpa_q3 bnpa_q3_sg bnpa_fa_postbn_bnr_off_scaled lora_edge_optimized lora_c bn_tuning bias_tuning \
    --rank 2 4 8 \
    --target-subjects 1 \
    --window-size 60 --window-stride 30 \
    --label-column ml_both_arms --window-label-rule majority \
    --pretrain-epochs 20 --steps-adapt 1 --batch-size 64 \
    --seed 1 \
    --reuse-first-source-model true \
    --adapt-eval-every-steps 1 \
    --adapter-layers pointwise_only \
    --adabn-calibration-mode ema_no_reset \
    --proj-init random --bnpa-bottleneck-bn on \
    --profile-full-sram true \
    "${SG_OPTIONS[@]}" \
    --adabn-stat-source target --train_mode_adaBN off --adapt-lr 1e-3 \
    --adabn-calib-batches 1 16 \
    --profile-performance true \
    --results-root "$RESULTS_ROOT"
