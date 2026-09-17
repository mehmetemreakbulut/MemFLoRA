#!/usr/bin/env bash
# 06 - RealWorld + MobileNetV2
# Reference output shipped in: results/main_accuracy/realworld_mobilenet_v2/
source "$(dirname "$0")/common.sh"

"$PYTHON" experiments/minimal_methods_benchmark.py \
    --dataset realworld --data-root data/realworld \
    --backbone mobilenet_v2 --mobilenet-v2-pretrained true --width-mult 1.0 \
    --method zero_shot full bnpa_q3_sg bnpa_fa_postbn_bnr_off_scaled lora_edge_optimized lora_c bn_tuning bias_tuning \
    --rank 2 4 8 \
    --target-locations all \
    --realworld-feature-set imu9 \
    --realworld-split-strategy chronological \
    --window-size 500 --window-stride 250 \
    --window-label-rule majority \
    --pretrain-epochs 20 --steps-adapt 50 --batch-size 64 \
    --seed {1..20} \
    --reuse-first-source-model true \
    --adapt-eval-every-steps 50 \
    --adapter-layers pointwise_only \
    --adabn-calibration-mode ema_no_reset \
    --adabn-calib-batches 1 \
    --proj-init random --bnpa-bottleneck-bn on \
    --profile-full-sram false \
    "${SG_OPTIONS[@]}" \
    --adabn-stat-source target --train_mode_adaBN off --adapt-lr 1e-3 \
    --results-root "$RESULTS_ROOT"
