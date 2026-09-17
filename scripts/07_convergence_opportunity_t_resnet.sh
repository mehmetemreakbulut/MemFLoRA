#!/usr/bin/env bash
# 07 - Convergence, T-ResNet
# Reference output shipped in: results/convergence/opportunity_t_resnet/
source "$(dirname "$0")/common.sh"

"$PYTHON" experiments/minimal_methods_benchmark.py \
    --dataset opportunity --data-root data/opportunity \
    --backbone t_resnet_official --t-resnet-feature-maps 64 \
    --method zero_shot full bnpa bnpa_sg bnpa_q1 bnpa_q3 bnpa_q3_sg bnpa_fa_postbn_bnr_off_scaled fixed_custom_bitmask_adabn_q3matched lora_edge_optimized lora_c bn_tuning bias_tuning bnpa_q3init bnpa_sg_q3init \
    --rank 2 4 8 16 \
    --target-subjects all \
    --window-size 60 --window-stride 30 \
    --label-column ml_both_arms --window-label-rule majority \
    --pretrain-epochs 20 --steps-adapt 50 100 150 200 250 --batch-size 64 \
    --seed {1..5} \
    --reuse-first-source-model true \
    --adapt-eval-every-steps 50 \
    --adapter-layers all \
    --adabn-calibration-mode ema_no_reset \
    --proj-init random --bnpa-bottleneck-bn on \
    --profile-full-sram false \
    "${SG_OPTIONS[@]}" \
    --adabn-stat-source target --train_mode_adaBN off --adapt-lr 1e-3 \
    --results-root "$RESULTS_ROOT"
