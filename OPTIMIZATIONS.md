# Memory optimizations since the paper's code

The paper's results come from commit `0715b37`. The changes below came after it,
while measuring MemFLoRA on a Jetson Orin Nano. None of them changes the math:

- gradients are bitwise identical to that commit in all 74 method, backbone and
  input-gradient combinations (checked on the CPU, where the torch path runs);
- the saved tensors are the same, and the paper's own profiler still reproduces
  Table 3 exactly;
- the adaptation loop restores the same best step as before, with the same
  weights and metrics.

What they reduce is the working memory around training, which the paper's
accounting leaves out.

## Training code

1. **Backward passes free buffers early.** Every hand-written backward frees each
   full-width buffer as soon as it has been used, as PyTorch's own backward does:
   `torch.where` instead of a float mask, the adapter scale applied to the small
   results instead of a full-width copy, per-channel scaling and `grad_x`
   accumulation in place, and masks read through zero-copy views. This covers
   MemFLoRA, the frozen blocks, the bit-packed ReLU, and the fixed-P and
   LoRA-Edge baselines (`src/adapters/`).
2. **Forward passes free buffers early.** The adapter's output is released as soon
   as it has been added, and the frozen blocks apply BN and the activation in
   place in the convolution's output buffer.
3. **Fused activation and mask packing.** An optional CUDA kernel applies
   ReLU/ReLU6 in place and writes the packed gates in the same pass, so the
   full-size boolean mask never exists (`src/utils/csrc/`). The torch fallback
   packs without an int64 copy and unpacks with one allocation instead of two.
4. **Calibration cache released.** PyTorch's cached blocks are returned after AdaBN
   calibration (`experiments/benchmark_adabn.py`).
5. **Smaller best-step checkpoint.** Adaptation keeps only what it changes, the
   optimized parameters and the buffers, instead of a copy of the whole model
   (`experiments/benchmark_training.py`).

## Runtime settings in the Jetson profiler

6. **Small cuBLAS workspaces:** `CUBLAS_WORKSPACE_CONFIG=:16:8`.
7. **Expandable allocator segments:**
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
8. **AdaBN calibration off by default;** `--use-adabn` turns it on. Table 4 of the
   paper shows that calibration has a negligible effect.

## Measured effect

New tensors during one MemFLoRA training step at batch 64, on the CPU (1 and 2):

| | Paper code | Now |
|---|---|---|
| MobileNetV2, forward peak | 120.4 MB | 84.3 MB |
| MobileNetV2, backward peak | 175.2 MB | 84.1 MB |
| T-ResNet, forward peak | 16.1 MB | 14.1 MB |
| T-ResNet, backward peak | 19.1 MB | 10.5 MB |

Best-step checkpoint (5): MemFLoRA's copy shrinks from 9.27 to 0.22 MB on
MobileNetV2 and from 2.31 to 0.06 MB on T-ResNet. Full fine-tuning trains every
weight, so its copy stays the same.

cuBLAS workspaces (6): the two handles, one for the forward and one for the
backward thread, held 8.52 MB each, 17.04 MB together. They now hold 128 KiB each.

On the Jetson (MobileNetV2, rank 2, batch 64), items 1–3 with AdaBN off took
MemFLoRA's peak from 229.9 to 121.5 MB of GPU memory used and from 398.5 to
167.8 MB reserved; the earlier run had calibration on, which accounts for part of
the reserved drop. Full fine-tuning stayed at 699.3 MB used and 792.7 MB reserved.
Items 5–7 have not been measured on the Jetson yet.
