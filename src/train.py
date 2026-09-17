from __future__ import annotations

from typing import Optional, Union

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.utils.adabn import (
    ADABN_CALIBRATION_MODES,
    assert_batch_norm_eval_frozen_finite,
    batch_norm_modules,
    enable_fused_adabn_calibration,
    first_tensor_from_batch,
    freeze_batch_norm_eval as freeze_bn_eval,
    reset_batch_norm_running_stats,
)


@torch.no_grad()
def calibrate_adabn(
    model: nn.Module,
    loader: DataLoader,
    device: Union[torch.device, str] = "cpu",
    num_batches: Optional[int] = None,
    calibration_mode: str = "ema_reset",
) -> None:
    """Estimate target-domain BN running statistics with a forward-only pass."""

    if calibration_mode not in ADABN_CALIBRATION_MODES:
        raise ValueError(
            f"Unknown AdaBN calibration_mode={calibration_mode!r}; "
            f"expected one of {ADABN_CALIBRATION_MODES}"
        )
    device = torch.device(device)
    model.to(device)
    if calibration_mode == "ema_reset":
        reset_batch_norm_running_stats(model)

    model.eval()
    enable_fused_adabn_calibration(model, True, calibration_mode=calibration_mode)
    for module in batch_norm_modules(model):
        module.train()
        if module.weight is not None:
            module.weight.requires_grad_(False)
        if module.bias is not None:
            module.bias.requires_grad_(False)

    try:
        for batch_index, batch in enumerate(loader):
            if num_batches is not None and batch_index >= num_batches:
                break
            inputs = first_tensor_from_batch(batch).to(device)
            model(inputs)
    finally:
        enable_fused_adabn_calibration(model, False)
        model.train()
        freeze_bn_eval(model)

    assert_batch_norm_eval_frozen_finite(model)
