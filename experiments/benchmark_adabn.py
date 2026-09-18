"""Calibrate target-domain BatchNorm statistics before adaptation."""

from __future__ import annotations

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments.benchmark_common import (
    BNPA_ADABN_AUDIT_METHODS,
    PreparedInputModel,
    method_uses_adabn,
)
from src.train import calibrate_adabn, freeze_bn_eval
from src.utils.adabn import (
    assert_batch_norm_eval_frozen_finite,
    first_tensor_from_batch,
)


def calibrate_minimal_adabn_if_needed(
    model: nn.Module,
    method: str,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Apply the method's target-calibration and BN-freezing policy."""
    minimal = method in BNPA_ADABN_AUDIT_METHODS
    uses_adabn = minimal or method_uses_adabn(method)
    if args.adabn_calib_batches == 0 and uses_adabn:
        model.train()
        freeze_bn_eval(model)
        return
    if not minimal:
        if not uses_adabn:
            return
        # Official calibration uses its full loader, as in the paper runs.
        calibrate_adabn(
            PreparedInputModel(model, args.backbone),
            loader,
            device=device,
            calibration_mode=args.adabn_calibration_mode,
        )
        release_cached_memory(device)
        return

    model.eval()
    set_eval_adabn_stat_collection(model, enabled=True)
    wrapped = PreparedInputModel(model, args.backbone).to(device)
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if (
                    args.adabn_calib_batches is not None
                    and batch_index >= args.adabn_calib_batches
                ):
                    break
                inputs = first_tensor_from_batch(batch).to(device, non_blocking=True)
                wrapped(inputs)
    finally:
        set_eval_adabn_stat_collection(model, enabled=False)
        model.train()
        freeze_bn_eval(model)
    assert_batch_norm_eval_frozen_finite(model)
    release_cached_memory(device)


def release_cached_memory(device: torch.device) -> None:
    """Return the blocks the calibration pass left in PyTorch's cache to the device.

    Training never reuses most of them, and the allocator would keep them reserved.
    """
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


def set_eval_adabn_stat_collection(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        setter = getattr(module, "set_eval_adabn_stat_collection", None)
        if setter is not None:
            setter(enabled, None)
