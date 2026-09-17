from __future__ import annotations
import torch
from torch import nn

BatchNormTypes = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
ADABN_CALIBRATION_MODES = ("ema_reset", "ema_no_reset")


def is_adabn_batch_norm(module: nn.Module) -> bool:
    return isinstance(module, BatchNormTypes) and not bool(
        getattr(module, "_exclude_from_adabn", False)
    )


def reset_batch_norm_running_stats(module: nn.Module) -> None:
    """Reset BatchNorm running statistics."""
    for child in batch_norm_modules(module):
        if child.running_mean is not None:
            child.running_mean.zero_()
        if child.running_var is not None:
            child.running_var.fill_(1.0)
        if getattr(child, "num_batches_tracked", None) is not None:
            child.num_batches_tracked.zero_()


def freeze_batch_norm_eval(module: nn.Module) -> None:
    """Force BatchNorm modules to eval mode and apply the affine-freezing policy."""
    for child in batch_norm_modules(module):
        child.eval()
        if child.weight is not None:
            child.weight.requires_grad_(False)
        if child.bias is not None:
            child.bias.requires_grad_(
                bool(getattr(child, "_keep_bias_trainable_in_eval", False))
            )


def enable_fused_adabn_calibration(
    module: nn.Module, enabled: bool, calibration_mode: str = "ema_reset"
) -> None:
    """Toggle AdaBN calibration mode on custom fused modules that expose it."""
    for child in module.modules():
        setter = getattr(child, "set_adabn_calibration", None)
        if setter is None:
            continue
        mode_setter = getattr(child, "set_adabn_calibration_mode", None)
        if mode_setter is not None:
            mode_setter(calibration_mode)
        setter(enabled)


def batch_norm_train_output_from_preactivation(
    preactivation: torch.Tensor, batch_norm: nn.modules.batchnorm._BatchNorm
) -> torch.Tensor:
    """Update BN buffers from ``preactivation`` and return train-mode BN output.

    The running-stat update follows the policy requested for AdaBN in this
    repository: per-channel mean/variance are computed over all non-channel
    dimensions with ``unbiased=False`` and then applied with the module's
    momentum, or cumulative averaging when ``momentum is None``.
    """
    if batch_norm.running_mean is None or batch_norm.running_var is None:
        raise RuntimeError("AdaBN calibration requires BatchNorm running statistics")
    if preactivation.ndim < 2:
        raise ValueError(
            f"Expected activation with channel dimension, got shape={tuple(preactivation.shape)}"
        )
    reduce_dims = tuple(dim for dim in range(preactivation.ndim) if dim != 1)
    batch_mean = preactivation.mean(dim=reduce_dims)
    batch_var = preactivation.var(dim=reduce_dims, unbiased=False)
    update_batch_norm_running_stats(batch_norm, batch_mean.detach(), batch_var.detach())
    view_shape = [1] * preactivation.ndim
    view_shape[1] = -1
    mean = batch_mean.to(device=preactivation.device, dtype=preactivation.dtype).view(
        *view_shape
    )
    var = batch_var.to(device=preactivation.device, dtype=preactivation.dtype).view(
        *view_shape
    )
    if batch_norm.affine:
        gamma = batch_norm.weight.to(
            device=preactivation.device, dtype=preactivation.dtype
        ).view(*view_shape)
        beta = batch_norm.bias.to(
            device=preactivation.device, dtype=preactivation.dtype
        ).view(*view_shape)
    else:
        gamma = torch.ones_like(mean)
        beta = torch.zeros_like(mean)
    return (preactivation - mean) * torch.rsqrt(var + batch_norm.eps) * gamma + beta


def update_batch_norm_running_stats(
    batch_norm: nn.modules.batchnorm._BatchNorm,
    batch_mean: torch.Tensor,
    batch_var: torch.Tensor,
) -> None:
    with torch.no_grad():
        if getattr(batch_norm, "num_batches_tracked", None) is not None:
            batch_norm.num_batches_tracked.add_(1)
            batches = int(batch_norm.num_batches_tracked.item())
        else:
            batches = 1
        momentum = (
            1.0 / float(max(1, batches))
            if batch_norm.momentum is None
            else float(batch_norm.momentum)
        )
        running_mean = batch_norm.running_mean
        running_var = batch_norm.running_var
        if running_mean is None or running_var is None:
            raise RuntimeError(
                "AdaBN calibration requires BatchNorm running statistics"
            )
        mean = batch_mean.to(device=running_mean.device, dtype=running_mean.dtype)
        var = batch_var.to(device=running_var.device, dtype=running_var.dtype)
        running_mean.mul_(1.0 - momentum).add_(mean, alpha=momentum)
        running_var.mul_(1.0 - momentum).add_(var, alpha=momentum)


def batch_norm_modules(module: nn.Module):
    for child in module.modules():
        if is_adabn_batch_norm(child):
            yield child


def assert_batch_norm_eval_frozen_finite(module: nn.Module) -> None:
    violations = []
    for name, child in module.named_modules():
        if not is_adabn_batch_norm(child):
            continue
        if child.training:
            violations.append(f"{name}: training=True")
        if child.weight is not None and child.weight.requires_grad:
            violations.append(f"{name}.weight requires_grad=True")
        if (
            child.bias is not None
            and child.bias.requires_grad
            and not bool(getattr(child, "_keep_bias_trainable_in_eval", False))
        ):
            violations.append(f"{name}.bias requires_grad=True")
        if (
            child.running_mean is not None
            and not torch.isfinite(child.running_mean).all()
        ):
            violations.append(f"{name}.running_mean non-finite")
        if (
            child.running_var is not None
            and not torch.isfinite(child.running_var).all()
        ):
            violations.append(f"{name}.running_var non-finite")
    if violations:
        raise RuntimeError(
            "BatchNorm AdaBN sanity check failed: " + "; ".join(violations)
        )


def first_tensor_from_batch(batch) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, dict):
        for key in ("inputs", "input", "x", "data"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                return value
        raise TypeError("Could not extract input tensor from batch dict")
    if isinstance(batch, (tuple, list)) and batch:
        if isinstance(batch[0], torch.Tensor):
            return batch[0]
    raise TypeError(
        f"Could not extract input tensor from batch of type {type(batch)!r}"
    )


def snapshot_running_buffers(
    model: nn.Module,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Save all BN-style buffers, including those excluded from AdaBN calibration."""
    snapshots = []
    for module in model.modules():
        for attr in ("running_mean", "running_var", "num_batches_tracked"):
            tensor = getattr(module, attr, None)
            if isinstance(tensor, torch.Tensor):
                snapshots.append((tensor, tensor.detach().clone()))
    return snapshots


def restore_running_buffers(snapshots: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
    for tensor, value in snapshots:
        tensor.copy_(value)
