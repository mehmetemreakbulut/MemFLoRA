"""Helpers shared by the adapter modules."""

from __future__ import annotations

from typing import NamedTuple, Literal, Optional, Tuple

import torch
from torch import nn

from src.utils.adabn import update_batch_norm_running_stats
from src.utils.bitpack import activate_and_pack_, pack_bool_mask, unpack_bool_mask

ActivationMaskMode = Literal["bool", "bitpack"]


class ConvSpec(NamedTuple):
    """Geometry of a Conv2d: everything F.conv2d needs besides the weights."""

    stride: Tuple[int, int]
    padding: Tuple[int, int]
    dilation: Tuple[int, int]
    groups: int


def conv_spec(conv: nn.Conv2d) -> ConvSpec:
    return ConvSpec(
        _as_2tuple(conv.stride),
        _as_2tuple(conv.padding),
        _as_2tuple(conv.dilation),
        int(conv.groups),
    )


def _activation_mask_mode_code(mode: ActivationMaskMode) -> int:
    if mode == "bool":
        return 0
    if mode == "bitpack":
        return 1
    raise ValueError(f"Unknown activation_mask_mode: {mode}")


def _as_2tuple(value) -> Tuple[int, int]:
    if isinstance(value, tuple):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _channel_view(vector: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    shape = [1] * reference.ndim
    shape[1] = -1
    return vector.view(*shape)


def _needs_grouped_projection(base_conv: nn.Conv2d) -> bool:
    return (
        base_conv.groups != 1
        or _as_2tuple(base_conv.kernel_size) != (1, 1)
        or _as_2tuple(base_conv.stride) != (1, 1)
        or _as_2tuple(base_conv.padding) != (0, 0)
        or _as_2tuple(base_conv.dilation) != (1, 1)
    )


def _prepare_activation_mask(
    mask: torch.Tensor, activation_mask_mode: int
) -> torch.Tensor:
    if activation_mask_mode == 0:
        return mask.to(torch.uint8)
    if activation_mask_mode == 1:
        packed, _shape = pack_bool_mask(mask)
        return packed
    raise ValueError(f"Unknown activation mask mode code: {activation_mask_mode}")


def _restore_activation_mask(
    activation_mask: torch.Tensor,
    activation_mask_mode: int,
    activation_mask_shape: Tuple[int, ...],
) -> torch.Tensor:
    if activation_mask_mode == 0:
        return activation_mask.view(torch.bool)  # stored as 0/1 bytes, no copy
    if activation_mask_mode == 1:
        return unpack_bool_mask(activation_mask, torch.Size(activation_mask_shape))
    raise ValueError(f"Unknown activation mask mode code: {activation_mask_mode}")


# The hand-written backward passes free each full-width intermediate, or overwrite
# it in place, as soon as the next operation has used it, as PyTorch's do.
def activation_grad(
    grad_output: torch.Tensor,
    activation_mask: torch.Tensor,
    activation: int,
    activation_mask_mode: int,
    activation_mask_shape: Tuple[int, ...],
) -> torch.Tensor:
    """The gradient before the activation: `grad_output` where it passed, else 0.

    Without an activation this is `grad_output` itself; otherwise it is a new
    tensor the caller owns and may overwrite. The unpacked mask is freed on return.
    """
    if activation == 0:
        return grad_output
    mask = _restore_activation_mask(
        activation_mask, activation_mask_mode, activation_mask_shape
    )
    return torch.where(mask, grad_output, 0)


def scale_channels(
    grad: torch.Tensor, scale: torch.Tensor, grad_output: torch.Tensor
) -> torch.Tensor:
    """`grad` times a per-channel scale, in place unless it is `grad_output`.

    Autograd may share `grad_output` with other nodes, so it is never overwritten.
    """
    view = _channel_view(scale, grad)
    return grad * view if grad is grad_output else grad.mul_(view)


def stash_fused_geometry(
    ctx,
    x: torch.Tensor,
    u_weight: torch.Tensor,
    p: ConvSpec,
    u: ConvSpec,
    base: ConvSpec,
    scale: float,
    activation: int,
) -> None:
    """Record on `ctx` the shapes and conv geometry every fused backward re-derives."""
    ctx.input_shape = tuple(x.shape)
    ctx.u_weight_shape = tuple(u_weight.shape)
    ctx.p = p
    ctx.u = u
    ctx.base = base
    ctx.scale = float(scale)
    ctx.activation = int(activation)


def fused_activation_inplace(
    ctx, s: torch.Tensor, activation: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply a fused activation in place and bit-pack the mask its backward needs.

    Records the unpacked mask shape on `ctx` and returns the activated tensor
    alongside the packed mask. `s` is a freshly computed pre-activation, so
    overwriting it keeps the saved-tensor footprint at one activation per site.
    """
    ctx.activation_mask_shape = tuple()
    if activation == 0:
        return s, torch.empty(0, device=s.device, dtype=torch.uint8)
    if activation not in (1, 2):
        raise NotImplementedError(f"Unsupported activation code: {activation}")
    ctx.activation_mask_shape = tuple(s.shape)
    return s, activate_and_pack_(s, activation)


def _activation_code(activation: Optional[nn.Module]) -> int:
    if activation is None or isinstance(activation, nn.Identity):
        return 0
    if isinstance(activation, nn.ReLU):
        return 1
    if isinstance(activation, nn.ReLU6):
        return 2
    raise NotImplementedError(
        "fused adapters support only None/Identity, ReLU, and ReLU6"
    )


def make_projection_conv(
    base_conv: nn.Conv2d,
    in_channels: int,
    out_channels: int,
    spatial: bool,
    groups: int,
    bias: bool = False,
    **factory,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=base_conv.kernel_size if spatial else 1,
        stride=base_conv.stride if spatial else 1,
        padding=base_conv.padding if spatial else 0,
        dilation=base_conv.dilation if spatial else 1,
        groups=groups,
        bias=bias,
        **factory,
    )


class AdapterBlockCommon:
    """Channel accessors, activation replay, folded-BN stats and AdaBN plumbing."""

    @property
    def in_channels(self) -> int:
        return self.base_conv.in_channels

    @property
    def out_channels(self) -> int:
        return self.base_conv.out_channels

    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_type == 0:
            return x
        if self.activation_type == 1:
            return torch.relu(x)
        if self.activation_type == 2:
            return torch.clamp(x, min=0, max=6)
        raise NotImplementedError(
            f"Unsupported activation code: {self.activation_type}"
        )

    def _bn_scale_shift(self) -> Tuple[torch.Tensor, torch.Tensor]:
        base = getattr(self, "base_conv", None)
        ref = base.weight if base is not None else self.batch_norm.running_mean
        device, dtype = ref.device, ref.dtype
        if self.batch_norm is None:
            return (
                torch.ones(self.base_conv.out_channels, device=device, dtype=dtype),
                torch.zeros(self.base_conv.out_channels, device=device, dtype=dtype),
            )
        if self.batch_norm.training:
            raise RuntimeError(
                f"{type(self).__name__} requires BatchNorm2d in eval mode"
            )
        if self.batch_norm.running_mean is None or self.batch_norm.running_var is None:
            raise RuntimeError(
                f"{type(self).__name__} requires BatchNorm2d running stats"
            )
        running_mean = self.batch_norm.running_mean.to(device=device, dtype=dtype)
        running_var = self.batch_norm.running_var.to(device=device, dtype=dtype)
        if self.batch_norm.affine:
            gamma = self.batch_norm.weight.to(device=device, dtype=dtype)
            beta = self.batch_norm.bias.to(device=device, dtype=dtype)
        else:
            gamma = torch.ones_like(running_mean)
            beta = torch.zeros_like(running_mean)
        scale = gamma * torch.rsqrt(running_var + self.batch_norm.eps)
        return scale, beta - running_mean * scale

    def _update_source_bn_running_stats_from_preactivation(
        self, preactivation: torch.Tensor, momentum_override: Optional[float]
    ) -> None:
        if self.batch_norm is None:
            return
        reduce_dims = tuple(dim for dim in range(preactivation.ndim) if dim != 1)
        batch_mean = preactivation.mean(dim=reduce_dims)
        batch_var = preactivation.var(dim=reduce_dims, unbiased=False)
        if momentum_override is None:
            update_batch_norm_running_stats(
                self.batch_norm, batch_mean.detach(), batch_var.detach()
            )
            return
        if self.batch_norm.running_mean is None or self.batch_norm.running_var is None:
            raise RuntimeError(
                "eval AdaBN stat collection requires BatchNorm running statistics"
            )
        momentum = float(momentum_override)
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum_override must be in [0, 1], got {momentum}")
        with torch.no_grad():
            if getattr(self.batch_norm, "num_batches_tracked", None) is not None:
                self.batch_norm.num_batches_tracked.add_(1)
            mean = batch_mean.detach().to(
                device=self.batch_norm.running_mean.device,
                dtype=self.batch_norm.running_mean.dtype,
            )
            var = batch_var.detach().to(
                device=self.batch_norm.running_var.device,
                dtype=self.batch_norm.running_var.dtype,
            )
            self.batch_norm.running_mean.mul_(1.0 - momentum).add_(mean, alpha=momentum)
            self.batch_norm.running_var.mul_(1.0 - momentum).add_(var, alpha=momentum)

    def set_adabn_calibration(self, enabled: bool) -> None:
        self._adabn_calibrating = bool(enabled)
        if self.batch_norm is not None:
            self.batch_norm.eval()

    def set_adabn_calibration_mode(self, mode: str) -> None:
        self._adabn_calibration_mode = mode


class ProjectionAdapterCommon:
    """Orthogonal projection initialization and eval-mode AdaBN controls."""

    def set_eval_adabn_stat_collection(
        self, enabled: bool, momentum: Optional[float] = None
    ) -> None:
        self._eval_adabn_stat_collecting = bool(enabled)
        self._eval_adabn_stat_momentum = None if momentum is None else float(momentum)
        if self.batch_norm is not None:
            self.batch_norm.eval()

    def _init_random_orthogonal_projection(self) -> None:
        in_channels_per_group = self.P.in_channels // self.P.groups
        flat_dim = in_channels_per_group * self.P.kernel_size[0] * self.P.kernel_size[1]
        rank = self.rank
        dtype = self.P.weight.dtype
        qr_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        with torch.no_grad():
            group_weights = []
            for _group in range(self.P.groups):
                if rank <= flat_dim:
                    matrix = torch.randn(
                        flat_dim, rank, device=self.P.weight.device, dtype=qr_dtype
                    )
                    q, _ = torch.linalg.qr(matrix, mode="reduced")
                    weight = q.transpose(0, 1)
                else:
                    matrix = torch.randn(
                        rank, flat_dim, device=self.P.weight.device, dtype=qr_dtype
                    )
                    q, _ = torch.linalg.qr(matrix, mode="reduced")
                    weight = q
                group_weights.append(
                    weight.to(dtype=dtype).reshape(
                        rank,
                        in_channels_per_group,
                        self.P.kernel_size[0],
                        self.P.kernel_size[1],
                    )
                )
            self.P.weight.copy_(torch.cat(group_weights, dim=0))
