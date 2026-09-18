"""Small fused adapter blocks used by the MemFLoRA experiments."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function

from src.utils.adabn import batch_norm_train_output_from_preactivation
from src.utils.bitpack import pack_bool_mask, unpack_bool_mask
from src.adapters._common import AdapterBlockCommon, ConvSpec, conv_spec, _channel_view


class _FrozenConv2dNoInputSaveFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        spec: ConvSpec,
    ) -> torch.Tensor:
        ctx.input_shape = tuple(x.shape)
        ctx.stride = spec.stride
        ctx.padding = spec.padding
        ctx.dilation = spec.dilation
        ctx.groups = int(spec.groups)
        ctx.weight = weight
        ctx.has_bias = bias is not None
        return F.conv2d(
            x, weight, bias, spec.stride, spec.padding, spec.dilation, spec.groups
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_x = None
        grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_x = torch.nn.grad.conv2d_input(
                ctx.input_shape,
                ctx.weight,
                grad_output,
                ctx.stride,
                ctx.padding,
                ctx.dilation,
                ctx.groups,
            )
        if ctx.needs_input_grad[2] and ctx.has_bias:
            grad_bias = grad_output.sum(dim=(0, 2, 3))
        return (grad_x, None, grad_bias, None)


class FrozenConvNoSave2D(nn.Module):
    """Frozen Conv2d with input-gradient support but no saved input activation."""

    def __init__(
        self, conv: nn.Conv2d, *, keep_base_bias_trainable: bool = False
    ) -> None:
        super().__init__()
        if not isinstance(conv, nn.Conv2d):
            raise TypeError(f"FrozenConvNoSave2D expects nn.Conv2d, got {type(conv)!r}")
        if conv.padding_mode != "zeros":
            raise ValueError(
                "FrozenConvNoSave2D supports only zero-padding Conv2d modules"
            )
        self.base_conv = deepcopy(conv)
        self.base_conv.weight.requires_grad_(False)
        if self.base_conv.bias is not None:
            self.base_conv.bias.requires_grad_(bool(keep_base_bias_trainable))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _FrozenConv2dNoInputSaveFunction.apply(
            x, self.base_conv.weight, self.base_conv.bias, conv_spec(self.base_conv)
        )

    @property
    def in_channels(self) -> int:
        return int(self.base_conv.in_channels)

    @property
    def out_channels(self) -> int:
        return int(self.base_conv.out_channels)

    @property
    def kernel_size(self) -> Tuple[int, int]:
        return tuple(self.base_conv.kernel_size)

    @property
    def stride(self) -> Tuple[int, int]:
        return tuple(self.base_conv.stride)

    @property
    def padding(self) -> Tuple[int, int]:
        return tuple(self.base_conv.padding)

    @property
    def dilation(self) -> Tuple[int, int]:
        return tuple(self.base_conv.dilation)

    @property
    def groups(self) -> int:
        return int(self.base_conv.groups)


class BitpackedReLUFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        mask = x > 0
        packed, shape = pack_bool_mask(mask)
        ctx.save_for_backward(packed)
        ctx.mask_shape = tuple(shape)
        return torch.relu(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (packed,) = ctx.saved_tensors
        mask = unpack_bool_mask(packed, torch.Size(ctx.mask_shape))
        return torch.where(mask, grad_output, 0)


class TResNetResidualReLUBitpack2D(nn.Module):
    """Residual ReLU that saves only a bitpacked activation mask."""

    activation_mask_mode = "bitpack"

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return BitpackedReLUFunction.apply(x)


class EvalBatchNormAffineFunction(Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        ctx.save_for_backward(scale)
        ctx.needs_shift_grad = bool(ctx.needs_input_grad[2])
        return x * _channel_view(scale, x) + _channel_view(shift, x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (scale,) = ctx.saved_tensors
        grad_x = grad_output * _channel_view(scale, grad_output)
        grad_shift = None
        if ctx.needs_shift_grad:
            reduce_dims = tuple(
                index for index in range(grad_output.ndim) if index != 1
            )
            grad_shift = grad_output.sum(dim=reduce_dims)
        return grad_x, None, grad_shift


class TResNetEvalBatchNormMinimal2D(AdapterBlockCommon, nn.Module):
    """Standalone BatchNorm2d in eval-affine form with AdaBN calibration support."""

    def __init__(
        self, batch_norm: nn.BatchNorm2d, keep_bn_bias_trainable_in_eval: bool = False
    ) -> None:
        super().__init__()
        if not isinstance(batch_norm, nn.BatchNorm2d):
            raise TypeError("batch_norm must be nn.BatchNorm2d")
        self.batch_norm = deepcopy(batch_norm)
        self._adabn_calibrating = False
        self._adabn_calibration_mode: Optional[str] = None
        self.keep_bn_bias_trainable_in_eval = bool(keep_bn_bias_trainable_in_eval)
        self.batch_norm.eval()
        if self.batch_norm.bias is not None:
            self.batch_norm._keep_bias_trainable_in_eval = False  # type: ignore[attr-defined]
        for parameter in self.batch_norm.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.batch_norm.eval()
        for parameter in self.batch_norm.parameters():
            parameter.requires_grad_(False)
        if self.batch_norm.bias is not None:
            self.batch_norm.bias.requires_grad_(False)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._adabn_calibrating:
            return batch_norm_train_output_from_preactivation(x, self.batch_norm)
        scale, shift = self._bn_scale_shift()
        return EvalBatchNormAffineFunction.apply(
            x,
            scale.to(device=x.device, dtype=x.dtype),
            shift.to(device=x.device, dtype=x.dtype),
        )


def _conv2d_modules(module: nn.Module) -> List[nn.Conv2d]:
    return [child for child in module.modules() if isinstance(child, nn.Conv2d)]


def _group_count(channels: int, preferred_groups: int) -> int:
    preferred_groups = max(1, min(int(preferred_groups), int(channels)))
    for groups in range(preferred_groups, 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _norm_group_count(channels: int, channels_per_group: int = 8) -> int:
    target_groups = max(1, int(channels) // max(1, int(channels_per_group)))
    return _group_count(channels, target_groups)


class TinyTLLiteResidualBranch2D(nn.Module):
    """TinyTL-style lite residual branch for frozen MobileNetV2 blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        conv_groups: int = 2,
        norm_channels_per_group: int = 8,
        zero_init_output_scale: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(
                "TinyTL lite residual kernel_size must be odd for same padding"
            )
        group_count = _group_count(in_channels, conv_groups)
        padding = kernel_size // 2
        self.group_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=group_count,
            bias=False,
        )
        self.group_norm = nn.GroupNorm(
            num_groups=_norm_group_count(in_channels, norm_channels_per_group),
            num_channels=in_channels,
        )
        self.activation = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.output_norm = nn.GroupNorm(
            num_groups=_norm_group_count(out_channels, norm_channels_per_group),
            num_channels=out_channels,
        )
        if zero_init_output_scale:
            nn.init.zeros_(self.output_norm.weight)
            nn.init.zeros_(self.output_norm.bias)

    def forward(self, x: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        h = x
        if h.shape[-2] > 1 or h.shape[-1] > 1:
            kernel_h = 2 if h.shape[-2] > 1 else 1
            kernel_w = 2 if h.shape[-1] > 1 else 1
            h = F.avg_pool2d(
                h,
                kernel_size=(kernel_h, kernel_w),
                stride=(kernel_h, kernel_w),
                ceil_mode=True,
            )
        h = self.group_conv(h)
        h = self.group_norm(h)
        h = self.activation(h)
        h = self.pointwise(h)
        h = self.output_norm(h)
        if tuple(h.shape[-2:]) != tuple(output_size):
            h = F.interpolate(h, size=output_size, mode="bilinear", align_corners=False)
        return h


class TinyTLLiteResidualWrapper2D(nn.Module):
    """Wrap a frozen MobileNetV2 block with a trainable TinyTL lite residual."""

    def __init__(
        self,
        block: nn.Module,
        kernel_size: int = 5,
        conv_groups: int = 2,
        norm_channels_per_group: int = 8,
    ) -> None:
        super().__init__()
        convs = _conv2d_modules(block)
        if not convs:
            raise ValueError(
                "TinyTLLiteResidualWrapper2D requires a block with Conv2d layers"
            )
        self.block = block
        self.lite_residual = TinyTLLiteResidualBranch2D(
            in_channels=int(convs[0].in_channels),
            out_channels=int(convs[-1].out_channels),
            kernel_size=kernel_size,
            conv_groups=conv_groups,
            norm_channels_per_group=norm_channels_per_group,
        )
        self.lite_residual.to(
            device=convs[0].weight.device, dtype=convs[0].weight.dtype
        )
        for parameter in self.block.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        return y + self.lite_residual(x, output_size=tuple(y.shape[-2:]))

    @property
    def in_channels(self) -> int:
        return int(self.lite_residual.group_conv.in_channels)

    @property
    def out_channels(self) -> int:
        return int(self.lite_residual.pointwise.out_channels)


class LoRACConv2d(nn.Module):
    """LoRA-C layer-wise factorization for Conv2d."""

    def __init__(
        self, base_conv: nn.Conv2d, rank: int, alpha: Optional[float] = None
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if base_conv.groups != 1:
            raise ValueError("LoRA-C Conv2d currently supports groups=1 only")
        if base_conv.padding_mode != "zeros":
            raise ValueError(
                "LoRA-C Conv2d currently supports padding_mode='zeros' only"
            )
        self.rank = int(rank)
        self.alpha = float(alpha) if alpha is not None else float(2 * rank)
        self.scale = self.alpha / float(rank)
        self.adapter_mode = "lora_c"
        self.base_conv = deepcopy(base_conv)
        for parameter in self.base_conv.parameters():
            parameter.requires_grad_(False)
        factory_kwargs = {
            "device": base_conv.weight.device,
            "dtype": base_conv.weight.dtype,
        }
        kernel_h, kernel_w = int(base_conv.kernel_size[0]), int(
            base_conv.kernel_size[1]
        )
        self.A = nn.Parameter(
            torch.empty(rank, base_conv.in_channels, kernel_w, **factory_kwargs)
        )
        self.B = nn.Parameter(
            torch.zeros(base_conv.out_channels, kernel_h, rank, **factory_kwargs)
        )
        self.reset_parameters()

    @property
    def in_channels(self) -> int:
        return self.base_conv.in_channels

    @property
    def out_channels(self) -> int:
        return self.base_conv.out_channels

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def delta_weight(self) -> torch.Tensor:
        return torch.einsum("oha,aiw->oihw", self.B, self.A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.base_conv.weight + self.scale * self.delta_weight()
        return F.conv2d(
            x,
            weight,
            self.base_conv.bias,
            self.base_conv.stride,
            self.base_conv.padding,
            self.base_conv.dilation,
            self.base_conv.groups,
        )
