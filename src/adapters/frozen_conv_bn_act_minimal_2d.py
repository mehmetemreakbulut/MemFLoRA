from __future__ import annotations
import copy
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
from torch.nn import grad as nn_grad
from src.utils.adabn import batch_norm_train_output_from_preactivation
from src.adapters._common import (
    ConvSpec,
    conv_spec,
    ActivationMaskMode,
    AdapterBlockCommon,
    _activation_code,
)
from src.adapters._common import (
    _activation_mask_mode_code,
    _prepare_activation_mask,
    activation_grad,
    scale_channels,
)


class FrozenConvBNActMinimal2DFunction(Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        base_weight: torch.Tensor,
        base_bias: Optional[torch.Tensor],
        bn_scale: torch.Tensor,
        bn_shift: torch.Tensor,
        base: ConvSpec,
        activation: int,
        activation_mask_mode: int,
    ) -> torch.Tensor:
        conv_out = F.conv2d(
            x,
            base_weight,
            base_bias,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            groups=base.groups,
        )
        bn_out = conv_out * bn_scale.view(1, -1, 1, 1) + bn_shift.view(1, -1, 1, 1)
        ctx.activation_mask_shape = tuple()
        if activation == 0:
            y = bn_out
            activation_mask = torch.empty(0, device=x.device, dtype=torch.uint8)
        elif activation == 1:
            mask = bn_out > 0
            activation_mask = _prepare_activation_mask(mask, activation_mask_mode)
            ctx.activation_mask_shape = tuple(mask.shape)
            y = torch.relu(bn_out)
        elif activation == 2:
            mask = (bn_out > 0) & (bn_out < 6)
            activation_mask = _prepare_activation_mask(mask, activation_mask_mode)
            ctx.activation_mask_shape = tuple(mask.shape)
            y = torch.clamp(bn_out, min=0, max=6)
        else:
            raise NotImplementedError(f"Unsupported activation code: {activation}")
        ctx.needs_x_grad = bool(ctx.needs_input_grad[0])
        ctx.needs_base_bias_grad = bool(ctx.needs_input_grad[2])
        ctx.needs_bn_shift_grad = bool(ctx.needs_input_grad[4])
        ctx.needs_any_backward = bool(
            ctx.needs_x_grad or ctx.needs_base_bias_grad or ctx.needs_bn_shift_grad
        )
        ctx.has_base_bias = base_bias is not None
        if ctx.needs_any_backward:
            ctx.save_for_backward(base_weight, bn_scale, activation_mask)
        ctx.input_shape = tuple(x.shape)
        ctx.base = base
        ctx.activation = int(activation)
        ctx.activation_mask_mode = int(activation_mask_mode)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_x = None
        grad_base_bias = None
        grad_bn_shift = None
        if ctx.needs_any_backward:
            base_weight, bn_scale, activation_mask = ctx.saved_tensors
            g = activation_grad(
                grad_output,
                activation_mask,
                ctx.activation,
                ctx.activation_mask_mode,
                ctx.activation_mask_shape,
            )
            if ctx.needs_bn_shift_grad:
                grad_bn_shift = g.sum(dim=(0, 2, 3))
            g = scale_channels(g, bn_scale, grad_output)  # now dL/d(conv output)
            if ctx.needs_base_bias_grad and ctx.has_base_bias:
                grad_base_bias = g.sum(dim=(0, 2, 3))
            if ctx.needs_x_grad:
                grad_x = nn_grad.conv2d_input(
                    ctx.input_shape,
                    base_weight,
                    g,
                    stride=ctx.base.stride,
                    padding=ctx.base.padding,
                    dilation=ctx.base.dilation,
                    groups=ctx.base.groups,
                )
        return (grad_x, None, grad_base_bias, None, grad_bn_shift, None, None, None)


class FrozenConvBNActMinimal2D(AdapterBlockCommon, nn.Module):
    """Frozen Conv2d + frozen BatchNorm2d eval + activation with minimal backward."""

    def __init__(
        self,
        base_conv: nn.Conv2d,
        batch_norm: Optional[nn.BatchNorm2d],
        activation: Optional[nn.Module],
        activation_mask_mode: ActivationMaskMode = "bool",
        keep_base_bias_trainable: bool = False,
        keep_bn_bias_trainable_in_eval: bool = False,
    ) -> None:
        super().__init__()
        if base_conv.padding_mode != "zeros":
            raise ValueError(
                "FrozenConvBNActMinimal2D currently supports only zero padding"
            )
        if batch_norm is not None and not isinstance(batch_norm, nn.BatchNorm2d):
            raise TypeError("batch_norm must be nn.BatchNorm2d or None")
        self.base_conv = copy.deepcopy(base_conv)
        self.batch_norm = copy.deepcopy(batch_norm) if batch_norm is not None else None
        self.activation = (
            copy.deepcopy(activation) if activation is not None else nn.Identity()
        )
        self.activation_type = _activation_code(self.activation)
        self.activation_mask_mode = activation_mask_mode
        self.activation_mask_mode_code = _activation_mask_mode_code(
            activation_mask_mode
        )
        self._adabn_calibrating = False
        self._adabn_calibration_mode = "ema_reset"
        self.keep_base_bias_trainable = bool(keep_base_bias_trainable)
        for parameter in self.base_conv.parameters():
            parameter.requires_grad_(False)
        if self.base_conv.bias is not None:
            self.base_conv.bias.requires_grad_(self.keep_base_bias_trainable)
        if self.batch_norm is not None:
            self.batch_norm.eval()
            self.keep_bn_bias_trainable_in_eval = bool(keep_bn_bias_trainable_in_eval)
            if self.batch_norm.bias is not None:
                keep_bias = self.keep_bn_bias_trainable_in_eval
                self.batch_norm._keep_bias_trainable_in_eval = keep_bias  # type: ignore
            for parameter in self.batch_norm.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.batch_norm is not None:
            self.batch_norm.eval()
            if self.batch_norm.bias is not None:
                self.batch_norm.bias.requires_grad_(self.keep_bn_bias_trainable_in_eval)
        for parameter in self.base_conv.parameters():
            parameter.requires_grad_(False)
        if self.base_conv.bias is not None:
            self.base_conv.bias.requires_grad_(self.keep_base_bias_trainable)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._adabn_calibrating and self.batch_norm is not None:
            conv_out = F.conv2d(
                x,
                self.base_conv.weight,
                self.base_conv.bias,
                stride=self.base_conv.stride,
                padding=self.base_conv.padding,
                dilation=self.base_conv.dilation,
                groups=self.base_conv.groups,
            )
            bn_out = batch_norm_train_output_from_preactivation(
                conv_out, self.batch_norm
            )
            return self._apply_activation(bn_out)
        bn_scale, bn_shift = self._bn_scale_shift()
        return FrozenConvBNActMinimal2DFunction.apply(
            x,
            self.base_conv.weight,
            self.base_conv.bias,
            bn_scale,
            bn_shift,
            conv_spec(self.base_conv),
            self.activation_type,
            self.activation_mask_mode_code,
        )
