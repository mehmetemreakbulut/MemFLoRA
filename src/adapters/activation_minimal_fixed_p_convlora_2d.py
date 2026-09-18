from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
from torch.nn import grad as nn_grad

from src.utils.adabn import batch_norm_train_output_from_preactivation
from src.utils.bitpack import pack_bool_mask
from src.adapters._common import (
    ConvSpec,
    activation_grad,
    conv_spec,
    make_projection_conv,
    AdapterBlockCommon,
    ProjectionAdapterCommon,
    scale_channels,
    stash_fused_geometry,
    _activation_code,
    _needs_grouped_projection,
)


class ActivationMinimalFixedPConvBNAct2DFunction(Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        base_weight: torch.Tensor,
        base_bias: Optional[torch.Tensor],
        p_weight: torch.Tensor,
        p_bias: Optional[torch.Tensor],
        u_weight: torch.Tensor,
        u_bias: Optional[torch.Tensor],
        bn_scale: torch.Tensor,
        bn_shift: torch.Tensor,
        p: ConvSpec,
        u: ConvSpec,
        base: ConvSpec,
        scale: float,
        activation: int,
    ) -> torch.Tensor:
        base_out = F.conv2d(
            x,
            base_weight,
            base_bias,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            groups=base.groups,
        )
        z = F.conv2d(
            x,
            p_weight,
            p_bias,
            stride=p.stride,
            padding=p.padding,
            dilation=p.dilation,
            groups=p.groups,
        )
        adapter_out = F.conv2d(
            z,
            u_weight,
            u_bias,
            stride=u.stride,
            padding=u.padding,
            dilation=u.dilation,
            groups=u.groups,
        )
        q = base_out + float(scale) * adapter_out
        bn_scale_view = bn_scale.view(1, -1, 1, 1)
        bn_shift_view = bn_shift.view(1, -1, 1, 1)
        bn_out = q * bn_scale_view + bn_shift_view
        ctx.activation_mask_shape = tuple()
        if activation == 0:
            y = bn_out
            activation_mask = torch.empty(0, device=x.device, dtype=torch.uint8)
        elif activation == 1:
            mask = bn_out > 0
            activation_mask = pack_bool_mask(mask)[0]
            ctx.activation_mask_shape = tuple(mask.shape)
            y = torch.relu(bn_out)
        elif activation == 2:
            mask = (bn_out > 0) & (bn_out < 6)
            activation_mask = pack_bool_mask(mask)[0]
            ctx.activation_mask_shape = tuple(mask.shape)
            y = torch.clamp(bn_out, min=0, max=6)
        else:
            raise NotImplementedError(f"Unsupported activation code: {activation}")
        ctx.needs_x_grad = bool(ctx.needs_input_grad[0])
        ctx.has_u_bias = u_bias is not None
        if ctx.needs_x_grad:
            ctx.save_for_backward(
                z, base_weight, p_weight, u_weight, bn_scale, activation_mask
            )
        else:
            ctx.save_for_backward(z, bn_scale, activation_mask)
        stash_fused_geometry(ctx, x, u_weight, p, u, base, scale, activation)
        ctx.z_shape = tuple(z.shape)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        saved_tensors = ctx.saved_tensors
        z = saved_tensors[0]
        if ctx.needs_x_grad:
            base_weight, p_weight, u_weight, bn_scale, activation_mask = saved_tensors[
                1:
            ]
        else:
            bn_scale, activation_mask = saved_tensors[1:]
            base_weight = p_weight = u_weight = None
        g = activation_grad(
            grad_output, activation_mask, ctx.activation, 1, ctx.activation_mask_shape
        )
        # dL/dq; the adapter output's gradient is this times scale, which is
        # applied to the small results instead of a full-width copy.
        grad_q = scale_channels(g, bn_scale, grad_output)
        del g
        grad_u_weight = nn_grad.conv2d_weight(
            z,
            ctx.u_weight_shape,
            grad_q,
            stride=ctx.u.stride,
            padding=ctx.u.padding,
            dilation=ctx.u.dilation,
            groups=ctx.u.groups,
        ).mul_(ctx.scale)
        grad_u_bias = grad_q.sum(dim=(0, 2, 3)).mul_(ctx.scale) if ctx.has_u_bias else None
        grad_x = None
        if ctx.needs_x_grad:
            grad_z = nn_grad.conv2d_input(
                ctx.z_shape,
                u_weight,
                grad_q,
                stride=ctx.u.stride,
                padding=ctx.u.padding,
                dilation=ctx.u.dilation,
                groups=ctx.u.groups,
            ).mul_(ctx.scale)
            grad_x = nn_grad.conv2d_input(
                ctx.input_shape,
                base_weight,
                grad_q,
                stride=ctx.base.stride,
                padding=ctx.base.padding,
                dilation=ctx.base.dilation,
                groups=ctx.base.groups,
            )
            del grad_q
            grad_x += nn_grad.conv2d_input(
                ctx.input_shape,
                p_weight,
                grad_z,
                stride=ctx.p.stride,
                padding=ctx.p.padding,
                dilation=ctx.p.dilation,
                groups=ctx.p.groups,
            )
        return (
            grad_x,
            None,
            None,
            None,
            None,
            grad_u_weight,
            grad_u_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class ActivationMinimalFixedPConvLoRA2D(
    AdapterBlockCommon, ProjectionAdapterCommon, nn.Module
):
    """Frozen Conv2d + frozen BatchNorm2d + activation + Fixed-P adapter.

    P is frozen and normal-initialized; the custom backward stores activation
    masks bit-packed. Neither is configurable.
    """

    adapter_mode = "pointwise"

    def __init__(
        self,
        base_conv: nn.Conv2d,
        batch_norm: Optional[nn.BatchNorm2d],
        activation: Optional[nn.Module],
        rank: int,
        alpha: Optional[float] = None,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if base_conv.padding_mode != "zeros":
            raise ValueError(
                "ActivationMinimalFixedPConvLoRA2D currently supports only zero padding"
            )
        if batch_norm is not None and not isinstance(batch_norm, nn.BatchNorm2d):
            raise TypeError("batch_norm must be nn.BatchNorm2d or None")
        self.rank = int(rank)
        self.alpha = float(alpha) if alpha is not None else float(rank)
        self.scale = self.alpha / float(rank)
        self.adapter_mode = (
            "grouped" if _needs_grouped_projection(base_conv) else "pointwise"
        )
        self.activation_type = _activation_code(activation)
        self._adabn_calibrating = False
        self._adabn_calibration_mode = "ema_reset"
        self._eval_adabn_stat_collecting = False
        self._eval_adabn_stat_momentum: Optional[float] = None
        self.base_conv = copy.deepcopy(base_conv)
        self.batch_norm = copy.deepcopy(batch_norm) if batch_norm is not None else None
        self.activation = (
            copy.deepcopy(activation) if activation is not None else nn.Identity()
        )
        for parameter in self.base_conv.parameters():
            parameter.requires_grad_(False)
        if self.batch_norm is not None:
            self.batch_norm.eval()
            for parameter in self.batch_norm.parameters():
                parameter.requires_grad_(False)
        factory_kwargs = {
            "device": base_conv.weight.device,
            "dtype": base_conv.weight.dtype,
        }
        grouped = self.adapter_mode == "grouped"
        groups = base_conv.groups if grouped else 1
        self.P = make_projection_conv(
            base_conv,
            base_conv.in_channels,
            groups * rank,
            grouped,
            groups,
            bias=use_bias,
            **factory_kwargs,
        )
        self.U = make_projection_conv(
            base_conv,
            self.P.out_channels,
            base_conv.out_channels,
            False,
            groups,
            bias=use_bias,
            **factory_kwargs,
        )
        self.reset_parameters()
        for parameter in self.P.parameters():
            parameter.requires_grad_(False)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.P.weight, mean=0.0, std=1.0 / math.sqrt(float(self.rank)))
        if self.P.bias is not None:
            nn.init.zeros_(self.P.bias)
        nn.init.zeros_(self.U.weight)
        if self.U.bias is not None:
            nn.init.zeros_(self.U.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.batch_norm is not None:
            self.batch_norm.eval()
        return self

    def _forward_eval_adabn_stat_collection(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            raise RuntimeError(
                "Fixed-P eval AdaBN stat collection must run under torch.no_grad()"
            )
        if self.batch_norm is None:
            return self._apply_activation(self._pre_bn_activation(x))
        if self.training or self.batch_norm.training:
            raise RuntimeError(
                "Fixed-P eval AdaBN stat collection requires eval-mode BatchNorm"
            )
        z = self._base_pre_bn_activation(x)
        pre_bn = self._pre_bn_activation(x)
        bn_scale, bn_shift = self._bn_scale_shift()
        bn_out = pre_bn * bn_scale.view(1, -1, 1, 1) + bn_shift.view(1, -1, 1, 1)
        y = self._apply_activation(bn_out)
        self._update_source_bn_running_stats_from_preactivation(
            z, momentum_override=self._eval_adabn_stat_momentum
        )
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._eval_adabn_stat_collecting:
            return self._forward_eval_adabn_stat_collection(x)
        if self._adabn_calibrating and self.batch_norm is not None:
            q = self._pre_bn_activation(x)
            bn_out = batch_norm_train_output_from_preactivation(q, self.batch_norm)
            return self._apply_activation(bn_out)
        bn_scale, bn_shift = self._bn_scale_shift()
        return ActivationMinimalFixedPConvBNAct2DFunction.apply(
            x,
            self.base_conv.weight,
            self.base_conv.bias,
            self.P.weight,
            self.P.bias,
            self.U.weight,
            self.U.bias,
            bn_scale,
            bn_shift,
            conv_spec(self.P),
            conv_spec(self.U),
            conv_spec(self.base_conv),
            self.scale,
            self.activation_type,
        )

    def _pre_bn_activation(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self._base_pre_bn_activation(x)
        z = F.conv2d(
            x,
            self.P.weight,
            self.P.bias,
            stride=self.P.stride,
            padding=self.P.padding,
            dilation=self.P.dilation,
            groups=self.P.groups,
        )
        adapter_out = F.conv2d(
            z,
            self.U.weight,
            self.U.bias,
            stride=self.U.stride,
            padding=self.U.padding,
            dilation=self.U.dilation,
            groups=self.U.groups,
        )
        return base_out + self.scale * adapter_out

    def _base_pre_bn_activation(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self.base_conv.weight,
            self.base_conv.bias,
            stride=self.base_conv.stride,
            padding=self.base_conv.padding,
            dilation=self.base_conv.dilation,
            groups=self.base_conv.groups,
        )
