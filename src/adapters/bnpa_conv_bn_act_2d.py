from __future__ import annotations

import copy
import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
from torch.nn import grad as nn_grad

from src.utils.adabn import batch_norm_train_output_from_preactivation
from src.utils.bitpack import unpack_bool_mask
from src.adapters._common import (
    fused_activation_inplace,
    stash_fused_geometry,
    ConvSpec,
    conv_spec,
    make_projection_conv,
    AdapterBlockCommon,
    ProjectionAdapterCommon,
    _activation_code,
    _channel_view,
    _needs_grouped_projection,
)


class BNPAConvBNAct2DFunction(Function):
    """Custom autograd for BNPA."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        base_weight: torch.Tensor,
        base_bias: Optional[torch.Tensor],
        p_weight: torch.Tensor,
        p_bias: Optional[torch.Tensor],
        projection_shift: torch.Tensor,
        u_weight: torch.Tensor,
        u_bias: Optional[torch.Tensor],
        bn_scale: torch.Tensor,
        bn_shift: torch.Tensor,
        bottleneck_weight: torch.Tensor,
        bottleneck_bias: torch.Tensor,
        bottleneck_running_mean: torch.Tensor,
        bottleneck_running_var: torch.Tensor,
        bottleneck_num_batches_tracked: torch.Tensor,
        p: ConvSpec,
        u: ConvSpec,
        base: ConvSpec,
        scale: float,
        activation: int,
        bottleneck_training: bool,
        bottleneck_bn_enabled: bool,
        bottleneck_eps: float,
        bottleneck_momentum: float,
        g_capture_callback: Optional[Callable[[torch.Tensor], None]],
        adapter_pre_bn: bool,
        post_bn_adapter_scale_by_source_bn: bool,
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
        q = F.conv2d(
            x,
            p_weight,
            p_bias,
            stride=p.stride,
            padding=p.padding,
            dilation=p.dilation,
            groups=p.groups,
        )
        if projection_shift.numel() > 0:
            q = q + _channel_view(
                projection_shift.to(device=q.device, dtype=q.dtype), q
            )
        if bottleneck_bn_enabled:
            reduce_dims = _channel_reduce_dims(q)
            if bottleneck_training:
                q_mean = q.mean(dim=reduce_dims)
                q_var = q.var(dim=reduce_dims, unbiased=False)
                _update_running_stats(
                    bottleneck_running_mean,
                    bottleneck_running_var,
                    bottleneck_num_batches_tracked,
                    q_mean,
                    q_var,
                    bottleneck_momentum,
                )
            else:
                q_mean = bottleneck_running_mean.to(device=q.device, dtype=q.dtype)
                q_var = bottleneck_running_var.to(device=q.device, dtype=q.dtype)
            q_invstd = torch.rsqrt(q_var + bottleneck_eps)
            q_hat = (q - _channel_view(q_mean, q)) * _channel_view(q_invstd, q)
            q_tilde = q_hat * _channel_view(bottleneck_weight, q) + _channel_view(
                bottleneck_bias, q
            )
        else:
            q_mean = torch.empty(0, device=q.device, dtype=q.dtype)
            q_invstd = torch.empty(0, device=q.device, dtype=q.dtype)
            q_tilde = q
        adapter_out = F.conv2d(
            q_tilde,
            u_weight,
            u_bias,
            stride=u.stride,
            padding=u.padding,
            dilation=u.dilation,
            groups=u.groups,
        )
        bn_scale_view = bn_scale.view(1, -1, 1, 1)
        bn_shift_view = bn_shift.view(1, -1, 1, 1)
        # Reuse base_out as the full-width workspace; BNPA does not need this
        # pre-activation tensor saved for backward.
        if adapter_pre_bn:
            base_out.add_(adapter_out, alpha=float(scale))
            base_out.mul_(bn_scale_view).add_(bn_shift_view)
        else:
            base_out.mul_(bn_scale_view).add_(bn_shift_view)
            if post_bn_adapter_scale_by_source_bn:
                adapter_out.mul_(bn_scale_view)
            base_out.add_(adapter_out, alpha=float(scale))
        s = base_out
        y, activation_mask = fused_activation_inplace(ctx, s, activation)
        ctx.needs_x_grad = bool(ctx.needs_input_grad[0])
        ctx.has_u_bias = u_bias is not None
        ctx.bottleneck_training = bool(bottleneck_training)
        ctx.bottleneck_bn_enabled = bool(bottleneck_bn_enabled)
        ctx.adapter_pre_bn = bool(adapter_pre_bn)
        ctx.post_bn_adapter_scale_by_source_bn = bool(
            post_bn_adapter_scale_by_source_bn
        )
        post_scale_to_save = torch.empty(0, device=x.device, dtype=x.dtype)
        u_bias_to_save = torch.empty(0, device=x.device, dtype=x.dtype)
        if ctx.needs_x_grad:
            ctx.save_for_backward(
                q,
                q_mean,
                q_invstd,
                base_weight,
                p_weight,
                u_weight,
                bn_scale,
                bottleneck_weight,
                bottleneck_bias,
                post_scale_to_save,
                u_bias_to_save,
                activation_mask,
            )
        else:
            ctx.save_for_backward(
                q,
                q_mean,
                q_invstd,
                u_weight,
                bn_scale,
                bottleneck_weight,
                bottleneck_bias,
                post_scale_to_save,
                u_bias_to_save,
                activation_mask,
            )
        stash_fused_geometry(ctx, x, u_weight, p, u, base, scale, activation)
        ctx.g_capture_callback = g_capture_callback
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        saved_tensors = ctx.saved_tensors
        if ctx.needs_x_grad:
            (
                q,
                q_mean,
                q_invstd,
                base_weight,
                p_weight,
                u_weight,
                bn_scale,
                bottleneck_weight,
                bottleneck_bias,
                post_bn_adapter_channel_scale,
                u_bias_for_scale_grad,
                activation_mask,
            ) = saved_tensors
        else:
            (
                q,
                q_mean,
                q_invstd,
                u_weight,
                bn_scale,
                bottleneck_weight,
                bottleneck_bias,
                post_bn_adapter_channel_scale,
                u_bias_for_scale_grad,
                activation_mask,
            ) = saved_tensors
            base_weight = p_weight = None
        if ctx.activation == 0:
            g = grad_output
        else:
            mask = unpack_bool_mask(
                activation_mask, torch.Size(ctx.activation_mask_shape)
            )
            g = grad_output * mask.to(dtype=grad_output.dtype)
        bn_scale_view = bn_scale.view(1, -1, 1, 1) if bn_scale is not None else None
        if ctx.adapter_pre_bn:
            adapter_output_scale_view = bn_scale_view
        elif ctx.post_bn_adapter_scale_by_source_bn:
            adapter_output_scale_view = bn_scale_view
        else:
            adapter_output_scale_view = None
        grad_adapter_out = (
            g * adapter_output_scale_view * ctx.scale
            if adapter_output_scale_view is not None
            else g * ctx.scale
        )
        grad_u_adapter_out = grad_adapter_out
        if ctx.bottleneck_bn_enabled:
            q_tilde, grad_q, grad_bn_weight, grad_bn_bias = _bottleneck_bn_backward(
                q=q,
                q_mean=q_mean,
                q_invstd=q_invstd,
                bottleneck_weight=bottleneck_weight,
                bottleneck_bias=bottleneck_bias,
                u_weight=u_weight,
                grad_adapter_out=grad_adapter_out,
                u_stride=ctx.u.stride,
                u_padding=ctx.u.padding,
                u_dilation=ctx.u.dilation,
                u_groups=ctx.u.groups,
                training=ctx.bottleneck_training,
            )
        else:
            q_tilde = q
            grad_q = nn_grad.conv2d_input(
                tuple(q.shape),
                u_weight,
                grad_adapter_out,
                stride=ctx.u.stride,
                padding=ctx.u.padding,
                dilation=ctx.u.dilation,
                groups=ctx.u.groups,
            )
            grad_bn_weight = None
            grad_bn_bias = None
        g_capture_callback = getattr(ctx, "g_capture_callback", None)
        if g_capture_callback is not None:
            g_capture_callback(grad_q)
        grad_u_weight = nn_grad.conv2d_weight(
            q_tilde.detach(),
            ctx.u_weight_shape,
            grad_u_adapter_out,
            stride=ctx.u.stride,
            padding=ctx.u.padding,
            dilation=ctx.u.dilation,
            groups=ctx.u.groups,
        )
        grad_u_bias = grad_u_adapter_out.sum(dim=(0, 2, 3)) if ctx.has_u_bias else None
        grad_x = None
        if ctx.needs_x_grad:
            grad_z = g * bn_scale_view
            grad_x_base = nn_grad.conv2d_input(
                ctx.input_shape,
                base_weight,
                grad_z,
                stride=ctx.base.stride,
                padding=ctx.base.padding,
                dilation=ctx.base.dilation,
                groups=ctx.base.groups,
            )
            grad_x_adapter = nn_grad.conv2d_input(
                ctx.input_shape,
                p_weight,
                grad_q,
                stride=ctx.p.stride,
                padding=ctx.p.padding,
                dilation=ctx.p.dilation,
                groups=ctx.p.groups,
            )
            grad_x = grad_x_base + grad_x_adapter
        grad_bn_shift = g.sum(dim=(0, 2, 3))
        return (
            grad_x,
            None,
            None,
            None,
            None,
            None,
            grad_u_weight,
            grad_u_bias,
            None,
            grad_bn_shift,
            grad_bn_weight,
            grad_bn_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class BNPAConvBNAct2DPostBNScaledOptimizedFunction(Function):
    """Specialized path for post-BN, no-BN_r, fixed source-scale BNPA."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        base_weight: torch.Tensor,
        base_bias: Optional[torch.Tensor],
        p_weight: torch.Tensor,
        p_bias: Optional[torch.Tensor],
        projection_shift: torch.Tensor,
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
            groups=1,
        )
        q = F.conv2d(
            x,
            p_weight,
            p_bias,
            stride=p.stride,
            padding=p.padding,
            dilation=p.dilation,
            groups=1,
        )
        if projection_shift.numel() > 0:
            q = q + _channel_view(
                projection_shift.to(device=q.device, dtype=q.dtype), q
            )
        adapter_out = F.conv2d(
            q,
            u_weight,
            u_bias,
            stride=u.stride,
            padding=u.padding,
            dilation=u.dilation,
            groups=1,
        )
        bn_scale_view = bn_scale.view(1, -1, 1, 1)
        base_out.mul_(bn_scale_view).add_(bn_shift.view(1, -1, 1, 1))
        adapter_out.mul_(bn_scale_view)
        base_out.add_(adapter_out, alpha=float(scale))
        s = base_out
        y, activation_mask = fused_activation_inplace(ctx, s, activation)
        ctx.needs_x_grad = bool(ctx.needs_input_grad[0])
        ctx.has_u_bias = u_bias is not None
        if ctx.needs_x_grad:
            ctx.save_for_backward(
                q, base_weight, p_weight, u_weight, bn_scale, activation_mask
            )
        else:
            ctx.save_for_backward(q, u_weight, bn_scale, activation_mask)
        stash_fused_geometry(ctx, x, u_weight, p, u, base, scale, activation)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.needs_x_grad:
            q, base_weight, p_weight, u_weight, bn_scale, activation_mask = (
                ctx.saved_tensors
            )
        else:
            q, u_weight, bn_scale, activation_mask = ctx.saved_tensors
            base_weight = p_weight = None
        if ctx.activation == 0:
            g = grad_output
        else:
            mask = unpack_bool_mask(
                activation_mask, torch.Size(ctx.activation_mask_shape)
            )
            g = grad_output * mask.to(dtype=grad_output.dtype)
        bn_scale_view = bn_scale.view(1, -1, 1, 1)
        grad_adapter_out = g * bn_scale_view * ctx.scale
        grad_q = nn_grad.conv2d_input(
            tuple(q.shape),
            u_weight,
            grad_adapter_out,
            stride=ctx.u.stride,
            padding=ctx.u.padding,
            dilation=ctx.u.dilation,
            groups=1,
        )
        grad_u_weight = nn_grad.conv2d_weight(
            q.detach(),
            ctx.u_weight_shape,
            grad_adapter_out,
            stride=ctx.u.stride,
            padding=ctx.u.padding,
            dilation=ctx.u.dilation,
            groups=1,
        )
        grad_u_bias = grad_adapter_out.sum(dim=(0, 2, 3)) if ctx.has_u_bias else None
        grad_x = None
        if ctx.needs_x_grad:
            grad_z = g * bn_scale_view
            grad_x_base = nn_grad.conv2d_input(
                ctx.input_shape,
                base_weight,
                grad_z,
                stride=ctx.base.stride,
                padding=ctx.base.padding,
                dilation=ctx.base.dilation,
                groups=1,
            )
            grad_x_adapter = nn_grad.conv2d_input(
                ctx.input_shape,
                p_weight,
                grad_q,
                stride=ctx.p.stride,
                padding=ctx.p.padding,
                dilation=ctx.p.dilation,
                groups=1,
            )
            grad_x = grad_x_base + grad_x_adapter
        grad_bn_shift = g.sum(dim=(0, 2, 3))
        return (
            grad_x,
            None,
            None,
            None,
            None,
            None,
            grad_u_weight,
            grad_u_bias,
            None,
            grad_bn_shift,
            None,
            None,
            None,
            None,
            None,
        )


class BNPAConvBNAct2D(AdapterBlockCommon, ProjectionAdapterCommon, nn.Module):
    """Bottleneck-Normalized Projected Adapter for frozen Conv-BN-Act blocks."""

    adapter_mode = "bnpa"

    def __init__(
        self,
        base_conv: nn.Conv2d,
        batch_norm: nn.BatchNorm2d,
        activation: Optional[nn.Module],
        rank: int,
        projection_init: str = "random_orthogonal",
        proj_mean_center: Optional[bool] = None,
        bnpa_bottleneck_bn: str = "on",
        adapter_pre_bn: bool = False,
        fa_port_layout: bool = False,
        post_bn_adapter_scale_by_source_bn: bool = False,
        optimized_post_bn_bnr_off_scaled: bool = False,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if base_conv.padding_mode != "zeros":
            raise ValueError("BNPAConvBNAct2D currently supports only zero padding")
        if not isinstance(batch_norm, nn.BatchNorm2d):
            raise TypeError("batch_norm must be nn.BatchNorm2d")
        self.rank = int(rank)
        self.alpha = float(rank)
        self.scale = 1.0
        self.projection_init = projection_init
        if bnpa_bottleneck_bn not in ("on", "off"):
            raise ValueError("bnpa_bottleneck_bn must be 'on' or 'off'")
        self.proj_mean_center = (
            bool(proj_mean_center) if proj_mean_center is not None else False
        )
        self.bnpa_bottleneck_bn = bnpa_bottleneck_bn
        self.bottleneck_bn_enabled = bnpa_bottleneck_bn != "off"
        self.bottleneck_bn_affine_trainable = self.bottleneck_bn_enabled
        self.adapter_pre_bn = bool(adapter_pre_bn)
        self.fa_port_layout = bool(fa_port_layout)
        self.post_bn_adapter_scale_by_source_bn = bool(
            post_bn_adapter_scale_by_source_bn
        )
        self.optimized_post_bn_bnr_off_scaled = bool(optimized_post_bn_bnr_off_scaled)
        if self.fa_port_layout and base_conv.groups != 1:
            raise ValueError("BNPA FA-port layout currently supports groups=1 only")
        self.adapter_mode = (
            "fa_port"
            if self.fa_port_layout
            else "grouped" if _needs_grouped_projection(base_conv) else "pointwise"
        )
        self.activation = (
            copy.deepcopy(activation) if activation is not None else nn.Identity()
        )
        self.activation_type = _activation_code(self.activation)
        self._adabn_calibrating = False
        self._adabn_calibration_mode: Optional[str] = None
        self._eval_adabn_stat_collecting = False
        self._eval_adabn_stat_momentum: Optional[float] = None
        self._bnpa_grad_q_capture_callback: Optional[Callable[[torch.Tensor], None]] = (
            None
        )
        self.base_conv = copy.deepcopy(base_conv)
        self.batch_norm = copy.deepcopy(batch_norm)
        self.batch_norm._keep_bias_trainable_in_eval = False  # type: ignore[attr-defined]
        for parameter in self.base_conv.parameters():
            parameter.requires_grad_(False)
        self.batch_norm.eval()
        for parameter in self.batch_norm.parameters():
            parameter.requires_grad_(False)
        factory_kwargs = {
            "device": base_conv.weight.device,
            "dtype": base_conv.weight.dtype,
        }
        grouped = self.adapter_mode == "grouped"
        adapter_groups = base_conv.groups if grouped else 1
        self.P = make_projection_conv(
            base_conv,
            base_conv.in_channels,
            base_conv.groups * rank if grouped else rank,
            grouped,
            adapter_groups,
            **factory_kwargs,
        )
        self.register_buffer(
            "projection_shift", torch.zeros(self.P.out_channels, **factory_kwargs)
        )
        self.bottleneck_bn = nn.BatchNorm2d(
            self.P.out_channels,
            eps=batch_norm.eps,
            momentum=batch_norm.momentum,
            affine=True,
            track_running_stats=True,
            **factory_kwargs,
        )
        self.bottleneck_bn._exclude_from_adabn = True  # type: ignore[attr-defined]
        self.U = make_projection_conv(
            base_conv,
            self.P.out_channels,
            base_conv.out_channels,
            self.adapter_mode == "fa_port",
            adapter_groups,
            **factory_kwargs,
        )
        self.register_buffer(
            "post_bn_adapter_channel_scale", torch.empty(0, **factory_kwargs)
        )
        self.reset_parameters()
        for parameter in self.P.parameters():
            parameter.requires_grad_(False)
        if not self.bottleneck_bn_affine_trainable:
            for parameter in self.bottleneck_bn.parameters():
                parameter.requires_grad_(False)

    def reset_parameters(self) -> None:
        self.projection_shift.zero_()
        if self.fa_port_layout and self.projection_init == "fa_port_random_orthogonal":
            self._init_random_orthogonal_projection()
        elif self.fa_port_layout or self.projection_init == "fa_port_normal":
            nn.init.normal_(
                self.P.weight, mean=0.0, std=1.0 / math.sqrt(float(self.rank))
            )
        elif self.projection_init == "random_orthogonal":
            self._init_random_orthogonal_projection()
        else:
            raise ValueError(
                f"Unknown BNPA projection initialization: {self.projection_init!r}"
            )
        if self.P.bias is not None:
            nn.init.zeros_(self.P.bias)
        nn.init.ones_(self.bottleneck_bn.weight)
        nn.init.zeros_(self.bottleneck_bn.bias)
        self.bottleneck_bn.running_mean.zero_()
        self.bottleneck_bn.running_var.fill_(1.0)
        if self.bottleneck_bn.num_batches_tracked is not None:
            self.bottleneck_bn.num_batches_tracked.zero_()
        nn.init.zeros_(self.U.weight)
        if self.U.bias is not None:
            nn.init.zeros_(self.U.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.batch_norm.eval()
        for module in (self.base_conv, self.batch_norm, self.P):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        for parameter in self.bottleneck_bn.parameters():
            parameter.requires_grad_(
                self.bottleneck_bn_enabled and self.bottleneck_bn_affine_trainable
            )
        return self

    def _forward_eval_adabn_stat_collection(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            raise RuntimeError(
                "BNPA eval AdaBN stat collection must run under torch.no_grad()"
            )
        if self.training or self.batch_norm.training:
            raise RuntimeError(
                "BNPA eval AdaBN stat collection requires eval-mode source BatchNorm"
            )
        bn_scale, bn_shift = self._bn_scale_shift()
        z = F.conv2d(
            x,
            self.base_conv.weight,
            self.base_conv.bias,
            stride=self.base_conv.stride,
            padding=self.base_conv.padding,
            dilation=self.base_conv.dilation,
            groups=self.base_conv.groups,
        )
        h = z * bn_scale.view(1, -1, 1, 1) + bn_shift.view(1, -1, 1, 1)
        projection_shift = (
            self.projection_shift
            if self.proj_mean_center
            else self.projection_shift[:0]
        )
        q = self.P(x)
        if projection_shift.numel() > 0:
            q = q + _channel_view(
                projection_shift.to(device=q.device, dtype=q.dtype), q
            )
        q_tilde = self.bottleneck_bn(q) if self.bottleneck_bn_enabled else q
        adapter_out = self.U(q_tilde)
        if self.adapter_pre_bn:
            s = (z + self.scale * adapter_out) * bn_scale.view(
                1, -1, 1, 1
            ) + bn_shift.view(1, -1, 1, 1)
        elif self.post_bn_adapter_scale_by_source_bn:
            s = h + self.scale * adapter_out * bn_scale.view(1, -1, 1, 1)
        else:
            s = h + self.scale * adapter_out
        y = self._apply_activation(s)
        self._update_source_bn_running_stats_from_preactivation(
            z, momentum_override=self._eval_adabn_stat_momentum
        )
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._eval_adabn_stat_collecting:
            return self._forward_eval_adabn_stat_collection(x)
        if self._adabn_calibrating:
            z = F.conv2d(
                x,
                self.base_conv.weight,
                self.base_conv.bias,
                stride=self.base_conv.stride,
                padding=self.base_conv.padding,
                dilation=self.base_conv.dilation,
                groups=self.base_conv.groups,
            )
            h = batch_norm_train_output_from_preactivation(z, self.batch_norm)
            return self._apply_activation(h)
        bn_scale, bn_shift = self._bn_scale_shift()
        projection_shift = (
            self.projection_shift
            if self.proj_mean_center
            else self.projection_shift[:0]
        )
        if self.optimized_post_bn_bnr_off_scaled:
            if (
                self.bottleneck_bn_enabled
                or self.adapter_pre_bn
                or not self.post_bn_adapter_scale_by_source_bn
            ):
                raise RuntimeError(
                    "optimized_post_bn_bnr_off_scaled requires post-BN placement, BN_r off, "
                    "fixed source-BN adapter scaling, and no SG/gpre/learnable-scale variants"
                )
            return BNPAConvBNAct2DPostBNScaledOptimizedFunction.apply(
                x,
                self.base_conv.weight,
                self.base_conv.bias,
                self.P.weight,
                self.P.bias,
                projection_shift,
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
        return BNPAConvBNAct2DFunction.apply(
            x,
            self.base_conv.weight,
            self.base_conv.bias,
            self.P.weight,
            self.P.bias,
            projection_shift,
            self.U.weight,
            self.U.bias,
            bn_scale,
            bn_shift,
            self.bottleneck_bn.weight,
            self.bottleneck_bn.bias,
            self.bottleneck_bn.running_mean,
            self.bottleneck_bn.running_var,
            self.bottleneck_bn.num_batches_tracked,
            conv_spec(self.P),
            conv_spec(self.U),
            conv_spec(self.base_conv),
            self.scale,
            self.activation_type,
            self.training,
            self.bottleneck_bn_enabled,
            self.bottleneck_bn.eps,
            (
                -1.0
                if self.bottleneck_bn.momentum is None
                else float(self.bottleneck_bn.momentum)
            ),
            self._bnpa_grad_q_capture_callback,
            self.adapter_pre_bn,
            self.post_bn_adapter_scale_by_source_bn,
        )


def _bottleneck_bn_backward(
    q: torch.Tensor,
    q_mean: torch.Tensor,
    q_invstd: torch.Tensor,
    bottleneck_weight: torch.Tensor,
    bottleneck_bias: torch.Tensor,
    u_weight: torch.Tensor,
    grad_adapter_out: torch.Tensor,
    u_stride: Tuple[int, int],
    u_padding: Tuple[int, int],
    u_dilation: Tuple[int, int],
    u_groups: int,
    training: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    reduce_dims = _channel_reduce_dims(q)
    q_hat = (q - _channel_view(q_mean, q)) * _channel_view(q_invstd, q)
    q_tilde = q_hat * _channel_view(bottleneck_weight, q) + _channel_view(
        bottleneck_bias, q
    )
    grad_q_tilde = nn_grad.conv2d_input(
        tuple(q_tilde.shape),
        u_weight,
        grad_adapter_out,
        stride=u_stride,
        padding=u_padding,
        dilation=u_dilation,
        groups=u_groups,
    )
    grad_gamma = (grad_q_tilde * q_hat).sum(dim=reduce_dims)
    grad_beta = grad_q_tilde.sum(dim=reduce_dims)
    grad_qhat = grad_q_tilde * _channel_view(bottleneck_weight, q)
    if training:
        elements_per_channel = _elements_per_channel(q)
        sum_grad_qhat = grad_qhat.sum(dim=reduce_dims)
        sum_grad_qhat_qhat = (grad_qhat * q_hat).sum(dim=reduce_dims)
        grad_q = (
            _channel_view(q_invstd, q)
            / float(elements_per_channel)
            * (
                float(elements_per_channel) * grad_qhat
                - _channel_view(sum_grad_qhat, q)
                - q_hat * _channel_view(sum_grad_qhat_qhat, q)
            )
        )
    else:
        grad_q = grad_qhat * _channel_view(q_invstd, q)
    return q_tilde.detach(), grad_q, grad_gamma, grad_beta


def _channel_reduce_dims(tensor: torch.Tensor) -> Tuple[int, ...]:
    return tuple(dim for dim in range(tensor.ndim) if dim != 1)


def _elements_per_channel(tensor: torch.Tensor) -> int:
    elements = 1
    for dim in _channel_reduce_dims(tensor):
        elements *= int(tensor.shape[dim])
    return elements


def _update_running_stats(
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    num_batches_tracked: torch.Tensor,
    batch_mean: torch.Tensor,
    batch_var: torch.Tensor,
    momentum: float,
) -> None:
    with torch.no_grad():
        if num_batches_tracked.numel() > 0:
            num_batches_tracked.add_(1)
            batches = int(num_batches_tracked.item())
        else:
            batches = 1
        factor = 1.0 / float(max(1, batches)) if momentum < 0 else float(momentum)
        mean = batch_mean.detach().to(
            device=running_mean.device, dtype=running_mean.dtype
        )
        var = batch_var.detach().to(device=running_var.device, dtype=running_var.dtype)
        running_mean.mul_(1.0 - factor).add_(mean, alpha=factor)
        running_var.mul_(1.0 - factor).add_(var, alpha=factor)
