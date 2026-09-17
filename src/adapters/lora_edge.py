from __future__ import annotations
import copy
from typing import Tuple
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
from torch.nn import grad as nn_grad
from src.adapters._common import ConvSpec, conv_spec
from typing import List, Sequence


class _LoRAEdgeOptimizedConv2dFunction(Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        base_weight: torch.Tensor,
        base_bias: torch.Tensor | None,
        tt_suffix: torch.Tensor,
        g1: torch.Tensor,
        spec: ConvSpec,
        scale: float,
    ) -> torch.Tensor:
        base_out = F.conv2d(
            x,
            base_weight,
            base_bias,
            stride=spec.stride,
            padding=spec.padding,
            dilation=spec.dilation,
            groups=spec.groups,
        )
        tt_suffix_grouped = tt_suffix.repeat(spec.groups, 1, 1, 1)
        suffix_projection = F.conv2d(
            x,
            tt_suffix_grouped,
            None,
            stride=spec.stride,
            padding=spec.padding,
            dilation=spec.dilation,
            groups=spec.groups,
        )
        batch_size, _channels, out_height, out_width = suffix_projection.shape
        rank = int(g1.shape[1])
        out_channels = int(g1.shape[0])
        out_channels_per_group = out_channels // spec.groups
        suffix_projection_view = suffix_projection.view(
            batch_size, spec.groups, rank, out_height, out_width
        )
        g1_view = g1.view(spec.groups, out_channels_per_group, rank)
        adapter_out = torch.einsum(
            "bgqhw,goq->bgohw", suffix_projection_view, g1_view
        ).reshape(batch_size, out_channels, out_height, out_width)
        ctx.needs_x_grad = bool(ctx.needs_input_grad[0])
        ctx.input_shape = tuple(x.shape)
        ctx.g1_shape = tuple(g1.shape)
        ctx.suffix_projection_shape = tuple(suffix_projection.shape)
        ctx.stride = spec.stride
        ctx.padding = spec.padding
        ctx.dilation = spec.dilation
        ctx.groups = spec.groups
        ctx.scale = float(scale)
        if ctx.needs_x_grad:
            ctx.save_for_backward(suffix_projection, base_weight, tt_suffix, g1)
        else:
            ctx.save_for_backward(suffix_projection)
        return base_out + ctx.scale * adapter_out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        saved_tensors = ctx.saved_tensors
        suffix_projection = saved_tensors[0]
        batch_size, _channels, out_height, out_width = suffix_projection.shape
        groups = int(ctx.groups)
        _out_channels, rank = ctx.g1_shape
        out_channels_per_group = int(ctx.g1_shape[0] // groups)
        scaled_grad_output = grad_output * ctx.scale
        scaled_grad_view = scaled_grad_output.view(
            batch_size, groups, out_channels_per_group, out_height, out_width
        )
        suffix_projection_view = suffix_projection.view(
            batch_size, groups, rank, out_height, out_width
        )
        grad_g1 = torch.einsum(
            "bgohw,bgqhw->goq", scaled_grad_view, suffix_projection_view
        ).reshape(ctx.g1_shape)
        grad_x = None
        if ctx.needs_x_grad:
            base_weight, tt_suffix, g1 = saved_tensors[1:]
            g1_view = g1.view(groups, out_channels_per_group, rank)
            grad_t = torch.einsum(
                "bgohw,goq->bgqhw", scaled_grad_view, g1_view
            ).reshape(ctx.suffix_projection_shape)
            tt_suffix_grouped = tt_suffix.repeat(groups, 1, 1, 1)
            grad_x_adapter = nn_grad.conv2d_input(
                ctx.input_shape,
                tt_suffix_grouped,
                grad_t,
                stride=ctx.stride,
                padding=ctx.padding,
                dilation=ctx.dilation,
                groups=groups,
            )
            grad_x_base = nn_grad.conv2d_input(
                ctx.input_shape,
                base_weight,
                grad_output,
                stride=ctx.stride,
                padding=ctx.padding,
                dilation=ctx.dilation,
                groups=groups,
            )
            grad_x = grad_x_base + grad_x_adapter
        return (grad_x, None, None, None, grad_g1, None, None)


class LoRAEdgeConv2d(nn.Module):
    """Pure LoRA-Edge wrapper for Conv2d."""

    adapter_mode = "lora_edge"

    def __init__(self, base_conv: nn.Conv2d, tt_rank: int) -> None:
        super().__init__()
        if tt_rank <= 0:
            raise ValueError("tt_rank must be positive")
        if base_conv.padding_mode != "zeros":
            raise ValueError("LoRAEdgeConv2d currently supports only zero padding")
        self.rank = int(tt_rank)
        self.tt_rank = int(tt_rank)
        self.scale = float(1.0)
        self.base_conv = copy.deepcopy(base_conv)
        for parameter in self.base_conv.parameters():
            parameter.requires_grad_(False)
        suffix, actual_rank = _tt_suffix_from_weight(
            self.base_conv.weight.detach(), self.tt_rank
        )
        dtype = self.base_conv.weight.dtype
        device = self.base_conv.weight.device
        self.actual_tt_rank = int(actual_rank)
        self.register_buffer("tt_suffix", suffix.to(device=device, dtype=dtype))
        self.g1 = nn.Parameter(
            torch.zeros(
                self.base_conv.out_channels,
                self.actual_tt_rank,
                device=device,
                dtype=dtype,
            )
        )

    @property
    def in_channels(self) -> int:
        return self.base_conv.in_channels

    @property
    def out_channels(self) -> int:
        return self.base_conv.out_channels

    def delta_weight(self) -> torch.Tensor:
        return torch.einsum("or,rijk->oijk", self.g1, self.tt_suffix)

    def merged_weight(self) -> torch.Tensor:
        return self.base_conv.weight.detach() + self.scale * self.delta_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self.merged_weight(),
            self.base_conv.bias,
            stride=self.base_conv.stride,
            padding=self.base_conv.padding,
            dilation=self.base_conv.dilation,
            groups=self.base_conv.groups,
        )


class LoRAEdgeConv2dOptimized(LoRAEdgeConv2d):
    """LoRA-Edge Conv2d that saves the frozen TT suffix projection, not input x."""

    adapter_mode = "lora_edge_optimized"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _LoRAEdgeOptimizedConv2dFunction.apply(
            x,
            self.base_conv.weight,
            self.base_conv.bias,
            self.tt_suffix,
            self.g1,
            conv_spec(self.base_conv),
            1.0,
        )


class LoRAEdgeConv2dOptimizedV2(LoRAEdgeConv2d):
    """Speed-oriented LoRA-Edge Conv2d using the fused merged-weight forward."""

    adapter_mode = "lora_edge_optimized_v2"


def _tt_suffix_from_weight(
    weight: torch.Tensor, target_rank: int
) -> Tuple[torch.Tensor, int]:
    cores = tt_svd(weight.detach(), target_rank=target_rank)
    suffix = contract_tt_suffix(cores[1:])
    return suffix, int(cores[0].shape[-1])


def tt_svd(tensor: torch.Tensor, target_rank: int) -> List[torch.Tensor]:
    if target_rank < 1:
        raise ValueError("target_rank must be >= 1")
    if tensor.dim() < 3:
        raise NotImplementedError(
            "tt_svd currently requires tensors with at least 3 dimensions"
        )
    shape = tuple(int(dim) for dim in tensor.shape)
    current = tensor.detach().to(dtype=torch.float32)
    r_prev = 1
    cores: List[torch.Tensor] = []
    for axis, n_k in enumerate(shape[:-1]):
        current = current.reshape(r_prev * n_k, -1)
        u, singular_values, vh = torch.linalg.svd(current, full_matrices=False)
        attainable_rank = min(int(u.shape[1]), int(target_rank))
        u = u[:, :attainable_rank]
        singular_values = singular_values[:attainable_rank]
        vh = vh[:attainable_rank, :]
        cores.append(u.reshape(r_prev, n_k, attainable_rank))
        current = singular_values.unsqueeze(1) * vh
        r_prev = attainable_rank
    cores.append(current.reshape(r_prev, shape[-1], 1))
    return cores


def contract_tt_suffix(cores_without_g1: Sequence[torch.Tensor]) -> torch.Tensor:
    if not cores_without_g1:
        raise ValueError("cores_without_g1 must be non-empty")
    current = cores_without_g1[0]
    for core in cores_without_g1[1:]:
        current = torch.tensordot(current, core, dims=([-1], [0]))
    if current.shape[-1] != 1:
        raise ValueError("TT suffix must end in rank 1")
    return current.squeeze(-1)
