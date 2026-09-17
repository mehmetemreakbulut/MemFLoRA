from __future__ import annotations
from typing import Dict, Optional, Sequence, Tuple
import torch
from torch import nn
from src.utils.csv import write_csv


def append_performance_rows(path, rows) -> None:
    write_csv(path, rows, PERFORMANCE_FIELDNAMES, append=True, extrasaction="ignore")


PERFORMANCE_FIELDNAMES = [
    "dataset",
    "target_domain",
    "batch_size",
    "backbone",
    "method",
    "rank",
    "seed",
    "forward_macs",
    "adapter_forward_macs",
    "backward_activation_macs_estimated",
    "trainable_weight_grad_macs_estimated",
    "adapter_backward_macs_estimated",
    "total_train_step_macs_estimated",
    "forward_fp32_bitops_proxy",
    "total_train_step_fp32_bitops_proxy",
    "trainable_params",
    "optimizer_update_bytes",
    "total_params",
]


def profile_one_batch_performance(
    model: nn.Module, inputs: torch.Tensor, metadata: Dict[str, object]
) -> Dict[str, object]:
    """Estimate one-batch abstract operation costs without changing training semantics.

    The profiler runs a single no-grad forward only to collect tensor shapes.
    Costs are analytical estimates from module shapes. MAC counts are multiply-
    accumulate counts, not wall-clock timings and not hardware-specific kernel
    measurements.
    """
    input_tensor = inputs.detach()
    row: Dict[str, object] = {
        field: metadata.get(field, "") for field in PERFORMANCE_FIELDNAMES
    }
    stats = _PerformanceStats()
    module_training = {module: bool(module.training) for module in model.modules()}
    handles = []

    def hook(module: nn.Module, module_inputs, module_output) -> None:
        first_input = _first_tensor(module_inputs)
        first_output = _first_tensor(module_output)
        if first_input is None or first_output is None:
            return
        _accumulate_module_cost(module, first_input, first_output, stats)

    custom_descendant_ids = _custom_counted_descendant_ids(model)
    for module in model.modules():
        if module is model:
            continue
        if id(module) in custom_descendant_ids:
            continue
        if _is_custom_counted_module(module):
            handles.append(module.register_forward_hook(hook))
            continue
        if _is_counted_module(module):
            handles.append(module.register_forward_hook(hook))
    try:
        model.eval()
        with torch.no_grad():
            model(input_tensor)
    finally:
        for handle in handles:
            handle.remove()
        for module, training in module_training.items():
            module.train(training)
    trainable_params = int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )
    total_params = int(sum(parameter.numel() for parameter in model.parameters()))
    bytes_per_param = _infer_parameter_bytes(model)
    method = str(metadata.get("base_method", metadata.get("method", "")))
    backward_activation_macs = (
        0 if method == "zero_shot" or trainable_params == 0 else int(stats.forward_macs)
    )
    trainable_weight_grad_macs = (
        0 if method == "zero_shot" else int(stats.trainable_weight_grad_macs)
    )
    sg_extra = _estimate_sg_extra_macs(stats.forward_macs, metadata)
    total_train_step_macs = int(
        stats.forward_macs
        + backward_activation_macs
        + trainable_weight_grad_macs
        + sg_extra
    )
    row.update(
        {
            "forward_macs": int(stats.forward_macs),
            "adapter_forward_macs": int(stats.adapter_forward_macs),
            "backward_activation_macs_estimated": int(backward_activation_macs),
            "trainable_weight_grad_macs_estimated": int(trainable_weight_grad_macs),
            "adapter_backward_macs_estimated": int(stats.adapter_backward_macs),
            "total_train_step_macs_estimated": int(total_train_step_macs),
            "forward_fp32_bitops_proxy": int(stats.forward_macs * 32 * 32),
            "total_train_step_fp32_bitops_proxy": int(total_train_step_macs * 32 * 32),
            "trainable_params": int(trainable_params),
            "optimizer_update_bytes": int(trainable_params * bytes_per_param),
            "total_params": int(total_params),
        }
    )
    return row


class _PerformanceStats:
    def __init__(self) -> None:
        self.forward_macs = 0
        self.adapter_forward_macs = 0
        self.trainable_weight_grad_macs = 0
        self.adapter_backward_macs = 0


def _first_tensor(value) -> Optional[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _is_counted_module(module: nn.Module) -> bool:
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        return True
    return _is_custom_counted_module(module)


def _is_custom_counted_module(module: nn.Module) -> bool:
    name = module.__class__.__name__
    return name in {
        "BNPAConvBNAct2D",
        "BNPASGConvBNAct2D",
        "ActivationMinimalFixedPConvLoRA2D",
        "FrozenConvBNActMinimal2D",
        "LoRACConv2d",
        "LoRAEdgeConv2d",
        "LoRAEdgeConv2dOptimized",
        "LoRAEdgeConv2dOptimizedV2",
    }


def _custom_counted_descendant_ids(model: nn.Module) -> set[int]:
    ids: set[int] = set()
    for module in model.modules():
        if not _is_custom_counted_module(module):
            continue
        for child in module.modules():
            if child is not module:
                ids.add(id(child))
    return ids


def _accumulate_module_cost(
    module: nn.Module,
    inputs: torch.Tensor,
    output: torch.Tensor,
    stats: _PerformanceStats,
) -> None:
    name = module.__class__.__name__
    if name in {
        "BNPAConvBNAct2D",
        "BNPASGConvBNAct2D",
        "ActivationMinimalFixedPConvLoRA2D",
    }:
        _accumulate_projected_cost(module, inputs, output, stats)
        return
    if name == "FrozenConvBNActMinimal2D":
        _accumulate_frozen_minimal_cost(module, inputs, output, stats)
        return
    if name == "LoRACConv2d":
        _accumulate_lora_c_cost(module, inputs, output, stats)
        return
    if name in {
        "LoRAEdgeConv2d",
        "LoRAEdgeConv2dOptimized",
        "LoRAEdgeConv2dOptimizedV2",
    }:
        _accumulate_lora_edge_cost(module, inputs, output, stats)
        return
    if isinstance(module, (nn.Conv1d, nn.Conv2d)):
        macs = _conv_macs_from_shape(module, inputs.shape, output.shape)
        stats.forward_macs += macs
        if module.weight.requires_grad:
            stats.trainable_weight_grad_macs += macs
        return
    if isinstance(module, nn.Linear):
        macs = _linear_macs(module, inputs, output)
        stats.forward_macs += macs
        if module.weight.requires_grad:
            stats.trainable_weight_grad_macs += macs
        return


def _accumulate_projected_cost(
    module: nn.Module,
    inputs: torch.Tensor,
    output: torch.Tensor,
    stats: _PerformanceStats,
) -> None:
    base_macs = _conv_macs_from_shape(module.base_conv, inputs.shape, output.shape)
    q_shape = _conv_output_shape_from_shape(module.P, inputs.shape)
    p_macs = _conv_macs_from_shape(module.P, inputs.shape, q_shape)
    u_shape = _conv_output_shape_from_shape(module.U, q_shape)
    u_macs = _conv_macs_from_shape(module.U, q_shape, u_shape)
    adapter_macs = p_macs + u_macs
    stats.forward_macs += base_macs + adapter_macs
    stats.adapter_forward_macs += adapter_macs
    stats.trainable_weight_grad_macs += u_macs
    stats.adapter_backward_macs += u_macs


def _accumulate_frozen_minimal_cost(
    module: nn.Module,
    inputs: torch.Tensor,
    output: torch.Tensor,
    stats: _PerformanceStats,
) -> None:
    conv = getattr(module, "conv", None) or getattr(module, "base_conv", None)
    if isinstance(conv, (nn.Conv1d, nn.Conv2d)):
        macs = _conv_macs_from_shape(conv, inputs.shape, output.shape)
        stats.forward_macs += macs


def _accumulate_lora_c_cost(
    module: nn.Module,
    inputs: torch.Tensor,
    output: torch.Tensor,
    stats: _PerformanceStats,
) -> None:
    base_macs = _conv_macs_from_shape(module.base_conv, inputs.shape, output.shape)
    delta_macs = _lora_c_delta_weight_macs(module)
    stats.forward_macs += base_macs + delta_macs
    stats.adapter_forward_macs += delta_macs
    stats.trainable_weight_grad_macs += base_macs + delta_macs
    stats.adapter_backward_macs += base_macs + delta_macs


def _accumulate_lora_edge_cost(
    module: nn.Module,
    inputs: torch.Tensor,
    output: torch.Tensor,
    stats: _PerformanceStats,
) -> None:
    base_macs = _conv_macs_from_shape(module.base_conv, inputs.shape, output.shape)
    if module.__class__.__name__.endswith("Optimized"):
        groups = int(module.base_conv.groups)
        suffix_shape = (
            int(output.shape[0]),
            groups * int(module.actual_tt_rank),
            *tuple(int(size) for size in output.shape[2:]),
        )
        suffix_macs = _conv_macs_from_weight_shape(
            input_shape=inputs.shape,
            output_shape=suffix_shape,
            kernel_size=_as_tuple(module.base_conv.kernel_size, len(inputs.shape) - 2),
            groups=groups,
        )
        g1_macs = int(output.numel() * int(module.actual_tt_rank))
        adapter_macs = suffix_macs + g1_macs
        trainable_grad = g1_macs
    else:
        adapter_macs = _lora_edge_delta_weight_macs(module)
        trainable_grad = base_macs + adapter_macs
    stats.forward_macs += base_macs + adapter_macs
    stats.adapter_forward_macs += adapter_macs
    stats.trainable_weight_grad_macs += trainable_grad
    stats.adapter_backward_macs += trainable_grad


def _conv_macs_from_shape(
    conv: nn.Module, input_shape: Sequence[int], output_shape: Sequence[int]
) -> int:
    kernel_size = _as_tuple(conv.kernel_size, len(input_shape) - 2)
    return _conv_macs_from_weight_shape(
        input_shape=input_shape,
        output_shape=output_shape,
        kernel_size=kernel_size,
        groups=int(conv.groups),
    )


def _conv_macs_from_weight_shape(
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    kernel_size: Sequence[int],
    groups: int,
) -> int:
    if len(input_shape) < 3 or len(output_shape) < 3:
        return 0
    batch = int(output_shape[0])
    out_channels = int(output_shape[1])
    out_positions = 1
    for size in output_shape[2:]:
        out_positions *= int(size)
    in_channels_per_group = int(input_shape[1]) // int(groups)
    kernel_ops = int(in_channels_per_group)
    for size in kernel_size:
        kernel_ops *= int(size)
    return int(batch * out_channels * out_positions * kernel_ops)


def _linear_macs(module: nn.Linear, inputs: torch.Tensor, output: torch.Tensor) -> int:
    output_elements = int(output.numel())
    return int(output_elements * int(module.in_features))


def _conv_output_shape_from_shape(
    conv: nn.Module, input_shape: Sequence[int]
) -> Tuple[int, ...]:
    spatial_dims = len(input_shape) - 2
    geometry = zip(
        input_shape[2:],
        _as_tuple(conv.kernel_size, spatial_dims),
        _as_tuple(conv.stride, spatial_dims),
        _as_tuple(conv.padding, spatial_dims),
        _as_tuple(conv.dilation, spatial_dims),
    )
    spatial = [
        int((size + 2 * pad - dil * (kernel - 1) - 1) // step + 1)
        for size, kernel, step, pad, dil in geometry
    ]
    return (int(input_shape[0]), int(conv.out_channels), *spatial)


def _as_tuple(value, ndim: int) -> Tuple[int, ...]:
    if isinstance(value, tuple):
        return tuple(int(item) for item in value[:ndim])
    return tuple(int(value) for _ in range(ndim))


def _lora_c_delta_weight_macs(module: nn.Module) -> int:
    rank = int(module.rank)
    if module.__class__.__name__.endswith("Conv1d"):
        return int(
            module.base_conv.out_channels
            * module.base_conv.in_channels
            * module.base_conv.kernel_size[0]
            * rank
        )
    kernel_h, kernel_w = module.base_conv.kernel_size
    return int(
        module.base_conv.out_channels
        * module.base_conv.in_channels
        * kernel_h
        * kernel_w
        * rank
    )


def _lora_edge_delta_weight_macs(module: nn.Module) -> int:
    rank = int(module.actual_tt_rank)
    out_channels = int(module.base_conv.out_channels)
    in_per_group = int(module.base_conv.in_channels // module.base_conv.groups)
    kernel_ops = in_per_group
    for size in _as_tuple(
        module.base_conv.kernel_size, len(module.base_conv.weight.shape) - 2
    ):
        kernel_ops *= int(size)
    return int(out_channels * rank * kernel_ops)


def _estimate_sg_extra_macs(forward_macs: int, metadata: Dict[str, object]) -> int:
    method = str(metadata.get("base_method", metadata.get("method", "")))
    if "sg" not in method:
        return 0
    return int(forward_macs)


def _infer_parameter_bytes(model: nn.Module) -> int:
    for parameter in model.parameters():
        return int(parameter.element_size())
    return 4
