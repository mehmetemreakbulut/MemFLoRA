"""Activation-memory (SRAM) profiling for the MemFLoRA experiments."""

from __future__ import annotations
from typing import Dict, Iterator, Optional, Sequence, Tuple
from torch import nn
from contextlib import contextmanager
import torch
from contextlib import nullcontext
from src.adapters.bnpa_sg import (
    bnpa_sg_capture,
    clear_bnpa_sg_captures,
    iter_bnpa_sg_modules,
    iter_bnpa_sg_projection_parameters,
)
from src.train import freeze_bn_eval
from src.utils.adabn import snapshot_running_buffers, restore_running_buffers
from src.utils.csv import write_csv


class SavedTensorProfiler:
    """Count saved bytes by storage kind without retaining diagnostic records."""

    def __init__(self, model: nn.Module) -> None:
        self.bytes_by_kind = {
            "activation": 0,
            "bitmask": 0,
            "parameter_or_buffer": 0,
            "constant": 0,
        }
        self._module_stack: list[tuple[str, str]] = []
        self._model_storages = {
            key
            for tensor in (*model.parameters(), *model.buffers())
            if (key := tensor_storage_key(tensor)) is not None
        }

    def add(self, tensor: torch.Tensor) -> tuple[str, int]:
        name, module_type = self._module_stack[-1] if self._module_stack else ("", "")
        if tensor_storage_key(tensor) in self._model_storages:
            kind = "parameter_or_buffer"
        elif tensor.dtype in (torch.bool, torch.uint8) and tensor.numel() > 0:
            kind = "bitmask"
        elif tensor.numel() == 0 or (
            tensor.dim() <= 1
            and tensor.numel() <= 8192
            and _is_constant_context(name, module_type)
        ):
            kind = "constant"
        else:
            kind = "activation"
        size = tensor_bytes(tensor)
        self.bytes_by_kind[kind] += size
        return kind, size

    @contextmanager
    def module_contexts(self, model: nn.Module) -> Iterator[None]:
        handles = []
        names = {module: name for name, module in model.named_modules()}

        def pre_hook(module: nn.Module, _inputs) -> None:
            self._module_stack.append((names[module], module.__class__.__name__))

        def post_hook(_module: nn.Module, _inputs, _output) -> None:
            if self._module_stack:
                self._module_stack.pop()

        for module in model.modules():
            handles.append(module.register_forward_pre_hook(pre_hook))
            handles.append(module.register_forward_hook(post_hook))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
            self._module_stack.clear()


def _is_constant_context(module_name: str, module_type: str) -> bool:
    """Preserve the paper profiler's small BN/custom-backward constant policy."""
    name = module_name.lower()
    module = module_type.lower()
    if "crossentropyloss" in f"{name}:{module}" or name == "loss":
        return False
    if any(
        token in module
        for token in (
            "activationminimalfixedpconvlora2d",
            "frozenconvbnactminimal2d",
            "bnpaconvbnact2d",
        )
    ):
        return True
    if "tresnetresidualrelubitpack2d" in module:
        return False
    if "tresnetevalbatchnormminimal2d" in module:
        return True
    if "loraedgeconv" in module or name.split(".")[-1] in {"a", "p", "b", "u"}:
        return False
    return is_batch_norm_context(name, module_type)


def tensor_storage_key(tensor: torch.Tensor) -> Optional[Tuple[str, int]]:
    if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        return None
    try:
        storage = tensor.untyped_storage()
        ptr = int(storage.data_ptr())
    except Exception:
        try:
            ptr = int(tensor.data_ptr())
        except Exception:
            return None
    if ptr == 0:
        return None
    return str(tensor.device), ptr


def is_batch_norm_context(name: str, module_type: str) -> bool:
    module = module_type.lower()
    if module_type in {"BatchNorm1d", "BatchNorm2d", "SyncBatchNorm"}:
        return True
    combined = f"{name}:{module}"
    if "batchnorm" in combined or "batch_norm" in combined:
        return True
    if (
        module == "bn"
        or module.startswith("bn")
        or module.endswith("bn")
        or "_bn" in module
    ):
        return True
    tokens = name.replace("/", ".").split(".")
    return any(
        token == "bn" or token.endswith("_bn") or token.startswith("bn_")
        for token in tokens
    )


FULL_SRAM_FIELDNAMES = [
    "dataset",
    "target_domain",
    "backbone",
    "method",
    "rank",
    "seed",
    "batch_size",
    "model_total_bytes",
    "trainable_parameter_bytes",
    "trainable_gradient_bytes_estimated",
    "optimizer_state_bytes_estimated",
    "activation_saved_bytes",
    "bitmask_saved_bytes",
    "constant_saved_bytes",
    "parameter_or_buffer_saved_bytes",
    "saved_tensor_total_bytes",
    "saved_backward_state_bytes",
    "saved_backward_peak_bytes",
    "gradient_live_peak_bytes",
    "bnpa_sg_overhead_bytes",
    "bnpa_sg_peak_sram_estimated_bytes",
    "full_sram_estimated_bytes",
    "peak_sram_estimated_bytes",
]


def tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def count_model_parameter_bytes(model: nn.Module) -> int:
    return int(sum(tensor_bytes(parameter) for parameter in model.parameters()))


def count_model_buffer_bytes(model: nn.Module) -> int:
    return int(sum(tensor_bytes(buffer) for buffer in model.buffers()))


def count_trainable_parameter_bytes(model: nn.Module) -> int:
    return int(
        sum(
            tensor_bytes(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )


def get_actual_gradient_bytes(model: nn.Module) -> int:
    return int(
        sum(
            tensor_bytes(parameter.grad)
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        )
    )


def profile_bnpa_sg_overhead(model: nn.Module, inputs: torch.Tensor) -> Dict[str, int]:
    modules = list(iter_bnpa_sg_modules(model))
    if not modules:
        return {"bnpa_sg_overhead_bytes": 0}
    captured_grad_q_bytes = int(
        sum(
            tensor_bytes(module.last_grad_q)
            for _name, module in modules
            if isinstance(module.last_grad_q, torch.Tensor)
        )
    )
    p_gradient_bytes = int(
        sum(
            tensor_bytes(parameter)
            for parameter in iter_bnpa_sg_projection_parameters(model)
        )
    )
    p_optimizer_state_bytes = 2 * p_gradient_bytes  # Adam's two moment buffers.
    remat_peak_overhead_bytes = captured_grad_q_bytes
    p_gradient_live_bytes = 0
    handles = []

    def make_hook(module: nn.Module):
        def hook(_module: nn.Module, hook_inputs: tuple[torch.Tensor, ...]) -> None:
            nonlocal remat_peak_overhead_bytes
            nonlocal p_gradient_live_bytes
            if not hook_inputs or not isinstance(hook_inputs[0], torch.Tensor):
                return
            x = hook_inputs[0]
            input_bytes = tensor_bytes(x)
            site_p_gradient_bytes = tensor_bytes(module.P.weight)
            if module.P.bias is not None:
                site_p_gradient_bytes += tensor_bytes(module.P.bias)
            remat_peak_overhead_bytes = max(
                remat_peak_overhead_bytes,
                captured_grad_q_bytes
                + p_gradient_live_bytes
                + input_bytes
                + site_p_gradient_bytes,
            )
            p_gradient_live_bytes += site_p_gradient_bytes
            remat_peak_overhead_bytes = max(
                remat_peak_overhead_bytes, captured_grad_q_bytes + p_gradient_live_bytes
            )

        return hook

    snapshots = snapshot_running_buffers(model)
    try:
        for _name, module in modules:
            handles.append(module.register_forward_pre_hook(make_hook(module)))
        with torch.no_grad():
            _ = model(inputs)
    finally:
        for handle in handles:
            handle.remove()
        restore_running_buffers(snapshots)
    sg_live_overhead_bytes = max(
        captured_grad_q_bytes + p_gradient_bytes, remat_peak_overhead_bytes
    )
    return {
        "bnpa_sg_remat_peak_overhead_bytes": int(remat_peak_overhead_bytes),
        "bnpa_sg_overhead_bytes": int(p_optimizer_state_bytes + sg_live_overhead_bytes),
    }


def profile_full_sram_for_step(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    metadata: Dict[str, object],
    force_bn_eval: bool = False,
) -> Dict[str, object]:
    """Profile the paper's analytical Adam SRAM accounting for one step.

    Saved parameter/buffer references are counted separately and excluded from
    ``saved_backward_state_bytes`` to avoid double-counting model memory.
    Workspace overhead is excluded, as in the benchmark's original defaults.
    """
    model_parameter_bytes = count_model_parameter_bytes(model)
    model_buffer_bytes = count_model_buffer_bytes(model)
    model_total_bytes = model_parameter_bytes + model_buffer_bytes
    trainable_parameter_bytes = count_trainable_parameter_bytes(model)
    gradient_bytes_estimated = trainable_parameter_bytes
    optimizer_state_estimated = (
        2 * trainable_parameter_bytes
    )  # Adam's two moment buffers.
    saved_backward_live_bytes = 0
    saved_backward_peak_bytes = 0
    gradient_live_bytes = 0
    gradient_live_peak_bytes = 0
    gradient_bytes_actual: object = ""
    has_trainable_parameters = any(
        parameter.requires_grad for parameter in model.parameters()
    )
    sg_modules = list(iter_bnpa_sg_modules(model))
    has_bnpa_sg = bool(sg_modules)
    bnpa_sg_peak_sram_estimated_bytes = 0
    peak_sram_estimated_bytes = model_total_bytes + optimizer_state_estimated
    grad_handles = []
    gradient_seen_parameter_ids: set[int] = set()

    def update_peak_sram_estimate() -> None:
        nonlocal peak_sram_estimated_bytes
        peak_sram_estimated_bytes = max(
            peak_sram_estimated_bytes,
            int(
                model_total_bytes
                + optimizer_state_estimated
                + saved_backward_live_bytes
                + gradient_live_bytes
            ),
        )

    def make_gradient_hook(parameter: nn.Parameter):
        parameter_id = id(parameter)

        def hook(grad: torch.Tensor) -> torch.Tensor:
            nonlocal gradient_live_bytes, gradient_live_peak_bytes
            if parameter_id not in gradient_seen_parameter_ids:
                gradient_seen_parameter_ids.add(parameter_id)
                gradient_live_bytes += tensor_bytes(grad)
                gradient_live_peak_bytes = max(
                    gradient_live_peak_bytes, gradient_live_bytes
                )
                update_peak_sram_estimate()
            return grad

        return hook

    if force_bn_eval:
        freeze_bn_eval(model)
    criterion = nn.CrossEntropyLoss()
    profiler = SavedTensorProfiler(model)
    model.zero_grad(set_to_none=True)
    if has_trainable_parameters:
        for parameter in model.parameters():
            if parameter.requires_grad:
                grad_handles.append(
                    parameter.register_hook(make_gradient_hook(parameter))
                )
    try:
        if has_trainable_parameters:

            def pack_hook(tensor: torch.Tensor):
                nonlocal saved_backward_live_bytes, saved_backward_peak_bytes
                kind, size = profiler.add(tensor)
                live_bytes = 0 if kind == "parameter_or_buffer" else size
                saved_backward_live_bytes += live_bytes
                saved_backward_peak_bytes = max(
                    saved_backward_peak_bytes, saved_backward_live_bytes
                )
                update_peak_sram_estimate()
                return {"tensor": tensor, "live_bytes": live_bytes, "released": False}

            def unpack_hook(payload) -> torch.Tensor:
                nonlocal saved_backward_live_bytes
                if isinstance(payload, dict):
                    live_bytes = int(payload.get("live_bytes", 0))
                    if live_bytes and not bool(payload.get("released", False)):
                        update_peak_sram_estimate()
                        saved_backward_live_bytes -= live_bytes
                        payload["released"] = True
                    return payload["tensor"]
                return payload

            sg_capture_context = (
                bnpa_sg_capture(model, enabled=True) if has_bnpa_sg else nullcontext()
            )
            with sg_capture_context:
                with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
                    with profiler.module_contexts(model):
                        logits = model(inputs)
                    loss = criterion(logits, labels)
                    loss.backward()
            gradient_bytes_actual = get_actual_gradient_bytes(model)
    finally:
        for handle in grad_handles:
            handle.remove()
    kind_bytes = profiler.bytes_by_kind
    activation_saved_bytes = int(kind_bytes["activation"])
    bitmask_saved_bytes = int(kind_bytes["bitmask"])
    constant_saved_bytes = int(kind_bytes["constant"])
    parameter_or_buffer_saved_bytes = int(kind_bytes["parameter_or_buffer"])
    saved_tensor_total_bytes = int(sum(kind_bytes.values()))
    saved_backward_state_bytes = (
        activation_saved_bytes + bitmask_saved_bytes + constant_saved_bytes
    )
    gradient_for_estimate = (
        int(gradient_bytes_actual)
        if isinstance(gradient_bytes_actual, int) and gradient_bytes_actual > 0
        else gradient_bytes_estimated
    )
    bnpa_sg_fields = {"bnpa_sg_overhead_bytes": 0}
    if has_bnpa_sg:
        bnpa_sg_fields = profile_bnpa_sg_overhead(model, inputs)
        bnpa_sg_peak_sram_estimated_bytes = int(
            model_total_bytes
            + optimizer_state_estimated
            + gradient_for_estimate
            + int(bnpa_sg_fields["bnpa_sg_overhead_bytes"])
        )
        clear_bnpa_sg_captures(model)
    full_sram_estimated_bytes = (
        model_total_bytes
        + gradient_bytes_estimated
        + optimizer_state_estimated
        + saved_backward_state_bytes
        + int(bnpa_sg_fields["bnpa_sg_overhead_bytes"])
    )
    peak_sram_estimated_bytes = max(
        peak_sram_estimated_bytes,
        model_total_bytes
        + optimizer_state_estimated
        + max(saved_backward_peak_bytes, saved_backward_state_bytes),
        model_total_bytes + optimizer_state_estimated + gradient_for_estimate,
        bnpa_sg_peak_sram_estimated_bytes,
    )
    model.zero_grad(set_to_none=True)
    row = {
        **metadata,
        "model_total_bytes": model_total_bytes,
        "trainable_parameter_bytes": trainable_parameter_bytes,
        "trainable_gradient_bytes_estimated": gradient_bytes_estimated,
        "optimizer_state_bytes_estimated": optimizer_state_estimated,
        "activation_saved_bytes": activation_saved_bytes,
        "bitmask_saved_bytes": bitmask_saved_bytes,
        "constant_saved_bytes": constant_saved_bytes,
        "parameter_or_buffer_saved_bytes": parameter_or_buffer_saved_bytes,
        "saved_tensor_total_bytes": saved_tensor_total_bytes,
        "saved_backward_state_bytes": saved_backward_state_bytes,
        "saved_backward_peak_bytes": saved_backward_peak_bytes,
        "gradient_live_peak_bytes": gradient_live_peak_bytes,
        "bnpa_sg_overhead_bytes": int(bnpa_sg_fields["bnpa_sg_overhead_bytes"]),
        "bnpa_sg_peak_sram_estimated_bytes": bnpa_sg_peak_sram_estimated_bytes,
        "full_sram_estimated_bytes": full_sram_estimated_bytes,
        "peak_sram_estimated_bytes": peak_sram_estimated_bytes,
    }
    return row


def append_full_sram_rows(path, rows: Sequence[Dict[str, object]]) -> None:
    write_csv(path, rows, FULL_SRAM_FIELDNAMES, append=True, extrasaction="ignore")
