from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, List, Tuple
import torch
from torch import nn
from torch.nn import grad as nn_grad
from src.adapters.bnpa_conv_bn_act_2d import BNPAConvBNAct2D
from src.utils.adabn import snapshot_running_buffers, restore_running_buffers


@dataclass(frozen=True)
class BNPASGConfig:
    p_lr: float = 1e-3
    p_weight_decay: float = 0.0


@dataclass
class BNPASGStepStats:
    performed: bool = False
    sites_accumulated: int = 0


class BNPASGConvBNAct2D(BNPAConvBNAct2D):
    """BNPA block whose frozen projection receives a rematerialized exact gradient."""

    adapter_mode = "bnpa_sg"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.updated_by_sg = True
        self.adapter_method_name = "bnpa_sg"
        self._sg_capture_enabled = False
        self.last_grad_q: torch.Tensor | None = None
        self._bnpa_grad_q_capture_callback = self._capture_bottleneck_grad

    def clear_sg_capture(self) -> None:
        self.last_grad_q = None

    def set_sg_capture(self, enabled: bool) -> None:
        self._sg_capture_enabled = bool(enabled)

    def _capture_bottleneck_grad(self, grad_q: torch.Tensor) -> None:
        if not self._sg_capture_enabled:
            return
        with torch.no_grad():
            self.last_grad_q = grad_q.detach().clone()


def iter_bnpa_sg_modules(model: nn.Module) -> Iterable[tuple[str, BNPASGConvBNAct2D]]:
    for name, module in model.named_modules():
        if isinstance(module, BNPASGConvBNAct2D) and getattr(
            module, "updated_by_sg", False
        ):
            yield name, module


def iter_bnpa_sg_projection_parameters(model: nn.Module) -> List[nn.Parameter]:
    params: List[nn.Parameter] = []
    for _name, module in iter_bnpa_sg_modules(model):
        params.append(module.P.weight)
        if module.P.bias is not None:
            params.append(module.P.bias)
    return params


def set_bnpa_sg_projection_frozen(model: nn.Module) -> None:
    for parameter in iter_bnpa_sg_projection_parameters(model):
        parameter.requires_grad_(False)


def make_bnpa_sg_p_optimizer(
    model: nn.Module, config: BNPASGConfig
) -> torch.optim.Optimizer:
    params = iter_bnpa_sg_projection_parameters(model)
    if not params:
        raise RuntimeError("bnpa_sg found no managed projection parameters")
    return torch.optim.Adam(
        params, lr=float(config.p_lr), weight_decay=float(config.p_weight_decay)
    )


@contextmanager
def bnpa_sg_capture(model: nn.Module, enabled: bool = True):
    changed: List[Tuple[BNPASGConvBNAct2D, bool]] = []
    for _name, module in iter_bnpa_sg_modules(model):
        changed.append((module, bool(module._sg_capture_enabled)))
        module.clear_sg_capture()
        module.set_sg_capture(bool(enabled))
    try:
        yield
    finally:
        for module, previous in changed:
            module.set_sg_capture(previous)


def clear_bnpa_sg_captures(model: nn.Module) -> None:
    for _name, module in iter_bnpa_sg_modules(model):
        module.clear_sg_capture()


def accumulate_bnpa_sg_p_grad(
    model: nn.Module, inputs: torch.Tensor
) -> BNPASGStepStats:
    modules = list(iter_bnpa_sg_modules(model))
    if not modules:
        return BNPASGStepStats(performed=False)
    for name, module in modules:
        if module.last_grad_q is None:
            raise RuntimeError(
                "bnpa_sg requires a fresh dL/dq capture before "
                "rematerialized P-gradient accumulation "
                f"(missing site={getattr(module, 'adapter_site_name', name)})"
            )
    stats = BNPASGStepStats(performed=True)
    handles = []
    bn_buffers = snapshot_running_buffers(model)

    def make_hook(site_name: str, module: BNPASGConvBNAct2D):
        def hook(_module: nn.Module, hook_inputs: tuple[torch.Tensor, ...]) -> None:
            if not hook_inputs:
                raise RuntimeError(
                    f"bnpa_sg remat hook at {site_name} received no input tensor"
                )
            x = hook_inputs[0]
            if not isinstance(x, torch.Tensor):
                raise TypeError(
                    f"bnpa_sg remat hook at {site_name} expected tensor input, got {type(x)}"
                )
            grad_q = module.last_grad_q
            if grad_q is None:
                raise RuntimeError(
                    f"bnpa_sg remat hook at {site_name} has no captured dL/dq"
                )
            if int(grad_q.shape[0]) != int(x.shape[0]):
                raise RuntimeError(
                    f"bnpa_sg remat batch mismatch at {site_name}: "
                    f"x={tuple(x.shape)} grad_q={tuple(grad_q.shape)}"
                )
            grad_q = grad_q.to(device=x.device, dtype=x.dtype)
            grad_weight = nn_grad.conv2d_weight(
                x.detach(),
                tuple(module.P.weight.shape),
                grad_q,
                stride=module.P.stride,
                padding=module.P.padding,
                dilation=module.P.dilation,
                groups=module.P.groups,
            ).to(device=module.P.weight.device, dtype=module.P.weight.dtype)
            if module.P.weight.grad is None:
                module.P.weight.grad = grad_weight
            else:
                module.P.weight.grad.add_(grad_weight)
            if module.P.bias is not None:
                grad_bias = grad_q.sum(dim=(0, 2, 3)).to(
                    device=module.P.bias.device, dtype=module.P.bias.dtype
                )
                if module.P.bias.grad is None:
                    module.P.bias.grad = grad_bias
                else:
                    module.P.bias.grad.add_(grad_bias)
            stats.sites_accumulated += 1

        return hook

    try:
        for name, module in modules:
            site_name = str(getattr(module, "adapter_site_name", name))
            handles.append(
                module.register_forward_pre_hook(make_hook(site_name, module))
            )
        with torch.no_grad():
            _ = model(inputs)
    finally:
        for handle in handles:
            handle.remove()
        restore_running_buffers(bn_buffers)
        clear_bnpa_sg_captures(model)
    return stats
