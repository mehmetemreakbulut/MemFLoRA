from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, TypeVar, Union, get_args
import torch.nn.functional as F
from torch import nn
from src.adapters.activation_minimal_fixed_p_convlora_2d import (
    ActivationMinimalFixedPConvLoRA2D,
)
from src.adapters.bnpa_conv_bn_act_2d import BNPAConvBNAct2D
from src.adapters.bnpa_sg import BNPASGConvBNAct2D
from src.adapters.frozen_conv_bn_act_minimal_2d import FrozenConvBNActMinimal2D
from src.adapters.lora_edge import (
    LoRAEdgeConv2d,
    LoRAEdgeConv2dOptimized,
    LoRAEdgeConv2dOptimizedV2,
)
from src.adapters.minimal_blocks import (
    FrozenConvNoSave2D,
    LoRACConv2d,
    TResNetEvalBatchNormMinimal2D,
    TResNetResidualReLUBitpack2D,
    TinyTLLiteResidualWrapper2D,
)
from src.methods import (
    ADAPTER_METHODS,
    BNPA_METHODS,
    BNPA_SG_METHODS,
    FIXED_P_METHODS,
    FUSED_METHODS,
    TINYTL_METHODS,
    LORA_EDGE_METHODS,
)

BackboneName = Literal["mobilenet_v2", "t_resnet"]
LayerSelection = Literal[
    "all",
    "last",
    "middle_last",
    "pointwise_only",
    "depthwise_only",
    "stem_only",
    "depthwise_excluded",
]
ConvModule = Union[nn.Conv1d, nn.Conv2d]
BATCH_NORM_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
Candidate = TypeVar("Candidate")


@dataclass
class _FusedSite:
    parent: nn.Module
    conv_key: str
    conv: nn.Conv2d
    batch_norm: nn.BatchNorm2d
    activation: nn.Module
    bn_key: str | None = None
    act_key: str | None = None
    same_pad_conv: nn.Module | None = None

    def replace(self, fused: nn.Module) -> None:
        setattr(self.parent, self.conv_key, fused)
        if self.bn_key is not None:
            setattr(self.parent, self.bn_key, nn.Identity())
        if self.act_key is not None:
            setattr(self.parent, self.act_key, nn.Identity())


def freeze_all_parameters(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def unfreeze_batch_norm_affine(model: nn.Module) -> None:
    for module in model.modules():
        if not isinstance(module, BATCH_NORM_TYPES):
            continue
        if module.weight is not None:
            module.weight.requires_grad_(True)
        if module.bias is not None:
            module.bias.requires_grad_(True)


def unfreeze_batch_norm_bias(model: nn.Module) -> List[str]:
    names: List[str] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, BATCH_NORM_TYPES) or module.bias is None:
            continue
        module.bias.requires_grad_(True)
        prefix = f"{module_name}." if module_name else ""
        names.append(f"{prefix}bias")
    return names


def unfreeze_convolution_bias(model: nn.Module) -> List[str]:
    names: List[str] = []
    conv_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
    for module_name, module in model.named_modules():
        if not isinstance(module, conv_types) or module.bias is None:
            continue
        module.bias.requires_grad_(True)
        prefix = f"{module_name}." if module_name else ""
        names.append(f"{prefix}bias")
    return names


def inject_adapters(
    model: nn.Module,
    method: str,
    rank: int,
    backbone: BackboneName,
    adapter_layers: LayerSelection = "all",
    projection_init: str = "random_orthogonal",
    bnpa_bottleneck_bn: str = "on",
) -> List[str]:
    if method not in ADAPTER_METHODS:
        raise ValueError(f"Unsupported adapter method: {method}")
    freeze_all_parameters(model)
    if method == "bias_tuning":
        trainable_names = [
            f"bn_beta:{name}" for name in unfreeze_batch_norm_bias(model)
        ]
        trainable_names.extend(
            f"conv_bias:{name}" for name in unfreeze_convolution_bias(model)
        )
        if not trainable_names:
            raise RuntimeError(
                "bias_tuning did not find any BatchNorm or convolution bias parameters"
            )
        return trainable_names
    if method == "bn_tuning":
        unfreeze_batch_norm_affine(model)
        trainable_names = [
            f"bn_affine:{name}"
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        if not trainable_names:
            raise RuntimeError("bn_tuning did not find any BatchNorm affine parameters")
        return trainable_names
    if method in TINYTL_METHODS:
        if backbone != "mobilenet_v2":
            raise ValueError(
                f"{method} is currently implemented only for --backbone mobilenet_v2"
            )
        replaced = _inject_tinytl_lite_residual_mobilenet_v2(
            model,
            method=method,
            adapter_layers=adapter_layers,
            minimalize_frozen_blocks=method == "tinytl_lite_residual_bias_minimal",
            train_bn_minimalize_frozen_convs=method
            == "tinytl_lite_residual_bias_trainbn_minimal",
        )
        trainable_names = [
            f"bn_beta:{name}" for name in unfreeze_batch_norm_bias(model)
        ]
        trainable_names.extend(
            f"conv_bias:{name}" for name in unfreeze_convolution_bias(model)
        )
        trainable_names.extend(
            f"lite_residual:{name}"
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and ".lite_residual." in name
        )
        if not replaced:
            raise RuntimeError(
                f"{method} did not wrap any MobileNetV2 inverted residual blocks"
            )
        if not trainable_names:
            raise RuntimeError(
                f"{method} did not expose any trainable TinyTL parameters"
            )
        return replaced
    if method == "lora_c":
        replaced = _inject_lora_c(
            model, rank=rank, backbone=backbone, adapter_layers=adapter_layers
        )
        return _require_adapter_replacements(method, adapter_layers, replaced)
    if method in LORA_EDGE_METHODS:
        replaced = _inject_lora_edge(
            model,
            rank=rank,
            backbone=backbone,
            adapter_layers=adapter_layers,
            optimized_v2=method == "lora_edge_optimized_v2",
        )
        return _require_adapter_replacements(method, adapter_layers, replaced)
    if backbone == "mobilenet_v2" and method in FUSED_METHODS:
        replaced = _inject_mobilenet_v2_fused_sites(
            model,
            method=method,
            rank=rank,
            adapter_layers=adapter_layers,
            projection_init=projection_init,
            bnpa_bottleneck_bn=bnpa_bottleneck_bn,
        )
        return _require_adapter_replacements(method, adapter_layers, replaced)
    if backbone == "t_resnet" and method in FUSED_METHODS:
        replaced = _inject_t_resnet_fused_conv_bn_sites(
            model,
            method=method,
            rank=rank,
            adapter_layers=adapter_layers,
            projection_init=projection_init,
            bnpa_bottleneck_bn=bnpa_bottleneck_bn,
        )
        return _require_adapter_replacements(method, adapter_layers, replaced)
    raise ValueError(f"Unsupported adapter method: {method}")


def _require_adapter_replacements(
    method: str, adapter_layers: LayerSelection, replaced: List[str]
) -> List[str]:
    if not replaced:
        raise RuntimeError(
            f"{method} with adapter_layers={adapter_layers} did not inject any adapters. "
            "Check the placement mode and backbone support."
        )
    return replaced


def _eligible_convolutions(
    model: nn.Module,
    backbone: BackboneName,
    adapter_layers: LayerSelection,
) -> List[Tuple[str, ConvModule]]:
    candidates: List[Tuple[str, ConvModule]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and backbone == "mobilenet_v2":
            if _mobilenet_v2_layer_name_is_selected(name, module, adapter_layers):
                candidates.append((name, module))
        elif isinstance(module, nn.Conv2d) and backbone == "t_resnet":
            if module.groups == 1:
                candidates.append((name, module))
    return candidates


def _conv_is_supported(conv: nn.Conv2d, adapter_layers: LayerSelection) -> bool:
    if conv.padding_mode != "zeros":
        return False
    if adapter_layers == "pointwise_only":
        return conv.groups == 1 and conv.kernel_size == (1, 1)
    if adapter_layers == "depthwise_only":
        return _is_depthwise_conv2d(conv)
    if adapter_layers == "depthwise_excluded":
        return conv.groups == 1
    return True


def _is_depthwise_conv2d(conv: nn.Conv2d) -> bool:
    return conv.groups == conv.in_channels and conv.out_channels == conv.in_channels


def _mobilenet_v2_layer_name_is_selected(
    name: str, conv: nn.Conv2d, adapter_layers: LayerSelection
) -> bool:
    if adapter_layers == "stem_only":
        return name == "features.0.0" and _conv_is_supported(conv, adapter_layers)
    return _conv_is_supported(conv, adapter_layers)


def _minimalize_mobilenet_v2_conv_bn_act_sites_for_tinytl(
    model: nn.Module, activation_mask_mode: str
) -> None:
    candidates = _mobilenet_v2_fused_adapter_candidates(model, "all")
    for _name, site in candidates:
        site.replace(
            FrozenConvBNActMinimal2D(
                site.conv,
                site.batch_norm,
                site.activation,
                activation_mask_mode=activation_mask_mode,
                keep_base_bias_trainable=True,
                keep_bn_bias_trainable_in_eval=True,
            )
        )


def _minimalize_mobilenet_v2_trainbn_conv_sites_for_tinytl(model: nn.Module) -> None:
    candidates = _mobilenet_v2_fused_adapter_candidates(model, "all")
    for _name, site in candidates:
        setattr(
            site.parent,
            site.conv_key,
            FrozenConvNoSave2D(site.conv, keep_base_bias_trainable=True),
        )


def _inject_tinytl_lite_residual_mobilenet_v2(
    model: nn.Module,
    method: str,
    adapter_layers: LayerSelection,
    minimalize_frozen_blocks: bool = False,
    train_bn_minimalize_frozen_convs: bool = False,
) -> List[str]:
    if adapter_layers not in (
        "all",
        "pointwise_only",
        "depthwise_excluded",
        "last",
        "middle_last",
    ):
        raise ValueError(
            f"{method} supports adapter_layers all, pointwise_only, "
            "depthwise_excluded, last, or middle_last for MobileNetV2"
        )
    features = getattr(model, "features", None)
    if not isinstance(features, nn.Sequential):
        raise ValueError(
            f"{method} expects a MobileNetV2 model with nn.Sequential features"
        )
    if minimalize_frozen_blocks:
        _minimalize_mobilenet_v2_conv_bn_act_sites_for_tinytl(
            model, activation_mask_mode=_activation_mask_mode_for_method(method)
        )
    if train_bn_minimalize_frozen_convs:
        _minimalize_mobilenet_v2_trainbn_conv_sites_for_tinytl(model)
    candidates = [
        (f"features.{name}", block)
        for name, block in features.named_children()
        if block.__class__.__name__ == "InvertedResidual"
    ]
    selected = _select_by_position(candidates, adapter_layers)
    replaced: List[str] = []
    for module_name, block in selected:
        _replace_module(model, module_name, TinyTLLiteResidualWrapper2D(block))
        replaced.append(module_name)
    return replaced


def _select_by_position(
    candidates: List[Tuple[str, Candidate]],
    adapter_layers: LayerSelection,
) -> List[Tuple[str, Candidate]]:
    if adapter_layers in (
        "all",
        "pointwise_only",
        "depthwise_only",
        "stem_only",
        "depthwise_excluded",
    ):
        return candidates
    if not candidates:
        return []
    if adapter_layers == "last":
        return [candidates[-1]]
    if adapter_layers == "middle_last":
        middle_index = len(candidates) // 2
        selected = [candidates[middle_index]]
        if candidates[-1][0] != selected[0][0]:
            selected.append(candidates[-1])
        return selected
    raise ValueError(f"Unknown adapter_layers: {adapter_layers}")


def _inject_lora_c(
    model: nn.Module, rank: int, backbone: BackboneName, adapter_layers: LayerSelection
) -> List[str]:
    candidates = _eligible_convolutions(model, backbone, adapter_layers)
    selected = _select_by_position(candidates, adapter_layers)
    for name, conv in selected:
        replacement = LoRACConv2d(conv, rank=rank, alpha=None)
        _replace_module(model, name, replacement)
    return [name for name, _conv in selected]


def _inject_lora_edge(
    model: nn.Module,
    rank: int,
    backbone: BackboneName,
    adapter_layers: LayerSelection,
    optimized: bool = False,
    optimized_v2: bool = False,
) -> List[str]:
    candidates = _eligible_lora_edge_convolutions(model, backbone, adapter_layers)
    selected = _select_by_position(candidates, adapter_layers)
    conv2d_cls = (
        LoRAEdgeConv2dOptimizedV2
        if optimized_v2
        else LoRAEdgeConv2dOptimized if optimized else LoRAEdgeConv2d
    )
    for name, conv in selected:
        replacement = conv2d_cls(conv, tt_rank=rank)
        _replace_module(model, name, replacement)
    return [name for name, _conv in selected]


def _eligible_lora_edge_convolutions(
    model: nn.Module,
    backbone: BackboneName,
    adapter_layers: LayerSelection,
) -> List[Tuple[str, ConvModule]]:
    if adapter_layers not in get_args(LayerSelection):
        raise ValueError(f"Unsupported adapter_layers for lora_edge: {adapter_layers}")
    if adapter_layers in ("depthwise_only", "stem_only") and backbone not in (
        "mobilenet",
        "mobilenet_v2",
    ):
        raise ValueError(
            f"adapter_layers={adapter_layers} is currently supported only for MobileNet backbones"
        )
    candidates: List[Tuple[str, ConvModule]] = []
    for name, module in model.named_modules():
        if isinstance(module, LoRAEdgeConv2d):
            continue
        elif isinstance(module, nn.Conv2d) and backbone == "mobilenet_v2":
            if _mobilenet_v2_layer_name_is_selected(name, module, adapter_layers):
                candidates.append((name, module))
        elif isinstance(module, nn.Conv2d) and backbone == "t_resnet":
            if _conv_is_supported(module, adapter_layers):
                candidates.append((name, module))
    return candidates


def _inject_mobilenet_v2_fused_sites(
    model: nn.Module,
    method: str,
    rank: int,
    adapter_layers: LayerSelection,
    projection_init: str,
    bnpa_bottleneck_bn: str = "on",
) -> List[str]:
    activation_mask_mode = _activation_mask_mode_for_method(method)
    # Adapter placement and frozen-block minimalization are separate choices:
    # pointwise_only should train only pointwise adapters, while depthwise/stem
    # Conv-BN sites still need minimalized frozen backwards.
    candidates = _mobilenet_v2_fused_adapter_candidates(model, "all")
    adapter_candidates = [
        (name, site)
        for name, site in candidates
        if _mobilenet_v2_layer_name_is_selected(name, site.conv, adapter_layers)
    ]
    selected_names = {
        name for name, _site in _select_by_position(adapter_candidates, adapter_layers)
    }
    replaced_names: List[str] = []
    for module_name, site in candidates:
        if module_name in selected_names:
            fused = _make_fused_adapter(
                site, method, rank, projection_init, bnpa_bottleneck_bn
            )
            if method in BNPA_METHODS:
                fused.adapter_site_name = module_name
            site.replace(fused)
            replaced_names.append(module_name)
        else:
            site.replace(
                FrozenConvBNActMinimal2D(
                    site.conv,
                    site.batch_norm,
                    site.activation,
                    activation_mask_mode=activation_mask_mode,
                )
            )
    return replaced_names


def _mobilenet_v2_fused_adapter_candidates(
    model: nn.Module,
    adapter_layers: LayerSelection,
) -> List[Tuple[str, _FusedSite]]:
    candidates: List[Tuple[str, _FusedSite]] = []
    for parent_name, parent in model.named_modules():
        if not isinstance(parent, nn.Sequential):
            continue
        items = list(parent._modules.items())
        index = 0
        while index < len(items) - 1:
            conv_key, conv = items[index]
            bn_key, batch_norm = items[index + 1]
            if not isinstance(conv, nn.Conv2d) or not isinstance(
                batch_norm, nn.BatchNorm2d
            ):
                index += 1
                continue
            act_key: Optional[str] = None
            activation: nn.Module = nn.Identity()
            if index + 2 < len(items):
                possible_act_key, possible_activation = items[index + 2]
                if isinstance(possible_activation, (nn.ReLU, nn.ReLU6, nn.Identity)):
                    act_key = str(possible_act_key)
                    activation = possible_activation
            module_name = f"{parent_name}.{conv_key}" if parent_name else str(conv_key)
            if _mobilenet_v2_layer_name_is_selected(module_name, conv, adapter_layers):
                candidates.append(
                    (
                        module_name,
                        _FusedSite(
                            parent,
                            str(conv_key),
                            conv,
                            batch_norm,
                            activation,
                            str(bn_key),
                            act_key,
                        ),
                    )
                )
            index += 3 if act_key is not None else 2
    return candidates


class _TResNetSamePadFusedAdapter2D(nn.Module):
    """Apply official temporal same-padding before an existing fused adapter block."""

    def __init__(self, same_pad_conv: nn.Module, fused_block: nn.Module) -> None:
        super().__init__()
        self.kernel_t = int(getattr(same_pad_conv, "kernel_t"))
        self.fused_block = fused_block

    @staticmethod
    def _same_pad_1d(kernel: int) -> Tuple[int, int]:
        pad = int(kernel) - 1
        return pad // 2, pad - (pad // 2)

    def forward(self, x):
        pad_before, pad_after = self._same_pad_1d(self.kernel_t)
        if pad_before or pad_after:
            x = F.pad(x, (0, 0, pad_before, pad_after))
        return self.fused_block(x)


def _inject_t_resnet_fused_conv_bn_sites(
    model: nn.Module,
    method: str,
    rank: int,
    adapter_layers: LayerSelection,
    projection_init: str,
    bnpa_bottleneck_bn: str = "on",
) -> List[str]:
    candidates = _t_resnet_fused_adapter_candidates(model)
    selected = _select_by_position(candidates, adapter_layers)
    # Candidate order determines projection initialization RNG consumption.
    selected_names = {name for name, _site in selected}
    replaced: List[str] = []
    for name, site in candidates:
        if name not in selected_names:
            continue
        fused = _make_fused_adapter(
            site, method, rank, projection_init, bnpa_bottleneck_bn
        )
        inner_fused = getattr(fused, "fused_block", fused)
        if isinstance(inner_fused, BNPAConvBNAct2D):
            inner_fused.adapter_site_name = name
        site.replace(fused)
        replaced.append(name)
    _minimalize_t_resnet_standalone_bn_relu(model)
    return replaced


def _minimalize_t_resnet_standalone_bn_relu(
    model: nn.Module,
    keep_bn_bias_trainable_in_eval: bool = False,
) -> None:
    for _block_name, block in model.named_modules():
        if not _looks_like_official_t_resnet_block(block):
            continue
        act = getattr(block, "act", None)
        if isinstance(act, nn.ReLU):
            setattr(block, "act", TResNetResidualReLUBitpack2D())
        pre_bn = getattr(block, "pre_bn", None)
        if isinstance(pre_bn, nn.BatchNorm2d):
            setattr(
                block,
                "pre_bn",
                TResNetEvalBatchNormMinimal2D(
                    pre_bn,
                    keep_bn_bias_trainable_in_eval=keep_bn_bias_trainable_in_eval,
                ),
            )
        shortcut = getattr(block, "shortcut", None)
        if isinstance(shortcut, nn.BatchNorm2d):
            setattr(
                block,
                "shortcut",
                TResNetEvalBatchNormMinimal2D(
                    shortcut,
                    keep_bn_bias_trainable_in_eval=keep_bn_bias_trainable_in_eval,
                ),
            )
        elif (
            isinstance(shortcut, nn.Sequential)
            and len(shortcut) >= 2
            and isinstance(shortcut[1], nn.BatchNorm2d)
        ):
            shortcut[1] = TResNetEvalBatchNormMinimal2D(
                shortcut[1],
                keep_bn_bias_trainable_in_eval=keep_bn_bias_trainable_in_eval,
            )


def _t_resnet_fused_adapter_candidates(
    model: nn.Module,
) -> List[Tuple[str, _FusedSite]]:
    candidates: List[Tuple[str, _FusedSite]] = []
    for block_name, block in model.named_modules():
        if not _looks_like_official_t_resnet_block(block):
            continue
        for attr in ("conv8", "conv5"):
            module = getattr(block, attr)
            same_pad_conv = getattr(module, "conv", None)
            conv = _inner_same_pad_conv2d(same_pad_conv)
            batch_norm = getattr(module, "bn", None)
            activation = getattr(module, "act", None)
            if (
                isinstance(conv, nn.Conv2d)
                and isinstance(batch_norm, nn.BatchNorm2d)
                and isinstance(activation, (nn.ReLU, nn.ReLU6, nn.Identity))
            ):
                candidates.append(
                    (
                        f"{block_name}.{attr}.conv.conv",
                        _FusedSite(
                            block,
                            attr,
                            conv,
                            batch_norm,
                            activation,
                            same_pad_conv=same_pad_conv,
                        ),
                    )
                )
        same_pad_conv = getattr(block, "conv3", None)
        conv = _inner_same_pad_conv2d(same_pad_conv)
        batch_norm = getattr(block, "bn3", None)
        if isinstance(conv, nn.Conv2d) and isinstance(batch_norm, nn.BatchNorm2d):
            site = _FusedSite(
                block,
                "conv3",
                conv,
                batch_norm,
                nn.Identity(),
                bn_key="bn3",
                same_pad_conv=same_pad_conv,
            )
            candidates.append((f"{block_name}.conv3.conv", site))
        shortcut = getattr(block, "shortcut", None)
        if isinstance(shortcut, nn.Sequential) and len(shortcut) >= 2:
            conv = shortcut[0]
            batch_norm = shortcut[1]
            if isinstance(conv, nn.Conv2d) and isinstance(batch_norm, nn.BatchNorm2d):
                candidates.append(
                    (
                        f"{block_name}.shortcut.0",
                        _FusedSite(
                            shortcut, "0", conv, batch_norm, nn.Identity(), bn_key="1"
                        ),
                    )
                )
    return candidates


def _looks_like_official_t_resnet_block(module: nn.Module) -> bool:
    return all(
        hasattr(module, attr)
        for attr in ("pre_bn", "conv8", "conv5", "conv3", "bn3", "shortcut")
    )


def _inner_same_pad_conv2d(module: object) -> Optional[nn.Conv2d]:
    conv = getattr(module, "conv", None)
    kernel_t = getattr(module, "kernel_t", None)
    if isinstance(conv, nn.Conv2d) and kernel_t is not None:
        return conv
    return None


def _make_fused_adapter(
    site: _FusedSite,
    method: str,
    rank: int,
    projection_init: str,
    bnpa_bottleneck_bn: str,
) -> nn.Module:
    conv, batch_norm, activation = site.conv, site.batch_norm, site.activation
    if method in FIXED_P_METHODS:
        fused: nn.Module = ActivationMinimalFixedPConvLoRA2D(
            conv, batch_norm, activation, rank=rank, alpha=None
        )
    elif method in BNPA_METHODS:
        spec = BNPA_METHODS[method]
        if method in BNPA_SG_METHODS:
            block_cls = BNPASGConvBNAct2D
        else:
            block_cls = BNPAConvBNAct2D
        fused = block_cls(
            conv,
            batch_norm,
            activation,
            rank=rank,
            projection_init=spec.projection_init or projection_init,
            proj_mean_center=spec.mean_center,
            bnpa_bottleneck_bn=spec.bottleneck_bn or bnpa_bottleneck_bn,
            adapter_pre_bn=spec.pre_bn,
            fa_port_layout=spec.fa_layout,
            post_bn_adapter_scale_by_source_bn=spec.scaled,
            optimized_post_bn_bnr_off_scaled=spec.optimized,
        )
        if method in BNPA_SG_METHODS:
            fused.updated_by_sg = True  # type: ignore[attr-defined]
            fused.adapter_method_name = method  # type: ignore[attr-defined]
    else:
        raise ValueError(f"Unsupported T-ResNet fused adapter method: {method}")
    if site.same_pad_conv is None:
        return fused
    return _TResNetSamePadFusedAdapter2D(site.same_pad_conv, fused)


def _activation_mask_mode_for_method(method: str) -> str:
    return "bitpack" if method in FUSED_METHODS else "bool"


def _replace_module(model: nn.Module, name: str, replacement: nn.Module) -> None:
    parent_name, _separator, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child_name, replacement)
