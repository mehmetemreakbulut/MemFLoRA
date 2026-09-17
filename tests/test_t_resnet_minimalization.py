from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.adapter_injection import inject_adapters
from src.models.official_t_resnet_2d import OfficialTResNet2D
from src.train import calibrate_adabn, freeze_bn_eval
from src.adapters.minimal_blocks import (
    TResNetEvalBatchNormMinimal2D,
    TResNetResidualReLUBitpack2D,
)


def test_t_resnet_activation_minimal_replaces_residual_relu_and_standalone_bn() -> None:
    model = OfficialTResNet2D(input_channels=16, num_classes=5, n_feature_maps=8)
    replaced = inject_adapters(
        model,
        method="fixed_custom_bitmask_adabn_q3matched",
        rank=2,
        backbone="t_resnet",
        adapter_layers="all",
    )
    assert replaced
    assert isinstance(model.block1.act, TResNetResidualReLUBitpack2D)
    assert isinstance(model.block2.act, TResNetResidualReLUBitpack2D)
    assert isinstance(model.block3.act, TResNetResidualReLUBitpack2D)
    assert isinstance(model.block1.pre_bn, TResNetEvalBatchNormMinimal2D)
    assert isinstance(model.block2.pre_bn, TResNetEvalBatchNormMinimal2D)
    assert isinstance(model.block3.pre_bn, TResNetEvalBatchNormMinimal2D)
    assert isinstance(model.block3.shortcut, TResNetEvalBatchNormMinimal2D)
    assert model(torch.randn(2, 1, 16, 90)).shape == (2, 5)


def test_t_resnet_minimal_standalone_bn_adabn_calibration_and_freeze() -> None:
    model = OfficialTResNet2D(input_channels=16, num_classes=5, n_feature_maps=8)
    inject_adapters(
        model,
        method="fixed_custom_bitmask_adabn_q3matched",
        rank=2,
        backbone="t_resnet",
        adapter_layers="all",
    )
    before = model.block2.pre_bn.batch_norm.running_mean.detach().clone()
    loader = DataLoader(
        TensorDataset(torch.randn(8, 1, 16, 90), torch.zeros(8, dtype=torch.long)),
        batch_size=4,
    )
    calibrate_adabn(model, loader, device="cpu", calibration_mode="ema_no_reset")
    after = model.block2.pre_bn.batch_norm.running_mean.detach().clone()
    assert not torch.allclose(before, after)
    model.train()
    freeze_bn_eval(model)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) and not getattr(
            module, "_exclude_from_adabn", False
        ):
            assert not module.training
            if module.weight is not None:
                assert not module.weight.requires_grad
            if module.bias is not None:
                assert not module.bias.requires_grad
