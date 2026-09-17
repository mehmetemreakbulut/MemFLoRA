"""Small argument/input helpers and shared method-policy imports."""

from __future__ import annotations

from itertools import product

from torch import nn

from src.adapters.bnpa_sg import BNPASGConfig
from src.methods import (
    METHODS,
    BNPA_ADABN_AUDIT_METHODS,
    BNPA_SG_METHODS,
    RANK_FREE_METHODS,
    method_uses_adabn,
    method_forces_bn_eval,
)


def prepare_batch_inputs(x, backbone):
    if backbone == "mobilenet_v2":
        return x.unsqueeze(1)
    if backbone == "t_resnet_official":
        return x
    raise ValueError(f"Unsupported backbone: {backbone}")


class PreparedInputModel(nn.Module):
    """Use the same input shape for calibration/profiling as for training."""

    def __init__(self, model, backbone):
        super().__init__()
        self.model, self.backbone = model, backbone

    def forward(self, x):
        return self.model(prepare_batch_inputs(x, self.backbone))


def parse_str_list(values, flag_name):
    values = values if isinstance(values, (list, tuple)) else [values]
    parsed = [
        item.strip()
        for value in values
        for item in str(value).split(",")
        if item.strip()
    ]
    if not parsed:
        raise ValueError(f"{flag_name} must contain at least one value")
    return parsed


def parse_float_list(values, flag_name):
    return [float(value) for value in parse_str_list(values, flag_name)]


def parse_int_list(values, flag_name):
    return [int(value) for value in parse_str_list(values, flag_name)]


def format_float_label(value):
    return f"{float(value):g}".replace("-", "m").replace("+", "").replace(".", "p")


def parse_adabn_calib_batch_value(value):
    if str(value).strip().lower() == "all":
        return None
    batches = int(value)
    if batches < 0:
        raise ValueError("--adabn-calib-batches must be non-negative or 'all'")
    return batches


def build_bnpa_sg_configs(args):
    values = product(
        parse_float_list(args.bnpa_sg_p_lr, "--bnpa-sg-p-lr"),
        parse_float_list(args.bnpa_sg_p_weight_decay, "--bnpa-sg-p-weight-decay"),
    )
    return [BNPASGConfig(*config) for config in dict.fromkeys(values)]
