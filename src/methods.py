"""Method names, adapter settings, and benchmark policies without model imports."""

from dataclasses import dataclass


@dataclass(frozen=True)
class _BNPAMethod:
    projection_init: str | None = None
    mean_center: bool | None = False
    bottleneck_bn: str | None = None
    pre_bn: bool = False
    fa_layout: bool = False
    scaled: bool = False
    optimized: bool = False


# Unspecified settings retain the caller's initialization and bottleneck BN.
BNPA_METHODS = {
    "bnpa": _BNPAMethod(mean_center=None),
    "bnpa_q3init": _BNPAMethod(projection_init="fa_port_normal"),
    "bnpa_sg": _BNPAMethod(mean_center=None),
    "bnpa_sg_q3init": _BNPAMethod(projection_init="fa_port_normal"),
    "bnpa_q1": _BNPAMethod(bottleneck_bn="on", pre_bn=True, fa_layout=True),
    "bnpa_q3": _BNPAMethod(bottleneck_bn="off", pre_bn=True, fa_layout=True),
    "bnpa_q3_sg": _BNPAMethod(bottleneck_bn="off", pre_bn=True, fa_layout=True),
    "bnpa_fa_postbn": _BNPAMethod(bottleneck_bn="on", fa_layout=True),
    "bnpa_fa_postbn_scaled": _BNPAMethod(
        bottleneck_bn="on", fa_layout=True, scaled=True
    ),
    "bnpa_fa_postbn_bnr_off": _BNPAMethod(bottleneck_bn="off", fa_layout=True),
    "bnpa_fa_postbn_bnr_off_scaled": _BNPAMethod(
        bottleneck_bn="off", fa_layout=True, scaled=True
    ),
    "bnpa_fa_postbn_bnr_off_optimized": _BNPAMethod(
        bottleneck_bn="off", fa_layout=True, scaled=True, optimized=True
    ),
    "bnpa_fa_postbn_bnr_off_scaled_orthinit": _BNPAMethod(
        projection_init="fa_port_random_orthogonal",
        bottleneck_bn="off",
        fa_layout=True,
        scaled=True,
    ),
    "fixed_custom_bitmask_adabn_audit_q3matched_postBN_scaled": _BNPAMethod(
        projection_init="fa_port_normal",
        bottleneck_bn="off",
        scaled=True,
    ),
}
BNPA_SG_METHODS = {"bnpa_sg", "bnpa_sg_q3init", "bnpa_q3_sg"}
FIXED_P_METHODS = {"fixed_custom_bitmask_adabn_q3matched"}
FUSED_METHODS = FIXED_P_METHODS | BNPA_METHODS.keys()
TINYTL_METHODS = {
    "tinytl_lite_residual_bias",
    "tinytl_lite_residual_bias_minimal",
    "tinytl_lite_residual_bias_trainbn_minimal",
}
LORA_EDGE_METHODS = {"lora_edge_optimized", "lora_edge_optimized_v2"}
ADAPTER_METHODS = (
    FUSED_METHODS
    | TINYTL_METHODS
    | LORA_EDGE_METHODS
    | {"bias_tuning", "bn_tuning", "lora_c"}
)

# Keep this tuple ordered: it defines CLI choices, not an unordered membership set.
METHODS = (
    "zero_shot",
    "full",
    "bias_tuning",
    "bn_tuning",
    "tinytl_lite_residual_bias",
    "tinytl_lite_residual_bias_minimal",
    "tinytl_lite_residual_bias_trainbn_minimal",
    "lora_c",
    "fixed_custom_bitmask_adabn_q3matched",
    "bnpa",
    "bnpa_q3init",
    "bnpa_fa_postbn",
    "bnpa_fa_postbn_bnr_off",
    "bnpa_fa_postbn_bnr_off_scaled",
    "bnpa_fa_postbn_bnr_off_optimized",
    "bnpa_fa_postbn_bnr_off_scaled_orthinit",
    "bnpa_fa_postbn_scaled",
    "fixed_custom_bitmask_adabn_audit_q3matched_postBN_scaled",
    "bnpa_q1",
    "bnpa_q3",
    "bnpa_q3_sg",
    "bnpa_sg",
    "bnpa_sg_q3init",
    "lora_edge_optimized",
    "lora_edge_optimized_v2",
)
BNPA_ADABN_AUDIT_METHODS = FUSED_METHODS & set(METHODS)
ADABN_METHODS = frozenset(
    BNPA_ADABN_AUDIT_METHODS | {"tinytl_lite_residual_bias_minimal"}
)
BN_EVAL_METHODS = (
    ADABN_METHODS  # Equal today; calibration and training remain separate policies.
)
RANK_FREE_METHODS = TINYTL_METHODS | {"bias_tuning", "bn_tuning"}


def method_uses_adabn(method):
    return method in ADABN_METHODS


def method_forces_bn_eval(method):
    return method in BN_EVAL_METHODS
