from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.adapters.bnpa_conv_bn_act_2d import BNPAConvBNAct2D


def make_block(activation: nn.Module = nn.ReLU6()) -> BNPAConvBNAct2D:
    torch.manual_seed(11)
    conv = nn.Conv2d(4, 6, kernel_size=3, padding=1, bias=True)
    batch_norm = nn.BatchNorm2d(6)
    with torch.no_grad():
        batch_norm.running_mean.copy_(torch.randn(6))
        batch_norm.running_var.copy_(torch.rand(6) + 0.5)
        batch_norm.weight.copy_(torch.randn(6))
        batch_norm.bias.copy_(torch.randn(6))
    block = BNPAConvBNAct2D(conv, batch_norm, activation, rank=3)
    with torch.no_grad():
        block.U.weight.normal_(mean=0.0, std=0.1)
    return block


def naive_bnpa_reference(
    block: BNPAConvBNAct2D,
    x: torch.Tensor,
    u_weight: torch.Tensor,
    bottleneck_weight: torch.Tensor,
    bottleneck_bias: torch.Tensor,
) -> torch.Tensor:
    z = F.conv2d(
        x,
        block.base_conv.weight.detach(),
        block.base_conv.bias.detach(),
        stride=block.base_conv.stride,
        padding=block.base_conv.padding,
        dilation=block.base_conv.dilation,
        groups=block.base_conv.groups,
    )
    bn_scale, bn_shift = block._bn_scale_shift()
    h = z * bn_scale.detach().view(1, -1, 1, 1) + bn_shift.detach().view(1, -1, 1, 1)
    q = F.conv2d(
        x,
        block.P.weight.detach(),
        block.P.bias.detach() if block.P.bias is not None else None,
        stride=block.P.stride,
        padding=block.P.padding,
        dilation=block.P.dilation,
        groups=block.P.groups,
    )
    mean = q.mean(dim=(0, 2, 3))
    var = q.var(dim=(0, 2, 3), unbiased=False)
    q_hat = (q - mean.view(1, -1, 1, 1)) * torch.rsqrt(
        var + block.bottleneck_bn.eps
    ).view(1, -1, 1, 1)
    q_tilde = q_hat * bottleneck_weight.view(1, -1, 1, 1) + bottleneck_bias.view(
        1, -1, 1, 1
    )
    delta = F.conv2d(
        q_tilde,
        u_weight,
        None,
        stride=block.U.stride,
        padding=block.U.padding,
        dilation=block.U.dilation,
        groups=block.U.groups,
    )
    s = h + block.scale * delta
    if block.activation_type == 0:
        return s
    if block.activation_type == 1:
        return torch.relu(s)
    if block.activation_type == 2:
        return torch.clamp(s, min=0, max=6)
    raise AssertionError("unsupported activation")


def test_bnpa_forward_and_gradients_match_naive_reference() -> None:
    block = make_block(nn.ReLU6())
    block.train()
    x = torch.randn(2, 4, 8, 8, requires_grad=True)
    grad_out = torch.randn(2, 6, 8, 8)

    y = block(x)
    y.backward(grad_out)
    grad_x = x.grad.detach().clone()
    grad_u = block.U.weight.grad.detach().clone()
    grad_gamma = block.bottleneck_bn.weight.grad.detach().clone()
    grad_beta = block.bottleneck_bn.bias.grad.detach().clone()

    x_ref = x.detach().clone().requires_grad_(True)
    u_ref = block.U.weight.detach().clone().requires_grad_(True)
    gamma_ref = block.bottleneck_bn.weight.detach().clone().requires_grad_(True)
    beta_ref = block.bottleneck_bn.bias.detach().clone().requires_grad_(True)
    y_ref = naive_bnpa_reference(block, x_ref, u_ref, gamma_ref, beta_ref)
    y_ref.backward(grad_out)

    assert torch.allclose(y, y_ref.detach(), atol=1e-5, rtol=1e-5)
    assert torch.allclose(grad_x, x_ref.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(grad_u, u_ref.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(grad_gamma, gamma_ref.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(grad_beta, beta_ref.grad, atol=1e-5, rtol=1e-5)
    assert block.base_conv.weight.grad is None
    assert block.P.weight.grad is None
    assert block.batch_norm.weight.grad is None


def test_bnpa_step_zero_matches_adabn_backbone() -> None:
    block = make_block(nn.ReLU())
    block.train()
    with torch.no_grad():
        block.U.weight.zero_()
    x = torch.randn(2, 4, 8, 8)
    y = block(x)
    y_ref = naive_bnpa_reference(
        block,
        x,
        block.U.weight,
        block.bottleneck_bn.weight,
        block.bottleneck_bn.bias,
    )
    assert torch.allclose(y, y_ref, atol=1e-6, rtol=1e-6)
