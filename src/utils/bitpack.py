from __future__ import annotations
import math
from typing import Tuple
import torch

from src.utils._bitpack_cuda import cuda_extension

_BIT_WEIGHTS_CACHE: dict[tuple[str, int | None], torch.Tensor] = {}


def _bit_weights(device: torch.device) -> torch.Tensor:
    key = (device.type, device.index)
    cached = _BIT_WEIGHTS_CACHE.get(key)
    if cached is None or cached.device != device:
        cached = torch.tensor(
            [1, 2, 4, 8, 16, 32, 64, 128], device=device, dtype=torch.uint8
        )
        _BIT_WEIGHTS_CACHE[key] = cached
    return cached


def pack_bool_mask(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
    """Pack a bool mask into bytes using least-significant-bit first order.

    Element ``i`` of the flattened mask is stored in bit ``i % 8`` of byte
    ``i // 8``. If the number of elements is not divisible by 8, the final byte
    is padded with zero bits.
    """
    if mask.dtype != torch.bool:
        raise TypeError(f"pack_bool_mask expects torch.bool input, got {mask.dtype}")
    original_shape = mask.shape
    if mask.is_cuda and mask.numel():
        extension = cuda_extension()
        if extension is not None:
            return extension.pack_bool(mask.contiguous()), original_shape
    flat = mask.reshape(-1)
    numel = int(flat.numel())
    if numel == 0:
        return torch.empty(0, device=mask.device, dtype=torch.uint8), original_shape
    padding = (-numel) % 8
    if padding:
        flat = torch.cat(
            [flat, torch.zeros(padding, device=mask.device, dtype=torch.bool)], dim=0
        )
    bits = flat.to(torch.uint8).view(-1, 8)
    weights = _bit_weights(mask.device).view(1, 8)
    # The sum of eight distinct bit weights is at most 255. Explicit uint8
    # avoids the default int64 conversion of the entire input to the reduction.
    packed = bits.mul_(weights).sum(dim=1, dtype=torch.uint8)
    return packed, original_shape


def activate_and_pack_(tensor: torch.Tensor, activation: int) -> torch.Tensor:
    """Activate a fresh preactivation in place and return its packed gradient gate.

    Intended for custom autograd forward functions, where grad recording is off.
    Non-contiguous layouts and CPU tensors use the equivalent torch path.
    """
    if activation not in (1, 2):
        raise ValueError("activation must be 1 (ReLU) or 2 (ReLU6)")
    if torch.is_grad_enabled() and tensor.requires_grad:
        raise RuntimeError("activate_and_pack_ must run inside a custom/no-grad forward")
    if tensor.is_cuda and tensor.is_contiguous() and tensor.numel():
        extension = cuda_extension()
        if extension is not None:
            return extension.activate_pack_(tensor, activation)
    mask = tensor > 0
    if activation == 2:
        mask.logical_and_(tensor < 6)
        tensor.clamp_(min=0, max=6)
    else:
        tensor.relu_()
    packed, _ = pack_bool_mask(mask)
    return packed


def unpack_bool_mask(packed: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    """Unpack a mask produced by :func:`pack_bool_mask`."""
    if packed.dtype != torch.uint8:
        raise TypeError(
            f"unpack_bool_mask expects torch.uint8 input, got {packed.dtype}"
        )
    numel = int(math.prod(tuple(int(dim) for dim in original_shape)))
    if numel == 0:
        return torch.empty(
            tuple(original_shape), device=packed.device, dtype=torch.bool
        )
    flat_packed = packed.reshape(-1)
    expected_bytes = (numel + 7) // 8
    if int(flat_packed.numel()) < expected_bytes:
        raise ValueError(
            f"Packed mask has {flat_packed.numel()} bytes, expected at least {expected_bytes}"
        )
    values = flat_packed[:expected_bytes].view(-1, 1)
    weights = _bit_weights(packed.device).view(1, 8)
    # Turn the byte of bits into 0/1 in place and read it as bool: one allocation.
    bits = torch.bitwise_and(values, weights).ne_(0).view(torch.bool)
    return bits.reshape(-1)[:numel].reshape(tuple(original_shape))


def masked_scaled_grad(grad: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply a packed gate and NCHW channel scale without a full-width mask.

    Writes a new output, never overwrites incoming gradients. Higher-order
    differentiation and unsupported layouts use the differentiable torch path.
    """
    if grad.ndim != 4 or scale.ndim != 1 or scale.numel() != grad.shape[1]:
        raise ValueError("Expected NCHW gradient and one scale per channel")
    if packed.dtype != torch.uint8 or packed.numel() < (grad.numel() + 7) // 8:
        raise ValueError("Invalid packed activation mask")
    if packed.device != grad.device or scale.device != grad.device:
        raise ValueError("Gradient, mask, and scale must share a device")
    if (not torch.is_grad_enabled() and grad.is_cuda and grad.numel()
            and grad.is_contiguous() and packed.is_contiguous() and scale.is_contiguous()
            and grad.dtype == scale.dtype
            and grad.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)):
        extension = cuda_extension()
        if extension is not None:
            return extension.masked_scaled_grad(grad, packed, scale)
    mask = unpack_bool_mask(packed, grad.shape)
    return torch.where(mask, grad, 0) * scale.view(1, -1, 1, 1)
