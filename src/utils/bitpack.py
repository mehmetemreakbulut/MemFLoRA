from __future__ import annotations
import math
from typing import Tuple
import torch

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
    packed = (bits * weights).sum(dim=1).to(torch.uint8)
    return packed, original_shape


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
