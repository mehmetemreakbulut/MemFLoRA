from __future__ import annotations

import pytest
import torch

from src.utils.bitpack import pack_bool_mask, unpack_bool_mask


@pytest.mark.parametrize(
    "shape",
    [
        (0,),
        (1,),
        (7,),
        (8,),
        (9,),
        (3, 5),
        (2, 3, 4),
        (2, 3, 5, 7),
    ],
)
def test_pack_unpack_bool_mask_roundtrip_cpu(shape: tuple[int, ...]) -> None:
    mask = torch.rand(shape) > 0.5
    packed, original_shape = pack_bool_mask(mask)
    unpacked = unpack_bool_mask(packed, original_shape)
    assert packed.dtype == torch.uint8
    assert packed.device == mask.device
    assert tuple(original_shape) == shape
    assert torch.equal(unpacked, mask)


def test_pack_unpack_all_zero_and_all_one_masks() -> None:
    for value in (False, True):
        mask = torch.full((3, 5, 11), value, dtype=torch.bool)
        packed, original_shape = pack_bool_mask(mask)
        unpacked = unpack_bool_mask(packed, original_shape)
        assert packed.numel() == (mask.numel() + 7) // 8
        assert torch.equal(unpacked, mask)


def test_pack_bool_mask_rejects_non_bool_input() -> None:
    with pytest.raises(TypeError):
        pack_bool_mask(torch.zeros(4, dtype=torch.uint8))


def test_unpack_bool_mask_rejects_non_uint8_input() -> None:
    with pytest.raises(TypeError):
        unpack_bool_mask(torch.zeros(1, dtype=torch.int16), torch.Size([8]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_pack_unpack_bool_mask_roundtrip_cuda() -> None:
    mask = torch.rand(2, 3, 17, device="cuda") > 0.5
    packed, original_shape = pack_bool_mask(mask)
    unpacked = unpack_bool_mask(packed, original_shape)
    assert packed.device.type == "cuda"
    assert torch.equal(unpacked, mask)
