from __future__ import annotations
from typing import Optional
import torch
from torch import nn


class TorchvisionMobileNetV2HAR(nn.Module):
    """Torchvision MobileNetV2 adapted to HAR window tensors.

    The runner prepares Opportunity tensors as [N, 1, C, T]. ImageNet MobileNetV2
    expects RGB inputs, so we repeat the single HAR channel to 3 channels and keep
    the pretrained first convolution unchanged.
    """

    def __init__(
        self, num_classes: int, pretrained: bool = True, width_mult: float = 1.0
    ) -> None:
        super().__init__()
        try:
            from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
        except ImportError as exc:
            raise ImportError(
                "TorchvisionMobileNetV2HAR requires torchvision. Install it "
                "or use --backbone t_resnet_official."
            ) from exc
        if pretrained and abs(float(width_mult) - 1.0) > 1e-12:
            raise ValueError(
                "Torchvision pretrained MobileNetV2 weights require width_mult=1.0. "
                "Use --mobilenet-v2-pretrained false for other width multipliers."
            )
        weights: Optional[MobileNet_V2_Weights] = (
            MobileNet_V2_Weights.DEFAULT if pretrained else None
        )
        base = mobilenet_v2(weights=weights, width_mult=width_mult)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(base.last_channel, int(num_classes))
        self.pretrained = bool(pretrained)
        self.width_mult = float(width_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"TorchvisionMobileNetV2HAR expects [N,1,C,T] or [N,3,C,T], got {tuple(x.shape)}"
            )
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError(
                f"TorchvisionMobileNetV2HAR expects 1 or 3 input channels, got {x.shape[1]}"
            )
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
