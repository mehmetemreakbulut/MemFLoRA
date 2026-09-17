from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn


class OfficialConv2dTimeSame(nn.Module):
    """Temporal same-padding Conv2d used by the official LoRA-Edge HAR model."""

    def __init__(self, in_channels: int, out_channels: int, kernel_t: int) -> None:
        super().__init__()
        self.kernel_t = int(kernel_t)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(self.kernel_t, 1),
            padding=(0, 0),
            bias=False,
        )

    @staticmethod
    def _same_pad_1d(kernel: int) -> tuple[int, int]:
        pad = kernel - 1
        return pad // 2, pad - (pad // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_before, pad_after = self._same_pad_1d(self.kernel_t)
        if pad_before or pad_after:
            x = F.pad(x, (0, 0, pad_before, pad_after))
        return self.conv(x)


class OfficialConvBNAct2d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_t: int, act: bool = True
    ) -> None:
        super().__init__()
        self.conv = OfficialConv2dTimeSame(in_channels, out_channels, kernel_t)
        self.bn = nn.BatchNorm2d(out_channels, track_running_stats=True)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class OfficialTResNetBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pre_bn = nn.BatchNorm2d(in_channels, track_running_stats=True)
        self.conv8 = OfficialConvBNAct2d(
            in_channels, out_channels, kernel_t=8, act=True
        )
        self.conv5 = OfficialConvBNAct2d(
            out_channels, out_channels, kernel_t=5, act=True
        )
        self.conv3 = OfficialConv2dTimeSame(out_channels, out_channels, kernel_t=3)
        self.bn3 = nn.BatchNorm2d(out_channels, track_running_stats=True)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels, track_running_stats=True),
            )
        else:
            self.shortcut = nn.BatchNorm2d(in_channels, track_running_stats=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.pre_bn(x)
        x = self.conv8(x)
        x = self.conv5(x)
        x = self.bn3(self.conv3(x))
        return self.act(x + residual)


class OfficialTResNet2D(nn.Module):
    """TResNet2D matching the official LoRA-Edge HAR architecture.

    Expected input shape is ``[batch, sensor_channels, time]``. The generic HAR
    runner may also pass ``[batch, 1, sensor_channels, time]``; both are mapped
    internally to ``[batch, sensor_channels, time, 1]``.
    """

    def __init__(
        self, input_channels: int, num_classes: int, n_feature_maps: int = 64
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.n_feature_maps = int(n_feature_maps)
        self.block1 = OfficialTResNetBlock2D(self.input_channels, self.n_feature_maps)
        self.block2 = OfficialTResNetBlock2D(
            self.n_feature_maps, self.n_feature_maps * 2
        )
        self.block3 = OfficialTResNetBlock2D(
            self.n_feature_maps * 2, self.n_feature_maps * 2
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.n_feature_maps * 2, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            if x.shape[1] != 1:
                raise ValueError(
                    f"OfficialTResNet2D expected channel singleton for 4D input, got {tuple(x.shape)}"
                )
            x = x.squeeze(1)
        if x.ndim != 3:
            raise ValueError(
                f"OfficialTResNet2D expected [B, C, T] or [B, 1, C, T], got {tuple(x.shape)}"
            )
        x = x.unsqueeze(3)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)
