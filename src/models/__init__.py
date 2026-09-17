"""HAR backbones used by the MemFLoRA experiments."""

from src.models.official_t_resnet_2d import OfficialTResNet2D
from src.models.torchvision_mobilenet_v2 import TorchvisionMobileNetV2HAR

__all__ = ["OfficialTResNet2D", "TorchvisionMobileNetV2HAR"]
