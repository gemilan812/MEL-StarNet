"""
文件说明：
这个文件集中定义论文主对比实验会用到的基线模型，
包括 torchvision 基线和表 4.2 里的 StarNet/MEL-StarNet 入口。
"""

from typing import Callable, Dict

import torch.nn as nn
from torchvision import models

from 公共核心.mel_starnet_variants import mel_starnet_s4, starnet_s4_baseline


def _lazy_import_timm():
    try:
        import timm
    except ImportError as error:
        raise ImportError("GhostNetV3 / FasterNet / MobileOne 基线需要先安装 timm。") from error
    return timm


def resnet50_benchmark(num_classes: int = 1000, pretrained: bool = False) -> nn.Module:
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def densenet121_benchmark(num_classes: int = 1000, pretrained: bool = False) -> nn.Module:
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def efficientnet_b0_benchmark(num_classes: int = 1000, pretrained: bool = False) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def mobilenet_v2_14_benchmark(num_classes: int = 1000, pretrained: bool = False) -> nn.Module:
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights, width_mult=1.4)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def ghostnetv3_130_benchmark(num_classes: int = 1000, pretrained: bool = False) -> nn.Module:
    timm = _lazy_import_timm()
    return timm.create_model("ghostnetv3_130", pretrained=pretrained, num_classes=num_classes)


def fasternet_t1_benchmark(num_classes: int = 1000, pretrained: bool = False) -> nn.Module:
    timm = _lazy_import_timm()
    return timm.create_model("fasternet_t1", pretrained=pretrained, num_classes=num_classes)


def mobileone_s2_benchmark(num_classes: int = 1000, pretrained: bool = False) -> nn.Module:
    timm = _lazy_import_timm()
    return timm.create_model("mobileone_s2", pretrained=pretrained, num_classes=num_classes)


BENCHMARK_MODEL_FACTORY: Dict[str, Callable[..., nn.Module]] = {
    "ResNet50": resnet50_benchmark,
    "DenseNet121": densenet121_benchmark,
    "EfficientNet-B0": efficientnet_b0_benchmark,
    "MobileNetV2-1.4": mobilenet_v2_14_benchmark,
    "GhostNetV3 1.3": ghostnetv3_130_benchmark,
    "FasterNet-T1": fasternet_t1_benchmark,
    "MobileOne-S2": mobileone_s2_benchmark,
    "StarNet_s4": starnet_s4_baseline,
    "Ours": mel_starnet_s4,
}


__all__ = [
    "BENCHMARK_MODEL_FACTORY",
    "densenet121_benchmark",
    "efficientnet_b0_benchmark",
    "fasternet_t1_benchmark",
    "ghostnetv3_130_benchmark",
    "mel_starnet_s4",
    "mobileone_s2_benchmark",
    "mobilenet_v2_14_benchmark",
    "resnet50_benchmark",
    "starnet_s4_baseline",
]
