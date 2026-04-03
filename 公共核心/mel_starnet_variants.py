"""
文件说明：
这个文件是所有 MEL-StarNet 变体的核心实现文件，
通过统一的可配置骨架生成主实验、消融实验和注意力替换实验所需模型。
"""

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from 公共核心.mel_starnet import ConvBN, CoordinateAttentionBranch, DCAFE, DSCBS, InvolutionRefine, InvLeaf, StarBlock


class CBS(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class StarStem(nn.Sequential):
    def __init__(self, out_channels: int = 32) -> None:
        super().__init__(ConvBN(3, out_channels, kernel_size=3, stride=2, padding=1), nn.ReLU6(inplace=True))


class StarDownsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2) -> None:
        super().__init__()
        self.block = ConvBN(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ChannelProjector(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class IdentityBridge(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class SEAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class ECAAttention(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.amax(dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CBAMAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channel_attn = ChannelAttention(channels)
        self.spatial_attn = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class CoordinateAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 32) -> None:
        super().__init__()
        mip = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU6(inplace=True)
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return x * a_h * a_w


class SCSEAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.channel_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_se = nn.Sequential(nn.Conv2d(channels, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.channel_se(x) + x * self.spatial_se(x)


class AttentionExpandBridge(nn.Module):
    def __init__(self, attention: nn.Module, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.attention = attention
        self.project = ChannelProjector(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x)
        x = self.project(x)
        return x


class ConfigurableStarNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        base_dim: int = 32,
        depths: Tuple[int, int, int, int] = (3, 3, 12, 5),
        mlp_ratio: int = 4,
        drop_path_rate: float = 0.0,
        stem_kind: str = "dscbs",
        stage_kind: str = "dscbs",
        bridge1_kind: str = "ledcg",
        bridge2_kind: str = "dcafe",
        refinement_kind: str = "residual_involution",
    ) -> None:
        super().__init__()
        if len(depths) != 4:
            raise ValueError("depths must contain four stage depths.")

        stage1_channels = base_dim
        stage2_channels = base_dim * 2
        stage3_channels = base_dim * 4
        stage4_channels = base_dim * 8

        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cursor = 0

        self.stem = self._build_stem(stem_kind, stage1_channels)
        self.stage1 = self._make_stage(stage_kind, stage1_channels, stage1_channels, depths[0], mlp_ratio, dpr[cursor : cursor + depths[0]])
        cursor += depths[0]
        self.stage2 = self._make_stage(stage_kind, stage1_channels, stage2_channels, depths[1], mlp_ratio, dpr[cursor : cursor + depths[1]])
        cursor += depths[1]

        self.bridge1, stage3_in_channels = self._build_bridge1(bridge1_kind, stage2_channels, stage3_channels)
        self.stage3 = self._make_stage(stage_kind, stage3_in_channels, stage3_channels, depths[2], mlp_ratio, dpr[cursor : cursor + depths[2]])
        cursor += depths[2]

        self.bridge2, stage4_in_channels = self._build_bridge2(bridge2_kind, stage3_channels, stage4_channels)
        self.stage4 = self._make_stage(stage_kind, stage4_in_channels, stage4_channels, depths[3], mlp_ratio, dpr[cursor : cursor + depths[3]])

        self.refinement = self._build_refinement(refinement_kind, stage4_channels)
        self.norm = nn.BatchNorm2d(stage4_channels)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(stage4_channels, num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(nn.init, "trunc_normal_"):
                nn.init.trunc_normal_(module.weight, std=0.02)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    @staticmethod
    def _build_stem(kind: str, out_channels: int) -> nn.Module:
        if kind == "star":
            return StarStem(out_channels)
        if kind == "cbs":
            return CBS(3, out_channels, stride=2)
        if kind == "dscbs":
            return DSCBS(3, out_channels, stride=2)
        raise ValueError(f"Unsupported stem kind: {kind}")

    @staticmethod
    def _build_transition(kind: str, in_channels: int, out_channels: int) -> nn.Module:
        if kind == "star":
            return StarDownsample(in_channels, out_channels, stride=2)
        if kind == "cbs":
            return CBS(in_channels, out_channels, stride=2)
        if kind == "dscbs":
            return DSCBS(in_channels, out_channels, stride=2)
        raise ValueError(f"Unsupported stage kind: {kind}")

    @classmethod
    def _make_stage(
        cls,
        stage_kind: str,
        in_channels: int,
        out_channels: int,
        depth: int,
        mlp_ratio: int,
        drop_path_values: Tuple[float, ...],
    ) -> nn.Sequential:
        blocks = [cls._build_transition(stage_kind, in_channels, out_channels)]
        blocks.extend(StarBlock(out_channels, mlp_ratio=mlp_ratio, drop_path_prob=drop_prob) for drop_prob in drop_path_values)
        return nn.Sequential(*blocks)

    @staticmethod
    def _build_bridge1(kind: str, in_channels: int, out_channels: int) -> Tuple[nn.Module, int]:
        if kind == "identity":
            return IdentityBridge(), in_channels
        if kind == "project":
            return ChannelProjector(in_channels, out_channels, kernel_size=1), out_channels
        if kind == "ledcg":
            from 公共核心.mel_starnet import LEDCGBlock

            return LEDCGBlock(in_channels, out_channels), out_channels
        raise ValueError(f"Unsupported bridge1 kind: {kind}")

    @staticmethod
    def _build_bridge2(kind: str, in_channels: int, out_channels: int) -> Tuple[nn.Module, int]:
        if kind == "identity":
            return IdentityBridge(), in_channels
        if kind == "project":
            return ChannelProjector(in_channels, out_channels, kernel_size=1), out_channels
        if kind == "dcafe":
            return DCAFE(in_channels, out_channels), out_channels
        if kind == "avg_ca":
            branch = CoordinateAttentionBranch(in_channels, pool_type="avg")
            return AttentionExpandBridge(branch, in_channels, out_channels), out_channels
        if kind == "max_ca":
            branch = CoordinateAttentionBranch(in_channels, pool_type="max")
            return AttentionExpandBridge(branch, in_channels, out_channels), out_channels
        if kind == "se":
            return AttentionExpandBridge(SEAttention(in_channels), in_channels, out_channels), out_channels
        if kind == "eca":
            return AttentionExpandBridge(ECAAttention(in_channels), in_channels, out_channels), out_channels
        if kind == "cbam":
            return AttentionExpandBridge(CBAMAttention(in_channels), in_channels, out_channels), out_channels
        if kind == "ca":
            return AttentionExpandBridge(CoordinateAttention(in_channels), in_channels, out_channels), out_channels
        if kind == "scse":
            return AttentionExpandBridge(SCSEAttention(in_channels), in_channels, out_channels), out_channels
        raise ValueError(f"Unsupported bridge2 kind: {kind}")

    @staticmethod
    def _build_refinement(kind: str, channels: int) -> nn.Module:
        if kind == "none":
            return nn.Identity()
        if kind == "involution":
            return InvolutionRefine(channels, kernel_size=5, stride=1, reduction_ratio=4, group_channels=16)
        if kind == "residual_involution":
            return InvLeaf(channels, kernel_size=5, stride=1, reduction_ratio=4, group_channels=16)
        raise ValueError(f"Unsupported refinement kind: {kind}")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.bridge1(x)
        x = self.stage3(x)
        x = self.bridge2(x)
        x = self.stage4(x)
        x = self.refinement(x)
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = torch.flatten(self.avgpool(x), 1)
        return self.head(x)


def _build_variant(
    num_classes: int = 1000,
    base_dim: int = 32,
    depths: Tuple[int, int, int, int] = (3, 3, 12, 5),
    mlp_ratio: int = 4,
    drop_path_rate: float = 0.0,
    stem_kind: str = "dscbs",
    stage_kind: str = "dscbs",
    bridge1_kind: str = "ledcg",
    bridge2_kind: str = "dcafe",
    refinement_kind: str = "residual_involution",
) -> ConfigurableStarNet:
    return ConfigurableStarNet(
        num_classes=num_classes,
        base_dim=base_dim,
        depths=depths,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path_rate,
        stem_kind=stem_kind,
        stage_kind=stage_kind,
        bridge1_kind=bridge1_kind,
        bridge2_kind=bridge2_kind,
        refinement_kind=refinement_kind,
    )


def starnet_s4_baseline(num_classes: int = 1000, **kwargs) -> ConfigurableStarNet:
    return _build_variant(num_classes=num_classes, stem_kind="star", stage_kind="star", bridge1_kind="identity", bridge2_kind="identity", refinement_kind="none", **kwargs)


def starnet_s4_ds_cbs(num_classes: int = 1000, **kwargs) -> ConfigurableStarNet:
    return _build_variant(num_classes=num_classes, stem_kind="dscbs", stage_kind="dscbs", bridge1_kind="identity", bridge2_kind="identity", refinement_kind="none", **kwargs)


def starnet_s4_ds_led_cg(num_classes: int = 1000, **kwargs) -> ConfigurableStarNet:
    return _build_variant(num_classes=num_classes, stem_kind="dscbs", stage_kind="dscbs", bridge1_kind="ledcg", bridge2_kind="identity", refinement_kind="none", **kwargs)


def starnet_s4_ds_led_cg_dcafe(num_classes: int = 1000, **kwargs) -> ConfigurableStarNet:
    return _build_variant(num_classes=num_classes, stem_kind="dscbs", stage_kind="dscbs", bridge1_kind="ledcg", bridge2_kind="dcafe", refinement_kind="none", **kwargs)


def mel_starnet_s4(num_classes: int = 1000, **kwargs) -> ConfigurableStarNet:
    return _build_variant(num_classes=num_classes, stem_kind="dscbs", stage_kind="dscbs", bridge1_kind="ledcg", bridge2_kind="dcafe", refinement_kind="residual_involution", **kwargs)


def starnet_s4_wo_ds_cbs(num_classes: int = 1000, **kwargs) -> ConfigurableStarNet:
    return _build_variant(num_classes=num_classes, stem_kind="star", stage_kind="star", bridge1_kind="ledcg", bridge2_kind="dcafe", refinement_kind="residual_involution", **kwargs)


def starnet_s4_wo_led_cg(num_classes: int = 1000, **kwargs) -> ConfigurableStarNet:
    return _build_variant(num_classes=num_classes, stem_kind="dscbs", stage_kind="dscbs", bridge1_kind="project", bridge2_kind="dcafe", refinement_kind="residual_involution", **kwargs)


def starnet_s4_wo_dcafe(num_classes: int = 1000, **kwargs) -> ConfigurableStarNet:
    return _build_variant(num_classes=num_classes, stem_kind="dscbs", stage_kind="dscbs", bridge1_kind="ledcg", bridge2_kind="project", refinement_kind="residual_involution", **kwargs)


def starnet_s4_wo_inv_leaf(num_classes: int = 1000, **kwargs) -> ConfigurableStarNet:
    return _build_variant(num_classes=num_classes, stem_kind="dscbs", stage_kind="dscbs", bridge1_kind="ledcg", bridge2_kind="dcafe", refinement_kind="none", **kwargs)


def starnet_s4_wo_refinement(num_classes: int = 1000, **kwargs) -> ConfigurableStarNet:
    return starnet_s4_wo_inv_leaf(num_classes=num_classes, **kwargs)


def starnet_s4_original_cbs(num_classes: int = 1000, **kwargs) -> ConfigurableStarNet:
    return _build_variant(num_classes=num_classes, stem_kind="cbs", stage_kind="cbs", bridge1_kind="ledcg", bridge2_kind="dcafe", refinement_kind="residual_involution", **kwargs)


def starnet_s4_se(num_classes: int = 1000, **kwargs) -> ConfigurableStarNet:
    return _build_variant(num_classes=num_classes, stem_kind="dscbs", stage_kind="dscbs", bridge1_kind="ledcg", bridge2_kind="se", refinement_kind="residual_involution", **kwargs)


def starnet_s4_eca(num_classes: int = 1000, **kwargs) -> ConfigurableStarNet:
    return _build_variant(num_classes=num_classes, stem_kind="dscbs", stage_kind="dscbs", bridge1_kind="ledcg", bridge2_kind="eca", refinement_kind="residual_involution", **kwargs)


def starnet_s4_cbam(num_classes: int = 1000, **kwargs) -> ConfigurableStarNet:
    return _build_variant(num_classes=num_classes, stem_kind="dscbs", stage_kind="dscbs", bridge1_kind="ledcg", bridge2_kind="cbam", refinement_kind="residual_involution", **kwargs)


def starnet_s4_ca(num_classes: int = 1000, **kwargs) -> ConfigurableStarNet:
    return _build_variant(num_classes=num_classes, stem_kind="dscbs", stage_kind="dscbs", bridge1_kind="ledcg", bridge2_kind="ca", refinement_kind="residual_involution", **kwargs)


def starnet_s4_scse(num_classes: int = 1000, **kwargs) -> ConfigurableStarNet:
    return _build_variant(num_classes=num_classes, stem_kind="dscbs", stage_kind="dscbs", bridge1_kind="ledcg", bridge2_kind="scse", refinement_kind="residual_involution", **kwargs)


def starnet_s4_avg_ca(num_classes: int = 1000, **kwargs) -> ConfigurableStarNet:
    return _build_variant(num_classes=num_classes, stem_kind="dscbs", stage_kind="dscbs", bridge1_kind="ledcg", bridge2_kind="avg_ca", refinement_kind="residual_involution", **kwargs)


def starnet_s4_max_ca(num_classes: int = 1000, **kwargs) -> ConfigurableStarNet:
    return _build_variant(num_classes=num_classes, stem_kind="dscbs", stage_kind="dscbs", bridge1_kind="ledcg", bridge2_kind="max_ca", refinement_kind="residual_involution", **kwargs)


def starnet_s4_involution_only(num_classes: int = 1000, **kwargs) -> ConfigurableStarNet:
    return _build_variant(num_classes=num_classes, stem_kind="dscbs", stage_kind="dscbs", bridge1_kind="ledcg", bridge2_kind="dcafe", refinement_kind="involution", **kwargs)


CUSTOM_MODEL_FACTORY: Dict[str, Callable[..., ConfigurableStarNet]] = {
    "StarNet_s4": starnet_s4_baseline,
    "+ DS-CBS": starnet_s4_ds_cbs,
    "+ DS-CBS + LED-CG": starnet_s4_ds_led_cg,
    "+ DS-CBS + LED-CG + DCAFE": starnet_s4_ds_led_cg_dcafe,
    "Ours": mel_starnet_s4,
    "w/o DS-CBS": starnet_s4_wo_ds_cbs,
    "w/o LED-CG": starnet_s4_wo_led_cg,
    "w/o DCAFE": starnet_s4_wo_dcafe,
    "w/o Inv-Leaf": starnet_s4_wo_inv_leaf,
    "w/o refinement": starnet_s4_wo_refinement,
    "Original CBS": starnet_s4_original_cbs,
    "DS-CBS": mel_starnet_s4,
    "SE": starnet_s4_se,
    "ECA": starnet_s4_eca,
    "CBAM": starnet_s4_cbam,
    "CA": starnet_s4_ca,
    "scSE": starnet_s4_scse,
    "Avg-CA only": starnet_s4_avg_ca,
    "Max-CA only": starnet_s4_max_ca,
    "Avg-CA + Max-CA (DCAFE)": mel_starnet_s4,
    "Involution only": starnet_s4_involution_only,
    "Residual Involution (Inv-Leaf)": mel_starnet_s4,
}


__all__ = [
    "ConfigurableStarNet",
    "CUSTOM_MODEL_FACTORY",
    "mel_starnet_s4",
    "starnet_s4_avg_ca",
    "starnet_s4_baseline",
    "starnet_s4_ca",
    "starnet_s4_cbam",
    "starnet_s4_dcafe",
    "starnet_s4_ds_cbs",
    "starnet_s4_ds_led_cg",
    "starnet_s4_ds_led_cg_dcafe",
    "starnet_s4_eca",
    "starnet_s4_involution_only",
    "starnet_s4_max_ca",
    "starnet_s4_original_cbs",
    "starnet_s4_scse",
    "starnet_s4_se",
    "starnet_s4_wo_dcafe",
    "starnet_s4_wo_ds_cbs",
    "starnet_s4_wo_inv_leaf",
    "starnet_s4_wo_led_cg",
    "starnet_s4_wo_refinement",
]


def starnet_s4_dcafe(num_classes: int = 1000, **kwargs) -> ConfigurableStarNet:
    return starnet_s4_ds_led_cg_dcafe(num_classes=num_classes, **kwargs)
