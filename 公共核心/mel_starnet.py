"""
文件说明：
这个文件定义完整的 MEL-StarNet 主模型，
包含 DS-CBS、LED-CG、DCAFE、Inv-Leaf 以及最终分类头，是训练与测试的核心网络文件。
"""

from typing import Tuple

import torch
import torch.nn as nn


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class ConvBN(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        with_bn: bool = True,
    ) -> None:
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=not with_bn,
            )
        ]
        if with_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        super().__init__(*layers)


class DSCBS(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.act(x)
        return x


class StarBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 4, drop_path_prob: float = 0.0) -> None:
        super().__init__()
        hidden_dim = mlp_ratio * dim
        self.dwconv = ConvBN(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, hidden_dim, kernel_size=1, with_bn=False)
        self.f2 = ConvBN(dim, hidden_dim, kernel_size=1, with_bn=False)
        self.g = ConvBN(hidden_dim, dim, kernel_size=1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, with_bn=False)
        self.act = nn.ReLU6(inplace=True)
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x1 = self.f1(x)
        x2 = self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = residual + self.drop_path(x)
        return x


class LEDExpansion(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if out_channels <= in_channels:
            raise ValueError("LEDExpansion expects out_channels > in_channels.")

        expanded_channels = out_channels - in_channels
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.expand(x)], dim=1)


class GroupConvBlock(nn.Sequential):
    def __init__(self, channels: int) -> None:
        super().__init__(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )


class LEDCGBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 3) -> None:
        super().__init__()
        if num_groups < 2:
            raise ValueError("LEDCGBlock expects at least 2 groups.")

        self.led = LEDExpansion(in_channels, out_channels)
        base = out_channels // num_groups
        remainder = out_channels % num_groups
        split_sizes = [base] * num_groups
        split_sizes[-1] += remainder
        self.split_sizes = split_sizes
        self.group_blocks = nn.ModuleList(GroupConvBlock(size) for size in split_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.led(x)
        chunks = torch.split(x, self.split_sizes, dim=1)
        outputs = [block(chunk) for block, chunk in zip(self.group_blocks, chunks)]
        return torch.cat(outputs, dim=1)


class CoordinateAttentionBranch(nn.Module):
    def __init__(self, channels: int, pool_type: str = "avg", reduction: int = 32) -> None:
        super().__init__()
        if pool_type not in {"avg", "max"}:
            raise ValueError("pool_type must be 'avg' or 'max'.")

        mip = max(8, channels // reduction)
        self.pool_type = pool_type
        self.pre = DSCBS(channels, channels, stride=1)
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU6(inplace=True)
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)

    def _pool_h(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool_type == "avg":
            return x.mean(dim=3, keepdim=True)
        return x.amax(dim=3, keepdim=True)

    def _pool_w(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool_type == "avg":
            return x.mean(dim=2, keepdim=True)
        return x.amax(dim=2, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        _, _, h, w = x.shape

        x_h = self._pool_h(x)
        x_w = self._pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return x * a_h * a_w


class DCAFE(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if out_channels != in_channels * 2:
            raise ValueError("DCAFE expects out_channels to be exactly 2 * in_channels.")

        self.avg_branch = CoordinateAttentionBranch(in_channels, pool_type="avg")
        self.max_branch = CoordinateAttentionBranch(in_channels, pool_type="max")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.avg_branch(x)
        max_out = self.max_branch(x)
        return torch.cat([avg_out, max_out], dim=1)


class InvolutionRefine(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        reduction_ratio: int = 4,
        group_channels: int = 16,
    ) -> None:
        super().__init__()
        if channels % group_channels != 0:
            raise ValueError("channels must be divisible by group_channels.")
        if stride != 1:
            raise ValueError("This implementation expects stride=1.")

        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.group_channels = group_channels
        self.groups = channels // group_channels

        reduced_channels = max(channels // reduction_ratio, 1)
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.SiLU(inplace=True),
        )
        self.span = nn.Conv2d(reduced_channels, self.groups * kernel_size * kernel_size, kernel_size=1)
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        weights = self.reduce(x)
        weights = self.span(weights)
        weights = weights.view(b, self.groups, self.kernel_size * self.kernel_size, h, w)

        patches = self.unfold(x)
        patches = patches.view(
            b,
            self.groups,
            self.group_channels,
            self.kernel_size * self.kernel_size,
            h,
            w,
        )
        out = (weights.unsqueeze(2) * patches).sum(dim=3)
        out = out.view(b, self.channels, h, w)
        return out


class InvLeaf(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        reduction_ratio: int = 4,
        group_channels: int = 16,
    ) -> None:
        super().__init__()
        self.refine = InvolutionRefine(
            channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            reduction_ratio=reduction_ratio,
            group_channels=group_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.refine(x)


class MELStarNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        base_dim: int = 32,
        depths: Tuple[int, int, int, int] = (3, 3, 12, 5),
        mlp_ratio: int = 4,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if len(depths) != 4:
            raise ValueError("depths must contain 4 stage depths.")

        stage1_channels = base_dim
        stage2_channels = base_dim * 2
        stage3_channels = base_dim * 4
        stage4_channels = base_dim * 8

        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cursor = 0

        self.stem = DSCBS(3, stage1_channels, stride=2)
        self.stage1 = self._make_stage(
            in_channels=stage1_channels,
            out_channels=stage1_channels,
            depth=depths[0],
            mlp_ratio=mlp_ratio,
            drop_path_values=dpr[cursor : cursor + depths[0]],
        )
        cursor += depths[0]

        self.stage2 = self._make_stage(
            in_channels=stage1_channels,
            out_channels=stage2_channels,
            depth=depths[1],
            mlp_ratio=mlp_ratio,
            drop_path_values=dpr[cursor : cursor + depths[1]],
        )
        cursor += depths[1]

        self.led_cg = LEDCGBlock(stage2_channels, stage3_channels)
        self.stage3 = self._make_stage(
            in_channels=stage3_channels,
            out_channels=stage3_channels,
            depth=depths[2],
            mlp_ratio=mlp_ratio,
            drop_path_values=dpr[cursor : cursor + depths[2]],
        )
        cursor += depths[2]

        self.dcafe = DCAFE(stage3_channels, stage4_channels)
        self.stage4 = self._make_stage(
            in_channels=stage4_channels,
            out_channels=stage4_channels,
            depth=depths[3],
            mlp_ratio=mlp_ratio,
            drop_path_values=dpr[cursor : cursor + depths[3]],
        )

        self.inv_leaf = InvLeaf(
            stage4_channels,
            kernel_size=5,
            stride=1,
            reduction_ratio=4,
            group_channels=16,
        )
        self.norm = nn.BatchNorm2d(stage4_channels)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(stage4_channels, num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        depth: int,
        mlp_ratio: int,
        drop_path_values: Tuple[float, ...],
    ) -> nn.Sequential:
        blocks = [DSCBS(in_channels, out_channels, stride=2)]
        blocks.extend(
            StarBlock(out_channels, mlp_ratio=mlp_ratio, drop_path_prob=drop_prob)
            for drop_prob in drop_path_values
        )
        return nn.Sequential(*blocks)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(nn.init, "trunc_normal_"):
                nn.init.trunc_normal_(module.weight, std=0.02)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.led_cg(x)
        x = self.stage3(x)
        x = self.dcafe(x)
        x = self.stage4(x)
        x = self.inv_leaf(x)
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = torch.flatten(self.avgpool(x), 1)
        x = self.head(x)
        return x


__all__ = ["MELStarNet"]
