"""
文件说明：
这个文件用于快速验证 MEL-StarNet 是否能正常跑通，
包括前向传播、中间层形状检查、模型结构概览以及 FLOPs/参数量统计。
"""

import argparse
import sys
from pathlib import Path

import torch
from thop import clever_format, profile
from torchsummary import summary

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from 公共核心.mel_starnet import MELStarNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate MEL-StarNet forward, summary, and FLOPs.")
    parser.add_argument("--num_classes", type=int, default=1000, help="Number of output classes.")
    parser.add_argument("--input_size", type=int, default=224, help="Square input image size.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for forward test.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the validation on.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA 不可用，自动切换到 CPU。")
        return torch.device("cpu")
    return torch.device(device_arg)


def print_stage_shapes(model: MELStarNet, input_tensor: torch.Tensor) -> None:
    print("\n===> 中间层形状检查...")
    with torch.no_grad():
        x = model.stem(input_tensor)
        print(f"stem 输出 shape: {tuple(x.shape)}")

        x = model.stage1(x)
        print(f"stage1 输出 shape: {tuple(x.shape)}")

        x = model.stage2(x)
        print(f"stage2 输出 shape: {tuple(x.shape)}")

        x = model.led_cg(x)
        print(f"LED-CG 输出 shape: {tuple(x.shape)}")

        x = model.stage3(x)
        print(f"stage3 输出 shape: {tuple(x.shape)}")

        x = model.dcafe(x)
        print(f"DCAFE 输出 shape: {tuple(x.shape)}")

        x = model.stage4(x)
        print(f"stage4 输出 shape: {tuple(x.shape)}")

        x = model.inv_leaf(x)
        print(f"Inv-Leaf 输出 shape: {tuple(x.shape)}")

        x = model.norm(x)
        x = model.avgpool(x)
        print(f"GAP 输出 shape: {tuple(x.shape)}")


def test_mel_starnet(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)

    print("===> 创建 MEL-StarNet 模型...")
    model = MELStarNet(
        num_classes=args.num_classes,
        base_dim=32,
        depths=(3, 3, 12, 5),
        mlp_ratio=4,
        drop_path_rate=0.0,
    ).to(device)
    model.eval()

    input_tensor = torch.randn(args.batch_size, 3, args.input_size, args.input_size, device=device)

    print("\n===> 执行前向传播测试...")
    try:
        with torch.no_grad():
            output = model(input_tensor)
        print(f"前向传播成功，输出 shape: {tuple(output.shape)}")
    except Exception as error:
        print(f"前向传播失败: {error}")
        return

    try:
        print_stage_shapes(model, input_tensor[:1])
    except Exception as error:
        print(f"中间层形状检查失败: {error}")

    print("\n===> 模型结构概览...")
    try:
        summary(model, (3, args.input_size, args.input_size), device=device.type)
    except Exception as error:
        print(f"torchsummary 解析失败，改为直接打印模型：{error}\n")
        print(model)

    print("\n===> 计算 FLOPs 和参数量...")
    try:
        macs, params = profile(model, inputs=(input_tensor,), verbose=False)
        macs, params = clever_format([macs, params], "%.3f")
        print("[统计结果]")
        print(f"模型总参数量: {params}")
        print(f"模型 FLOPs (MACs): {macs}")
    except Exception as error:
        print(f"计算 FLOPs 失败: {error}")


if __name__ == "__main__":
    test_mel_starnet(parse_args())
