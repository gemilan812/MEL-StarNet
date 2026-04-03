"""
文件说明：
这个文件是通用模型统计脚本，
用于计算任意模型的参数量、MACs 和 FLOPs，适合批量评估论文中的对照模型。
"""

import argparse
import csv
import importlib
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from thop import clever_format, profile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from 公共核心.model_zoo import MODEL_ZOO, get_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Universal FLOPs/params profiler for local models.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Model names from model_zoo. Example: --models Ours 'w/o DCAFE'",
    )
    parser.add_argument(
        "--all_zoo",
        action="store_true",
        help="Profile every model currently registered in model_zoo.",
    )
    parser.add_argument(
        "--builder",
        type=str,
        default="",
        help="Dynamic builder path in the form module:function. Example: 公共核心.mel_starnet:MELStarNet",
    )
    parser.add_argument("--num_classes", type=int, default=1000, help="Number of output classes.")
    parser.add_argument("--batch_size", type=int, default=1, help="Input batch size.")
    parser.add_argument("--channels", type=int, default=3, help="Number of input channels.")
    parser.add_argument("--height", type=int, default=224, help="Input image height.")
    parser.add_argument("--width", type=int, default=224, help="Input image width.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Profiling device.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--drop_path_rate",
        type=float,
        default=0.0,
        help="Optional common kwarg for models that accept drop_path_rate.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print all available model_zoo names and exit.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA 不可用，自动切换到 CPU。")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_builder(builder_path: str):
    if ":" not in builder_path:
        raise ValueError("--builder 必须是 module:function 形式。")
    module_name, function_name = builder_path.split(":", 1)
    module = importlib.import_module(module_name)
    builder = getattr(module, function_name)
    return builder


def manual_param_count(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def build_model_from_zoo(model_name: str, num_classes: int, drop_path_rate: float):
    kwargs: Dict[str, object] = {"num_classes": num_classes}
    try:
        return get_model(model_name, **kwargs, drop_path_rate=drop_path_rate)
    except TypeError:
        return get_model(model_name, **kwargs)


def build_model_from_builder(builder, num_classes: int, drop_path_rate: float):
    try:
        return builder(num_classes=num_classes, drop_path_rate=drop_path_rate)
    except TypeError:
        try:
            return builder(num_classes=num_classes)
        except TypeError:
            return builder()


def profile_one_model(
    model_name: str,
    model: torch.nn.Module,
    input_shape: Tuple[int, int, int, int],
    device: torch.device,
) -> Dict[str, object]:
    model = model.to(device)
    model.eval()
    dummy_input = torch.randn(*input_shape, device=device)

    with torch.no_grad():
        output = model(dummy_input)

    macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
    flops = macs * 2.0
    total_params, trainable_params = manual_param_count(model)
    macs_fmt, flops_fmt, total_params_fmt, trainable_params_fmt = clever_format(
        [macs, flops, total_params, trainable_params], "%.3f"
    )

    return {
        "model": model_name,
        "output_shape": tuple(output.shape),
        "macs": macs,
        "flops": flops,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "macs_fmt": macs_fmt,
        "flops_fmt": flops_fmt,
        "total_params_fmt": total_params_fmt,
        "trainable_params_fmt": trainable_params_fmt,
    }


def print_result(result: Dict[str, object]) -> None:
    print(f"\n=== {result['model']} ===")
    print(f"Output shape: {result['output_shape']}")
    print(f"Params: {result['total_params_fmt']} (raw: {result['total_params']})")
    print(f"Trainable Params: {result['trainable_params_fmt']} (raw: {result['trainable_params']})")
    print(f"MACs: {result['macs_fmt']} (raw: {result['macs']})")
    print(f"FLOPs ~= 2 x MACs: {result['flops_fmt']} (raw: {result['flops']})")


def write_csv(csv_path: Path, results: Iterable[Dict[str, object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "output_shape",
        "total_params",
        "trainable_params",
        "macs",
        "flops",
        "total_params_fmt",
        "trainable_params_fmt",
        "macs_fmt",
        "flops_fmt",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)


def main() -> None:
    args = parse_args()

    if args.list:
        print("Available model_zoo names:")
        for name in sorted(MODEL_ZOO.keys()):
            print(name)
        return

    device = resolve_device(args.device)
    input_shape = (args.batch_size, args.channels, args.height, args.width)
    results: List[Dict[str, object]] = []

    zoo_models = args.models or []
    if args.all_zoo:
        zoo_models = sorted(MODEL_ZOO.keys())

    for model_name in zoo_models:
        try:
            model = build_model_from_zoo(model_name, args.num_classes, args.drop_path_rate)
            result = profile_one_model(model_name, model, input_shape, device)
            print_result(result)
            results.append(result)
        except Exception as error:
            print(f"\n=== {model_name} ===")
            print(f"评估失败: {error}")

    if args.builder:
        builder = load_builder(args.builder)
        try:
            model = build_model_from_builder(builder, args.num_classes, args.drop_path_rate)
            result = profile_one_model(args.builder, model, input_shape, device)
            print_result(result)
            results.append(result)
        except Exception as error:
            print(f"\n=== {args.builder} ===")
            print(f"评估失败: {error}")

    if not results and not zoo_models and not args.builder:
        print("没有指定模型。你可以使用 --models、--all_zoo 或 --builder。")
        print("例如: python profile_models.py --models Ours 'w/o DCAFE'")
        return

    if args.output_csv and results:
        csv_path = Path(args.output_csv)
        write_csv(csv_path, results)
        print(f"\nCSV 已保存到: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
