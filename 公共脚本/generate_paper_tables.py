"""
文件说明：
这个文件用于一键生成论文里的对比表和消融表，
会自动统计 Params/FLOPs，并可把 Accuracy/Precision/Recall/F1 合并进表格。
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from thop import profile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from 公共核心.model_zoo import get_model


TABLES = [
    {
        "id": "table_4_2",
        "title": "表4.2 实验结果对比表",
        "columns": ["Model", "Accuracy", "Precision", "Recall", "F1", "Params (M)", "Flops(M)"],
        "rows": [
            "ResNet50",
            "DenseNet121",
            "EfficientNet-B0",
            "MobileNetV2-1.4",
            "GhostNetV3 1.3",
            "FasterNet-T1",
            "MobileOne-S2",
            "StarNet_s4",
            "Ours",
        ],
    },
    {
        "id": "table_4_3",
        "title": "表4.3 总体模块累加式消融对比表",
        "columns": ["Model", "Accuracy", "Precision", "Recall", "F1", "Params (M)", "Flops(M)"],
        "rows": [
            "StarNet_s4",
            "+ DS-CBS",
            "+ DS-CBS + LED-CG",
            "+ DS-CBS + LED-CG + DCAFE",
            "Ours",
        ],
    },
    {
        "id": "table_4_4",
        "title": "表4.4 总体模块去除式消融对比表",
        "columns": ["Model", "Accuracy", "Precision", "Recall", "F1", "Params (M)", "Flops(M)"],
        "rows": [
            "w/o DS-CBS",
            "w/o LED-CG",
            "w/o DCAFE",
            "w/o Inv-Leaf",
            "Ours",
        ],
    },
    {
        "id": "table_4_5",
        "title": "表4.5 DCAFE 与经典注意力模块替换对比表",
        "columns": ["Attention module", "Accuracy", "Precision", "Recall", "F1", "Params (M)", "Flops(M)"],
        "rows": ["SE", "ECA", "CBAM", "CA", "scSE", "Ours"],
    },
    {
        "id": "table_4_6",
        "title": "表4.6 DS-CBS 小消融对比表",
        "columns": ["Variant", "Accuracy", "Params (M)", "Flops(M)"],
        "rows": ["Original CBS", "DS-CBS"],
    },
    {
        "id": "table_4_7",
        "title": "表4.7 DCAFE 内部小消融对比表",
        "columns": ["Variant", "Accuracy", "Params (M)", "Flops(M)"],
        "rows": ["Avg-CA only", "Max-CA only", "Avg-CA + Max-CA (DCAFE)"],
    },
    {
        "id": "table_4_8",
        "title": "表4.8 Inv-Leaf 小消融对比表",
        "columns": ["Variant", "Accuracy", "Params (M)", "Flops(M)"],
        "rows": ["w/o refinement", "Involution only", "Residual Involution (Inv-Leaf)"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-ready profiling tables for all ablation models.")
    parser.add_argument("--num_classes", type=int, default=1000, help="Number of model output classes.")
    parser.add_argument("--height", type=int, default=224, help="Input image height.")
    parser.add_argument("--width", type=int, default=224, help="Input image width.")
    parser.add_argument("--batch_size", type=int, default=1, help="Input batch size for profiling.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Profiling device.",
    )
    parser.add_argument(
        "--metrics_file",
        type=str,
        default="",
        help="Optional JSON or CSV containing Accuracy/Precision/Recall/F1 keyed by model name.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="paper_tables",
        help="Directory to save markdown/csv tables.",
    )
    parser.add_argument(
        "--drop_path_rate",
        type=float,
        default=0.0,
        help="Optional common kwarg for custom models.",
    )
    parser.add_argument(
        "--write_metrics_template",
        action="store_true",
        help="Also write a blank metrics template JSON for later manual filling.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA 不可用，自动切换到 CPU。")
        return torch.device("cpu")
    return torch.device(device_arg)


def normalize_metric_keys(metrics: Dict[str, object]) -> Dict[str, object]:
    normalized = {}
    for key, value in metrics.items():
        lowered = key.strip().lower()
        if lowered in {"accuracy", "acc"}:
            normalized["Accuracy"] = value
        elif lowered in {"precision", "prec"}:
            normalized["Precision"] = value
        elif lowered in {"recall", "rec"}:
            normalized["Recall"] = value
        elif lowered in {"f1", "f1-score", "f1_score"}:
            normalized["F1"] = value
    return normalized


def load_metrics_file(metrics_path: Path) -> Dict[str, Dict[str, object]]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    if metrics_path.suffix.lower() == ".json":
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        return {name: normalize_metric_keys(metrics) for name, metrics in payload.items()}

    if metrics_path.suffix.lower() == ".csv":
        metrics = {}
        with metrics_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                name = row.get("model") or row.get("Model") or row.get("name") or row.get("Name")
                if not name:
                    continue
                metrics[name] = normalize_metric_keys(row)
        return metrics

    raise ValueError("metrics_file only supports .json or .csv")


def manual_param_count(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def profile_model(model_name: str, num_classes: int, input_shape: Tuple[int, int, int, int], device: torch.device, drop_path_rate: float):
    kwargs = {"num_classes": num_classes}
    try:
        model = get_model(model_name, **kwargs, drop_path_rate=drop_path_rate)
    except TypeError:
        model = get_model(model_name, **kwargs)

    model = model.to(device)
    model.eval()
    dummy_input = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        _ = model(dummy_input)
    macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
    flops = macs * 2.0
    params = manual_param_count(model)
    return params, flops


def format_decimal(value: object, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def format_millions(value: object, digits: int = 3) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value) / 1e6:.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def collect_profiles(args: argparse.Namespace, device: torch.device) -> Tuple[Dict[str, Dict[str, object]], Dict[str, str]]:
    unique_models = []
    for table in TABLES:
        for name in table["rows"]:
            if name not in unique_models:
                unique_models.append(name)

    input_shape = (args.batch_size, 3, args.height, args.width)
    profile_results: Dict[str, Dict[str, object]] = {}
    failures: Dict[str, str] = {}

    for model_name in unique_models:
        print(f"Profiling: {model_name}")
        try:
            params, flops = profile_model(model_name, args.num_classes, input_shape, device, args.drop_path_rate)
            profile_results[model_name] = {"Params (M)": params, "Flops(M)": flops}
        except Exception as error:
            failures[model_name] = str(error)
            profile_results[model_name] = {"Params (M)": "", "Flops(M)": ""}
            print(f"  failed: {error}")

    return profile_results, failures


def build_table_rows(
    table: Dict[str, object],
    profile_results: Dict[str, Dict[str, object]],
    metrics_results: Dict[str, Dict[str, object]],
) -> List[Dict[str, str]]:
    rows = []
    first_col = table["columns"][0]

    for model_name in table["rows"]:
        row = {column: "" for column in table["columns"]}
        row[first_col] = model_name

        model_metrics = metrics_results.get(model_name, {})
        for metric_name in ["Accuracy", "Precision", "Recall", "F1"]:
            if metric_name in row:
                row[metric_name] = format_decimal(model_metrics.get(metric_name, ""))

        profile_metrics = profile_results.get(model_name, {})
        if "Params (M)" in row:
            row["Params (M)"] = format_millions(profile_metrics.get("Params (M)", ""))
        if "Flops(M)" in row:
            row["Flops(M)"] = format_millions(profile_metrics.get("Flops(M)", ""))

        rows.append(row)

    return rows


def write_csv_table(path: Path, columns: List[str], rows: List[Dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(columns: List[str], rows: List[Dict[str, str]]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, separator] + body)


def write_metrics_template(path: Path) -> None:
    template = {}
    for table in TABLES:
        for model_name in table["rows"]:
            template.setdefault(
                model_name,
                {
                    "Accuracy": "",
                    "Precision": "",
                    "Recall": "",
                    "F1": "",
                },
            )
    path.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_results: Dict[str, Dict[str, object]] = {}
    if args.metrics_file:
        metrics_results = load_metrics_file(Path(args.metrics_file))

    profile_results, failures = collect_profiles(args, device)

    markdown_sections = []
    for table in TABLES:
        rows = build_table_rows(table, profile_results, metrics_results)
        csv_path = output_dir / f"{table['id']}.csv"
        write_csv_table(csv_path, table["columns"], rows)
        markdown_sections.append(f"## {table['title']}\n")
        markdown_sections.append(markdown_table(table["columns"], rows))
        markdown_sections.append("")

    if failures:
        markdown_sections.append("## Profiling Failures\n")
        for model_name, error in failures.items():
            markdown_sections.append(f"- `{model_name}`: {error}")
        markdown_sections.append("")

        failure_log = output_dir / "profiling_failures.txt"
        failure_log.write_text(
            "\n".join(f"{name}: {error}" for name, error in failures.items()),
            encoding="utf-8",
        )

    markdown_path = output_dir / "paper_tables.md"
    markdown_path.write_text("\n".join(markdown_sections), encoding="utf-8")

    if args.write_metrics_template:
        template_path = output_dir / "metrics_template.json"
        write_metrics_template(template_path)
        print(f"Metrics template saved to: {template_path.resolve()}")

    print(f"Markdown tables saved to: {markdown_path.resolve()}")
    print(f"CSV tables saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
