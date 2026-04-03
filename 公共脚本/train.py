"""
文件说明：
这个文件是 MEL-StarNet 的通用训练脚本，
基于 ImageFolder 数据集组织方式完成训练、验证、保存 checkpoint 和输出分类映射。
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from PIL import ImageFile
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from 公共核心.mel_starnet import MELStarNet

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class GaussianNoise:
    def __init__(self, mean: float = 0.0, std: float = 0.05, p: float = 0.5) -> None:
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return tensor
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MEL-StarNet with ImageFolder datasets.")
    parser.add_argument("--train_dir", type=str, default="C:\\Users\\gml\\Desktop\\Indian Medicinal Leaves Image Datasets\\Medicinal Leaf dataset_spilt\\train", help="Path to the training dataset.")
    parser.add_argument("--val_dir", type=str, default="C:\\Users\\gml\\Desktop\\Indian Medicinal Leaves Image Datasets\\Medicinal Leaf dataset_spilt\\val", help="Path to the validation dataset.")
    parser.add_argument("--save_dir", type=str, default="runs/mel_starnet", help="Directory to save outputs.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.02, help="Weight decay for AdamW.")
    parser.add_argument("--img_size", type=int, default=224, help="Square image size.")
    parser.add_argument("--workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Training device.")
    parser.add_argument("--lrf", type=float, default=0.2, help="Final cosine LR ratio.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience on validation accuracy.")
    parser.add_argument("--drop_path_rate", type=float, default=0.0, help="Drop path rate for MEL-StarNet.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def normalize_input_path(path_text: str) -> str:
    return path_text.strip().strip('"').strip("'")


def prompt_directory_path(arg_value: str, prompt_text: str) -> str:
    candidate = normalize_input_path(arg_value)
    while not candidate:
        candidate = normalize_input_path(input(prompt_text))
        if not candidate:
            print("路径不能为空，请重新输入。")
            continue
        if not Path(candidate).is_dir():
            print(f"路径不存在或不是文件夹: {candidate}")
            candidate = ""

    if not Path(candidate).is_dir():
        raise FileNotFoundError(f"路径不存在或不是文件夹: {candidate}")
    return candidate


def resolve_dataset_paths(args: argparse.Namespace) -> argparse.Namespace:
    args.train_dir = prompt_directory_path(args.train_dir, "请输入训练集路径 train_dir: ")
    args.val_dir = prompt_directory_path(args.val_dir, "请输入验证集路径 val_dir: ")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(img_size: int) -> Dict[str, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            GaussianNoise(std=0.05, p=0.5),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return {"train": train_transform, "val": val_transform}


def build_dataloaders(args: argparse.Namespace) -> Dict[str, object]:
    transforms_map = build_transforms(args.img_size)
    train_dataset = datasets.ImageFolder(args.train_dir, transform=transforms_map["train"])
    val_dataset = datasets.ImageFolder(args.val_dir, transform=transforms_map["val"])

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError("Training and validation folders must contain the same class names.")

    pin_memory = args.device.startswith("cuda") and torch.cuda.is_available()
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": pin_memory,
        "persistent_workers": args.workers > 0,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
    }


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable. Falling back to CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(device_arg)


def update_confusion_matrix(
    confusion: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    num_classes: int,
) -> None:
    targets = targets.to(torch.int64).cpu()
    predictions = predictions.to(torch.int64).cpu()
    encoded = targets * num_classes + predictions
    batch_confusion = torch.bincount(encoded, minlength=num_classes * num_classes)
    confusion += batch_confusion.view(num_classes, num_classes)


def compute_metrics(confusion: torch.Tensor, loss_sum: float, sample_count: int) -> Dict[str, float]:
    confusion = confusion.to(torch.float32)
    tp = confusion.diag()
    fp = confusion.sum(dim=0) - tp
    fn = confusion.sum(dim=1) - tp

    precision = tp / torch.clamp(tp + fp, min=1.0)
    recall = tp / torch.clamp(tp + fn, min=1.0)
    f1 = 2.0 * precision * recall / torch.clamp(precision + recall, min=1e-12)
    accuracy = tp.sum() / torch.clamp(confusion.sum(), min=1.0)

    return {
        "loss": loss_sum / max(sample_count, 1),
        "accuracy": accuracy.item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1.mean().item(),
    }


def compute_running_accuracy(confusion: torch.Tensor) -> float:
    total = torch.clamp(confusion.sum(), min=1).item()
    correct = confusion.diag().sum().item()
    return correct / total


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    epoch: int,
    total_epochs: int,
    stage_name: str,
    optimizer: AdamW = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    loss_sum = 0.0
    sample_count = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    total_batches = len(dataloader)
    progress_desc = f"{stage_name} {epoch}/{total_epochs}"
    iterator = dataloader

    if tqdm is not None:
        iterator = tqdm(dataloader, total=total_batches, desc=progress_desc, dynamic_ncols=True, leave=False)

    for batch_idx, (images, targets) in enumerate(iterator, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        predictions = outputs.argmax(dim=1)
        batch_size = images.size(0)
        loss_sum += loss.item() * batch_size
        sample_count += batch_size
        update_confusion_matrix(confusion, targets, predictions, num_classes)

        running_loss = loss_sum / max(sample_count, 1)
        running_acc = compute_running_accuracy(confusion)

        if tqdm is not None:
            iterator.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.4f}")
        else:
            print(
                f"\r{progress_desc} Batch [{batch_idx}/{total_batches}] "
                f"Loss: {running_loss:.4f} Acc: {running_acc:.4f}",
                end="",
                flush=True,
            )

    if tqdm is None:
        print(flush=True)
    else:
        iterator.close()

    return compute_metrics(confusion, loss_sum, sample_count)


def save_json(path: Path, payload: Dict[str, int]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def save_checkpoint(path: Path, state: Dict[str, object]) -> None:
    torch.save(state, path)


def main() -> None:
    args = resolve_dataset_paths(parse_args())
    set_seed(args.seed)

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    data = build_dataloaders(args)
    train_dataset = data["train_dataset"]
    val_dataset = data["val_dataset"]
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]

    num_classes = len(train_dataset.classes)
    model_kwargs = {
        "num_classes": num_classes,
        "base_dim": 32,
        "depths": (3, 3, 12, 5),
        "mlp_ratio": 4,
        "drop_path_rate": args.drop_path_rate,
    }

    model = MELStarNet(**model_kwargs).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: ((1.0 + math.cos(math.pi * epoch / args.epochs)) / 2.0) * (1.0 - args.lrf) + args.lrf,
    )

    class_to_idx_path = save_dir / "class_to_idx.json"
    save_json(class_to_idx_path, train_dataset.class_to_idx)

    best_acc = 0.0
    best_epoch = -1
    epochs_without_improve = 0

    print(f"Train samples: {len(train_dataset)}", flush=True)
    print(f"Val samples: {len(val_dataset)}", flush=True)
    print(f"Classes: {num_classes}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Outputs will be saved to: {save_dir.resolve()}", flush=True)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            num_classes,
            epoch=epoch,
            total_epochs=args.epochs,
            stage_name="Train",
            optimizer=optimizer,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            num_classes,
            epoch=epoch,
            total_epochs=args.epochs,
            stage_name="Val",
        )
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"LR: {current_lr:.6f} "
            f"Train Loss: {train_metrics['loss']:.4f} "
            f"Train Acc: {train_metrics['accuracy']:.4f} "
            f"Val Loss: {val_metrics['loss']:.4f} "
            f"Val Acc: {val_metrics['accuracy']:.4f} "
            f"Val Precision: {val_metrics['precision']:.4f} "
            f"Val Recall: {val_metrics['recall']:.4f} "
            f"Val F1: {val_metrics['f1']:.4f}",
            flush=True,
        )

        improved = val_metrics["accuracy"] > best_acc
        if improved:
            best_acc = val_metrics["accuracy"]
            best_epoch = epoch
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        checkpoint = {
            "epoch": epoch,
            "best_acc": best_acc,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "class_to_idx": train_dataset.class_to_idx,
            "model_kwargs": model_kwargs,
            "args": vars(args),
            "metrics": {
                "train": train_metrics,
                "val": val_metrics,
            },
        }

        save_checkpoint(save_dir / "last.pt", checkpoint)

        if improved:
            save_checkpoint(save_dir / "best.pt", checkpoint)
            print(f"Best checkpoint updated at epoch {epoch} with val_acc={best_acc:.4f}", flush=True)

        if epochs_without_improve >= args.patience:
            print(f"Early stopping triggered after {args.patience} epochs without improvement.", flush=True)
            break

    print(f"Training finished. Best epoch: {best_epoch}, best val_acc: {best_acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
