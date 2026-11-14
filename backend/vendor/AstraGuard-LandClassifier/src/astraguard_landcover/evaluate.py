"""Evaluate a trained checkpoint on the untouched geographic test split."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .classes import CLASS_NAMES, IGNORE_INDEX
from .dataset import H5LandCoverDataset
from .engine import validate
from .losses import CombinedSegmentationLoss
from .models import build_model
from .utils import select_device, write_json


def _plot_confusion(matrix: list[list[int]], output: Path) -> None:
    values = np.asarray(matrix, dtype=np.float64)
    row_total = values.sum(axis=1, keepdims=True)
    normalized = np.divide(values, row_total, out=np.zeros_like(values), where=row_total > 0)
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    figure.colorbar(image, ax=axis, label="Fraction of true class")
    axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=35, ha="right")
    axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Normalized confusion matrix")
    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            value = normalized[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.55 else "black",
                fontsize=8,
            )
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def evaluate_checkpoint(
    checkpoint_path: Path,
    data_dir: Path | None,
    output_dir: Path | None,
    *,
    batch_size: int | None = None,
    workers: int | None = None,
    regions_file: Path | None = None,
) -> dict[str, Any]:
    device = select_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    stats = checkpoint["stats"]
    processed_dir = data_dir or Path(config["data"]["processed_dir"])
    destination = output_dir or checkpoint_path.parent / "test"
    destination.mkdir(parents=True, exist_ok=True)
    dataset = H5LandCoverDataset(
        processed_dir / "test.h5",
        band_mean=stats["band_mean"],
        band_std=stats["band_std"],
        scale_divisor=float(stats["scale_divisor"]),
        augment=False,
    )
    workers = int(config["data"].get("num_workers", 4)) if workers is None else workers
    with h5py.File(processed_dir / "test.h5", "r") as handle:
        if "regions" not in handle:
            raise ValueError("test.h5 has no region metadata; regenerate it with prepare_tiles")
        region_by_key = dict(zip(handle["keys"].asstr()[:], handle["regions"].asstr()[:]))
        region_names = list(handle["regions"].asstr()[:])
    if regions_file is not None:
        selected_regions = {line.strip() for line in regions_file.read_text(encoding="utf-8").splitlines()
                            if line.strip()}
        if not selected_regions or selected_regions - set(region_names):
            raise ValueError("Reporting region list is empty or contains AOIs absent from test.h5")
        dataset = Subset(dataset, [i for i, region in enumerate(region_names) if region in selected_regions])
    if len(dataset) == 0:
        raise ValueError("The test split contains no tiles")
    loader = DataLoader(
        dataset,
        batch_size=batch_size or int(config["training"].get("batch_size", 16)),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    model = build_model(config["model"], initialize_pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    class_weights = torch.tensor(stats["class_weights"], dtype=torch.float32, device=device)
    criterion = CombinedSegmentationLoss(
        num_classes=len(CLASS_NAMES),
        ignore_index=IGNORE_INDEX,
        class_weights=class_weights,
        ce_weight=float(config["training"].get("ce_weight", 0.6)),
        dice_weight=float(config["training"].get("dice_weight", 0.4)),
    )
    result = validate(
        model,
        loader,
        criterion,
        device,
        num_classes=len(CLASS_NAMES),
        ignore_index=IGNORE_INDEX,
        use_amp=bool(config["training"].get("amp", True)) and device.type == "cuda",
        region_by_key=region_by_key,
    )
    result["checkpoint"] = str(checkpoint_path)
    result["test_tiles"] = len(dataset)
    result["regions_file"] = str(regions_file) if regions_file is not None else None
    result["parameter_count"] = sum(p.numel() for p in model.parameters())
    result["metric_protocol"] = "F1=2TP/(target+predicted); absent-in-both classes excluded from macro means"
    write_json(destination / "metrics.json", result)
    model_name = config["model"]["name"]
    summaries = []
    classes = []
    for region, values in [("ALL", result), *result["per_region"].items()]:
        summaries.append({
            "model": model_name, "region": region,
            "tiles": len(dataset) if region == "ALL" else values["tiles"],
            **{name: values[name] for name in ("miou", "macro_f1", "overall_accuracy")},
        })
        for class_name, scores in values["per_class"].items():
            classes.append({"model": model_name, "region": region, "class": class_name, **scores})
    for filename, rows in (("model_comparison.csv", summaries), ("per_class_metrics.csv", classes)):
        with (destination / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    _plot_confusion(result["confusion_matrix"], destination / "confusion_matrix.png")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--regions-file", type=Path, help="Optional newline-separated test AOI reporting subset")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate_checkpoint(args.checkpoint, args.data_dir, args.output_dir,
                                 batch_size=args.batch_size, workers=args.workers,
                                 regions_file=args.regions_file)
    agriculture = result["per_class"]["agriculture"]
    print(
        f"mIoU={result['miou']} macro_F1={result['macro_f1']} "
        f"agriculture_IoU={agriculture['iou']} "
        f"agriculture_F1={agriculture['f1']} "
        f"area_error={result['agriculture_area_error_percent']}%"
    )


if __name__ == "__main__":
    main()
