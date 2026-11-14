"""Training and validation loops shared by CLI entry points."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from tqdm import tqdm

from .metrics import SegmentationMetrics


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor, list[str]]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    *,
    use_amp: bool,
) -> float:
    model.train()
    loss_sum = 0.0
    example_count = 0
    progress = tqdm(loader, desc="train", leave=False)
    for images, masks, _keys in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = images.shape[0]
        loss_sum += float(loss.detach()) * batch_size
        example_count += batch_size
        progress.set_postfix(loss=f"{loss_sum / example_count:.4f}")
    return loss_sum / max(example_count, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor, list[str]]],
    criterion: nn.Module,
    device: torch.device,
    *,
    num_classes: int,
    ignore_index: int,
    use_amp: bool,
    region_by_key: dict[str, str] | None = None,
) -> dict[str, Any]:
    model.eval()
    metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index)
    loss_sum = 0.0
    example_count = 0
    regions: dict[str, SegmentationMetrics] = {}
    region_tiles: dict[str, int] = {}
    for images, masks, keys in tqdm(loader, desc="validate", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = criterion(logits, masks)
        batch_size = images.shape[0]
        loss_sum += float(loss) * batch_size
        example_count += batch_size
        metrics.update(logits, masks)
        if region_by_key is not None:
            prediction = logits.argmax(dim=1).cpu()
            targets = masks.cpu()
            for index, key in enumerate(keys):
                region = region_by_key[key]
                if region not in regions:
                    regions[region] = SegmentationMetrics(num_classes, ignore_index)
                    region_tiles[region] = 0
                regions[region].update(prediction[index], targets[index])
                region_tiles[region] += 1
    result = metrics.compute()
    result["loss"] = loss_sum / max(example_count, 1)
    if region_by_key is not None:
        result["per_region"] = {
            name: {**metric.compute(), "tiles": region_tiles[name]}
            for name, metric in sorted(regions.items())
        }
    return result
