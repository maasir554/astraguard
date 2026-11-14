"""Streaming raster coverage audits and reusable training quality gates."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .classes import BAND_NAMES, CLASS_NAMES, remap_worldcover


def signature(folder: Path) -> dict:
    return {name: {"size": (folder / name).stat().st_size,
                   "mtime_ns": (folder / name).stat().st_mtime_ns}
            for name in ("image.tif", "worldcover.tif", "metadata.json")}


def audit_region(folder: Path, *, min_coverage: float = 0.95, tile_size: int = 256,
                 min_valid_fraction: float = 0.95) -> dict:
    import rasterio
    from rasterio.windows import Window

    if not 0 < min_coverage <= 1 or not 0 < min_valid_fraction <= 1 or tile_size < 1:
        raise ValueError("Coverage fractions must be in (0,1] and tile size positive")
    counts = {"pixels": 0, "image_valid": 0, "label_valid": 0, "joint_valid": 0}
    class_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    accepted_classes = np.zeros_like(class_counts)
    accepted = candidates = 0
    with rasterio.open(folder / "image.tif") as image, rasterio.open(folder / "worldcover.tif") as labels:
        aligned = (image.crs == labels.crs and image.shape == labels.shape
                   and image.transform.almost_equals(labels.transform))
        if not aligned or image.count != len(BAND_NAMES) or labels.count != 1 or image.crs is None:
            raise ValueError("Image/label grids, CRS or band counts do not match the data contract")
        for row in range(0, image.height, tile_size):
            height = min(tile_size, image.height - row)
            values = image.read(window=Window(0, row, image.width, height))
            raw = labels.read(1, window=Window(0, row, image.width, height))
            present = np.isfinite(values).all(axis=0) & (values != 0).any(axis=0)
            mapped = remap_worldcover(raw)
            reference = (raw != 0) & (mapped != 255)
            joint = present & reference
            counts["pixels"] += int(raw.size)
            counts["image_valid"] += int(present.sum())
            counts["label_valid"] += int(reference.sum())
            counts["joint_valid"] += int(joint.sum())
            class_counts += np.bincount(mapped[joint], minlength=len(CLASS_NAMES))
            if height == tile_size:
                for column in range(0, image.width - tile_size + 1, tile_size):
                    candidates += 1
                    valid = joint[:, column:column + tile_size]
                    if float(valid.mean()) >= min_valid_fraction:
                        accepted += 1
                        tile_mask = mapped[:, column:column + tile_size]
                        accepted_classes += np.bincount(tile_mask[valid], minlength=len(CLASS_NAMES))
        fractions = {name + "_fraction": counts[name] / counts["pixels"]
                     for name in ("image_valid", "label_valid", "joint_valid")}
        return {"passed": fractions["joint_valid_fraction"] >= min_coverage and accepted > 0,
                "min_coverage": min_coverage, "min_valid_fraction": min_valid_fraction,
                "tile_size": tile_size, "aligned": True, "pixel_counts": counts, **fractions,
                "accepted_tiles": accepted, "candidate_tiles": candidates,
                "class_counts": dict(zip(CLASS_NAMES, map(int, class_counts))),
                "accepted_tile_class_counts": dict(zip(CLASS_NAMES, map(int, accepted_classes))),
                "signature": signature(folder)}


def manifest_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_report(report: dict, raw_dir: Path, manifest: Path, region_ids: list[str]) -> None:
    """Lightweight gate: reject missing, failed or stale audits before GPU submission."""
    if not report.get("passed"):
        raise ValueError("Quality report is incomplete or failed")
    if report.get("manifest_sha256") != manifest_hash(manifest):
        raise ValueError("Quality report belongs to a different manifest")
    if Path(report.get("raw_dir", "")).resolve() != raw_dir.resolve():
        raise ValueError("Quality report belongs to a different raw directory")
    regions = report.get("regions", {})
    if set(regions) != set(region_ids):
        raise ValueError("Quality report does not cover exactly the requested AOIs")
    for region in region_ids:
        item = regions[region]
        if not item.get("passed") or item.get("signature") != signature(raw_dir / region):
            raise ValueError(f"Failed or stale raster quality audit: {region}")
