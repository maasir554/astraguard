"""Bind the passing AOIs of an existing audit to an explicit subset manifest.

This does not re-audit raster pixels or alter the original failed report. It
reuses coverage evidence only after verifying current file signatures, metadata,
the parent manifest, and unchanged geographic split assignments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from .aoi_manifest import load_aoi_manifest
from .coverage import manifest_hash, verify_report
from .download_manifest import region_is_complete
from .utils import write_json


def freeze_subset(parent_manifest: Path, manifest: Path, audit_path: Path,
                  raw_dir: Path, output: Path) -> dict:
    if output.resolve() == audit_path.resolve():
        raise ValueError("Subset report must not overwrite the parent audit")
    parent = {r.region_id: r for r in load_aoi_manifest(parent_manifest)}
    selected = load_aoi_manifest(manifest)
    evidence = audit_path.read_bytes()
    source = json.loads(evidence)
    if source.get("manifest_sha256") != manifest_hash(parent_manifest):
        raise ValueError("Parent audit belongs to a different manifest")
    if Path(source.get("raw_dir", "")).resolve() != raw_dir.resolve():
        raise ValueError("Parent audit belongs to a different raw directory")
    audited = source.get("regions", {})
    if set(audited) != set(parent):
        raise ValueError("Parent audit does not cover exactly the parent manifest")
    threshold = source.get("min_coverage", 0)
    if not isinstance(threshold, (int, float)) or not 0.95 <= threshold <= 1:
        raise ValueError("Parent audit must enforce at least 95% usable coverage")
    passing = {name for name, item in audited.items() if item.get("passed") is True}
    ids = [r.region_id for r in selected]
    if set(ids) != passing:
        raise ValueError("Subset must contain exactly the passing audited AOIs")
    for record in selected:
        if parent.get(record.region_id) != record:
            raise ValueError(f"Changed geographic split or geometry: {record.region_id}")
        item = audited[record.region_id]
        coverage = item.get("joint_valid_fraction", 0)
        if (not isinstance(coverage, (int, float)) or not math.isfinite(coverage)
                or not threshold <= coverage <= 1 or item.get("accepted_tiles", 0) < 1):
            raise ValueError(f"Insufficient recorded coverage: {record.region_id}")
        if not region_is_complete(record, raw_dir, "2021-01-01", "2021-12-31"):
            raise ValueError(f"Missing or mismatched source metadata: {record.region_id}")
    excluded = sorted(set(parent) - passing)
    report = {
        "passed": True,
        "raw_dir": str(raw_dir.resolve()),
        "manifest_sha256": manifest_hash(manifest),
        "min_coverage": threshold,
        "regions": {name: audited[name] for name in ids},
        "failed_regions": [],
        "split_counts": dict(Counter(r.split for r in selected)),
        "accepted_tiles_by_split": {
            split: sum(audited[r.region_id]["accepted_tiles"]
                       for r in selected if r.split == split)
            for split in ("train", "val", "test")
        },
        "provenance": {
            "method": "passing_audited_subset_with_current_signature_and_metadata_checks",
            "parent_manifest_sha256": source["manifest_sha256"],
            "parent_audit_sha256": hashlib.sha256(evidence).hexdigest(),
            "parent_audit_path": str(audit_path.resolve()),
            "excluded_regions": [{"region_id": name, "split": parent[name].split,
                                  "reason": "not_passing_parent_acquisition_and_coverage_audit"}
                                 for name in excluded],
        },
    }
    # Reject stale files even if the previous audit recorded a pass.
    verify_report(report, raw_dir, manifest, ids)
    write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = freeze_subset(args.parent_manifest, args.manifest, args.audit,
                           args.raw_dir, args.output)
    print(f"READY: {len(report['regions'])} verified AOIs; splits={report['split_counts']}")
    print(f"Recorded accepted tiles: {report['accepted_tiles_by_split']}")
    print(f"Subset quality report: {args.output}")


if __name__ == "__main__":
    main()
