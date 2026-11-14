"""Audit geographic separation and optionally write a buffered subset of existing AOIs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path

from .aoi_manifest import AOIRecord, load_aoi_manifest
from .utils import write_json


def conflicts(left: AOIRecord, right: AOIRecord, buffer_km: float = 0) -> bool:
    """Conservative latitude/longitude rectangle separation with a kilometre buffer.

    Suitable for MP AOIs; this is not a geodesic distance or district-boundary test.
    """
    west, south, east, north = left.bbox
    other_west, other_south, other_east, other_north = right.bbox
    latitude = (south + north + other_south + other_north) / 4
    lon_buffer = buffer_km / (111.32 * math.cos(math.radians(latitude)))
    lat_buffer = buffer_km / 110.57
    return (west < other_east + lon_buffer and east > other_west - lon_buffer
            and south < other_north + lat_buffer and north > other_south - lat_buffer)


def audit(records: list[AOIRecord], buffer_km: float = 0) -> dict:
    pairs = []
    for index, left in enumerate(records):
        for right in records[index + 1:]:
            if left.split != right.split and conflicts(left, right, buffer_km):
                pairs.append({"left": left.region_id, "left_split": left.split,
                              "right": right.region_id, "right_split": right.split})
    return {"buffer_km": buffer_km, "region_count": len(records),
            "split_counts": {split: sum(r.split == split for r in records)
                             for split in ("train", "val", "test")},
            "cross_split_conflicts": pairs, "passed": not pairs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--buffer-km", type=float, default=5)
    parser.add_argument("--clean-output", type=Path,
                        help="Write a NEW manifest: retain test, prune nearby validation, then training")
    args = parser.parse_args()
    if not math.isfinite(args.buffer_km) or args.buffer_km < 0:
        parser.error("buffer-km must be finite and nonnegative")
    records = load_aoi_manifest(args.manifest)
    report = audit(records, args.buffer_km)
    report["manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if args.clean_output:
        if args.clean_output.resolve() == args.manifest.resolve() or args.clean_output.exists():
            parser.error("clean-output must be a new file distinct from the source manifest")
        test = [r for r in records if r.split == "test"]
        validation = [r for r in records if r.split == "val"
                      and not any(conflicts(r, other, args.buffer_km) for other in test)]
        held_out = test + validation
        kept = [r for r in records if r.split != "train"
                or not any(conflicts(r, other, args.buffer_km) for other in held_out)]
        kept = [r for r in kept if r.split != "val" or r in validation]
        cleaned = audit(kept, args.buffer_km)
        if not cleaned["passed"] or any(not any(r.split == split for r in kept)
                                       for split in ("train", "val", "test")):
            write_json(args.report, report)
            raise SystemExit("Cannot clean: held-out splits conflict or all training AOIs would be removed")
        args.clean_output.parent.mkdir(parents=True, exist_ok=True)
        with args.clean_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["region_id", "split", "west", "south", "east", "north"])
            writer.writerows([r.region_id, r.split, *r.bbox] for r in kept)
        report["cleaned"] = cleaned
        report["removed_regions"] = [r.region_id for r in records if r not in kept]
    write_json(args.report, report)
    print(f"AOIs: {len(records)}; cross-split conflicts: {len(report['cross_split_conflicts'])}")
    if not report["passed"] and not args.clean_output:
        raise SystemExit("Split audit failed. Inspect report before training.")


if __name__ == "__main__":
    main()
