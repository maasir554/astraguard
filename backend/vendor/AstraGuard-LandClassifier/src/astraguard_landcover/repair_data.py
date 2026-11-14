"""Audit all manifest AOIs, then build a separate coverage-checked raw-data version."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from .aoi_manifest import load_aoi_manifest
from .coverage import audit_region, manifest_hash, verify_report
from .download_manifest import REQUIRED_REGION_FILES, region_is_complete
from .utils import write_json

GIB = 1024**3


def unique_bytes(*roots: Path) -> int:
    seen = set()
    total = 0
    for root in roots:
        if root.exists():
            for path in root.rglob("*"):
                if path.is_file():
                    stat = path.stat()
                    key = (stat.st_dev, stat.st_ino)
                    if key not in seen:
                        seen.add(key)
                        total += stat.st_size
    return total


def inspect(record, root, min_coverage):
    if not region_is_complete(record, root, "2021-01-01", "2021-12-31"):
        return {"passed": False, "error": "Missing/mismatched files or 2021 source metadata"}
    try:
        return audit_region(root / record.region_id, min_coverage=min_coverage)
    except Exception as exc:
        return {"passed": False, "error": f"{type(exc).__name__}: {exc}"}


def audit_all(records, root, manifest, report_path, min_coverage):
    report = {"raw_dir": str(root.resolve()), "manifest_sha256": manifest_hash(manifest),
              "min_coverage": min_coverage, "regions": {}}
    for index, record in enumerate(records, 1):
        result = inspect(record, root, min_coverage)
        report["regions"][record.region_id] = result
        write_json(report_path, report)
        print(f"Audit {index}/{len(records)} {record.region_id}: "
              f"{'PASS' if result['passed'] else 'REPAIR'} "
              f"coverage={result.get('joint_valid_fraction', 0):.2%}", flush=True)
    report["failed_regions"] = [name for name, values in report["regions"].items() if not values["passed"]]
    report["passed"] = not report["failed_regions"]
    write_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "repair", "verify"), default="audit")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path)
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--max-raw-gb", type=float, default=50)
    parser.add_argument("--scenes-per-granule", type=int, default=12)
    parser.add_argument("--max-total-scenes", type=int, default=96)
    parser.add_argument("--max-cloud", type=float, default=20)
    args = parser.parse_args()
    if not 0 < args.min_coverage <= 1 or args.max_raw_gb < 2:
        parser.error("min-coverage must be in (0,1]; storage budget must be >= 2 GiB")
    records = load_aoi_manifest(args.manifest)
    if args.mode == "verify":
        if not args.quality_report:
            parser.error("verify requires --quality-report")
        verify_report(json.loads(args.quality_report.read_text()), args.raw_dir, args.manifest,
                      [r.region_id for r in records])
        print("Quality report verified against manifest and current file signatures", flush=True)
        return
    if args.mode == "repair":
        if not args.target_dir:
            parser.error("repair requires a separate --target-dir")
        source, target = args.raw_dir.resolve(), args.target_dir.resolve()
        if source == target or source in target.parents or target in source.parents:
            parser.error("Source and target directories must be separate, non-overlapping roots")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    source_report = audit_all(records, args.raw_dir, args.manifest,
                              args.report_dir / "source_audit.json", args.min_coverage)
    if args.mode == "audit":
        print(f"Audit complete: {len(source_report['failed_regions'])}/{len(records)} AOIs need repair")
        return

    from .download_aoi import download_region
    import dask
    target = args.target_dir
    target.mkdir(parents=True, exist_ok=True)
    staging = target / ".staging"
    staging.mkdir(exist_ok=True)
    limit = int(args.max_raw_gb * GIB)
    actions = {}
    failures = {}
    # Existing files and raw v1 are never overwritten. Only this process's newly
    # created staging directories are removed by TemporaryDirectory cleanup.
    for index, record in enumerate(records, 1):
        name = record.region_id
        final = target / name
        if final.exists():
            result = inspect(record, target, args.min_coverage)
            if not result["passed"]:
                failures[name] = "Existing target fails quality; preserve it and choose a new target directory"
            else:
                actions[name] = "resumed_existing_target"
            continue
        print(f"Versioning {index}/{len(records)} {name}", flush=True)
        try:
            with tempfile.TemporaryDirectory(prefix=f"{name}-", dir=staging) as temporary:
                temporary_root = Path(temporary)
                candidate = temporary_root / name
                if source_report["regions"][name]["passed"]:
                    candidate.mkdir()
                    for filename in REQUIRED_REGION_FILES:
                        source_file = args.raw_dir / name / filename
                        try:
                            os.link(source_file, candidate / filename)
                        except OSError:
                            if unique_bytes(args.raw_dir, target) + source_file.stat().st_size > limit:
                                raise RuntimeError("Combined raw-data budget prevents fallback file copy")
                            shutil.copy2(source_file, candidate / filename)
                    action = "reused_good_source_files"
                else:
                    if unique_bytes(args.raw_dir, target) + GIB > limit:
                        raise RuntimeError("Combined v1+v2 raw-data budget has less than 1 GiB staging headroom")
                    with dask.config.set(scheduler="threads", num_workers=2):
                        download_region(region_id=name, split=record.split, bbox=list(record.bbox),
                                        start_date="2021-01-01", end_date="2021-12-31",
                                        output_root=temporary_root, max_cloud=args.max_cloud,
                                        max_scenes=args.max_total_scenes, spatial_coverage=True,
                                        scenes_per_granule=args.scenes_per_granule,
                                        write_observation_count=True)
                    action = "repaired_spatial_monthly_v2"
                quality = inspect(record, temporary_root, args.min_coverage)
                if not quality["passed"]:
                    raise RuntimeError(f"Candidate failed coverage gate: {quality}")
                if unique_bytes(args.raw_dir, target) > limit:
                    raise RuntimeError("Candidate exceeds combined v1+v2 raw-data budget")
                candidate.rename(final)
                actions[name] = action
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
            print(f"Not promoted: {name}: {failures[name]}", flush=True)
        write_json(args.report_dir / "repair_progress.json", {"actions": actions, "failures": failures})
    final_report = audit_all(records, target, args.manifest,
                             args.report_dir / "target_audit.json", args.min_coverage)
    final_report["actions"] = actions
    final_report["repair_failures"] = failures
    final_report["combined_unique_raw_bytes"] = unique_bytes(args.raw_dir, target)
    final_report["max_raw_gb"] = args.max_raw_gb
    final_report["passed"] = final_report["passed"] and final_report["combined_unique_raw_bytes"] <= limit
    write_json(args.report_dir / "target_audit.json", final_report)
    if not final_report["passed"]:
        raise SystemExit("Repair incomplete. Inspect target_audit.json; do not start training.")
    print("Versioned raw data passed all coverage gates; ready for fresh training runs", flush=True)


if __name__ == "__main__":
    main()
