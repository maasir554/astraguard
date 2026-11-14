"""Run the paper split audit, optional preprocessing, training and AOI evaluation."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

from .utils import write_json


def run(*arguments: str) -> None:
    command = [sys.executable, "-m", *map(str, arguments)]
    print("Running:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, help="Preprocess this raw directory first")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=["deeplabv3plus", "segformer"],
                        default=["deeplabv3plus", "segformer"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume matching last.pt files explicitly")
    parser.add_argument("--buffer-km", type=float, default=5)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    run("astraguard_landcover.audit_split", "--manifest", args.manifest,
        "--buffer-km", str(args.buffer_km), "--report", args.output_root / "split_audit.json")
    stamp_path = args.data_dir / "experiment_provenance.json"
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if args.raw_dir:
        if any(args.data_dir.glob("*.h5")):
            parser.error("For preprocessing choose a new empty data-dir to preserve existing HDF5 files")
        run("astraguard_landcover.prepare_tiles", "--raw-dir", args.raw_dir,
            "--output-dir", args.data_dir, "--aoi-manifest", args.manifest)
        write_json(stamp_path, {"manifest_sha256": digest})
    # Validate prepared split membership even when using PBS-produced HDF5 files.
    import json
    import h5py
    from .aoi_manifest import load_aoi_manifest
    records = load_aoi_manifest(args.manifest)
    for split in ("train", "val", "test"):
        expected = {r.region_id for r in records if r.split == split}
        with h5py.File(args.data_dir / f"{split}.h5", "r") as handle:
            actual = set(handle["regions"].asstr()[:])
        if actual != expected:
            parser.error(f"{split}.h5 AOIs differ from manifest (including regions with zero accepted tiles)")
    if stamp_path.exists() and json.loads(stamp_path.read_text())["manifest_sha256"] != digest:
        parser.error("Prepared data was generated with a different manifest")
    for model in dict.fromkeys(args.models):
        destination = args.output_root / f"{model}_seed{args.seed}"
        if destination.exists() and not args.resume:
            parser.error(f"Run directory exists: {destination}; choose a new output-root or --resume")
        if args.resume and destination.exists():
            previous = destination / "experiment_provenance.json"
            if not previous.exists() or json.loads(previous.read_text())["manifest_sha256"] != digest:
                parser.error("Resume requires a matching experiment provenance file")
        destination.mkdir(parents=True, exist_ok=True)
        write_json(destination / "experiment_provenance.json", {"manifest_sha256": digest})
        shutil.copy2(args.manifest, destination / "aoi_manifest.tsv")
        for source, target in (("stats.json", "dataset_stats.json"), ("manifest.json", "dataset_manifest.json")):
            shutil.copy2(args.data_dir / source, destination / target)
        command = ["astraguard_landcover.train", "--config", f"configs/{model}.yaml",
                   "--data-dir", args.data_dir, "--output-dir", destination,
                   "--batch-size", str(args.batch_size), "--workers", str(args.workers),
                   "--epochs", str(args.epochs), "--seed", str(args.seed)]
        if args.resume and (destination / "last.pt").exists():
            command += ["--resume", destination / "last.pt"]
        run(*command)
        run("astraguard_landcover.evaluate", "--checkpoint", destination / "best.pt",
            "--data-dir", args.data_dir, "--workers", str(args.workers),
            "--batch-size", str(args.batch_size))


if __name__ == "__main__":
    main()
