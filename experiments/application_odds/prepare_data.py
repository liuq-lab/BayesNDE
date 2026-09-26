#!/usr/bin/env python3
"""Download and preprocess the Shuttle anomaly-detection data set."""
from __future__ import annotations

import argparse
from experiments.application_odds.common import CONFIG, DATA_ROOT, dataset_path, file_digest, load_odds, processed_path
from pathlib import Path
import numpy as np
import shutil
import tarfile
import tempfile
import urllib.request


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    base = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if base != target and base not in target.parents:
            raise ValueError(f"Unsafe archive member: {member.name}")
        if member.issym() or member.islnk():
            raise ValueError(f"Links are not allowed in archive: {member.name}")
    archive.extractall(destination)


def prepare(data_root: Path = DATA_ROOT, force_download: bool = False) -> None:
    raw_dir = data_root / "raw_archives"
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / CONFIG["archive"]["filename"]
    expected_md5 = CONFIG["archive"]["md5"]
    if force_download or not archive_path.is_file() or file_digest(archive_path, "md5") != expected_md5:
        with tempfile.NamedTemporaryFile(dir=raw_dir, prefix=".download.", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            with urllib.request.urlopen(CONFIG["archive"]["url"], timeout=120) as response:
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)
            if file_digest(temporary, "md5") != expected_md5:
                raise ValueError("Downloaded ODDS archive checksum mismatch")
            temporary.replace(archive_path)
        finally:
            temporary.unlink(missing_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        safe_extract(archive, data_root)
    for dataset in CONFIG["datasets"]:
        split = load_odds(dataset, data_root, verify=True)
        with np.load(dataset_path(dataset, data_root), allow_pickle=False) as raw:
            labels = np.asarray(raw["arr_1"]).reshape(-1)
        n_test = len(split.x_test)
        n_validation = len(split.x_validation)
        train_y = labels[: -(n_test + n_validation)]
        val_y = labels[-(n_test + n_validation) : -n_test]
        output = processed_path(dataset, data_root)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle, train_x=split.x_train, train_y=train_y,
                val_x=split.x_validation, val_y=val_y,
                test_x=split.x_test, test_y=split.y_test,
            )
        temporary.replace(output)
        print(
            f"{dataset}: train={len(split.x_train)} validation={len(split.x_validation)} "
            f"test={len(split.x_test)} dimension={split.dimension} k={int(split.y_test.sum())}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    prepare(args.data_root, args.force_download)


if __name__ == "__main__":
    main()
