#!/usr/bin/env python3
"""Every data set the UCI table reports, in one place."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "uci_expansion_v1"
RAW_ARCHIVES = DATA_ROOT / "raw_archives"
PROCESSED_ROOT = DATA_ROOT / "processed"


ZENODO_RECORD = "https://zenodo.org/api/records/4559067/files"


DATASET_FILES = {
    "BANK": {
        "url": f"{ZENODO_RECORD}/BANK.tar.gz/content",
        "archive": "BANK.tar.gz",
        "md5": "c6a2f0f47b99bab93838528e0f5603bc",
        "loader": "uci_npy",
        "shape": (45211, 17),
    },
}


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON serialize {type(value)!r}")


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, default=json_default)


def save_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {path}")
    return value


def append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {text}"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def md5_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def download_file(url: str, target: Path, expected_md5: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and md5_file(target) == expected_md5:
        return target
    partial = target.with_suffix(target.suffix + ".part")
    if partial.exists():
        partial.unlink()
    with urllib.request.urlopen(url, timeout=120) as response, open(partial, "wb") as handle:
        shutil.copyfileobj(response, handle, length=1 << 20)
    actual = md5_file(partial)
    if actual != expected_md5:
        raise ValueError(f"MD5 mismatch for {target.name}: expected {expected_md5}, got {actual}")
    partial.replace(target)
    return target


def safe_extract_tar(archive_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    root = target_dir.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            destination = (target_dir / member.name).resolve()
            if destination != root and root not in destination.parents:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        archive.extractall(target_dir)


@dataclass
class PaperSplit:
    dataset: str
    train_x: np.ndarray
    val_x: np.ndarray
    test_x: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    selected_test_x: np.ndarray
    selected_test_indices: np.ndarray
    metadata: dict[str, Any]


def paper_split(data: np.ndarray, dataset: str, test_points: int = 100) -> PaperSplit:
    data = np.asarray(data, dtype=np.float64)
    order = np.arange(len(data), dtype=np.int64)
    np.random.RandomState(42).shuffle(order)
    n_test = int(0.1 * len(order))
    test_indices = order[-n_test:]
    remaining = order[:-n_test]
    n_val = int(0.1 * len(remaining))
    val_indices = remaining[-n_val:]
    train_indices = remaining[:-n_val]
    if test_points < 1 or test_points > len(test_indices):
        raise ValueError(f"test_points must be in [1, {len(test_indices)}]")
    selected = test_indices[: int(test_points)]
    metadata = {
        "dataset": dataset,
        "split_protocol": "fixed seed-42 UCI split",
        "shuffle_seed": 42,
        "all_points": int(len(data)),
        "train_points": int(len(train_indices)),
        "validation_points": int(len(val_indices)),
        "test_points_full": int(len(test_indices)),
        "test_points_selected": int(len(selected)),
        "test_subset": "first points of the shuffled test split",
    }
    return PaperSplit(
        dataset=dataset,
        train_x=data[train_indices].astype(np.float32),
        val_x=data[val_indices].astype(np.float32),
        test_x=data[test_indices].astype(np.float32),
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        selected_test_x=data[selected].astype(np.float32),
        selected_test_indices=selected,
        metadata=metadata,
    )


def prepare_dataset(data_root: Path, dataset: str, test_points: int, *, smoke_max_rows: int | None = None) -> PaperSplit:
    info = DATASET_FILES[dataset]
    archive = download_file(str(info["url"]), data_root / "raw_archives" / str(info["archive"]), str(info["md5"]))
    extracted = data_root / "raw" / dataset
    loader = str(info.get("loader", "uci_npy"))
    if loader == "uci_npy":
        data_path = extracted / dataset / "data.npy"
        if not data_path.exists():
            safe_extract_tar(archive, extracted)
        data = np.load(data_path)
        expected_shape = tuple(info["shape"])
        if tuple(data.shape) != expected_shape:
            raise ValueError(f"Unexpected {dataset} shape {data.shape}; expected {expected_shape}")
        if not np.isfinite(data).all():
            raise ValueError(f"{dataset} contains non-finite values")
        if float(data.min()) < 0.0 or float(data.max()) > 1.0:
            raise ValueError(f"{dataset} is not in the paper's [0, 1] preprocessed scale")
        source_fingerprint = {
            "data_shape": list(data.shape),
            "data_dtype": str(data.dtype),
            "data_sha256": sha256_array(data),
            "data_path": str(data_path),
            "min": float(data.min()),
            "max": float(data.max()),
            "smoke_truncated": bool(smoke_max_rows is not None and len(data) > int(smoke_max_rows)),
        }
        if smoke_max_rows is not None and 0 < smoke_max_rows < len(data):
            data = np.asarray(data[: int(smoke_max_rows)], dtype=np.float64)
        split = paper_split(data, dataset, min(test_points, max(1, int(0.1 * len(data)))))
        expected_rows = expected_shape[0]
    else:
        raise ValueError(f"Unknown {dataset} loader {loader!r}")
    split_root = data_root / "processed" / dataset
    split_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        split_root / "paper_split.npz",
        train_x=split.train_x,
        val_x=split.val_x,
        test_x=split.test_x,
        train_indices=split.train_indices,
        val_indices=split.val_indices,
        test_indices=split.test_indices,
    )
    save_json(
        split_root / "dataset_fingerprint.json",
        {
            **split.metadata,
            "archive": str(archive),
            "archive_md5": md5_file(archive),
            "data_path": str(data_path),
            **source_fingerprint,
            "benchmark_rows_used": int(split.metadata["all_points"]),
            "smoke_truncated": bool(source_fingerprint.get("smoke_truncated", False)),
            "expected_rows": int(expected_rows),
        },
    )
    return split


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path, expected_sha256: str | None = None, timeout: int = 180) -> Path:
    import urllib.request

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        actual = sha256_file(target)
        if expected_sha256 is None or actual == expected_sha256:
            return target
    partial = target.with_suffix(target.suffix + ".part")
    if partial.exists():
        partial.unlink()
    with urllib.request.urlopen(url, timeout=timeout) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1 << 20)
    actual = sha256_file(partial)
    if expected_sha256 is not None and actual != expected_sha256:
        partial.unlink()
        raise ValueError(f"SHA-256 mismatch for {target.name}: expected {expected_sha256}, got {actual}")
    partial.replace(target)
    return target


DATA_SEED = 42
DEQUANT_SEED = 42
SPLIT_POLICY = "conditional UCI stratified 81/9/10 split"

MIN_SINGULAR_VALUE_RATIO = 1.0e-5
MAX_CORR_CONDITION = 1.0e4


PARKTELE = "ParkinsonsTelemonitoring"
PARKTELE_PREPROCESS_VERSION = "parktele_v1_minmax_drop_ddp_dda"
PARKTELE_URL = "https://archive.ics.uci.edu/static/public/189/parkinsons+telemonitoring.zip"
PARKTELE_ARCHIVE_SHA256 = "2f82bb0ef96fa7d8d7edf4d97b89173e07c2cca7723440a64bc38918a83345df"
PARKTELE_ARCHIVE = RAW_ARCHIVES / "ParkinsonsTelemonitoring.zip"
PARKTELE_MEMBER = "parkinsons_updrs.data"

PARKTELE_VOICE_COLUMNS = (
    "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
    "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11",
    "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "PPE",
)
PARKTELE_COLLINEAR_DROP = {
    "Jitter:DDP": ("Jitter:RAP", 3.0, 0.05),
    "Shimmer:DDA": ("Shimmer:APQ3", 3.0, 0.05),
}
PARKTELE_FEATURE_COLUMNS = tuple(
    c for c in PARKTELE_VOICE_COLUMNS if c not in PARKTELE_COLLINEAR_DROP
)

PARKTELE_X_DIM = 14
PARKTELE_RAW_ROWS = 5875
PARKTELE_SPLIT_SIZES = (4760, 528, 587)
PARKTELE_TEST_EVAL_POINTS = 587


def _parktele_paths() -> tuple[Path, Path, Path]:
    root = PROCESSED_ROOT / PARKTELE
    return root, root / "paper_split.npz", root / "dataset_fingerprint.json"


def _parktele_read_raw():
    import pandas as pd

    download(PARKTELE_URL, PARKTELE_ARCHIVE, PARKTELE_ARCHIVE_SHA256)
    with zipfile.ZipFile(PARKTELE_ARCHIVE) as archive:
        with archive.open(PARKTELE_MEMBER) as handle:
            frame = pd.read_csv(handle)
    if frame.shape != (PARKTELE_RAW_ROWS, 22):
        raise ValueError(f"Unexpected telemonitoring table: {frame.shape}")
    missing = [c for c in PARKTELE_VOICE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing voice columns: {missing}")
    return frame


def prepare_parktele(force: bool = False) -> PaperSplit:
    root, split_path, fingerprint_path = _parktele_paths()
    if split_path.exists() and fingerprint_path.exists() and not force:
        return load_parktele()

    frame = _parktele_read_raw()

    collinearity = []
    for dropped, (kept, ratio, tol) in PARKTELE_COLLINEAR_DROP.items():
        observed = frame[dropped].to_numpy(np.float64) / frame[kept].to_numpy(np.float64)
        deviation = float(np.abs(observed - ratio).max())
        if deviation > tol:
            raise ValueError(
                f"{dropped} is not a {ratio}x copy of {kept}: max deviation {deviation:.4f}"
            )
        collinearity.append({
            "dropped": dropped,
            "proxy_for": kept,
            "expected_ratio": ratio,
            "observed_ratio_min": float(observed.min()),
            "observed_ratio_max": float(observed.max()),
            "reason": "definitional identity in Praat; retaining it makes the density degenerate",
        })

    raw = frame[list(PARKTELE_FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    if raw.shape != (PARKTELE_RAW_ROWS, PARKTELE_X_DIM) or not np.isfinite(raw).all():
        raise ValueError(f"Unexpected feature matrix: {raw.shape}")
    duplicate_rows = int(len(raw) - len(np.unique(raw, axis=0)))
    if duplicate_rows:
        raise ValueError(
            f"Unexpected duplicate rows in telemonitoring voice features: {duplicate_rows}"
        )

    lower = raw.min(axis=0)
    upper = raw.max(axis=0)
    span = upper - lower
    if np.any(span <= 0.0):
        raise ValueError("A retained feature is constant")
    scaled = (raw - lower) / span

    correlation = np.corrcoef(scaled, rowvar=False)
    condition = float(np.linalg.cond(correlation))
    if condition >= MAX_CORR_CONDITION:
        raise ValueError(
            f"Correlation condition number {condition:.3g} >= {MAX_CORR_CONDITION}: "
            "the design matrix is near-degenerate, which invalidates density comparisons"
        )

    split = paper_split(scaled, PARKTELE, test_points=PARKTELE_TEST_EVAL_POINTS)
    observed = (len(split.train_x), len(split.val_x), len(split.test_x))
    if observed != PARKTELE_SPLIT_SIZES:
        raise ValueError(f"Unexpected split sizes {observed}")

    unique_counts = [int(len(np.unique(raw[:, j]))) for j in range(PARKTELE_X_DIM)]
    metadata = {
        **split.metadata,
        "preprocess_version": PARKTELE_PREPROCESS_VERSION,
        "source": "UCI Machine Learning Repository archive 189 (Parkinsons Telemonitoring)",
        "source_member": PARKTELE_MEMBER,
        "archive_path": str(PARKTELE_ARCHIVE),
        "archive_sha256": sha256_file(PARKTELE_ARCHIVE),
        "raw_rows": int(len(frame)),
        "used_rows": int(len(raw)),
        "duplicate_rows_dropped": 0,
        "x_dim": PARKTELE_X_DIM,
        "feature_columns": list(PARKTELE_FEATURE_COLUMNS),
        "dropped_columns": sorted(PARKTELE_COLLINEAR_DROP),
        "collinearity_evidence": collinearity,
        "dropped_metadata_columns": [
            "subject#", "age", "sex", "test_time", "motor_UPDRS", "total_UPDRS",
        ],
        "normalization": "per-feature min-max over the complete official table",
        "feature_min": lower.tolist(),
        "feature_max": upper.tolist(),
        "observed_space_log_jacobian": float(-np.sum(np.log(span))),
        "observed_space_log_jacobian_applied": False,
        "correlation_condition_number": condition,
        "unique_values_per_column": unique_counts,
        "rows_per_unique_value": [float(len(raw) / c) for c in unique_counts],
        "subjects": int(frame["subject#"].nunique()),
        "repeated_measures_caveat": (
            "42 subjects contribute ~140 recordings each; the row-level shuffle-42 split "
            "places the same subject in train and test.  Every method shares the split, so "
            "the comparison is paired, but the absolute numbers are not subject-held-out."
        ),
        "train_sha256": sha256_array(split.train_x),
        "validation_sha256": sha256_array(split.val_x),
        "test_sha256": sha256_array(split.test_x),
        "selected_test_sha256": sha256_array(split.selected_test_x),
    }
    split.metadata.update(metadata)

    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        split_path,
        train_x=split.train_x,
        val_x=split.val_x,
        test_x=split.test_x,
        train_indices=split.train_indices,
        val_indices=split.val_indices,
        test_indices=split.test_indices,
        selected_test_x=split.selected_test_x,
        selected_test_indices=split.selected_test_indices,
    )
    fingerprint_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return split


def load_parktele(dataset: str = PARKTELE) -> PaperSplit:
    if dataset != PARKTELE:
        raise ValueError(f"This loader only supports {PARKTELE}, got {dataset}")
    _, split_path, fingerprint_path = _parktele_paths()
    if not split_path.exists() or not fingerprint_path.exists():
        return prepare_parktele()
    metadata = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    if metadata.get("preprocess_version") != PARKTELE_PREPROCESS_VERSION:
        raise ValueError(
            f"Cached split was built by {metadata.get('preprocess_version')!r}, "
            f"expected {PARKTELE_PREPROCESS_VERSION!r}; rerun with --force"
        )
    with np.load(split_path) as saved:
        arrays = {key: np.asarray(saved[key]) for key in saved.files}
    return PaperSplit(dataset=PARKTELE, metadata=metadata, **arrays)


CONDITIONAL_SPECS: dict[str, dict[str, Any]] = {
    "Pendigits10": {
        "url": "https://archive.ics.uci.edu/static/public/81/pen+based+recognition+of+handwritten+digits.zip",
        "archive": "Pendigits.zip",
        "archive_sha256": "1e02bea023613c2b11c9492f6f34caf975420455934f3527d270cee9a1f03b64",
        "members": ("pendigits.tra", "pendigits.tes"),
        "member_rows": (7494, 3498),
        "source": (
            "UCI Machine Learning Repository archive 81 "
            "(Pen-Based Recognition of Handwritten Digits)"
        ),
        "preprocess_version": "pendigits10_v1_dequant_zscore",
        "x_dim": 16,
        "labels": [str(d) for d in range(10)],
        "dedup": False,
        "dequantize": 1.0,
        "guard": "corr_condition",
        "expected_rows": 10992,
        "expected_class_counts": (1143, 1143, 1144, 1055, 1144, 1055, 1056, 1142, 1055, 1055),
        "expected_split_sizes": (8902, 990, 1100),
        "extra_metadata": {
            "not_the_odds_variant": (
                "distinct from experiments/application_odds/data/ODDS/Pendigits, which is a "
                "6870-row binary anomaly-detection subsample of the same source"
            ),
        },
        "dequantization_extra": {
            "reference": "uniform dequantization for quantized columns",
            "interpretation": (
                "reported log-likelihood is for the dequantized variable and lower-bounds "
                "the log-pmf of the recorded 0..100 integer lattice"
            ),
        },
    },
    "EEGEye": {
        "url": "https://archive.ics.uci.edu/static/public/264/eeg+eye+state.zip",
        "archive": "EEGEye.zip",
        "member": "EEG Eye State.arff",
        "preprocess_version": "eegeye_v1_dropglitch_zscore",
        "x_dim": 14,
        "labels": ["eyes_open", "eyes_closed"],
        "dedup": True,
        "dequantize": None,
        "guard": "sv_ratio",
    },
    "Vehicle": {
        "url": "https://archive.ics.uci.edu/static/public/149/statlog+vehicle+silhouettes.zip",
        "archive": "Vehicle.zip",
        "member": "xa*.dat",
        "preprocess_version": "vehicle_v1_dequant_zscore",
        "x_dim": 18,
        "labels": ["bus", "opel", "saab", "van"],
        "dedup": True,
        "dequantize": 1.0,
        "guard": "sv_ratio",
    },
}


CONDITIONAL = tuple(CONDITIONAL_SPECS)
UNCONDITIONAL = (PARKTELE,)
ALL_DATASETS = UNCONDITIONAL + CONDITIONAL


def latent_dimension(x_dim: int) -> int:
    return min(int(round(x_dim / 2)), int(x_dim) - 1)


def _conditional_paths(dataset: str) -> tuple[Path, Path, Path]:
    root = PROCESSED_ROOT / dataset
    return root, root / "processed_split.npz", root / "dataset_fingerprint.json"


def _read_pendigits10(archive: Path, spec: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    x_dim = int(spec["x_dim"])
    blocks = []
    with zipfile.ZipFile(archive) as zf:
        for member, expected in zip(spec["members"], spec["member_rows"]):
            with zf.open(member) as handle:
                rows = [r for r in csv.reader(io.TextIOWrapper(handle, "utf-8")) if r]
            if len(rows) != expected:
                raise ValueError(f"{member}: expected {expected} rows, got {len(rows)}")
            blocks.append(np.array([[int(v) for v in r] for r in rows], dtype=np.int64))
    table = np.vstack(blocks)
    if table.shape != (spec["expected_rows"], x_dim + 1):
        raise ValueError(f"Unexpected Pendigits table: {table.shape}")
    x_int, y = table[:, :x_dim], table[:, x_dim]
    if x_int.min() < 0 or x_int.max() > 100:
        raise ValueError(f"Coordinates outside [0, 100]: [{x_int.min()}, {x_int.max()}]")
    joint = np.hstack([x_int, y[:, None]])
    duplicates = int(len(joint) - len(np.unique(joint, axis=0)))
    if duplicates:
        raise ValueError(f"Unexpected duplicate (x, y) rows: {duplicates}")
    note = {
        "source_members": list(spec["members"]),
        "source_member_rows": list(spec["member_rows"]),
    }
    return x_int.astype(np.float64), y, note


def _read_eegeye(archive: Path, spec: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    with zipfile.ZipFile(archive) as zf:
        text = zf.read(spec["member"]).decode("utf-8", errors="ignore").splitlines()
    start = next(i for i, l in enumerate(text) if l.strip().lower().startswith("@data"))
    table = np.array([[float(v) for v in l.split(",")] for l in text[start + 1:] if l.strip()])
    x, y = table[:, :-1], table[:, -1].astype(np.int64)
    median = np.median(x, axis=0)
    mad = np.median(np.abs(x - median), axis=0) + 1e-12
    bad = (np.abs(x - median) / mad > 100.0).any(axis=1)
    note = {"rows_dropped_as_sensor_glitch": int(bad.sum()),
            "rule": "robust z = |x - median| / MAD > 100 in any channel"}
    return x[~bad], y[~bad], note


def _read_vehicle(archive: Path, spec: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    names = sorted(n for n in zipfile.ZipFile(archive).namelist() if n.endswith(".dat"))
    rows: list[list[str]] = []
    with zipfile.ZipFile(archive) as zf:
        for name in names:
            rows += [l.split() for l in zf.read(name).decode().splitlines() if l.strip()]
    x = np.array([[float(v) for v in r[:-1]] for r in rows], dtype=np.float64)
    raw_labels = [r[-1].strip().lower() for r in rows]
    order = {lab: i for i, lab in enumerate(spec["labels"])}
    y = np.array([order[l] for l in raw_labels], dtype=np.int64)
    return x, y, {"source_members": names}


READERS: dict[str, Callable[[Path, dict], tuple[np.ndarray, np.ndarray, dict]]] = {
    "Pendigits10": _read_pendigits10,
    "EEGEye": _read_eegeye,
    "Vehicle": _read_vehicle,
}


def prepare_conditional(dataset: str, force: bool = False):
    from bayesnde.data.uci_conditional import split_indices, standardize_from_train

    spec = CONDITIONAL_SPECS[dataset]
    root, split_path, fingerprint_path = _conditional_paths(dataset)
    if split_path.exists() and fingerprint_path.exists() and not force:
        return load_conditional(dataset)

    archive = download(spec["url"], RAW_ARCHIVES / spec["archive"], spec.get("archive_sha256"))
    x, y, note = READERS[dataset](archive, spec)
    x = np.asarray(x, dtype=np.float64)
    n_labels = len(spec["labels"])
    if x.shape[1] != int(spec["x_dim"]):
        raise ValueError(f"{dataset}: expected {spec['x_dim']} features, got {x.shape[1]}")
    if not np.isfinite(x).all():
        raise ValueError(f"{dataset}: non-finite feature values")

    duplicates = 0
    if spec["dedup"]:
        _, first = np.unique(x, axis=0, return_index=True)
        keep = np.sort(first)
        duplicates = int(len(x) - len(keep))
        x, y = x[keep], y[keep]
    if "expected_rows" in spec and len(x) != int(spec["expected_rows"]):
        raise ValueError(f"{dataset}: expected {spec['expected_rows']} rows, got {len(x)}")
    counts = np.bincount(y, minlength=n_labels)
    if "expected_class_counts" in spec:
        observed = tuple(int(c) for c in counts)
        if observed != tuple(spec["expected_class_counts"]):
            raise ValueError(f"{dataset}: unexpected class counts {observed}")
    if len(counts) != n_labels or counts.min() < 20:
        raise ValueError(f"{dataset}: label counts {counts.tolist()} unusable for per-label experts")

    lattice_unique = [int(len(np.unique(x[:, j]))) for j in range(x.shape[1])]
    dequantization = None
    if spec["dequantize"] is not None:
        delta = float(spec["dequantize"])
        rng = np.random.default_rng(DEQUANT_SEED)
        x = x + rng.uniform(0.0, delta, x.shape)
        dequantization = {
            "rule": "x_j + U(0, delta) on the recorded integer lattice",
            "delta": delta,
            "seed": DEQUANT_SEED,
            "unique_values_per_column_before": lattice_unique,
            "discrete_lower_bound_offset": 0.0,
            **spec.get("dequantization_extra", {}),
        }

    scale = x.std(axis=0)
    if np.any(scale == 0):
        raise ValueError(f"{dataset}: a feature is constant")
    standardized = (x - x.mean(axis=0)) / scale
    condition = float(np.linalg.cond(np.corrcoef(standardized, rowvar=False)))
    singular = np.linalg.svd(standardized, compute_uv=False)
    sv_ratio = float(singular[-1] / singular[0])
    if spec["guard"] == "corr_condition":
        if condition >= MAX_CORR_CONDITION:
            raise ValueError(
                f"{dataset}: correlation condition number {condition:.3g} >= {MAX_CORR_CONDITION}"
            )
    elif sv_ratio <= MIN_SINGULAR_VALUE_RATIO:
        raise ValueError(
            f"{dataset}: smallest singular-value ratio {sv_ratio:.3e} <= "
            f"{MIN_SINGULAR_VALUE_RATIO:g}: the data is rank deficient, so a continuous "
            "density on R^p does not exist"
        )

    train_idx, val_idx, test_idx = split_indices(
        y, seed=DATA_SEED, test_fraction=0.1, val_fraction_of_remaining=0.1
    )
    observed = (len(train_idx), len(val_idx), len(test_idx))
    if "expected_split_sizes" in spec and observed != tuple(spec["expected_split_sizes"]):
        raise ValueError(f"{dataset}: unexpected split sizes {observed}")

    train_x, val_x, test_x, mean, std, jacobian = standardize_from_train(
        x[train_idx], x[val_idx], x[test_idx]
    )
    if abs(jacobian + float(np.sum(np.log(std)))) > 1.0e-12:
        raise ValueError(f"{dataset}: jacobian_correction != -sum(log(std))")

    metadata = {
        "dataset": dataset,
        "repeat": 0,
        "seed": DATA_SEED,
        "preprocess_version": spec["preprocess_version"],
        "source": spec.get("source", spec["url"]),
        "archive_path": str(archive),
        "archive_sha256": sha256_file(archive),
        "x_dim": int(x.shape[1]),
        "z_dim_for_this_round": latent_dimension(int(x.shape[1])),
        "label_names": list(spec["labels"]),
        "label_counts": counts.tolist(),
        "feature_columns": list(spec.get("feature_columns", [])) or None,
        "raw_rows": int(len(x) + duplicates),
        "used_rows": int(len(x)),
        "duplicate_rows_dropped": duplicates,
        "reader_note": note,
        "dequantization": dequantization,
        "unique_values_per_column": [int(len(np.unique(x[:, j]))) for j in range(x.shape[1])],
        "rows_per_unique_value": [float(len(x) / c) for c in lattice_unique],
        "degeneracy_guard": spec["guard"],
        "correlation_condition_number": condition,
        "smallest_singular_value_ratio": sv_ratio,
        "standardization_mean": mean.tolist(),
        "standardization_std": std.tolist(),
        "jacobian_correction": float(jacobian),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "split_policy": SPLIT_POLICY,
        "train_sha256": sha256_array(train_x),
        "validation_sha256": sha256_array(val_x),
        "test_sha256": sha256_array(test_x),
        **spec.get("extra_metadata", {}),
    }

    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        split_path,
        train_x=train_x, train_y=y[train_idx],
        val_x=val_x, val_y=y[val_idx],
        test_x=test_x, test_y=y[test_idx],
    )
    fingerprint_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return _build_conditional(dataset, metadata)


def _build_conditional(dataset: str, metadata: dict):
    from bayesnde.data.uci_conditional import ProcessedSplit

    _, split_path, _ = _conditional_paths(dataset)
    with np.load(split_path) as saved:
        arrays = {k: np.asarray(saved[k]) for k in saved.files}
    return ProcessedSplit(dataset=dataset, repeat=0, metadata=metadata, **arrays)


def load_conditional(dataset: str):
    if dataset not in CONDITIONAL_SPECS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {CONDITIONAL}")
    _, split_path, fingerprint_path = _conditional_paths(dataset)
    if not split_path.exists() or not fingerprint_path.exists():
        return prepare_conditional(dataset)
    metadata = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    expected = CONDITIONAL_SPECS[dataset]["preprocess_version"]
    if metadata.get("preprocess_version") != expected:
        raise ValueError(
            f"{dataset}: cached split built by {metadata.get('preprocess_version')!r}, "
            f"expected {expected!r}; rerun with --force"
        )
    return _build_conditional(dataset, metadata)


def is_conditional(dataset: str) -> bool:
    return dataset in CONDITIONAL_SPECS


def x_dim(dataset: str) -> int:
    if dataset == PARKTELE:
        return PARKTELE_X_DIM
    return int(CONDITIONAL_SPECS[dataset]["x_dim"])


def prepare(dataset: str, force: bool = False):
    if dataset == PARKTELE:
        return prepare_parktele(force=force)
    return prepare_conditional(dataset, force=force)


def load_split(dataset: str = PARKTELE):
    if dataset == PARKTELE:
        return load_parktele()
    return load_conditional(dataset)


def _describe(dataset: str) -> dict[str, Any]:
    split = prepare(dataset)
    paths = _parktele_paths() if dataset == PARKTELE else _conditional_paths(dataset)
    _, split_path, _ = paths
    common = {
        "dataset": dataset,
        "x_dim": x_dim(dataset),
        "train": int(len(split.train_x)),
        "validation": int(len(split.val_x)),
        "test": int(len(split.test_x)),
        "correlation_condition_number": split.metadata.get("correlation_condition_number"),
        "split_path": str(split_path),
    }
    if dataset == PARKTELE:
        common["selected_test"] = int(len(split.selected_test_x))
        return common
    common.update({
        "y_dim": split.y_dim,
        "z_dim_round2": latent_dimension(x_dim(dataset)),
        "jacobian_correction": split.jacobian_correction,
        "smallest_singular_value_ratio": split.metadata.get("smallest_singular_value_ratio"),
        "label_counts": split.metadata.get("label_counts"),
    })
    return common


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the UCI expansion splits")
    parser.add_argument("--dataset", choices=ALL_DATASETS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true", help="re-derive from the raw archive")
    args = parser.parse_args()
    targets = ALL_DATASETS if args.all else ((args.dataset,) if args.dataset else ())
    if not targets:
        parser.error("pass --dataset X or --all")
    out = []
    for name in targets:
        prepare(name, force=args.force)
        out.append(_describe(name))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
