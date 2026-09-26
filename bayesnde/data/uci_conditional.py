"""Data preparation for labeled UCI conditional density benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import tarfile
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OneHotEncoder


ZENODO_RECORD_URL = "https://zenodo.org/api/records/4559067/files"

DATASET_URLS: dict[str, str] = {
    "HEPMASS": f"{ZENODO_RECORD_URL}/HEPMASS.tar.gz/content",
    "AReM": "https://archive.ics.uci.edu/static/public/366/activity+recognition+system+based+on+multisensor+data+fusion+arem.zip",
    "BANK": "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
}

ARCHIVE_NAMES: dict[str, str] = {
    "HEPMASS": "HEPMASS.tar.gz",
    "AReM": "AReM.zip",
    "BANK": "BANK.zip",
}

PREPROCESS_VERSION = "conditional_uci_v2_bank_ordinal"


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=json_default)


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass
class RawConditionalDataset:
    name: str
    x_frame: pd.DataFrame
    y: np.ndarray
    label_names: list[str]
    categorical_columns: list[str] = field(default_factory=list)
    source: str = ""

    @property
    def numeric_columns(self) -> list[str]:
        return [col for col in self.x_frame.columns if col not in set(self.categorical_columns)]


@dataclass
class ProcessedSplit:
    dataset: str
    repeat: int
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    metadata: dict[str, Any]

    @property
    def x_dim(self) -> int:
        return int(self.train_x.shape[1])

    @property
    def y_dim(self) -> int:
        return int(len(self.metadata["label_names"]))

    @property
    def jacobian_correction(self) -> float:
        return float(self.metadata["jacobian_correction"])


def parse_dataset_list(text: str | Sequence[str]) -> list[str]:
    if isinstance(text, str):
        names = [part.strip() for part in text.split(",") if part.strip()]
    else:
        names = [str(part).strip() for part in text if str(part).strip()]
    if not names:
        raise ValueError("At least one dataset name is required.")
    normalized = []
    aliases = {"AREM": "AReM", "HEPMASS": "HEPMASS", "BANK": "BANK"}
    for name in names:
        key = name.upper()
        if key not in aliases:
            raise ValueError(f"Unknown dataset {name!r}; expected HEPMASS, AReM, or BANK.")
        normalized.append(aliases[key])
    return normalized


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, path: Path, *, force: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return path
    tmp_path = path.with_suffix(path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()
    with urllib.request.urlopen(url, timeout=120) as response, open(tmp_path, "wb") as handle:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total and total.isdigit() else None
        copied = 0
        next_report = 25 * 1024 * 1024
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
            copied += len(chunk)
            if copied >= next_report:
                if total_bytes:
                    print(f"downloaded {path.name}: {copied / 1e6:.1f}/{total_bytes / 1e6:.1f} MB", flush=True)
                else:
                    print(f"downloaded {path.name}: {copied / 1e6:.1f} MB", flush=True)
                next_report += 25 * 1024 * 1024
    tmp_path.replace(path)
    return path


def _safe_member_path(target_dir: Path, member_name: str) -> Path:
    target = (target_dir / member_name).resolve()
    root = target_dir.resolve()
    if root != target and root not in target.parents:
        raise RuntimeError(f"Unsafe archive member path: {member_name}")
    return target


def safe_extract_tar(path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            _safe_member_path(target_dir, member.name)
        archive.extractall(target_dir)


def safe_extract_zip(path: Path | io.BytesIO, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            _safe_member_path(target_dir, member)
        archive.extractall(target_dir)


def ensure_raw_dataset(data_root: Path, dataset: str, *, force_download: bool = False) -> Path:
    dataset = parse_dataset_list([dataset])[0]
    archive_dir = data_root / "raw_archives"
    raw_dir = data_root / "raw" / dataset
    archive_path = archive_dir / ARCHIVE_NAMES[dataset]
    download_file(DATASET_URLS[dataset], archive_path, force=force_download)

    sentinel = raw_dir / ".extracted.json"
    if sentinel.exists() and not force_download:
        return raw_dir
    if raw_dir.exists() and force_download:
        for child in raw_dir.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
            else:
                import shutil

                shutil.rmtree(child)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if dataset == "HEPMASS":
        safe_extract_tar(archive_path, raw_dir)
    elif dataset == "AReM":
        safe_extract_zip(archive_path, raw_dir)
    elif dataset == "BANK":
        with zipfile.ZipFile(archive_path) as outer:
            bank_zip = outer.read("bank.zip")
        safe_extract_zip(io.BytesIO(bank_zip), raw_dir)
    else:
        raise ValueError(dataset)

    save_json(
        sentinel,
        {
            "dataset": dataset,
            "archive": str(archive_path),
            "source_url": DATASET_URLS[dataset],
            "archive_size": archive_path.stat().st_size,
            "archive_sha256": sha256_file(archive_path),
        },
    )
    return raw_dir


def _find_file(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find {name} under {root}")
    return matches[0]


def load_hepmass_raw(data_root: Path, *, force_download: bool = False, max_rows: int | None = None) -> RawConditionalDataset:
    raw_dir = ensure_raw_dataset(data_root, "HEPMASS", force_download=force_download)
    train_path = _find_file(raw_dir, "1000_train.csv")
    test_path = _find_file(raw_dir, "1000_test.csv")
    read_kwargs: dict[str, Any] = {"comment": "#", "header": None}
    if max_rows is not None and int(max_rows) > 0:
        read_kwargs["nrows"] = max(1000, int(max_rows) * 4)
    train_df = pd.read_csv(train_path, **read_kwargs)
    test_df = pd.read_csv(test_path, **read_kwargs)

    label_col = train_df.columns[0]
    train_y = pd.to_numeric(train_df[label_col], errors="coerce").astype("Int64")
    test_y = pd.to_numeric(test_df[test_df.columns[0]], errors="coerce").astype("Int64")
    train_x = train_df.drop(columns=[label_col])
    test_x = test_df.drop(columns=[test_df.columns[0]])
    if test_x.shape[1] == train_x.shape[1] + 1:
        test_x = test_x.iloc[:, :-1]
    elif train_x.shape[1] == test_x.shape[1] + 1:
        train_x = train_x.iloc[:, :-1]
    if train_x.shape[1] != test_x.shape[1]:
        raise ValueError(f"HEPMASS train/test feature mismatch: {train_x.shape} vs {test_x.shape}")
    feature_names = [f"hepmass_{idx:02d}" for idx in range(train_x.shape[1])]
    train_x.columns = feature_names
    test_x.columns = feature_names
    frame = pd.concat([train_x, test_x], axis=0, ignore_index=True)
    y_raw = pd.concat([train_y, test_y], axis=0, ignore_index=True).to_numpy(dtype=np.int64)
    finite_mask = np.isfinite(frame.to_numpy(dtype=np.float64)).all(axis=1) & np.isfinite(y_raw)
    frame = frame.loc[finite_mask].reset_index(drop=True).astype(np.float32)
    y_raw = y_raw[finite_mask].astype(np.int64)
    label_values = sorted(int(value) for value in np.unique(y_raw))
    label_map = {value: idx for idx, value in enumerate(label_values)}
    y = np.asarray([label_map[int(value)] for value in y_raw], dtype=np.int64)
    label_names = [str(value) for value in label_values]
    return RawConditionalDataset(
        name="HEPMASS",
        x_frame=frame,
        y=y,
        label_names=label_names,
        categorical_columns=[],
        source=str(raw_dir),
    )


def load_arem_raw(data_root: Path, *, force_download: bool = False) -> RawConditionalDataset:
    raw_dir = ensure_raw_dataset(data_root, "AReM", force_download=force_download)
    activity_dirs = [path for path in sorted(raw_dir.iterdir()) if path.is_dir() and not path.name.startswith("__")]
    rows: list[pd.DataFrame] = []
    labels: list[int] = []
    label_names = [path.name for path in activity_dirs]
    label_map = {name: idx for idx, name in enumerate(label_names)}
    feature_names = ["avg_rss12", "var_rss12", "avg_rss13", "var_rss13", "avg_rss23", "var_rss23"]
    for activity_dir in activity_dirs:
        for csv_path in sorted(activity_dir.glob("*.csv")):
            frame = pd.read_csv(csv_path, comment="#", header=None, on_bad_lines="skip")
            frame = frame.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
            if frame.shape[1] < 7:
                continue
            features = frame.iloc[:, 1:7].copy()
            features.columns = feature_names
            rows.append(features.astype(np.float32))
            labels.extend([label_map[activity_dir.name]] * len(features))
    if not rows:
        raise RuntimeError(f"No AReM CSV rows parsed under {raw_dir}")
    x_frame = pd.concat(rows, axis=0, ignore_index=True)
    y = np.asarray(labels, dtype=np.int64)
    return RawConditionalDataset(
        name="AReM",
        x_frame=x_frame,
        y=y,
        label_names=label_names,
        categorical_columns=[],
        source=str(raw_dir),
    )


def load_bank_raw(data_root: Path, *, force_download: bool = False) -> RawConditionalDataset:
    raw_dir = ensure_raw_dataset(data_root, "BANK", force_download=force_download)
    bank_path = _find_file(raw_dir, "bank-full.csv")
    frame = pd.read_csv(bank_path, sep=";")
    if "y" not in frame.columns:
        raise ValueError(f"BANK file {bank_path} does not contain target column 'y'.")
    y_text = frame["y"].astype(str).to_numpy()
    label_names = sorted(np.unique(y_text).tolist())
    label_map = {name: idx for idx, name in enumerate(label_names)}
    y = np.asarray([label_map[value] for value in y_text], dtype=np.int64)
    x_frame = frame.drop(columns=["y"]).copy()
    categorical_columns = [col for col in x_frame.columns if not pd.api.types.is_numeric_dtype(x_frame[col])]
    return RawConditionalDataset(
        name="BANK",
        x_frame=x_frame,
        y=y,
        label_names=label_names,
        categorical_columns=categorical_columns,
        source=str(bank_path),
    )


def load_raw_dataset(data_root: Path, dataset: str, *, force_download: bool = False, max_rows: int | None = None) -> RawConditionalDataset:
    dataset = parse_dataset_list([dataset])[0]
    if dataset == "HEPMASS":
        return load_hepmass_raw(data_root, force_download=force_download, max_rows=max_rows)
    if dataset == "AReM":
        return load_arem_raw(data_root, force_download=force_download)
    if dataset == "BANK":
        return load_bank_raw(data_root, force_download=force_download)
    raise ValueError(dataset)


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)


def encode_split_frames(
    raw: RawConditionalDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, dict[str, Any]]:
    train_frame = raw.x_frame.iloc[train_idx].reset_index(drop=True)
    val_frame = raw.x_frame.iloc[val_idx].reset_index(drop=True)
    test_frame = raw.x_frame.iloc[test_idx].reset_index(drop=True)
    categorical = list(raw.categorical_columns)
    numeric = [col for col in raw.x_frame.columns if col not in set(categorical)]

    parts_train: list[np.ndarray] = []
    parts_val: list[np.ndarray] = []
    parts_test: list[np.ndarray] = []
    feature_names: list[str] = []
    discrete_mask_parts: list[np.ndarray] = []
    metadata: dict[str, Any] = {"categorical_columns": categorical, "numeric_columns": numeric}

    if numeric:
        train_numeric = train_frame[numeric].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        val_numeric = val_frame[numeric].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        test_numeric = test_frame[numeric].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        parts_train.append(train_numeric)
        parts_val.append(val_numeric)
        parts_test.append(test_numeric)
        feature_names.extend(numeric)
        discrete_mask_parts.append(np.zeros(len(numeric), dtype=bool))

    if categorical and raw.name == "BANK":
        train_cat_parts: list[np.ndarray] = []
        val_cat_parts: list[np.ndarray] = []
        test_cat_parts: list[np.ndarray] = []
        ordinal_categories: dict[str, list[str]] = {}
        unknown_codes: dict[str, int] = {}
        for col in categorical:
            train_values = train_frame[col].astype(str)
            categories = sorted(train_values.dropna().unique().tolist())
            mapping = {value: idx for idx, value in enumerate(categories)}
            unknown_code = len(categories)

            def encode_column(frame: pd.DataFrame) -> np.ndarray:
                return frame[col].astype(str).map(mapping).fillna(unknown_code).to_numpy(dtype=np.float32).reshape(-1, 1)

            train_cat_parts.append(encode_column(train_frame))
            val_cat_parts.append(encode_column(val_frame))
            test_cat_parts.append(encode_column(test_frame))
            ordinal_categories[col] = categories
            unknown_codes[col] = int(unknown_code)
        parts_train.append(np.concatenate(train_cat_parts, axis=1))
        parts_val.append(np.concatenate(val_cat_parts, axis=1))
        parts_test.append(np.concatenate(test_cat_parts, axis=1))
        feature_names.extend(categorical)
        discrete_mask_parts.append(np.ones(len(categorical), dtype=bool))
        metadata["categorical_encoding"] = "ordinal_train_codes"
        metadata["ordinal_categories"] = ordinal_categories
        metadata["unknown_category_codes"] = unknown_codes
    elif categorical:
        encoder = _make_one_hot_encoder()
        train_cat = train_frame[categorical].astype(str)
        val_cat = val_frame[categorical].astype(str)
        test_cat = test_frame[categorical].astype(str)
        encoder.fit(train_cat)
        parts_train.append(encoder.transform(train_cat).astype(np.float32))
        parts_val.append(encoder.transform(val_cat).astype(np.float32))
        parts_test.append(encoder.transform(test_cat).astype(np.float32))
        cat_feature_names = encoder.get_feature_names_out(categorical).tolist()
        feature_names.extend(cat_feature_names)
        discrete_mask_parts.append(np.ones(len(cat_feature_names), dtype=bool))
        metadata["categorical_encoding"] = "one_hot_train_categories"
        metadata["one_hot_categories"] = [cats.tolist() for cats in encoder.categories_]

    train_x = np.concatenate(parts_train, axis=1) if len(parts_train) > 1 else parts_train[0]
    val_x = np.concatenate(parts_val, axis=1) if len(parts_val) > 1 else parts_val[0]
    test_x = np.concatenate(parts_test, axis=1) if len(parts_test) > 1 else parts_test[0]
    discrete_mask = np.concatenate(discrete_mask_parts, axis=0) if discrete_mask_parts else np.zeros(train_x.shape[1], dtype=bool)
    return train_x, val_x, test_x, feature_names, discrete_mask, metadata


def drop_hepmass_repeated_features(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    feature_names: list[str],
    discrete_mask: np.ndarray,
    *,
    max_count_threshold: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, list[str]]:
    features_to_remove: list[int] = []
    for idx in range(train_x.shape[1]):
        values, _ = np.unique(train_x[:, idx], return_counts=True)
        if values.size <= int(max_count_threshold):
            features_to_remove.append(idx)
    if not features_to_remove:
        return train_x, val_x, test_x, feature_names, discrete_mask, []
    keep = np.asarray([idx for idx in range(train_x.shape[1]) if idx not in set(features_to_remove)], dtype=np.int64)
    removed_names = [feature_names[idx] for idx in features_to_remove]
    return train_x[:, keep], val_x[:, keep], test_x[:, keep], [feature_names[idx] for idx in keep], discrete_mask[keep], removed_names


def dequantize(x: np.ndarray, discrete_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64).copy()
    if np.any(discrete_mask):
        out[:, discrete_mask] += rng.uniform(-0.5, 0.5, size=(out.shape[0], int(discrete_mask.sum())))
    return out.astype(np.float32)


def standardize_from_train(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    *,
    min_std: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    mean = train_x.mean(axis=0, dtype=np.float64)
    std = train_x.std(axis=0, dtype=np.float64)
    std = np.maximum(std, float(min_std))
    train_std = ((train_x - mean) / std).astype(np.float32)
    val_std = ((val_x - mean) / std).astype(np.float32)
    test_std = ((test_x - mean) / std).astype(np.float32)
    jacobian_correction = -float(np.sum(np.log(std)))
    return train_std, val_std, test_std, mean.astype(np.float64), std.astype(np.float64), jacobian_correction


def maybe_subsample(raw: RawConditionalDataset, *, max_rows: int | None, seed: int) -> RawConditionalDataset:
    if max_rows is None or int(max_rows) <= 0 or len(raw.y) <= int(max_rows):
        return raw
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=int(max_rows), random_state=int(seed))
    idx, _ = next(splitter.split(np.zeros(len(raw.y)), raw.y))
    idx = np.sort(idx)
    return RawConditionalDataset(
        name=raw.name,
        x_frame=raw.x_frame.iloc[idx].reset_index(drop=True),
        y=raw.y[idx],
        label_names=list(raw.label_names),
        categorical_columns=list(raw.categorical_columns),
        source=raw.source,
    )


def split_indices(
    y: np.ndarray,
    *,
    seed: int,
    test_fraction: float = 0.1,
    val_fraction_of_remaining: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_idx = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(
        all_idx,
        test_size=float(test_fraction),
        random_state=int(seed),
        shuffle=True,
        stratify=y,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=float(val_fraction_of_remaining),
        random_state=int(seed) + 1009,
        shuffle=True,
        stratify=y[train_val_idx],
    )
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def _save_split(path: Path, x_std: np.ndarray, x_dequant: np.ndarray, y: np.ndarray, indices: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x_std=np.asarray(x_std, dtype=np.float32),
        x_dequant=np.asarray(x_dequant, dtype=np.float32),
        y=np.asarray(y, dtype=np.int64),
        indices=np.asarray(indices, dtype=np.int64),
    )


def prepare_processed_dataset(
    data_root: Path,
    dataset: str,
    *,
    n_repeats: int = 3,
    seed: int = 42,
    max_rows: int | None = None,
    force_download: bool = False,
    force_processed: bool = False,
    test_fraction: float = 0.1,
    val_fraction_of_remaining: float = 0.1,
) -> list[Path]:
    dataset = parse_dataset_list([dataset])[0]
    raw = maybe_subsample(
        load_raw_dataset(data_root, dataset, force_download=force_download, max_rows=max_rows),
        max_rows=max_rows,
        seed=seed,
    )
    out_root = data_root / "processed" / "conditional_uci" / dataset
    out_paths: list[Path] = []
    save_json(
        out_root / "metadata.json",
        {
            "dataset": raw.name,
            "source": raw.source,
            "n_rows": int(len(raw.y)),
            "raw_feature_count": int(raw.x_frame.shape[1]),
            "label_names": raw.label_names,
            "label_counts": {raw.label_names[int(k)]: int(v) for k, v in zip(*np.unique(raw.y, return_counts=True))},
            "categorical_columns": raw.categorical_columns,
            "max_rows": max_rows,
            "preprocess_version": PREPROCESS_VERSION,
        },
    )
    for repeat in range(int(n_repeats)):
        repeat_seed = int(seed) + repeat
        repeat_dir = out_root / f"repeat_{repeat}"
        preprocess_path = repeat_dir / "preprocess.json"
        if preprocess_path.exists() and not force_processed:
            existing = load_json(preprocess_path)
            existing_max_rows = existing.get("max_rows")
            requested_max_rows = None if max_rows is None else int(max_rows)
            if existing_max_rows == requested_max_rows and int(existing.get("x_dim", 0)) > 0 and existing.get("preprocess_version") == PREPROCESS_VERSION:
                out_paths.append(repeat_dir)
                continue

        train_idx, val_idx, test_idx = split_indices(
            raw.y,
            seed=repeat_seed,
            test_fraction=test_fraction,
            val_fraction_of_remaining=val_fraction_of_remaining,
        )
        train_x, val_x, test_x, feature_names, discrete_mask, encode_meta = encode_split_frames(raw, train_idx, val_idx, test_idx)
        removed_features: list[str] = []
        if dataset == "HEPMASS":
            train_x, val_x, test_x, feature_names, discrete_mask, removed_features = drop_hepmass_repeated_features(
                train_x,
                val_x,
                test_x,
                feature_names,
                discrete_mask,
                max_count_threshold=5,
            )
        rng = np.random.default_rng(repeat_seed)
        train_deq = dequantize(train_x, discrete_mask, rng)
        val_deq = dequantize(val_x, discrete_mask, rng)
        test_deq = dequantize(test_x, discrete_mask, rng)
        train_std, val_std, test_std, mean, std, jacobian_correction = standardize_from_train(train_deq, val_deq, test_deq)
        train_y = raw.y[train_idx]
        val_y = raw.y[val_idx]
        test_y = raw.y[test_idx]
        _save_split(repeat_dir / "train.npz", train_std, train_deq, train_y, train_idx)
        _save_split(repeat_dir / "val.npz", val_std, val_deq, val_y, val_idx)
        _save_split(repeat_dir / "test.npz", test_std, test_deq, test_y, test_idx)
        metadata = {
            "dataset": dataset,
            "preprocess_version": PREPROCESS_VERSION,
            "repeat": repeat,
            "seed": repeat_seed,
            "split_policy": {
                "test_fraction": test_fraction,
                "val_fraction_of_remaining": val_fraction_of_remaining,
                "train_fraction_effective": (1.0 - test_fraction) * (1.0 - val_fraction_of_remaining),
            },
            "x_dim": int(train_std.shape[1]),
            "label_names": raw.label_names,
            "feature_names": feature_names,
            "discrete_indices": np.flatnonzero(discrete_mask).astype(int).tolist(),
            "continuous_indices": np.flatnonzero(~discrete_mask).astype(int).tolist(),
            "standardization_mean": mean,
            "standardization_std": std,
            "jacobian_correction": jacobian_correction,
            "removed_features": removed_features,
            "encoding": encode_meta,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "max_rows": None if max_rows is None else int(max_rows),
        }
        save_json(preprocess_path, metadata)
        out_paths.append(repeat_dir)
    return out_paths


def load_processed_split(data_root: Path, dataset: str, repeat: int) -> ProcessedSplit:
    dataset = parse_dataset_list([dataset])[0]
    repeat_dir = data_root / "processed" / "conditional_uci" / dataset / f"repeat_{repeat}"
    if not repeat_dir.exists():
        raise FileNotFoundError(f"Processed split does not exist: {repeat_dir}")
    train = np.load(repeat_dir / "train.npz")
    val = np.load(repeat_dir / "val.npz")
    test = np.load(repeat_dir / "test.npz")
    metadata = load_json(repeat_dir / "preprocess.json")
    return ProcessedSplit(
        dataset=dataset,
        repeat=int(repeat),
        train_x=train["x_std"].astype(np.float32),
        train_y=train["y"].astype(np.int64),
        val_x=val["x_std"].astype(np.float32),
        val_y=val["y"].astype(np.int64),
        test_x=test["x_std"].astype(np.float32),
        test_y=test["y"].astype(np.int64),
        metadata=metadata,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare labeled UCI conditional density datasets.")
    parser.add_argument("command", choices=["prepare", "download-only"])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--datasets", default="HEPMASS,AReM,BANK")
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-processed", action="store_true")
    args = parser.parse_args(argv)

    data_root = Path(args.data_root)
    datasets = parse_dataset_list(args.datasets)
    if args.command == "download-only":
        for dataset in datasets:
            raw_dir = ensure_raw_dataset(data_root, dataset, force_download=bool(args.force_download))
            print(f"{dataset}: raw data ready at {raw_dir}")
        return
    for dataset in datasets:
        paths = prepare_processed_dataset(
            data_root,
            dataset,
            n_repeats=int(args.n_repeats),
            seed=int(args.seed),
            max_rows=args.max_rows,
            force_download=bool(args.force_download),
            force_processed=bool(args.force_processed),
        )
        print(f"{dataset}: processed repeats ready: {', '.join(str(path) for path in paths)}")


if __name__ == "__main__":
    main()
