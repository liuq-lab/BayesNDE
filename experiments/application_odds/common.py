"""Shared, dependency-light contracts for the ODDS anomaly experiment."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.stats import rankdata


APPLICATION_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = APPLICATION_ROOT / "config.json"
DATA_ROOT = APPLICATION_ROOT / "data"
RESULTS_ROOT = APPLICATION_ROOT / "results"


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


CONFIG = load_config()
DATASET_ALIASES = {
    "shuttle": "shuttle",
}


def canonical_dataset(name: str) -> str:
    if name in CONFIG["datasets"]:
        return name
    key = name.strip().lower().replace(" ", "")
    try:
        return DATASET_ALIASES[key]
    except KeyError as exc:
        allowed = ", ".join(CONFIG["datasets"])
        raise ValueError(f"Unknown dataset {name!r}; choose one of: {allowed}") from exc


def canonical_method(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {item["id"]: item["id"] for item in CONFIG["methods"]}
    aliases.update({"bgmbs": "bgm_bs", "bayesnde": "bgm_bs"})
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"Unknown method {name!r}") from exc


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_digest(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.view(np.uint8))
    return digest.hexdigest()


@dataclass(frozen=True)
class OddSplit:
    name: str
    x_train: np.ndarray
    x_validation: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    source_path: Path

    @property
    def dimension(self) -> int:
        return int(self.x_train.shape[1])


def dataset_path(name: str, data_root: Path | str = DATA_ROOT) -> Path:
    canonical = canonical_dataset(name)
    return Path(data_root) / CONFIG["datasets"][canonical]["relative_path"]


def split_arrays(data: np.ndarray, labels: np.ndarray, name: str = "dataset") -> OddSplit:
    x = np.asarray(data)
    y = np.asarray(labels).reshape(-1)
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError(f"Invalid data/label shapes: {x.shape}, {y.shape}")
    n_test = int(0.1 * len(x))
    if n_test == 0:
        raise ValueError("Dataset is too small for the canonical split")
    x_remaining = x[:-n_test]
    n_validation = int(0.1 * len(x_remaining))
    if n_validation == 0:
        raise ValueError("Dataset is too small for the canonical split")
    return OddSplit(
        name=name,
        x_train=x_remaining[:-n_validation],
        x_validation=x_remaining[-n_validation:],
        x_test=x[-n_test:],
        y_test=y[-n_test:],
        source_path=Path("<memory>"),
    )


def load_odds(name: str, data_root: Path | str = DATA_ROOT, verify: bool = True) -> OddSplit:
    canonical = canonical_dataset(name)
    path = dataset_path(canonical, data_root)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run prepare_data.py first")
    spec = CONFIG["datasets"][canonical]
    if verify and file_digest(path, "md5") != spec["file_md5"]:
        raise ValueError(f"Checksum mismatch for {path}")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"arr_0", "arr_1"}:
            raise ValueError(f"Unexpected keys in {path}: {archive.files}")
        split = split_arrays(archive["arr_0"], archive["arr_1"], canonical)
    split = OddSplit(
        name=canonical, x_train=split.x_train,
        x_validation=split.x_validation, x_test=split.x_test,
        y_test=split.y_test.astype(np.int64, copy=False), source_path=path,
    )
    if verify:
        observed = {
            "n_train": len(split.x_train), "n_validation": len(split.x_validation),
            "n_test": len(split.x_test), "dimension": split.dimension,
            "n_test_anomalies": int(split.y_test.sum()),
        }
        bad = {key: (observed[key], spec[key]) for key in observed if observed[key] != spec[key]}
        if bad or not np.isin(split.y_test, [0, 1]).all() or not np.isfinite(split.x_test).all():
            raise ValueError(f"Canonical dataset validation failed for {canonical}: {bad}")
    return split


def processed_path(name: str, data_root: Path | str = DATA_ROOT) -> Path:
    return Path(data_root) / "processed" / f"{canonical_dataset(name)}.npz"


def load_processed(name: str, data_root: Path | str = DATA_ROOT, verify: bool = True) -> OddSplit:
    canonical = canonical_dataset(name)
    path = processed_path(canonical, data_root)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run prepare_data.py first")
    with np.load(path, allow_pickle=False) as archive:
        expected_keys = {"train_x", "train_y", "val_x", "val_y", "test_x", "test_y"}
        if set(archive.files) != expected_keys:
            raise ValueError(f"Unexpected keys in {path}: {archive.files}")
        split = OddSplit(
            name=canonical,
            x_train=archive["train_x"], x_validation=archive["val_x"],
            x_test=archive["test_x"], y_test=archive["test_y"], source_path=path,
        )
    if verify:
        spec = CONFIG["datasets"][canonical]
        observed = (len(split.x_train), len(split.x_validation), len(split.x_test), split.dimension, int(split.y_test.sum()))
        expected = (spec["n_train"], spec["n_validation"], spec["n_test"], spec["dimension"], spec["n_test_anomalies"])
        if observed != expected:
            raise ValueError(f"Processed dataset validation failed for {canonical}: {observed} != {expected}")
    return split


load_dataset = load_processed


def precision_at_k(scores: Iterable[float], labels: Iterable[int]) -> float:
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    label = np.asarray(labels).reshape(-1)
    if score.shape != label.shape or score.size == 0:
        raise ValueError(f"Score/label shapes must match and be nonempty: {score.shape}, {label.shape}")
    if not np.isfinite(score).all() or not np.isin(label, [0, 1]).all():
        raise ValueError("Scores must be finite and labels must be binary")
    k = int(np.sum(label))
    if k <= 0:
        raise ValueError("precision@k is undefined when there are no anomalies")
    rank = rankdata(score)
    return float(np.sum((rank <= k) & (label == 1)) / float(k))


def auroc(scores: Iterable[float], labels: Iterable[int], *, lower_score_more_anomalous: bool = True) -> float:
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    label = np.asarray(labels).reshape(-1)
    if score.shape != label.shape or score.size == 0:
        raise ValueError(f"Score/label shapes must match and be nonempty: {score.shape}, {label.shape}")
    if not np.isfinite(score).all() or not np.isin(label, [0, 1]).all():
        raise ValueError("Scores must be finite and labels must be binary")
    n_positive = int(np.sum(label == 1))
    n_negative = int(np.sum(label == 0))
    if n_positive == 0 or n_negative == 0:
        raise ValueError("AUROC requires both normal and anomalous test points")
    anomaly_score = -score if lower_score_more_anomalous else score
    positive_rank_sum = float(np.sum(rankdata(anomaly_score, method="average")[label == 1]))
    return (positive_rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)


def result_dir(dataset: str, method: str, seed: int, root: Path | str = RESULTS_ROOT) -> Path:
    return Path(root) / canonical_dataset(dataset) / canonical_method(method) / f"seed_{int(seed)}"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_scores_and_metrics(
    dataset: str, method: str, seed: int, scores: Iterable[float], labels: Iterable[int],
    *, root: Path | str = RESULTS_ROOT, metadata: Mapping[str, Any] | None = None,
) -> Path:
    dataset = canonical_dataset(dataset)
    method = canonical_method(method)
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    label = np.asarray(labels, dtype=np.int64).reshape(-1)
    precision = precision_at_k(score, label)
    directory = result_dir(dataset, method, seed, root)
    directory.mkdir(parents=True, exist_ok=True)
    score_path = directory / "scores.npz"
    temporary = directory / ".scores.npz.tmp"
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, scores=score, labels=label)
    os.replace(temporary, score_path)
    payload: dict[str, Any] = {
        "schema_version": CONFIG["schema_version"], "status": "complete",
        "dataset": dataset, "method": method, "seed": int(seed),
        "lower_score_more_anomalous": True, "n_test": int(score.size),
        "k": int(label.sum()), "precision_at_k": precision,
        "scores_file": "scores.npz", "scores_sha256": file_digest(score_path),
        "labels_sha256": array_digest(label),
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    _atomic_json(directory / "metrics.json", payload)
    return directory / "metrics.json"
