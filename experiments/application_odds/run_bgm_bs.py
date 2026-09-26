#!/usr/bin/env python3
"""Training and scoring helpers for the Shuttle anomaly-detection application."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import tensorflow as tf
from scipy.stats import rankdata

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bayesgm.datasets import Base_sampler
from experiments.uci.data import append_log


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_weights(model: tf.keras.Model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.weights.h5")
    model.save_weights(str(temporary))
    os.replace(temporary, path)


def data_path(dataset: str, data_root: Path) -> Path:
    return data_root / f"{dataset}.npz"


def load_data(dataset: str, data_root: Path) -> dict[str, np.ndarray]:
    path = data_path(dataset, data_root)
    if not path.exists():
        raise FileNotFoundError(f"Canonical processed ODDS split not found: {path}")
    with np.load(path) as raw:
        required = {"train_x", "train_y", "val_x", "val_y", "test_x", "test_y"}
        missing = required.difference(raw.files)
        if missing:
            raise ValueError(f"{path} is missing keys: {sorted(missing)}")
        result = {key: np.asarray(raw[key]) for key in required}
    for split in ("train", "val", "test"):
        x = np.asarray(result[f"{split}_x"], dtype=np.float32)
        y = np.asarray(result[f"{split}_y"]).reshape(-1).astype(np.int64)
        if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all():
            raise ValueError(f"Invalid {split} arrays in {path}")
        result[f"{split}_x"], result[f"{split}_y"] = x, y
    dims = {result[f"{split}_x"].shape[1] for split in ("train", "val", "test")}
    if len(dims) != 1:
        raise ValueError(f"Feature dimensions disagree in {path}")
    return result


def decoder_log_density(model: Any, values: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    chunks = []
    for start in range(0, len(values), batch_size):
        x = tf.convert_to_tensor(values[start:start + batch_size], tf.float32)
        z = model.e_net(x, training=False)
        mean, variance = model._decode_generator(z, training=False)
        variance = tf.maximum(variance, tf.cast(1.0e-6, variance.dtype))
        logp = -0.5 * tf.reduce_sum(tf.math.log(tf.cast(2.0 * math.pi, variance.dtype) * variance)
                                    + tf.square(x - mean) / variance, axis=1)
        chunks.append(np.asarray(logp.numpy(), dtype=np.float64))
    return np.concatenate(chunks)


def decoder_log_likelihood(model: Any, values: np.ndarray, batch_size: int = 1024) -> float:
    return float(np.mean(decoder_log_density(model, values, batch_size)))


def load_curve(root: Path) -> list[dict[str, Any]]:
    path = root / "iterative_validation_curve.json"
    return [] if not path.exists() else list(json.loads(path.read_text())["rows"])


def save_curve(root: Path, rows: list[dict[str, Any]]) -> None:
    rows = sorted({int(row["epoch"]): row for row in rows}.values(), key=lambda row: int(row["epoch"]))
    atomic_json(root / "iterative_validation_curve.json", {"rows": rows})
    columns = ["epoch", "validation_decoder_mean_log_likelihood", "loss_x",
               "loss_z", "empirical_corr_loss", "generator_weights"]
    path = root / "iterative_validation_curve.csv"
    temporary = path.with_suffix(f".csv.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows([{key: row.get(key) for key in columns} for row in rows])
    os.replace(temporary, path)


def newest_egm_checkpoint(root: Path) -> tuple[int, Path, Path] | None:
    pattern = re.compile(r"egm_step_(\d+)_generator\.weights\.h5$")
    found = []
    for generator in (root / "checkpoint/egm").glob("egm_step_*_generator.weights.h5"):
        match = pattern.search(generator.name)
        if match:
            step = int(match.group(1)); encoder = generator.with_name(f"egm_step_{step:04d}_encoder.weights.h5")
            if encoder.exists(): found.append((step, generator, encoder))
    return max(found, default=None, key=lambda value: value[0])


def load_egm_curve(root: Path) -> list[dict[str, Any]]:
    path = root / "egm_validation_curve.json"
    return [] if not path.exists() else list(json.loads(path.read_text())["rows"])


def save_egm_curve(root: Path, rows: list[dict[str, Any]]) -> None:
    rows = sorted({int(row["step"]): row for row in rows}.values(), key=lambda row: int(row["step"]))
    atomic_json(root / "egm_validation_curve.json", {"rows": rows})


def train_egm(model: Any, train_x: np.ndarray, val_x: np.ndarray,
              config: Mapping[str, Any], root: Path,
              val_y: np.ndarray | None = None) -> dict[str, Any]:
    target, batch_size = int(config["training"]["egm_n_iter"]), int(config["training"]["batch_size"])
    frozen = root / "frozen_egm_selection.json"
    if frozen.exists():
        selected = json.loads(frozen.read_text(encoding="utf-8"))
        model.g_net.load_weights(selected["selected"]["generator_weights"])
        model.e_net.load_weights(selected["selected"]["encoder_weights"])
        return selected
    checkpoint = newest_egm_checkpoint(root); start = 0
    if checkpoint:
        start = checkpoint[0]; model.g_net.load_weights(str(checkpoint[1])); model.e_net.load_weights(str(checkpoint[2]))
    sampler = Base_sampler(x=train_x, y=train_x, v=train_x, batch_size=batch_size, normalize=False)
    model.data_sampler = sampler
    every = int(config["training"]["egm_batches_per_eval"])
    rows = load_egm_curve(root)
    for step in range(start + 1, target + 1):
        for _ in range(int(model.params["g_d_freq"])):
            batch_x, _, _ = sampler.next_batch(); batch_z = model.z_sampler.get_batch(batch_size)
            model.train_disc_step(batch_z, batch_x)
        batch_x, _, _ = sampler.next_batch(); batch_z = model.z_sampler.get_batch(batch_size)
        losses = model.train_gen_step(batch_z, batch_x)
        if step % every == 0 or step == target:
            prefix = root / "checkpoint/egm" / f"egm_step_{step:04d}"
            generator_path = Path(str(prefix) + "_generator.weights.h5")
            encoder_path = Path(str(prefix) + "_encoder.weights.h5")
            atomic_weights(model.g_net, generator_path); atomic_weights(model.e_net, encoder_path)
            decoder_scores = decoder_log_density(model, val_x)
            score = float(np.mean(decoder_scores))
            row = {"step": step, "validation_decoder_mean_log_likelihood": score,
                         "total_loss": float(losses[-1]),
                         "generator_weights": str(generator_path), "encoder_weights": str(encoder_path)}
            if val_y is not None:
                row["validation_decoder_precision_at_k"] = precision_at_k(decoder_scores, val_y)
                row["validation_k"] = int(np.sum(val_y))
            rows.append(row)
            save_egm_curve(root, rows)
            append_log(root / "run.log", f"EGM step={step}/{target} total_loss={float(losses[-1]):.8g} decoder_val_ll={score:.8g}")
    rows = [row for row in load_egm_curve(root) if 0 < int(row["step"]) <= target]
    if not rows or newest_egm_checkpoint(root) is None or newest_egm_checkpoint(root)[0] != target:
        raise RuntimeError(f"EGM did not reach step {target}")
    if config["egm_selection"]["criterion"] == "maximum validation decoder precision at k":
        best = max(rows, key=lambda row: (
            float(row["validation_decoder_precision_at_k"]),
            float(row["validation_decoder_mean_log_likelihood"]), -int(row["step"])))
    else:
        best = max(rows, key=lambda row: (float(row["validation_decoder_mean_log_likelihood"]),
                                          -int(row["step"])))
    payload = {"criterion": config["egm_selection"]["criterion"],
               "tie_break": config["egm_selection"]["tie_break"],
               "selected": best, "candidates": rows,
               "test_points_used_for_selection": 0, "frozen_before_iterative_training": True}
    atomic_json(frozen, payload)
    model.g_net.load_weights(best["generator_weights"]); model.e_net.load_weights(best["encoder_weights"])
    return payload


def point_seed(seed: int, global_test_index: int) -> int:
    return int((seed * 1_000_003 + global_test_index * 10_009 + 900_000) % 2_147_483_647)


def precision_at_k(log_density: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels).reshape(-1); k = int(np.sum(labels == 1))
    if k <= 0: raise ValueError("Test set contains no anomalies")
    ranks = rankdata(np.asarray(log_density), method="average")
    return float(np.sum((ranks <= k) & (labels == 1)) / k)
