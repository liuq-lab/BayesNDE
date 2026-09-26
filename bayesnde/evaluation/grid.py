"""Shared helpers for density evaluation configs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Tuple

import numpy as np
import yaml


def validate_indep_gmm_dimensions(config: Mapping[str, Any]) -> None:
    data_cfg = config.get("data", {})
    dataset_name = str(data_cfg.get("name", ""))
    if dataset_name != "indep_gmm":
        return

    model_cfg = config.get("model", {})
    dim = int(data_cfg["dim"])
    x_dim = int(model_cfg["x_dim"])
    z_dim = int(model_cfg["z_dim"])
    if dim != x_dim or z_dim < 1:
        raise ValueError(
            f"{dataset_name} dimension mismatch: expected data.dim == model.x_dim and "
            f"model.z_dim >= 1, got data.dim={dim}, model.x_dim={x_dim}, model.z_dim={z_dim}."
        )


def validate_checkpoint_matches_config(checkpoint_dir: Path, config: Mapping[str, Any]) -> None:
    checkpoint_path = Path(checkpoint_dir).resolve()
    run_config_path = None
    for candidate_dir in (checkpoint_path, *checkpoint_path.parents):
        candidate = candidate_dir / "run_config_used.yaml"
        if candidate.exists():
            run_config_path = candidate
            break
    if run_config_path is None:
        return

    with open(run_config_path, "r", encoding="utf-8") as f:
        checkpoint_config = yaml.safe_load(f) or {}

    data_cfg = checkpoint_config.get("data", {})
    model_cfg = checkpoint_config.get("model", {})
    if "dim" not in data_cfg or "x_dim" not in model_cfg or "z_dim" not in model_cfg:
        return

    expected = (
        int(config["data"]["dim"]),
        int(config["model"]["x_dim"]),
        int(config["model"]["z_dim"]),
    )
    observed = (
        int(data_cfg["dim"]),
        int(model_cfg["x_dim"]),
        int(model_cfg["z_dim"]),
    )
    if observed != expected:
        raise ValueError(
            "Checkpoint dimension mismatch: "
            f"checkpoint data/model dims={observed}, requested dims={expected}. "
            "Use a checkpoint trained with the same data.dim/model.x_dim/model.z_dim."
        )

    requested_model_cfg = config.get("model", {})
    architecture_keys = (
        "factorized_generator",
        "fixed_variance_head",
    )
    for key in architecture_keys:
        requested_value = bool(requested_model_cfg.get(key, False))
        observed_value = bool(model_cfg.get(key, False))
        if requested_value != observed_value:
            raise ValueError(
                f"Checkpoint architecture mismatch for model.{key}: "
                f"checkpoint has {observed_value}, requested config has {requested_value}."
            )

    if bool(requested_model_cfg.get("fixed_variance_head", False)):
        requested_value = float(requested_model_cfg.get("fixed_variance_value", 1.0e-2))
        observed_value = float(model_cfg.get("fixed_variance_value", 1.0e-2))
        if not np.isclose(requested_value, observed_value):
            raise ValueError(
                "Checkpoint fixed-variance mismatch: "
                f"checkpoint model.fixed_variance_value={observed_value}, "
                f"requested config has {requested_value}."
            )


def create_2d_slice_grid(
    grid_cfg: Mapping[str, Any],
    data_cfg: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_dim = int(data_cfg.get("dim", 2))
    if data_dim != 2:
        raise ValueError(f"2D grid evaluation requires data.dim == 2, got {data_dim}.")

    n = int(grid_cfg["n"])
    grid_x1 = np.linspace(float(grid_cfg["x1_min"]), float(grid_cfg["x1_max"]), n)
    grid_x2 = np.linspace(float(grid_cfg["x2_min"]), float(grid_cfg["x2_max"]), n)
    v1, v2 = np.meshgrid(grid_x1, grid_x2)
    data_grid = np.vstack((v1.ravel(), v2.ravel())).T.astype(np.float32)
    return v1, v2, data_grid


def remove_artifacts(run_dir: Path, names: Iterable[str]) -> None:
    for name in names:
        path = Path(run_dir) / name
        if path.exists() and path.is_file():
            path.unlink()


def array_fingerprint(array: np.ndarray) -> dict[str, Any]:
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256(arr.view(np.uint8)).hexdigest()
    finite = np.isfinite(arr) if np.issubdtype(arr.dtype, np.number) else np.ones(arr.shape, dtype=bool)
    summary: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": digest,
    }
    if np.issubdtype(arr.dtype, np.number):
        finite_values = arr[finite]
        summary.update(
            {
                "finite_count": int(finite_values.size),
                "mean": float(np.mean(finite_values)) if finite_values.size else None,
                "std": float(np.std(finite_values)) if finite_values.size else None,
                "min": float(np.min(finite_values)) if finite_values.size else None,
                "max": float(np.max(finite_values)) if finite_values.size else None,
            }
        )
    return summary


def save_dataset_fingerprint(
    run_dir: Path,
    *,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    selected_test_indices: Optional[np.ndarray] = None,
    selected_test_points: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "x_train": array_fingerprint(x_train),
        "x_val": array_fingerprint(x_val),
        "x_test": array_fingerprint(x_test),
    }
    if selected_test_indices is not None:
        indices = np.asarray(selected_test_indices, dtype=np.int64)
        payload["selected_test_indices"] = array_fingerprint(indices)
        payload["selected_test_indices_values"] = indices.tolist()
    if selected_test_points is not None:
        payload["selected_test_points"] = array_fingerprint(selected_test_points)

    run_path = Path(run_dir)
    with open(run_path / "dataset_fingerprint.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    if selected_test_indices is not None:
        np.savez(
            run_path / "selected_test_indices.npz",
            test_indices=np.asarray(selected_test_indices, dtype=np.int64),
            x_test=selected_test_points if selected_test_points is not None else np.empty((0,), dtype=np.float32),
        )
    return payload
