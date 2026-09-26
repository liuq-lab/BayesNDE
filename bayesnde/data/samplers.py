"""Simulation dataset helpers for density comparison scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from typing import Mapping
import hashlib

from bayesgm.datasets import GMM_indep_sampler, Swiss_roll_sampler


BASE_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "model.yaml"


def resolve_path(path_value, base_dir=None):
    p = Path(path_value)
    if p.is_absolute():
        return p
    base = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parents[2]
    return (base / p).resolve()


def build_sampler(data_cfg: Mapping[str, Any]) -> Any:
    np.random.seed(int(data_cfg["seed"]))
    name = str(data_cfg.get("name", "indep_gmm"))
    if name == "indep_gmm":
        return GMM_indep_sampler(
            N=int(data_cfg["n"]),
            sd=float(data_cfg["sd"]),
            dim=int(data_cfg["dim"]),
            n_components=int(data_cfg["n_components"]),
            bound=float(data_cfg["bound"]),
        )
    if name == "involute":
        return Swiss_roll_sampler(
            N=int(data_cfg["n"]),
            theta=float(data_cfg.get("theta", 2.0 * np.pi)),
            scale=float(data_cfg.get("scale", 2.0)),
            sigma=float(data_cfg.get("sigma", 0.4)),
        )
    raise ValueError(f"Unsupported dataset for density visualization: {name!r}")


def average_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)

    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def safe_log_density(px: np.ndarray, *, min_positive: float = 0.0) -> np.ndarray:
    values = np.asarray(px, dtype=np.float64)
    out = np.full(values.shape, -np.inf, dtype=np.float64)
    mask = np.isfinite(values) & (values > float(min_positive))
    out[mask] = np.log(values[mask])
    return out


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return float("nan")
    rx = average_rank(x)
    ry = average_rank(y)
    if np.std(rx) == 0.0 or np.std(ry) == 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _top_log_spearman(
    truth_log_px: np.ndarray,
    estimate_log_px: np.ndarray,
    finite: np.ndarray,
    quantile: float,
) -> float:
    if not np.any(finite):
        return float("nan")
    threshold = float(np.quantile(truth_log_px[finite], float(quantile)))
    mask = finite & (truth_log_px >= threshold)
    return spearman_corr(estimate_log_px[mask], truth_log_px[mask])


def _linear_fit(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    mask = np.isfinite(y) & np.isfinite(x)
    if int(np.sum(mask)) < 2:
        return float("nan"), float("nan"), float("nan")
    x_fit = np.asarray(x[mask], dtype=np.float64)
    y_fit = np.asarray(y[mask], dtype=np.float64)
    if float(np.var(x_fit)) == 0.0 or float(np.var(y_fit)) == 0.0:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x_fit, y_fit, deg=1)
    pred = slope * x_fit + intercept
    ss_res = float(np.sum(np.square(y_fit - pred)))
    ss_tot = float(np.sum(np.square(y_fit - float(np.mean(y_fit)))))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    return float(slope), float(intercept), r2


def masked_spearman_metrics(
    truth_px: np.ndarray,
    estimate_px: np.ndarray,
    finite: np.ndarray,
) -> Dict[str, float]:
    truth = np.asarray(truth_px, dtype=np.float64)
    estimate = np.asarray(estimate_px, dtype=np.float64)
    mask = np.asarray(finite, dtype=bool) & np.isfinite(truth) & np.isfinite(estimate)
    metrics = {
        "density_spearman_top75pct": float("nan"),
        "density_spearman_top50pct": float("nan"),
        "density_spearman_top25pct": float("nan"),
        "density_spearman_top10pct": float("nan"),
        "log_density_spearman_corr": float("nan"),
    }
    if not np.any(mask):
        return metrics

    truth_finite = truth[mask]
    for quantile, name in (
        (0.25, "density_spearman_top75pct"),
        (0.50, "density_spearman_top50pct"),
        (0.75, "density_spearman_top25pct"),
        (0.90, "density_spearman_top10pct"),
    ):
        threshold = float(np.quantile(truth_finite, quantile))
        q_mask = mask & (truth >= threshold)
        metrics[name] = spearman_corr(estimate[q_mask], truth[q_mask])

    positive = mask & (truth > 0.0) & (estimate > 0.0)
    metrics["log_density_spearman_corr"] = spearman_corr(
        np.log(estimate[positive]),
        np.log(truth[positive]),
    )
    return metrics


def density_calibration_metrics(
    truth_px: np.ndarray,
    estimate_px: np.ndarray,
    estimate_log_px: np.ndarray | None = None,
    truth_log_px: np.ndarray | None = None,
    finite: np.ndarray | None = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    truth = np.asarray(truth_px, dtype=np.float64).reshape(-1)
    estimate = np.asarray(estimate_px, dtype=np.float64).reshape(-1)
    if truth.shape != estimate.shape:
        raise ValueError("truth_px and estimate_px must have the same shape.")

    log_truth = safe_log_density(truth) if truth_log_px is None else np.asarray(truth_log_px, dtype=np.float64).reshape(-1)
    log_estimate = (
        safe_log_density(estimate)
        if estimate_log_px is None
        else np.asarray(estimate_log_px, dtype=np.float64).reshape(-1)
    )
    if log_truth.shape != truth.shape or log_estimate.shape != truth.shape:
        raise ValueError("log-density arrays must align with density arrays.")

    base_mask = np.isfinite(truth) & np.isfinite(estimate)
    if finite is not None:
        base_mask &= np.asarray(finite, dtype=bool).reshape(-1)
    log_mask = base_mask & np.isfinite(log_truth) & np.isfinite(log_estimate)

    metrics: Dict[str, Any] = {
        "density_points": int(truth.size),
        "finite_density_points": int(np.sum(base_mask)),
        "finite_log_density_points": int(np.sum(log_mask)),
    }

    if np.any(base_mask):
        metrics.update(
            {
                "density_spearman_corr": spearman_corr(estimate[base_mask], truth[base_mask]),
                "spearman_corr": spearman_corr(estimate[base_mask], truth[base_mask]),
            }
        )
    else:
        metrics.update(
            {
                "density_spearman_corr": float("nan"),
                "spearman_corr": float("nan"),
            }
        )

    if np.any(log_mask):
        log_error = log_estimate[log_mask] - log_truth[log_mask]
        slope_true_on_est, intercept_true_on_est, r2_true_on_est = _linear_fit(
            log_truth[log_mask], log_estimate[log_mask]
        )
        slope_est_on_true, intercept_est_on_true, r2_est_on_true = _linear_fit(
            log_estimate[log_mask], log_truth[log_mask]
        )
        metrics.update(
            {
                "mean_log_bias_est_minus_truth": float(np.mean(log_error)),
                "median_log_bias_est_minus_truth": float(np.median(log_error)),
                "log_density_spearman_corr": spearman_corr(log_estimate[log_mask], log_truth[log_mask]),
                "top25_spearman_log": _top_log_spearman(log_truth, log_estimate, log_mask, 0.75),
                "top10_spearman_log": _top_log_spearman(log_truth, log_estimate, log_mask, 0.90),
                "calibration_slope_true_on_est": slope_true_on_est,
                "calibration_intercept_true_on_est": intercept_true_on_est,
                "calibration_r2_true_on_est": r2_true_on_est,
                "calibration_slope_est_on_true": slope_est_on_true,
                "calibration_intercept_est_on_true": intercept_est_on_true,
                "calibration_r2_est_on_true": r2_est_on_true,
            }
        )
    else:
        metrics.update(
            {
                "mean_log_bias_est_minus_truth": float("nan"),
                "median_log_bias_est_minus_truth": float("nan"),
                "log_density_spearman_corr": float("nan"),
                "top25_spearman_log": float("nan"),
                "top10_spearman_log": float("nan"),
                "calibration_slope_true_on_est": float("nan"),
                "calibration_intercept_true_on_est": float("nan"),
                "calibration_r2_true_on_est": float("nan"),
                "calibration_slope_est_on_true": float("nan"),
                "calibration_intercept_est_on_true": float("nan"),
                "calibration_r2_est_on_true": float("nan"),
            }
        )

    metrics.update(masked_spearman_metrics(truth, estimate, base_mask))
    metrics.update(
        {
            "spearman": metrics["density_spearman_corr"],
            "log_spearman": metrics["log_density_spearman_corr"],
            "bias": metrics["mean_log_bias_est_minus_truth"],
            "slope": metrics["calibration_slope_true_on_est"],
            "spearman_log": metrics["log_density_spearman_corr"],
        }
    )
    if extra:
        metrics.update(extra)
    return metrics


def _array_digest(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode())
    digest.update(str(arr.dtype).encode())
    digest.update(arr.view(np.uint8))
    return digest.hexdigest()

def dataset_fingerprint(sampler: Any) -> dict[str, Any]:
    return {
        split: {"shape": list(np.asarray(values).shape), "sha256": _array_digest(np.asarray(values))}
        for split, values in (("train", sampler.X_train), ("validation", sampler.X_val), ("test", sampler.X_test))
    }

def assert_disjoint_splits(train: np.ndarray, validation: np.ndarray, test: np.ndarray) -> None:
    def row_tokens(values: np.ndarray) -> set[bytes]:
        arr = np.ascontiguousarray(values)
        return {row.tobytes() for row in arr}

    tokens = {"train": row_tokens(train), "validation": row_tokens(validation), "test": row_tokens(test)}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = tokens[left].intersection(tokens[right])
        if overlap:
            raise RuntimeError(f"Data leakage: {len(overlap)} exact rows overlap {left} and {right}.")
