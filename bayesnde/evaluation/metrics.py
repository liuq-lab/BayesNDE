"""Scoring helpers shared by the evaluation pipeline."""
from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

from bayesnde.data.samplers import spearman_corr


def bootstrap_spearman_ci(
    truth: np.ndarray,
    estimate: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 20260831,
) -> Dict[str, Any]:
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    estimate = np.asarray(estimate, dtype=np.float64).reshape(-1)
    finite = np.isfinite(truth) & np.isfinite(estimate)
    truth, estimate = truth[finite], estimate[finite]
    if truth.size < 2:
        return {"low": None, "high": None, "confidence": confidence, "n_bootstrap": 0}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(n_bootstrap)):
        indices = rng.integers(0, truth.size, size=truth.size)
        value = spearman_corr(estimate[indices], truth[indices])
        if np.isfinite(value):
            values.append(value)
    if not values:
        return {"low": None, "high": None, "confidence": confidence, "n_bootstrap": 0}
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "low": float(np.quantile(values, alpha)),
        "high": float(np.quantile(values, 1.0 - alpha)),
        "confidence": float(confidence),
        "n_bootstrap": len(values),
        "seed": int(seed),
    }


def generation_metric_summary(section: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    density = section.get("density", {})
    mode_residual = section.get("mode_residual", {})
    within_1sd = mode_residual.get("fraction_within_1sd_of_center") or []
    finite_within_1sd = [float(v) for v in within_1sd if v is not None and np.isfinite(float(v))]
    corr = section.get("corr", {})
    two_sample = section.get("two_sample_vs_validation", section.get("two_sample_vs_test", {}))
    return {
        f"{prefix}_mean_log_true_px": density.get("mean_log_true_px"),
        f"{prefix}_min_fraction_within_1sd": min(finite_within_1sd) if finite_within_1sd else None,
        f"{prefix}_mean_fraction_within_1sd": (
            float(np.mean(finite_within_1sd)) if finite_within_1sd else None
        ),
        f"{prefix}_corr_frobenius_error": two_sample.get("corr_matrix_frobenius_error"),
        f"{prefix}_corr_frobenius_error_to_identity": section.get(
            "corr_frobenius_error_to_independent"
        ),
        f"{prefix}_max_abs_offdiag_corr": corr.get("max_abs_offdiag_corr"),
        f"{prefix}_mean_abs_offdiag_corr": corr.get("mean_abs_offdiag_corr"),
        f"{prefix}_rho_gt_0p1_pairs": corr.get("n_pairs_abs_corr_gt_threshold"),
        f"{prefix}_wasserstein_mean": two_sample.get("wasserstein_mean"),
        f"{prefix}_wasserstein_max": two_sample.get("wasserstein_max"),
        f"{prefix}_ks_stat_mean": two_sample.get("ks_stat_mean"),
        f"{prefix}_ks_stat_max": two_sample.get("ks_stat_max"),
        f"{prefix}_sym_kl_mean": two_sample.get("sym_kl_mean"),
        f"{prefix}_jsd_mean": two_sample.get("jsd_mean"),
        f"{prefix}_mmd_rbf": two_sample.get("mmd_rbf"),
        f"{prefix}_lisi_mean": two_sample.get("lisi_mean"),
        f"{prefix}_lisi_normalized_mean": two_sample.get("lisi_normalized_mean"),
        f"{prefix}_peak_coverage_mean": two_sample.get("peak_coverage_mean"),
        f"{prefix}_peak_full_coverage_dim_fraction": two_sample.get(
            "peak_full_coverage_dim_fraction"
        ),
        f"{prefix}_marginal_histogram_correlation_mean": two_sample.get(
            "marginal_histogram_correlation_mean"
        ),
        f"{prefix}_peak_valley_ratio_error_mean": two_sample.get(
            "peak_valley_ratio_error_mean"
        ),
        f"{prefix}_peak_valley_separated_fraction": two_sample.get(
            "peak_valley_separated_fraction"
        ),
    }
