"""Generation diagnostics for frozen BGM models, and the checkpoint rule built on them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import ks_2samp, wasserstein_distance

try:
    import tensorflow as tf
except ImportError:
    tf = None


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _quantiles(values: np.ndarray) -> list[float | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [None] * 7
    return [float(x) for x in np.quantile(arr, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])]


def _per_dim_summary(values: np.ndarray) -> dict[str, list[float]]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": np.mean(arr, axis=0).tolist(),
        "std": np.std(arr, axis=0).tolist(),
        "q10": np.quantile(arr, 0.10, axis=0).tolist(),
        "q25": np.quantile(arr, 0.25, axis=0).tolist(),
        "q50": np.quantile(arr, 0.50, axis=0).tolist(),
        "q75": np.quantile(arr, 0.75, axis=0).tolist(),
        "q90": np.quantile(arr, 0.90, axis=0).tolist(),
    }


def _global_summary(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": None, "min": None, "max": None, "quantiles": [None] * 7}
    return {
        "mean": float(np.mean(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "quantiles": _quantiles(finite),
    }


def _mode_occupancy(values: np.ndarray, centers: np.ndarray) -> list[list[float]]:
    arr = np.asarray(values, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    nearest = np.argmin(np.abs(arr[:, :, None] - centers[None, None, :]), axis=2)
    out: list[list[float]] = []
    for dim in range(arr.shape[1]):
        counts = np.bincount(nearest[:, dim], minlength=len(centers)).astype(np.float64)
        out.append((counts / max(float(arr.shape[0]), 1.0)).tolist())
    return out


def _mode_residual_summary(values: np.ndarray, centers: np.ndarray, sd: float | None) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    residual = np.min(np.abs(arr[:, :, None] - centers[None, None, :]), axis=2)
    out: dict[str, Any] = {
        "mean_abs_distance_to_nearest_center": np.mean(residual, axis=0).tolist(),
        "q50_abs_distance_to_nearest_center": np.quantile(residual, 0.50, axis=0).tolist(),
        "q75_abs_distance_to_nearest_center": np.quantile(residual, 0.75, axis=0).tolist(),
        "q90_abs_distance_to_nearest_center": np.quantile(residual, 0.90, axis=0).tolist(),
    }
    if sd is not None and np.isfinite(sd) and sd > 0:
        out["fraction_within_1sd_of_center"] = np.mean(residual <= sd, axis=0).tolist()
        out["fraction_within_2sd_of_center"] = np.mean(residual <= 2.0 * sd, axis=0).tolist()
    return out


def _corr_summary(values: np.ndarray, threshold: float = 0.1) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[1] < 2:
        return {
            "corr_matrix": np.eye(arr.shape[1], dtype=np.float64).tolist(),
            "corr_frobenius_error_to_independent": 0.0,
            "max_abs_offdiag_corr": 0.0,
            "mean_abs_offdiag_corr": 0.0,
            "n_pairs_abs_corr_gt_threshold": 0,
            "threshold": float(threshold),
            "pairs_abs_corr_gt_threshold": [],
        }
    corr = np.corrcoef(arr, rowvar=False)
    offdiag = corr[np.triu_indices(corr.shape[0], k=1)]
    pairs = []
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[1]):
            rho = float(corr[i, j])
            if abs(rho) > threshold:
                pairs.append({"i": i, "j": j, "rho": rho, "abs_rho": abs(rho)})
    pairs.sort(key=lambda row: row["abs_rho"], reverse=True)
    target = np.eye(arr.shape[1], dtype=np.float64)
    return {
        "corr_matrix": corr.tolist(),
        "corr_frobenius_error_to_independent": float(np.linalg.norm(corr - target, ord="fro")),
        "max_abs_offdiag_corr": float(np.max(np.abs(offdiag))) if offdiag.size else 0.0,
        "mean_abs_offdiag_corr": float(np.mean(np.abs(offdiag))) if offdiag.size else 0.0,
        "n_pairs_abs_corr_gt_threshold": len(pairs),
        "threshold": float(threshold),
        "pairs_abs_corr_gt_threshold": pairs,
    }


def _corr_error(values: np.ndarray) -> float:
    return float(_corr_summary(values)["corr_frobenius_error_to_independent"])


def _density_summary(values: np.ndarray, sampler: Any) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if hasattr(sampler, "get_log_density"):
        log_px = np.asarray(sampler.get_log_density(arr), dtype=np.float64)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            px = np.exp(log_px)
    else:
        px = np.asarray(sampler.get_density(arr), dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_px = np.log(px)
    finite = np.isfinite(log_px)
    return {
        "n_points": int(arr.shape[0]),
        "mean_true_px": float(np.mean(px[np.isfinite(px)])) if np.any(np.isfinite(px)) else None,
        "mean_log_true_px": float(np.mean(log_px[finite])) if np.any(finite) else None,
        "finite_log_true_px_count": int(np.sum(finite)),
        "finite_log_true_px_fraction": float(np.mean(finite)),
        "log_true_px_quantiles": _quantiles(log_px),
        "true_px_quantiles": _quantiles(px),
    }


def _two_sample_metrics(reference: np.ndarray, generated: np.ndarray) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float64)
    gen = np.asarray(generated, dtype=np.float64)
    n_dim = ref.shape[1]
    wasserstein = []
    ks_stat = []
    ks_pvalue = []
    for dim in range(n_dim):
        wasserstein.append(float(wasserstein_distance(ref[:, dim], gen[:, dim])))
        ks = ks_2samp(ref[:, dim], gen[:, dim])
        ks_stat.append(float(ks.statistic))
        ks_pvalue.append(float(ks.pvalue))
    out = {
        "wasserstein_per_dim": wasserstein,
        "ks_stat_per_dim": ks_stat,
        "ks_pvalue_per_dim": ks_pvalue,
        "wasserstein_mean": float(np.mean(wasserstein)) if wasserstein else 0.0,
        "wasserstein_max": float(np.max(wasserstein)) if wasserstein else 0.0,
        "ks_stat_mean": float(np.mean(ks_stat)) if ks_stat else 0.0,
        "ks_stat_max": float(np.max(ks_stat)) if ks_stat else 0.0,
        "corr_matrix_frobenius_error": float(
            np.linalg.norm(np.corrcoef(ref, rowvar=False) - np.corrcoef(gen, rowvar=False), ord="fro")
        ) if n_dim > 1 else 0.0,
    }
    out.update(_distribution_evidence_metrics(ref, gen))
    return out


def _histogram_divergences(
    reference: np.ndarray,
    generated: np.ndarray,
    n_bins: int = 80,
    eps: float = 1.0e-12,
) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float64)
    gen = np.asarray(generated, dtype=np.float64)
    kl_ref_gen = []
    kl_gen_ref = []
    jsd = []
    for dim in range(ref.shape[1]):
        combined = np.concatenate([ref[:, dim], gen[:, dim]])
        combined = combined[np.isfinite(combined)]
        if combined.size < 2:
            kl_ref_gen.append(float("nan"))
            kl_gen_ref.append(float("nan"))
            jsd.append(float("nan"))
            continue
        low, high = np.quantile(combined, [0.001, 0.999])
        if not np.isfinite(low) or not np.isfinite(high) or low == high:
            low = float(np.min(combined))
            high = float(np.max(combined))
        if low == high:
            high = low + 1.0
        ref_counts, edges = np.histogram(ref[:, dim], bins=n_bins, range=(low, high))
        gen_counts, _ = np.histogram(gen[:, dim], bins=edges)
        p = ref_counts.astype(np.float64) + eps
        q = gen_counts.astype(np.float64) + eps
        p = p / np.sum(p)
        q = q / np.sum(q)
        m = 0.5 * (p + q)
        kl_pq = float(np.sum(p * np.log(p / q)))
        kl_qp = float(np.sum(q * np.log(q / p)))
        kl_ref_gen.append(kl_pq)
        kl_gen_ref.append(kl_qp)
        jsd.append(float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))))
    sym_kl = [0.5 * (a + b) for a, b in zip(kl_ref_gen, kl_gen_ref)]
    return {
        "kl_ref_to_gen_per_dim": kl_ref_gen,
        "kl_gen_to_ref_per_dim": kl_gen_ref,
        "sym_kl_per_dim": sym_kl,
        "jsd_per_dim": jsd,
        "kl_ref_to_gen_mean": _finite_mean(kl_ref_gen),
        "kl_gen_to_ref_mean": _finite_mean(kl_gen_ref),
        "sym_kl_mean": _finite_mean(sym_kl),
        "jsd_mean": _finite_mean(jsd),
    }


def _marginal_peak_coverage(
    reference: np.ndarray,
    generated: np.ndarray,
    n_bins: int = 100,
) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float64)
    gen = np.asarray(generated, dtype=np.float64)
    coverages: list[float] = []
    ref_counts: list[int] = []
    gen_counts: list[int] = []
    matched_counts: list[int] = []
    for dim in range(ref.shape[1]):
        combined = np.concatenate([ref[:, dim], gen[:, dim]])
        low, high = np.quantile(combined[np.isfinite(combined)], [0.001, 0.999])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            ref_counts.append(0); gen_counts.append(0); matched_counts.append(0); coverages.append(float("nan"))
            continue
        ref_hist, edges = np.histogram(ref[:, dim], bins=n_bins, range=(low, high), density=True)
        gen_hist, _ = np.histogram(gen[:, dim], bins=edges, density=True)
        ref_smooth = gaussian_filter1d(ref_hist.astype(np.float64), sigma=1.25)
        gen_smooth = gaussian_filter1d(gen_hist.astype(np.float64), sigma=1.25)
        ref_peaks, _ = find_peaks(ref_smooth, prominence=max(float(np.max(ref_smooth)) * 0.08, 1.0e-12), distance=max(3, n_bins // 10))
        gen_peaks, _ = find_peaks(gen_smooth, prominence=max(float(np.max(gen_smooth)) * 0.08, 1.0e-12), distance=max(3, n_bins // 10))
        centres = 0.5 * (edges[:-1] + edges[1:])
        tolerance = max(3.0 * float(edges[1] - edges[0]), 0.08 * float(high - low))
        unmatched = list(centres[gen_peaks])
        matched = 0
        for peak in centres[ref_peaks]:
            if not unmatched:
                break
            nearest = int(np.argmin(np.abs(np.asarray(unmatched) - peak)))
            if abs(unmatched[nearest] - peak) <= tolerance:
                matched += 1
                unmatched.pop(nearest)
        ref_count = int(len(ref_peaks))
        ref_counts.append(ref_count)
        gen_counts.append(int(len(gen_peaks)))
        matched_counts.append(matched)
        coverages.append(float(matched / ref_count) if ref_count else float("nan"))
    coverage_array = np.asarray(coverages, dtype=np.float64)
    eligible = np.asarray(ref_counts) > 0
    return {
        "reference_peak_count_per_dim": ref_counts,
        "generated_peak_count_per_dim": gen_counts,
        "matched_peak_count_per_dim": matched_counts,
        "peak_coverage_per_dim": coverages,
        "peak_coverage_mean": float(np.nanmean(coverage_array)) if np.any(np.isfinite(coverage_array)) else None,
        "peak_full_coverage_dim_fraction": float(np.mean(coverage_array[eligible] >= 1.0)) if np.any(eligible) else None,
        "reference_peak_count_median": float(np.median(np.asarray(ref_counts)[eligible])) if np.any(eligible) else None,
    }


def _marginal_shape_metrics(
    reference: np.ndarray,
    generated: np.ndarray,
    n_bins: int = 120,
) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float64)
    gen = np.asarray(generated, dtype=np.float64)
    correlations: list[float] = []
    reference_ratios: list[float] = []
    generated_ratios: list[float] = []
    ratio_errors: list[float] = []
    separated: list[float] = []
    eps = 1.0e-12
    for dim in range(ref.shape[1]):
        finite = np.concatenate([ref[:, dim], gen[:, dim]])
        finite = finite[np.isfinite(finite)]
        if finite.size < 2:
            correlations.append(float("nan"))
            continue
        low, high = np.quantile(finite, [0.001, 0.999])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            correlations.append(float("nan"))
            continue
        ref_hist, edges = np.histogram(ref[:, dim], bins=n_bins, range=(low, high), density=True)
        gen_hist, _ = np.histogram(gen[:, dim], bins=edges, density=True)
        ref_smooth = gaussian_filter1d(ref_hist.astype(np.float64), sigma=1.5)
        gen_smooth = gaussian_filter1d(gen_hist.astype(np.float64), sigma=1.5)
        if np.std(ref_smooth) > eps and np.std(gen_smooth) > eps:
            correlations.append(float(np.corrcoef(ref_smooth, gen_smooth)[0, 1]))
        else:
            correlations.append(float("nan"))
        peaks, _ = find_peaks(
            ref_smooth,
            prominence=max(float(np.max(ref_smooth)) * 0.08, eps),
            distance=max(3, n_bins // 10),
        )
        for left, right in zip(peaks[:-1], peaks[1:]):
            if right - left < 3:
                continue
            valley = int(left + np.argmin(ref_smooth[left:right + 1]))
            radius = max(1, n_bins // 60)
            left_gen = float(np.max(gen_smooth[max(0, left - radius):min(n_bins, left + radius + 1)]))
            right_gen = float(np.max(gen_smooth[max(0, right - radius):min(n_bins, right + radius + 1)]))
            valley_gen = float(np.mean(gen_smooth[max(0, valley - radius):min(n_bins, valley + radius + 1)]))
            ref_ratio = float(ref_smooth[valley] / max(min(ref_smooth[left], ref_smooth[right]), eps))
            gen_ratio = float(valley_gen / max(min(left_gen, right_gen), eps))
            reference_ratios.append(ref_ratio)
            generated_ratios.append(gen_ratio)
            ratio_errors.append(abs(gen_ratio - ref_ratio))
            separated.append(float(gen_ratio <= max(0.35, ref_ratio + 0.20)))
    return {
        "marginal_histogram_correlation_per_dim": correlations,
        "marginal_histogram_correlation_mean": _finite_mean(correlations),
        "reference_peak_valley_ratio_mean": _finite_mean(reference_ratios),
        "generated_peak_valley_ratio_mean": _finite_mean(generated_ratios),
        "peak_valley_ratio_error_mean": _finite_mean(ratio_errors),
        "peak_valley_separated_fraction": _finite_mean(separated),
        "peak_valley_pair_count": int(len(generated_ratios)),
    }


def _finite_mean(values: list[float]) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else None


def _subsample_rows(values: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[0] <= max_points:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(arr.shape[0], size=max_points, replace=False)
    return arr[idx]


def _density_eval_subset(values: np.ndarray, max_points: int | None, seed: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if max_points is None or max_points < 1 or arr.shape[0] <= max_points:
        return arr
    return _subsample_rows(arr, max_points=max_points, seed=seed)


def _density_points_per_summary(max_points: int | None, n_summaries: int) -> int | None:
    if max_points is None:
        return None
    if max_points < 1:
        return None
    return max(1, int(np.ceil(float(max_points) / max(int(n_summaries), 1))))


def _squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2 = np.sum(np.square(a), axis=1)[:, None]
    b2 = np.sum(np.square(b), axis=1)[None, :]
    return np.maximum(a2 + b2 - 2.0 * np.matmul(a, b.T), 0.0)


def _mmd_rbf(
    reference: np.ndarray,
    generated: np.ndarray,
    max_points: int = 2000,
    seed: int = 20260604,
) -> dict[str, Any]:
    ref = _subsample_rows(reference, max_points=max_points, seed=seed)
    gen = _subsample_rows(generated, max_points=max_points, seed=seed + 1)
    if ref.size == 0 or gen.size == 0:
        return {"mmd_rbf": None, "mmd_rbf_scales": [], "mmd_rbf_per_scale": []}
    d_xx = _squared_distances(ref, ref)
    d_yy = _squared_distances(gen, gen)
    d_xy = _squared_distances(ref, gen)
    median_sq = float(np.median(d_xy[np.isfinite(d_xy)]))
    if not np.isfinite(median_sq) or median_sq <= 0.0:
        median_sq = float(ref.shape[1])
    scales = [0.25 * median_sq, median_sq, 4.0 * median_sq]
    values = []
    for scale in scales:
        gamma = 1.0 / max(2.0 * scale, 1.0e-12)
        mmd = (
            float(np.mean(np.exp(-gamma * d_xx)))
            + float(np.mean(np.exp(-gamma * d_yy)))
            - 2.0 * float(np.mean(np.exp(-gamma * d_xy)))
        )
        values.append(max(mmd, 0.0))
    return {
        "mmd_rbf": float(np.mean(values)),
        "mmd_rbf_scales": scales,
        "mmd_rbf_per_scale": values,
        "mmd_rbf_points": int(min(ref.shape[0], gen.shape[0])),
    }


def _sliced_wasserstein(
    reference: np.ndarray,
    generated: np.ndarray,
    n_projections: int = 64,
    max_points: int = 2000,
    seed: int = 20260831,
) -> dict[str, Any]:
    ref = _subsample_rows(reference, max_points=max_points, seed=seed)
    gen = _subsample_rows(generated, max_points=max_points, seed=seed + 1)
    n = min(ref.shape[0], gen.shape[0])
    if n < 1:
        return {"sliced_wasserstein": None, "sliced_wasserstein_projections": int(n_projections)}
    ref = ref[:n]
    gen = gen[:n]
    combined = np.vstack([ref, gen])
    mean = np.mean(combined, axis=0, keepdims=True)
    std = np.std(combined, axis=0, keepdims=True)
    std = np.where(np.isfinite(std) & (std > 1.0e-12), std, 1.0)
    ref = (ref - mean) / std
    gen = (gen - mean) / std
    rng = np.random.default_rng(seed)
    projections = rng.normal(size=(ref.shape[1], int(n_projections)))
    projections /= np.maximum(np.linalg.norm(projections, axis=0, keepdims=True), 1.0e-12)
    distances = np.mean(np.abs(np.sort(ref @ projections, axis=0) - np.sort(gen @ projections, axis=0)), axis=0)
    return {
        "sliced_wasserstein": float(np.mean(distances)),
        "sliced_wasserstein_per_projection": distances.tolist(),
        "sliced_wasserstein_projections": int(n_projections),
        "sliced_wasserstein_points": int(n),
    }


def _lisi_two_sample(
    reference: np.ndarray,
    generated: np.ndarray,
    k: int = 30,
    max_points_per_group: int = 2000,
    seed: int = 20260604,
) -> dict[str, Any]:
    ref = _subsample_rows(reference, max_points=max_points_per_group, seed=seed)
    gen = _subsample_rows(generated, max_points=max_points_per_group, seed=seed + 1)
    values = np.vstack([ref, gen])
    labels = np.concatenate([np.zeros(ref.shape[0], dtype=np.int64), np.ones(gen.shape[0], dtype=np.int64)])
    if values.shape[0] <= 2:
        return {"lisi_mean": None, "lisi_normalized_mean": None, "lisi_k": int(k)}
    k_eff = min(int(k), values.shape[0] - 1)
    tree = cKDTree(values)
    _, indices = tree.query(values, k=k_eff + 1)
    neighbor_labels = labels[indices[:, 1:]]
    frac_ref = np.mean(neighbor_labels == 0, axis=1)
    frac_gen = 1.0 - frac_ref
    lisi = 1.0 / (np.square(frac_ref) + np.square(frac_gen) + 1.0e-12)
    return {
        "lisi_mean": float(np.mean(lisi)),
        "lisi_normalized_mean": float(np.mean(lisi - 1.0)),
        "lisi_k": int(k_eff),
        "lisi_points": int(values.shape[0]),
    }


def _distribution_evidence_metrics(reference: np.ndarray, generated: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out.update(_histogram_divergences(reference, generated))
    out.update(_mmd_rbf(reference, generated))
    out.update(_lisi_two_sample(reference, generated))
    out.update(_sliced_wasserstein(reference, generated))
    out.update(_marginal_peak_coverage(reference, generated))
    out.update(_marginal_shape_metrics(reference, generated))
    return out


def _save_marginal_overlay_plot(
    *,
    reference: np.ndarray,
    generated: np.ndarray,
    centers: np.ndarray,
    sd: float | None,
    out_path: Path,
    title: str,
    max_dims: int | None = 10,
    start_dim: int = 0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = np.asarray(reference, dtype=np.float64)
    gen = np.asarray(generated, dtype=np.float64)
    available_dims = min(ref.shape[1], gen.shape[1])
    start_dim = int(start_dim)
    n_dim = available_dims - start_dim if max_dims is None else min(available_dims - start_dim, int(max_dims))
    if n_dim < 1:
        return
    n_cols = min(5, n_dim)
    n_rows = int(np.ceil(n_dim / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.4 * n_rows), squeeze=False)
    for local_dim in range(n_rows * n_cols):
        ax = axes[local_dim // n_cols][local_dim % n_cols]
        if local_dim >= n_dim:
            ax.axis("off")
            continue
        dim = start_dim + local_dim
        combined = np.concatenate([ref[:, dim], gen[:, dim]])
        low, high = np.quantile(combined[np.isfinite(combined)], [0.001, 0.999])
        if not np.isfinite(low) or not np.isfinite(high) or low == high:
            low, high = -1.5, 1.5
        if sd is not None and np.isfinite(sd) and sd > 0.0:
            x_grid = np.linspace(low, high, 500)
            component_pdf = np.exp(
                -0.5 * np.square((x_grid[:, None] - centers[None, :]) / float(sd))
            ) / (float(sd) * np.sqrt(2.0 * np.pi))
            exact_pdf = np.mean(component_pdf, axis=1)
            ax.plot(x_grid, exact_pdf, color="black", linewidth=1.4, label="Exact PDF")
        else:
            ax.hist(ref[:, dim], bins=60, range=(low, high), density=True, alpha=0.35, label="truth")
        ax.hist(gen[:, dim], bins=60, range=(low, high), density=True, alpha=0.45, label="generated")
        for center in centers:
            ax.axvline(float(center), color="black", linewidth=0.6, alpha=0.35)
        ax.set_title(f"dim {dim}", fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_generic_marginal_overlay_plot(
    *,
    reference: np.ndarray,
    generated: np.ndarray,
    out_path: Path,
    title: str,
    reference_label: str = "real",
    generated_label: str = "generated",
    max_dims: int | None = 12,
    start_dim: int = 0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = np.asarray(reference, dtype=np.float64)
    gen = np.asarray(generated, dtype=np.float64)
    available_dims = min(ref.shape[1], gen.shape[1])
    start_dim = int(start_dim)
    n_dim = available_dims - start_dim if max_dims is None else min(available_dims - start_dim, int(max_dims))
    if n_dim < 1:
        return
    n_cols = min(4, n_dim)
    n_rows = int(np.ceil(n_dim / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.5 * n_rows), squeeze=False)
    for local_dim in range(n_rows * n_cols):
        ax = axes[local_dim // n_cols][local_dim % n_cols]
        if local_dim >= n_dim:
            ax.axis("off")
            continue
        dim = start_dim + local_dim
        combined = np.concatenate([ref[:, dim], gen[:, dim]])
        combined = combined[np.isfinite(combined)]
        if combined.size >= 2:
            low, high = np.quantile(combined, [0.001, 0.999])
        else:
            low, high = -1.0, 1.0
        if not np.isfinite(low) or not np.isfinite(high) or low == high:
            low, high = float(np.min(combined)), float(np.max(combined))
        if not np.isfinite(low) or not np.isfinite(high) or low == high:
            low, high = -1.0, 1.0
        ax.hist(ref[:, dim], bins=60, range=(low, high), density=True, alpha=0.38, label=reference_label)
        ax.hist(gen[:, dim], bins=60, range=(low, high), density=True, alpha=0.45, label=generated_label)
        ax.set_title(f"dim {dim}", fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _standardize_for_embedding(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    std = np.where(np.isfinite(std) & (std > 1.0e-12), std, 1.0)
    out = (arr - mean) / std
    return np.nan_to_num(out, copy=False).astype(np.float32)


def _fit_umap_or_pca(values: np.ndarray, *, seed: int, n_neighbors: int, min_dist: float) -> tuple[np.ndarray, dict[str, Any]]:
    values_std = _standardize_for_embedding(values)
    n_neighbors = max(2, min(int(n_neighbors), int(values_std.shape[0]) - 1))
    try:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=float(min_dist),
            metric="euclidean",
            random_state=int(seed),
        )
        embedding = reducer.fit_transform(values_std)
        metadata = {
            "method": "UMAP",
            "n_neighbors": int(n_neighbors),
            "min_dist": float(min_dist),
            "random_state": int(seed),
        }
    except Exception as exc:
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2, random_state=int(seed))
        embedding = reducer.fit_transform(values_std)
        metadata = {
            "method": "PCA fallback",
            "fallback_reason": repr(exc),
            "random_state": int(seed),
        }
    return np.asarray(embedding, dtype=np.float32), metadata


def _save_uci_umap_plot(
    *,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    generated: np.ndarray,
    out_dir: Path,
    dataset: str,
    seed: int,
    max_points_per_group: int,
    n_neighbors: int,
    min_dist: float,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_plot = _subsample_rows(train, max_points=max_points_per_group, seed=seed + 11)
    val_plot = _subsample_rows(validation, max_points=max_points_per_group, seed=seed + 12)
    test_plot = _subsample_rows(test, max_points=max_points_per_group, seed=seed + 13)
    gen_plot = _subsample_rows(generated, max_points=max_points_per_group, seed=seed + 14)
    values = np.vstack([train_plot, val_plot, test_plot, gen_plot])
    labels = np.concatenate(
        [
            np.full(train_plot.shape[0], "train", dtype=object),
            np.full(val_plot.shape[0], "validation", dtype=object),
            np.full(test_plot.shape[0], "test", dtype=object),
            np.full(gen_plot.shape[0], "generated", dtype=object),
        ]
    )
    embedding, metadata = _fit_umap_or_pca(values, seed=seed, n_neighbors=n_neighbors, min_dist=min_dist)
    np.savez_compressed(
        out_dir / "umap_embedding_generated_vs_real.npz",
        embedding=embedding,
        labels=labels,
        values=values.astype(np.float32),
        method=np.asarray(metadata["method"]),
    )

    colors = {
        "background": "#d6d6d6",
        "generated": "#2b8c7d",
        "real": "#e39a24",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), squeeze=False)
    ax_train, ax_test = axes[0]

    def draw_panel(ax: Any, real_label: str, title: str, background_labels: set[str]) -> None:
        bg_mask = np.asarray([label in background_labels for label in labels], dtype=bool)
        real_mask = labels == real_label
        gen_mask = labels == "generated"
        if np.any(bg_mask):
            ax.scatter(
                embedding[bg_mask, 0],
                embedding[bg_mask, 1],
                s=4,
                c=colors["background"],
                alpha=0.30,
                linewidths=0,
                label="Other real",
            )
        ax.scatter(
            embedding[gen_mask, 0],
            embedding[gen_mask, 1],
            s=5,
            c=colors["generated"],
            alpha=0.70,
            linewidths=0,
            label="Generated",
        )
        ax.scatter(
            embedding[real_mask, 0],
            embedding[real_mask, 1],
            s=5,
            c=colors["real"],
            alpha=0.70,
            linewidths=0,
            label="Real",
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_frame_on(False)

    draw_panel(ax_train, "train", "Train", {"validation", "test"})
    draw_panel(ax_test, "test", "Test", {"train", "validation"})
    handles, panel_labels = ax_test.get_legend_handles_labels()
    fig.legend(handles, panel_labels, loc="center right", frameon=False)
    fig.suptitle(f"{dataset}: generated vs real data ({metadata['method']})", fontsize=14)
    fig.tight_layout(rect=(0, 0, 0.88, 0.93))
    fig.savefig(out_dir / "fig_umap_generated_vs_real.png", dpi=220)
    plt.close(fig)
    return {
        **metadata,
        "max_points_per_group": int(max_points_per_group),
        "points": {
            "train": int(train_plot.shape[0]),
            "validation": int(val_plot.shape[0]),
            "test": int(test_plot.shape[0]),
            "generated": int(gen_plot.shape[0]),
        },
    }


def run_uci_bgm_generation_diagnostics(
    *,
    model: Any,
    train_x: np.ndarray,
    validation_x: np.ndarray,
    test_x: np.ndarray,
    run_dir: Path,
    config: Mapping[str, Any],
    seed: int = 20260803,
) -> dict[str, Any]:
    diag_cfg = config.get("generation_diagnostics", {})
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    n_samples = int(diag_cfg.get("n_samples", 5000))
    reference_max_points = int(diag_cfg.get("reference_max_points", 5000))
    umap_max_points = int(diag_cfg.get("umap_max_points_per_group", 1500))
    umap_neighbors = int(diag_cfg.get("umap_n_neighbors", 30))
    umap_min_dist = float(diag_cfg.get("umap_min_dist", 0.1))
    z_dim = int(config["model"]["z_dim"])
    rng = np.random.default_rng(seed)
    z_samples = rng.normal(size=(n_samples, z_dim)).astype(np.float32)
    z_tensor = tf.convert_to_tensor(z_samples, dtype=tf.float32)
    decoder_outputs = model.g_net(z_tensor, training=False)
    if not isinstance(decoder_outputs, (tuple, list)):
        raise ValueError("BGM UCI generation diagnostics require generator output (mu, var) or (mu, var, U).")
    if len(decoder_outputs) == 2:
        mu_x, sigma_square_x = decoder_outputs
        low_rank_u = None
    elif len(decoder_outputs) == 3:
        mu_x, sigma_square_x, low_rank_u = decoder_outputs
    else:
        raise ValueError("BGM UCI generation diagnostics require generator output (mu, var) or (mu, var, U).")
    x_mean = np.asarray(mu_x.numpy(), dtype=np.float64)
    sigma_square = np.asarray(sigma_square_x.numpy(), dtype=np.float64)
    x_sample = x_mean + rng.normal(size=x_mean.shape) * np.sqrt(np.maximum(sigma_square, 0.0))
    low_rank_array = None
    if low_rank_u is not None:
        low_rank_array = np.asarray(low_rank_u.numpy(), dtype=np.float64)
        eps_low_rank = rng.normal(size=(x_mean.shape[0], low_rank_array.shape[-1])).astype(np.float64)
        x_sample = x_sample + np.einsum("ndr,nr->nd", low_rank_array, eps_low_rank)

    train = np.asarray(train_x, dtype=np.float64)
    validation = np.asarray(validation_x, dtype=np.float64)
    test = np.asarray(test_x, dtype=np.float64)
    train_ref = _subsample_rows(train, reference_max_points, seed + 21)
    validation_ref = _subsample_rows(validation, reference_max_points, seed + 22)
    test_ref = _subsample_rows(test, reference_max_points, seed + 23)

    diagnostics: dict[str, Any] = {
        "n_generated": int(n_samples),
        "z_dim": z_dim,
        "x_dim": int(config["model"]["x_dim"]),
        "seed": int(seed),
        "reference_max_points": int(reference_max_points),
        "train": {
            "n_points": int(train.shape[0]),
            "summary_points": int(train_ref.shape[0]),
            "per_dim": _per_dim_summary(train_ref),
            "corr": _corr_summary(train_ref),
        },
        "validation": {
            "n_points": int(validation.shape[0]),
            "summary_points": int(validation_ref.shape[0]),
            "per_dim": _per_dim_summary(validation_ref),
            "corr": _corr_summary(validation_ref),
        },
        "test": {
            "n_points": int(test.shape[0]),
            "summary_points": int(test_ref.shape[0]),
            "per_dim": _per_dim_summary(test_ref),
            "corr": _corr_summary(test_ref),
        },
        "generated_mean": {
            "finite_value_fraction": float(np.mean(np.isfinite(x_mean))),
            "per_dim": _per_dim_summary(x_mean),
            "corr": _corr_summary(x_mean),
            "two_sample_vs_train": _two_sample_metrics(train_ref, x_mean),
            "two_sample_vs_validation": _two_sample_metrics(validation_ref, x_mean),
            "two_sample_vs_test": _two_sample_metrics(test_ref, x_mean),
        },
        "generated_sample": {
            "finite_value_fraction": float(np.mean(np.isfinite(x_sample))),
            "per_dim": _per_dim_summary(x_sample),
            "corr": _corr_summary(x_sample),
            "two_sample_vs_train": _two_sample_metrics(train_ref, x_sample),
            "two_sample_vs_validation": _two_sample_metrics(validation_ref, x_sample),
            "two_sample_vs_test": _two_sample_metrics(test_ref, x_sample),
        },
        "generator_sigma_square": _per_dim_summary(sigma_square),
        "generator_sigma_square_global": _global_summary(sigma_square),
    }
    if low_rank_array is not None:
        diagnostics["generator_low_rank_u_global"] = _global_summary(low_rank_array)
        diagnostics["generator_low_rank_cov_diag_global"] = _global_summary(np.sum(np.square(low_rank_array), axis=-1))

    save_payload = {
        "z_samples": z_samples,
        "x_gen_mean": x_mean.astype(np.float32),
        "x_gen_sample": x_sample.astype(np.float32),
        "sigma_square_x": sigma_square.astype(np.float32),
        "x_train_reference": train_ref.astype(np.float32),
        "x_validation_reference": validation_ref.astype(np.float32),
        "x_test_reference": test_ref.astype(np.float32),
    }
    if low_rank_array is not None:
        save_payload["low_rank_u"] = low_rank_array.astype(np.float32)
    np.savez_compressed(run_path / "generated_samples.npz", **save_payload)
    _save_generic_marginal_overlay_plot(
        reference=train_ref,
        generated=x_sample,
        out_path=run_path / "fig_marginals_generated_sample_vs_train.png",
        title="Generated sample vs UCI train marginals",
        reference_label="train",
        generated_label="generated",
    )
    _save_generic_marginal_overlay_plot(
        reference=test_ref,
        generated=x_sample,
        out_path=run_path / "fig_marginals_generated_sample_vs_test.png",
        title="Generated sample vs UCI test marginals",
        reference_label="test",
        generated_label="generated",
    )
    diagnostics["umap"] = _save_uci_umap_plot(
        train=train_ref,
        validation=validation_ref,
        test=test_ref,
        generated=x_sample,
        out_dir=run_path,
        dataset=str(config.get("data", {}).get("name", "UCI")),
        seed=seed + 31,
        max_points_per_group=umap_max_points,
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
    )
    with open(run_path / "generation_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, default=_json_default)
    return diagnostics


def run_uci_generation_diagnostics_from_arrays(
    *,
    generated_x: np.ndarray,
    train_x: np.ndarray,
    validation_x: np.ndarray,
    test_x: np.ndarray,
    run_dir: Path,
    config: Mapping[str, Any],
    seed: int = 20260803,
    generation_mode: str = "generated",
    z_samples: np.ndarray | None = None,
    x_mean: np.ndarray | None = None,
    sigma_square: np.ndarray | None = None,
    low_rank_u: np.ndarray | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    diag_cfg = config.get("generation_diagnostics", {})
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    reference_max_points = int(diag_cfg.get("reference_max_points", 5000))
    umap_enabled = bool(diag_cfg.get("umap_enabled", True))
    umap_max_points = int(diag_cfg.get("umap_max_points_per_group", 1500))
    umap_neighbors = int(diag_cfg.get("umap_n_neighbors", 30))
    umap_min_dist = float(diag_cfg.get("umap_min_dist", 0.1))

    generated = np.asarray(generated_x, dtype=np.float64)
    if generated.ndim != 2:
        raise ValueError("generated_x must have shape (n_samples, x_dim).")

    train = np.asarray(train_x, dtype=np.float64)
    validation = np.asarray(validation_x, dtype=np.float64)
    test = np.asarray(test_x, dtype=np.float64)
    train_ref = _subsample_rows(train, reference_max_points, seed + 21)
    validation_ref = _subsample_rows(validation, reference_max_points, seed + 22)
    test_ref = _subsample_rows(test, reference_max_points, seed + 23)

    mean_for_summary = generated if x_mean is None else np.asarray(x_mean, dtype=np.float64)
    diagnostics: dict[str, Any] = {
        "n_generated": int(generated.shape[0]),
        "generation_mode": str(generation_mode),
        "seed": int(seed),
        "reference_max_points": int(reference_max_points),
        "x_dim": int(generated.shape[1]),
        "train": {
            "n_points": int(train.shape[0]),
            "summary_points": int(train_ref.shape[0]),
            "per_dim": _per_dim_summary(train_ref),
            "corr": _corr_summary(train_ref),
        },
        "validation": {
            "n_points": int(validation.shape[0]),
            "summary_points": int(validation_ref.shape[0]),
            "per_dim": _per_dim_summary(validation_ref),
            "corr": _corr_summary(validation_ref),
        },
        "test": {
            "n_points": int(test.shape[0]),
            "summary_points": int(test_ref.shape[0]),
            "per_dim": _per_dim_summary(test_ref),
            "corr": _corr_summary(test_ref),
        },
        "generated_mean": {
            "per_dim": _per_dim_summary(mean_for_summary),
            "corr": _corr_summary(mean_for_summary),
            "two_sample_vs_train": _two_sample_metrics(train_ref, mean_for_summary),
            "two_sample_vs_validation": _two_sample_metrics(validation_ref, mean_for_summary),
            "two_sample_vs_test": _two_sample_metrics(test_ref, mean_for_summary),
        },
        "generated_sample": {
            "per_dim": _per_dim_summary(generated),
            "corr": _corr_summary(generated),
            "two_sample_vs_train": _two_sample_metrics(train_ref, generated),
            "two_sample_vs_validation": _two_sample_metrics(validation_ref, generated),
            "two_sample_vs_test": _two_sample_metrics(test_ref, generated),
        },
    }
    if sigma_square is not None:
        sigma_array = np.asarray(sigma_square, dtype=np.float64)
        diagnostics["generator_sigma_square"] = _per_dim_summary(sigma_array)
        diagnostics["generator_sigma_square_global"] = _global_summary(sigma_array)
    if low_rank_u is not None:
        low_rank_array = np.asarray(low_rank_u, dtype=np.float64)
        diagnostics["generator_low_rank_u_global"] = _global_summary(low_rank_array)
        diagnostics["generator_low_rank_cov_diag_global"] = _global_summary(np.sum(np.square(low_rank_array), axis=-1))
    if extra_metadata:
        diagnostics["metadata"] = dict(extra_metadata)

    save_payload = {
        "x_gen_sample": generated.astype(np.float32),
        "x_train_reference": train_ref.astype(np.float32),
        "x_validation_reference": validation_ref.astype(np.float32),
        "x_test_reference": test_ref.astype(np.float32),
    }
    if z_samples is not None:
        save_payload["z_samples"] = np.asarray(z_samples, dtype=np.float32)
    if x_mean is not None:
        save_payload["x_gen_mean"] = np.asarray(x_mean, dtype=np.float32)
    if sigma_square is not None:
        save_payload["sigma_square_x"] = np.asarray(sigma_square, dtype=np.float32)
    if low_rank_u is not None:
        save_payload["low_rank_u"] = np.asarray(low_rank_u, dtype=np.float32)
    np.savez_compressed(run_path / "generated_samples.npz", **save_payload)

    _save_generic_marginal_overlay_plot(
        reference=train_ref,
        generated=generated,
        out_path=run_path / "fig_marginals_generated_sample_vs_train.png",
        title=f"{generation_mode} vs UCI train marginals",
        reference_label="train",
        generated_label="generated",
    )
    _save_generic_marginal_overlay_plot(
        reference=test_ref,
        generated=generated,
        out_path=run_path / "fig_marginals_generated_sample_vs_test.png",
        title=f"{generation_mode} vs UCI test marginals",
        reference_label="test",
        generated_label="generated",
    )
    if umap_enabled:
        diagnostics["umap"] = _save_uci_umap_plot(
            train=train_ref,
            validation=validation_ref,
            test=test_ref,
            generated=generated,
            out_dir=run_path,
            dataset=str(config.get("data", {}).get("name", "UCI")),
            seed=seed + 31,
            max_points_per_group=umap_max_points,
            n_neighbors=umap_neighbors,
            min_dist=umap_min_dist,
        )
    else:
        diagnostics["umap"] = {"enabled": False}

    with open(run_path / "generation_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, default=_json_default)
    return diagnostics


def run_bgm_generation_diagnostics(
    *,
    model: Any,
    sampler: Any,
    run_dir: Path,
    config: Mapping[str, Any],
    seed: int = 20260528,
) -> dict[str, Any]:
    diag_cfg = config.get("generation_diagnostics", {})
    n_samples = int(diag_cfg.get("n_samples", min(5000, len(sampler.X_train) + len(sampler.X_val))))
    density_max_points = diag_cfg.get("density_max_points")
    density_max_points = int(density_max_points) if density_max_points is not None else None
    density_points_per_summary_cfg = diag_cfg.get("density_max_points_per_summary")
    if density_points_per_summary_cfg is not None:
        density_points_per_summary = int(density_points_per_summary_cfg)
    else:
        density_points_per_summary = _density_points_per_summary(density_max_points, 4)
    umap_enabled = bool(diag_cfg.get("umap_enabled", True))
    plots_enabled = bool(diag_cfg.get("plots_enabled", True))
    umap_max_points = int(diag_cfg.get("umap_max_points_per_group", 1500))
    umap_neighbors = int(diag_cfg.get("umap_n_neighbors", 30))
    umap_min_dist = float(diag_cfg.get("umap_min_dist", 0.1))
    z_dim = int(config["model"]["z_dim"])
    x_dim = int(config.get("model", {}).get("x_dim", config.get("data", {}).get("dim", 0)))
    rng = np.random.default_rng(seed)
    z_samples = rng.normal(size=(n_samples, z_dim)).astype(np.float32)

    z_tensor = tf.convert_to_tensor(z_samples, dtype=tf.float32)
    decoder_outputs = model.g_net(z_tensor, training=False)
    if not isinstance(decoder_outputs, (tuple, list)):
        raise ValueError("BGM generator diagnostics require (mu, var) or (mu, var, U).")
    if len(decoder_outputs) == 2:
        mu_x, sigma_square_x = decoder_outputs
        low_rank_u = None
    elif len(decoder_outputs) == 3:
        mu_x, sigma_square_x, low_rank_u = decoder_outputs
    else:
        raise ValueError("BGM generator diagnostics require (mu, var) or (mu, var, U).")
    x_mean = np.asarray(mu_x.numpy(), dtype=np.float64)
    sigma_square = np.asarray(sigma_square_x.numpy(), dtype=np.float64)

    eps = rng.normal(size=x_mean.shape).astype(np.float64)
    x_sample = x_mean + eps * np.sqrt(np.maximum(sigma_square, 0.0))
    low_rank_array = None
    if low_rank_u is not None:
        low_rank_array = np.asarray(low_rank_u.numpy(), dtype=np.float64)
        eps_low_rank = rng.normal(size=(x_mean.shape[0], low_rank_array.shape[-1])).astype(np.float64)
        x_sample = x_sample + np.einsum("ndr,nr->nd", low_rank_array, eps_low_rank)

    x_train = np.asarray(sampler.X_train, dtype=np.float64)
    x_val = np.asarray(getattr(sampler, "X_val", sampler.X_test), dtype=np.float64)
    x_test = np.asarray(sampler.X_test, dtype=np.float64)
    plot_reference_split = str(diag_cfg.get("plot_reference_split", "test")).lower()
    if plot_reference_split == "validation":
        plot_reference = x_val
        plot_reference_label = "validation"
    elif plot_reference_split == "train":
        plot_reference = x_train
        plot_reference_label = "train"
    elif plot_reference_split == "test":
        plot_reference = x_test
        plot_reference_label = "test"
    else:
        raise ValueError(
            "generation_diagnostics.plot_reference_split must be one of "
            "{'train', 'validation', 'test'}."
        )
    if x_dim <= 0:
        x_dim = int(x_train.shape[1])
    x_train_density = _density_eval_subset(x_train, density_points_per_summary, seed + 101)
    x_val_density = _density_eval_subset(x_val, density_points_per_summary, seed + 105)
    x_test_density = _density_eval_subset(x_test, density_points_per_summary, seed + 102)
    x_mean_density = _density_eval_subset(x_mean, density_points_per_summary, seed + 103)
    x_sample_density = _density_eval_subset(x_sample, density_points_per_summary, seed + 104)
    centers_value = getattr(sampler, "centers", None)
    shared_marginal_centers = centers_value is not None and np.asarray(centers_value).ndim == 1
    centers = (
        np.asarray(centers_value, dtype=np.float64)
        if shared_marginal_centers
        else np.asarray([], dtype=np.float64)
    )
    data_sd = config.get("data", {}).get("sd")
    data_sd = float(data_sd) if data_sd is not None else None

    def mode_occupancy(values: np.ndarray) -> list[list[float]] | None:
        return _mode_occupancy(values, centers) if shared_marginal_centers else None

    def mode_residual(values: np.ndarray) -> dict[str, Any]:
        if shared_marginal_centers:
            return _mode_residual_summary(values, centers, data_sd)
        return {"supported": False, "reason": "joint_gmm_has_dimension_specific_component_locations"}

    diagnostics: dict[str, Any] = {
        "n_generated": int(n_samples),
        "z_dim": z_dim,
        "x_dim": int(x_dim),
        "seed": int(seed),
        "density_max_points": density_max_points,
        "density_points_per_summary": density_points_per_summary,
        "centers": centers.tolist(),
        "train": {
            "per_dim": _per_dim_summary(x_train),
            "density": _density_summary(x_train_density, sampler),
            "corr_frobenius_error_to_independent": _corr_error(x_train),
            "corr": _corr_summary(x_train),
            "mode_occupancy_per_dim": mode_occupancy(x_train),
            "mode_residual": mode_residual(x_train),
        },
        "test": {
            "per_dim": _per_dim_summary(x_test),
            "density": _density_summary(x_test_density, sampler),
            "corr_frobenius_error_to_independent": _corr_error(x_test),
            "corr": _corr_summary(x_test),
            "mode_occupancy_per_dim": mode_occupancy(x_test),
            "mode_residual": mode_residual(x_test),
        },
        "validation": {
            "per_dim": _per_dim_summary(x_val),
            "density": _density_summary(x_val_density, sampler),
            "corr_frobenius_error_to_independent": _corr_error(x_val),
            "corr": _corr_summary(x_val),
            "mode_occupancy_per_dim": mode_occupancy(x_val),
            "mode_residual": mode_residual(x_val),
        },
        "generated_mean": {
            "finite_value_fraction": float(np.mean(np.isfinite(x_mean))),
            "per_dim": _per_dim_summary(x_mean),
            "density": _density_summary(x_mean_density, sampler),
            "corr_frobenius_error_to_independent": _corr_error(x_mean),
            "corr": _corr_summary(x_mean),
            "mode_occupancy_per_dim": mode_occupancy(x_mean),
            "mode_residual": mode_residual(x_mean),
            "two_sample_vs_train": _two_sample_metrics(x_train, x_mean),
            "two_sample_vs_validation": _two_sample_metrics(x_val, x_mean),
            "two_sample_vs_test": _two_sample_metrics(x_test, x_mean),
        },
        "generated_sample": {
            "finite_value_fraction": float(np.mean(np.isfinite(x_sample))),
            "per_dim": _per_dim_summary(x_sample),
            "density": _density_summary(x_sample_density, sampler),
            "corr_frobenius_error_to_independent": _corr_error(x_sample),
            "corr": _corr_summary(x_sample),
            "mode_occupancy_per_dim": mode_occupancy(x_sample),
            "mode_residual": mode_residual(x_sample),
            "two_sample_vs_train": _two_sample_metrics(x_train, x_sample),
            "two_sample_vs_validation": _two_sample_metrics(x_val, x_sample),
            "two_sample_vs_test": _two_sample_metrics(x_test, x_sample),
        },
        "generator_sigma_square": _per_dim_summary(sigma_square),
        "generator_sigma_square_global": _global_summary(sigma_square),
    }
    if low_rank_array is not None:
        diagnostics["generator_low_rank_u_global"] = _global_summary(low_rank_array)
        diagnostics["generator_low_rank_cov_diag_global"] = _global_summary(
            np.sum(np.square(low_rank_array), axis=-1)
        )

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    save_payload = {
        "z_samples": z_samples,
        "x_gen_mean": x_mean,
        "x_gen_sample": x_sample,
        "sigma_square_x": sigma_square,
    }
    if low_rank_array is not None:
        save_payload["low_rank_u"] = low_rank_array
    np.savez(run_path / "generated_samples.npz", **save_payload)
    dims_per_page = int(diag_cfg.get("marginal_plot_dims_per_page", 10))
    if dims_per_page < 1:
        raise ValueError("generation_diagnostics.marginal_plot_dims_per_page must be positive.")
    plot_manifest: dict[str, Any] = {
        "x_dim": int(x_dim),
        "dims_per_page": dims_per_page,
        "plots_enabled": plots_enabled,
        "pages": [],
        "covered_dimensions": [],
    }
    if plots_enabled:
        for start_dim in range(0, int(x_dim), dims_per_page):
            end_dim = min(start_dim + dims_per_page, int(x_dim))
            suffix = f"page_{start_dim // dims_per_page + 1:02d}_dims_{start_dim:03d}_{end_dim - 1:03d}"
            mean_name = f"fig_marginals_generated_mean_vs_{plot_reference_label}_{suffix}.png"
            sample_name = f"fig_marginals_generated_sample_vs_{plot_reference_label}_{suffix}.png"
            common = {
                "reference": plot_reference,
                "max_dims": dims_per_page,
                "start_dim": start_dim,
            }
            if shared_marginal_centers:
                _save_marginal_overlay_plot(
                    **common, generated=x_mean, centers=centers, sd=data_sd,
                    out_path=run_path / mean_name,
                    title=f"Generated mean vs {plot_reference_label}: dims {start_dim}-{end_dim - 1}",
                )
                _save_marginal_overlay_plot(
                    **common, generated=x_sample, centers=centers, sd=data_sd,
                    out_path=run_path / sample_name,
                    title=f"Generated sample vs {plot_reference_label}: dims {start_dim}-{end_dim - 1}",
                )
            else:
                _save_generic_marginal_overlay_plot(
                    **common, generated=x_mean, out_path=run_path / mean_name,
                    title=f"Generated mean vs {plot_reference_label}: dims {start_dim}-{end_dim - 1}",
                    reference_label=plot_reference_label, generated_label="generated mean",
                )
                _save_generic_marginal_overlay_plot(
                    **common, generated=x_sample, out_path=run_path / sample_name,
                    title=f"Generated sample vs {plot_reference_label}: dims {start_dim}-{end_dim - 1}",
                    reference_label=plot_reference_label, generated_label="generated sample",
                )
            dimensions = list(range(start_dim, end_dim))
            plot_manifest["covered_dimensions"].extend(dimensions)
            plot_manifest["pages"].append({
                "page": start_dim // dims_per_page + 1,
                "dimensions": dimensions,
                "generated_mean_figure": mean_name,
                "generated_sample_figure": sample_name,
            })
        if int(x_dim) <= int(diag_cfg.get("all_dimensions_sheet_max", 50)):
            all_mean_name = f"fig_marginals_all_{x_dim}d_generated_mean_vs_exact.png"
            all_sample_name = f"fig_marginals_all_{x_dim}d_generated_vs_exact.png"
            if shared_marginal_centers:
                _save_marginal_overlay_plot(
                    reference=plot_reference, generated=x_mean, centers=centers, sd=data_sd,
                    out_path=run_path / all_mean_name,
                    title=f"{x_dim}D BGM generation mean: all marginals vs exact mixture PDF",
                    max_dims=None,
                )
                _save_marginal_overlay_plot(
                    reference=plot_reference, generated=x_sample, centers=centers, sd=data_sd,
                    out_path=run_path / all_sample_name,
                    title=f"{x_dim}D BGM generation: all marginals vs exact mixture PDF",
                    max_dims=None,
                )
            else:
                _save_generic_marginal_overlay_plot(
                    reference=plot_reference, generated=x_mean, out_path=run_path / all_mean_name,
                    title=f"{x_dim}D BGM generation mean: all marginals vs {plot_reference_label}",
                    reference_label=plot_reference_label, generated_label="generated mean", max_dims=None,
                )
                _save_generic_marginal_overlay_plot(
                    reference=plot_reference, generated=x_sample, out_path=run_path / all_sample_name,
                    title=f"{x_dim}D BGM generation: all marginals vs {plot_reference_label}",
                    reference_label=plot_reference_label, generated_label="generated sample", max_dims=None,
                )
            plot_manifest["all_dimensions_sheet"] = {
                "dimensions": list(range(int(x_dim))),
                "generated_mean_figure": all_mean_name,
                "generated_sample_figure": all_sample_name,
            }
    plot_manifest["coverage_complete"] = plot_manifest["covered_dimensions"] == list(range(int(x_dim))) if plots_enabled else False
    diagnostics["marginal_plot_manifest"] = plot_manifest
    with open(run_path / "marginal_plot_manifest.json", "w", encoding="utf-8") as f:
        json.dump(plot_manifest, f, indent=2, default=_json_default)
    if plots_enabled and umap_enabled:
        try:
            umap_metadata = _save_uci_umap_plot(
                train=x_train,
                validation=x_val,
                test=x_test,
                generated=x_sample,
                out_dir=run_path,
                dataset=str(config.get("data", {}).get("name", "simulation")),
                seed=int(seed) + 777,
                max_points_per_group=int(umap_max_points),
                n_neighbors=int(umap_neighbors),
                min_dist=float(umap_min_dist),
            )
            diagnostics["umap_generated_sample"] = {
                **umap_metadata,
                "figure": str(run_path / "fig_umap_generated_vs_real.png"),
                "embedding": str(run_path / "umap_embedding_generated_vs_real.npz"),
            }
        except Exception as exc:
            diagnostics["umap_generated_sample"] = {
                "enabled": True,
                "failed": True,
                "error": repr(exc),
            }
    with open(run_path / "generation_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, default=_json_default)
    return diagnostics


def run_deterministic_generator_diagnostics(
    *,
    model: Any,
    sampler: Any,
    run_dir: Path,
    config: Mapping[str, Any],
    seed: int = 20260528,
) -> dict[str, Any]:
    diag_cfg = config.get("generation_diagnostics", {})
    n_samples = int(diag_cfg.get("n_samples", min(5000, len(sampler.X_train) + len(sampler.X_val))))
    density_max_points = diag_cfg.get("density_max_points")
    density_max_points = int(density_max_points) if density_max_points is not None else None
    density_points_per_summary = _density_points_per_summary(density_max_points, 3)
    model_cfg = config.get("model", {})
    z_dim = int(model_cfg["z_dim"])
    x_dim = int(model_cfg.get("x_dim", config.get("data", {}).get("dim", sampler.X_train.shape[1])))
    rng = np.random.default_rng(seed)
    z_samples = rng.normal(size=(n_samples, z_dim)).astype(np.float32)

    z_tensor = tf.convert_to_tensor(z_samples, dtype=tf.float32)
    x_gen = np.asarray(model.g_net(z_tensor, training=False).numpy(), dtype=np.float64)

    x_train = np.asarray(sampler.X_train, dtype=np.float64)
    x_test = np.asarray(sampler.X_test, dtype=np.float64)
    x_train_density = _density_eval_subset(x_train, density_points_per_summary, seed + 101)
    x_test_density = _density_eval_subset(x_test, density_points_per_summary, seed + 102)
    x_gen_density = _density_eval_subset(x_gen, density_points_per_summary, seed + 103)
    centers = np.asarray(getattr(sampler, "centers", np.linspace(-1.0, 1.0, 3)), dtype=np.float64)
    data_sd = config.get("data", {}).get("sd")
    data_sd = float(data_sd) if data_sd is not None else None

    generated_summary = {
        "per_dim": _per_dim_summary(x_gen),
        "density": _density_summary(x_gen_density, sampler),
        "corr_frobenius_error_to_independent": _corr_error(x_gen),
        "corr": _corr_summary(x_gen),
        "mode_occupancy_per_dim": _mode_occupancy(x_gen, centers),
        "mode_residual": _mode_residual_summary(x_gen, centers, data_sd),
        "two_sample_vs_train": _two_sample_metrics(x_train, x_gen),
        "two_sample_vs_test": _two_sample_metrics(x_test, x_gen),
    }

    diagnostics: dict[str, Any] = {
        "n_generated": int(n_samples),
        "z_dim": z_dim,
        "x_dim": x_dim,
        "seed": int(seed),
        "density_max_points": density_max_points,
        "density_points_per_summary": density_points_per_summary,
        "centers": centers.tolist(),
        "generator_type": "deterministic",
        "train": {
            "per_dim": _per_dim_summary(x_train),
            "density": _density_summary(x_train_density, sampler),
            "corr_frobenius_error_to_independent": _corr_error(x_train),
            "corr": _corr_summary(x_train),
            "mode_occupancy_per_dim": _mode_occupancy(x_train, centers),
            "mode_residual": _mode_residual_summary(x_train, centers, data_sd),
        },
        "test": {
            "per_dim": _per_dim_summary(x_test),
            "density": _density_summary(x_test_density, sampler),
            "corr_frobenius_error_to_independent": _corr_error(x_test),
            "corr": _corr_summary(x_test),
            "mode_occupancy_per_dim": _mode_occupancy(x_test, centers),
            "mode_residual": _mode_residual_summary(x_test, centers, data_sd),
        },
        "generated_mean": generated_summary,
        "generated_sample": generated_summary,
        "generator_sigma_square": None,
    }

    run_path = Path(run_dir)
    np.savez(
        run_path / "generated_samples.npz",
        z_samples=z_samples,
        x_gen_mean=x_gen,
        x_gen_sample=x_gen,
    )
    _save_marginal_overlay_plot(
        reference=x_test,
        generated=x_gen,
        centers=centers,
        sd=data_sd,
        out_path=run_path / "fig_marginals_generated_vs_test.png",
        title="Generated vs exact marginal distributions",
    )
    with open(run_path / "generation_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, default=_json_default)
    return diagnostics


LOWER_IS_BETTER = (
    "mmd_rbf",
    "sym_kl_mean",
    "sliced_wasserstein",
    "wasserstein_mean",
    "ks_stat_mean",
)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    ranks = np.empty_like(values, dtype=np.float64)
    for column in range(values.shape[1]):
        column_values = values[:, column]
        order = np.argsort(column_values, kind="mergesort")
        sorted_values = column_values[order]
        column_ranks = np.empty(len(column_values), dtype=np.float64)
        start = 0
        while start < len(sorted_values):
            stop = start + 1
            while stop < len(sorted_values) and sorted_values[stop] == sorted_values[start]:
                stop += 1
            column_ranks[order[start:stop]] = 0.5 * (start + stop - 1)
            start = stop
        ranks[:, column] = column_ranks / max(len(column_values) - 1, 1)
    return ranks


def generation_rank_features(
    two_sample: Mapping[str, Any],
    dim: int,
    std_ratio_fraction: float | None = None,
) -> dict[str, float] | None:
    try:
        record = {name: float(two_sample[name]) for name in LOWER_IS_BETTER}
        record["corr_error_per_sqrt_dim"] = float(
            two_sample["corr_matrix_frobenius_error"]
        ) / np.sqrt(float(dim))
        record["one_minus_lisi"] = 1.0 - float(two_sample["lisi_normalized_mean"])
    except (KeyError, TypeError, ValueError):
        return None
    if std_ratio_fraction is not None:
        record["one_minus_std_ratio_fraction"] = 1.0 - float(std_ratio_fraction)
    if not all(np.isfinite(v) for v in record.values()):
        return None
    return record


def rank_candidates(candidates: Sequence[Mapping[str, float]]) -> np.ndarray:
    if not candidates:
        raise ValueError("No candidates to rank.")
    names = sorted(set.intersection(*(set(c) for c in candidates)))
    names = [n for n in names if all(isinstance(c[n], (int, float)) for c in candidates)]
    if not names:
        raise ValueError("Candidates share no numeric diagnostic columns.")
    values = np.asarray([[float(c[n]) for n in names] for c in candidates], dtype=np.float64)
    return _average_ranks(values).mean(axis=1)


def std_ratio_fraction(payload: Mapping[str, Any], section: str = "generated_sample") -> float | None:
    try:
        generated = np.asarray(payload[section]["per_dim"]["std"], dtype=np.float64)
        validation = np.asarray(payload["validation"]["per_dim"]["std"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    if generated.shape != validation.shape or generated.size == 0:
        return None
    ratios = generated / np.maximum(validation, 1.0e-12)
    if not np.all(np.isfinite(ratios)):
        return None
    return float(np.mean((ratios >= 0.75) & (ratios <= 1.25)))


def load_generation_diagnostics(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["generated_sample"]["two_sample_vs_validation"]


def load_candidates(
    directories: Iterable[str | Path],
    key: str,
    **extra: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for directory in sorted(Path(d) for d in directories):
        path = directory / "generation_diagnostics.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        generated = payload.get("generated_sample", payload)
        two_sample = generated.get("two_sample_vs_validation")
        if two_sample is None:
            continue
        digits = "".join(c for c in directory.name if c.isdigit())
        if not digits:
            continue
        rows.append({
            key: int(digits),
            "two_sample_vs_validation": two_sample,
            "std_ratio_fraction": std_ratio_fraction(payload),
            "diagnostics_path": str(path.resolve()),
            **extra,
        })
    return rows


def select_pooled(
    candidates: Sequence[Mapping[str, Any]],
    dim: int,
    *,
    key: str = "epoch",
) -> dict[str, Any]:
    scored: list[tuple[dict[str, float], Mapping[str, Any]]] = []
    for candidate in candidates:
        if int(candidate.get(key, 0)) <= 0:
            continue
        features = generation_rank_features(
            candidate["two_sample_vs_validation"], dim, candidate.get("std_ratio_fraction")
        )
        if features is not None:
            scored.append((features, candidate))
    if not scored:
        raise RuntimeError("No selectable checkpoint: every candidate had a missing or non-finite diagnostic.")
    scores = rank_candidates([f for f, _ in scored])
    best = int(np.lexsort((np.asarray([int(c[key]) for _, c in scored]), scores))[0])
    ranked = sorted(
        ({**{k: v for k, v in c.items() if k != "two_sample_vs_validation"},
          "score": float(s)} for (_, c), s in zip(scored, scores)),
        key=lambda row: row["score"],
    )
    return {
        "criterion": "minimum equal-weight average percentile rank over two-sample generation diagnostics",
        "lower_is_better": True,
        "uses_true_density": False,
        "uses_known_mixture_centres": False,
        "pooled_candidates": len(scored),
        "metrics": sorted(scored[best][0]),
        key: int(scored[best][1][key]),
        "selected": ranked[0],
        "score": float(scores[best]),
        "all_scores": ranked,
    }
