"""Run bridge-sampling density evaluation on a frozen BGM checkpoint."""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]

from bayesgm.models import BGM
from bayesnde.training.generation_optimized import build_bgm_model
from bayesnde.estimators.bridge import BGM_BridgeDensityEstimator
from bayesnde.diagnostics.generation import run_bgm_generation_diagnostics
from bayesnde.evaluation.grid import (
    create_2d_slice_grid,
    remove_artifacts,
    save_dataset_fingerprint,
    validate_checkpoint_matches_config,
    validate_indep_gmm_dimensions,
)
from bayesnde.data.samplers import (
    build_sampler as build_simulation_sampler,
    density_calibration_metrics,
    masked_spearman_metrics,
    safe_log_density,
)
from bayesnde.evaluation.metrics import bootstrap_spearman_ci


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Path, data: Mapping[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(data), f, sort_keys=False)


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(path: Path, data: Mapping[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=json_default)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bind_density_cache_to_run(
    *,
    cache_path: Path,
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    weights = {}
    for key in ("generator_weights", "encoder_weights"):
        value = manifest.get(key)
        if value is None:
            continue
        path = Path(str(value)).resolve()
        weights[key] = {"path": str(path), "sha256": _file_sha256(path)}
    identity = {
        "schema_version": 1,
        "weights": weights,
        "density_bridge_eval": config.get("density_bridge_eval", {}),
        "hmc_settings": config.get("hmc_settings", {}),
    }
    identity_path = cache_path.with_name(cache_path.stem + "_identity.json")
    if cache_path.exists():
        if not identity_path.exists():
            raise RuntimeError(
                f"Refusing to reuse legacy density cache without checkpoint identity: {cache_path}. "
                "Use a new output directory."
            )
        recorded = json.loads(identity_path.read_text(encoding="utf-8"))
        if recorded != identity:
            raise RuntimeError(
                f"Density cache identity does not match the requested checkpoint/protocol: {cache_path}. "
                "Use a new output directory."
            )
    else:
        save_json(identity_path, identity)


def generation_metric_summary(section: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    density = section.get("density", {})
    mode_residual = section.get("mode_residual", {})
    within_1sd = mode_residual.get("fraction_within_1sd_of_center") or []
    finite_within_1sd = [float(v) for v in within_1sd if v is not None and np.isfinite(float(v))]
    corr = section.get("corr", {})
    two_sample = section.get("two_sample_vs_test", {})
    return {
        f"{prefix}_mean_log_true_px": density.get("mean_log_true_px"),
        f"{prefix}_min_fraction_within_1sd": min(finite_within_1sd) if finite_within_1sd else None,
        f"{prefix}_mean_fraction_within_1sd": (
            float(np.mean(finite_within_1sd)) if finite_within_1sd else None
        ),
        f"{prefix}_corr_frobenius_error": section.get("corr_frobenius_error_to_independent"),
        f"{prefix}_max_abs_offdiag_corr": corr.get("max_abs_offdiag_corr"),
        f"{prefix}_mean_abs_offdiag_corr": corr.get("mean_abs_offdiag_corr"),
        f"{prefix}_rho_gt_0p1_pairs": corr.get("n_pairs_abs_corr_gt_threshold"),
        f"{prefix}_wasserstein_mean": two_sample.get("wasserstein_mean"),
        f"{prefix}_wasserstein_max": two_sample.get("wasserstein_max"),
        f"{prefix}_ks_stat_mean": two_sample.get("ks_stat_mean"),
        f"{prefix}_ks_stat_max": two_sample.get("ks_stat_max"),
    }


def deep_update(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def resolve_path(path_value: str | Path, base_dir: Path = REPO_ROOT) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def materialize_visual_config(config: Dict[str, Any], preset: str) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    presets = cfg.pop("presets", {})
    if presets:
        if preset not in presets:
            raise ValueError(f"Unknown preset={preset!r}. Available: {sorted(presets)}")
        preset_update = copy.deepcopy(presets[preset])
        preset_update.pop("model", None)
        preset_update.pop("training", None)
        deep_update(cfg, preset_update)
    cfg["preset"] = preset
    return cfg


def merge_bridge_config(visual_cfg: Dict[str, Any], bridge_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(visual_cfg)
    model_setting = copy.deepcopy(bridge_cfg.get("model_setting", {}))

    if model_setting.get("params_override"):
        cfg.setdefault("model", {})
        deep_update(cfg["model"], model_setting["params_override"])
    if model_setting.get("output_dir"):
        cfg.setdefault("model", {})
        cfg["model"]["output_dir"] = model_setting["output_dir"]

    cfg["model_setting"] = model_setting
    cfg["hmc_settings"] = copy.deepcopy(bridge_cfg.get("hmc_settings", cfg.get("hmc_settings", {})))
    cfg["density_bridge_eval"] = copy.deepcopy(bridge_cfg.get("density_bridge_eval", {}))
    cfg["density_eval"] = copy.deepcopy(bridge_cfg.get("density_eval", {}))
    cfg["run_management"] = copy.deepcopy(bridge_cfg.get("run_management", {}))
    visual_density_cfg = cfg.get("density", {})
    for key in (
        "K",
        "S",
        "nu",
        "epsilon",
        "n_repeats",
        "eval_batch_size",
        "likelihood",
        "variance_floor",
        "seed",
        "save_every",
    ):
        if key in visual_density_cfg:
            cfg["density_bridge_eval"][key] = copy.deepcopy(visual_density_cfg[key])
    if "hmc_settings" in visual_cfg:
        deep_update(cfg["hmc_settings"], visual_cfg["hmc_settings"])
    cfg.setdefault("outputs", {})
    cfg["outputs"].setdefault("root_dir", "outputs/bridge_indep_gmm")
    cfg["outputs"].setdefault("cmap_density", "Blues")
    cfg["outputs"].setdefault("cmap_log_px", "viridis")
    cfg["outputs"].setdefault("cmap_ess", "cividis")
    cfg["outputs"].setdefault("cmap_iter", "plasma")
    cfg["outputs"].setdefault("cmap_converged", "Greens")
    return cfg


def validate_bridge_config(config: Dict[str, Any]) -> None:
    density_cfg = config.setdefault("density_bridge_eval", {})
    hmc_cfg = config.setdefault("hmc_settings", {})

    required_density_keys = ("K", "S", "nu", "epsilon", "tol", "max_iter", "fit_fraction", "use_neff", "n_repeats")
    required_hmc_keys = (
        "M",
        "burn_in",
        "step_size",
        "num_leapfrog_steps",
        "target_accept_prob",
        "num_chains",
        "initial_state",
        "initial_state_scale",
        "min_effective_samples",
        "max_retries",
        "retry_multiplier",
        "strict_ess",
        "proposal_covariance_floor",
    )
    missing_density = [key for key in required_density_keys if key not in density_cfg]
    missing_hmc = [key for key in required_hmc_keys if key not in hmc_cfg]
    if missing_density or missing_hmc:
        raise ValueError(
            "bridge config is missing required keys: "
            f"density_bridge_eval={missing_density}, hmc_settings={missing_hmc}"
        )

    density_cfg["K"] = int(density_cfg["K"])
    density_cfg["S"] = int(density_cfg["S"])
    density_cfg["n_repeats"] = int(density_cfg["n_repeats"])
    density_cfg["max_iter"] = int(density_cfg["max_iter"])
    hmc_cfg["M"] = int(hmc_cfg["M"])
    hmc_cfg["burn_in"] = int(hmc_cfg["burn_in"])
    hmc_cfg["num_leapfrog_steps"] = int(hmc_cfg["num_leapfrog_steps"])
    hmc_cfg["num_chains"] = int(hmc_cfg["num_chains"])
    hmc_cfg["min_effective_samples"] = int(hmc_cfg["min_effective_samples"])
    hmc_cfg["max_retries"] = int(hmc_cfg["max_retries"])

    fit_fraction = float(density_cfg.get("fit_fraction", 0.5))
    if not (0.0 < fit_fraction < 1.0):
        raise ValueError("density_bridge_eval.fit_fraction must be strictly between 0 and 1.")
    density_cfg["fit_fraction"] = fit_fraction
    density_cfg["effective_fit_budget"] = int(round(hmc_cfg["M"] * fit_fraction))
    covariance_mode = str(density_cfg.get("proposal_covariance_mode", "fitted_full"))
    if covariance_mode not in {"fitted_full", "fixed_isotropic"}:
        raise ValueError("density_bridge_eval.proposal_covariance_mode is invalid.")
    density_cfg["proposal_covariance_mode"] = covariance_mode
    if covariance_mode == "fixed_isotropic":
        scale = float(density_cfg.get("proposal_scale", 0.0))
        if scale <= 0.0:
            raise ValueError("fixed_isotropic requires a positive density_bridge_eval.proposal_scale.")
        density_cfg["proposal_scale"] = scale


def require_gpu(config: Mapping[str, Any]) -> list[tf.config.PhysicalDevice]:
    gpu_cfg = config.get("gpu", {})
    gpu_required = bool(gpu_cfg.get("required", True))

    gpus = tf.config.list_physical_devices("GPU")
    if not gpu_required:
        if gpus and bool(gpu_cfg.get("memory_growth", True)):
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        return gpus

    if not gpus:
        build_info = tf.sysconfig.get_build_info()
        raise RuntimeError(
            "GPU is required, but TensorFlow sees no GPU. Stopping before inference.\n"
            f"TensorFlow version: {tf.__version__}\n"
            f"CUDA build info: {build_info}\n"
            "Check the active Python environment, CUDA/cuDNN libraries, and NVIDIA driver visibility."
        )

    if bool(gpu_cfg.get("memory_growth", True)):
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    return gpus


def build_sampler(data_cfg: Mapping[str, Any]) -> Any:
    return build_simulation_sampler(data_cfg)


def load_cached_truth_iid_test_points(
    path: Path,
    *,
    test_points: np.ndarray,
    test_indices: np.ndarray,
) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        with np.load(path) as data:
            if not {"x_test", "test_indices", "px"}.issubset(data.files):
                return None
            cached_x = np.asarray(data["x_test"])
            cached_indices = np.asarray(data["test_indices"], dtype=np.int64)
            cached_px = np.asarray(data["px"], dtype=np.float64)
    except Exception:
        return None
    n_points = int(len(test_indices))
    if cached_x.shape[0] < n_points or cached_indices.shape[0] < n_points or cached_px.shape[0] < n_points:
        return None
    cached_x = cached_x[:n_points]
    cached_indices = cached_indices[:n_points]
    cached_px = cached_px[:n_points]
    if cached_x.shape != test_points.shape:
        return None
    if not np.array_equal(cached_indices, np.asarray(test_indices, dtype=np.int64)):
        return None
    if not np.allclose(cached_x.astype(np.float32), np.asarray(test_points, dtype=np.float32), rtol=0.0, atol=1.0e-6):
        return None
    if not np.all(np.isfinite(cached_px)):
        return None
    return cached_px.astype(np.float64, copy=True)


def build_model_params(config: Mapping[str, Any], run_dir: Path) -> Dict[str, Any]:
    params = copy.deepcopy(dict(config["model"]))
    if "x_dim" not in params:
        params["x_dim"] = int(config["data"]["dim"])
    params["use_bnn"] = False
    params["save_model"] = False
    params["save_res"] = False
    params["output_dir"] = str(run_dir)
    return params


def load_frozen_model(
    config: Mapping[str, Any],
    run_dir: Path,
    checkpoint_dir: Path,
    epoch: Optional[int],
    egm_iter: Optional[int],
) -> tuple[BGM, Dict[str, Any]]:
    params = build_model_params(config, run_dir)
    seed = int(config.get("data", {}).get("seed", 1024))
    density_cfg = config["density_bridge_eval"]
    model = build_bgm_model(params=params, random_seed=seed)
    estimator = BGM_BridgeDensityEstimator(
        model=model,
        likelihood=str(density_cfg.get("likelihood", "gaussian")),
        random_seed=int(density_cfg.get("seed", 42)),
        variance_floor=float(density_cfg.get("variance_floor", 1e-6)),
        fixed_likelihood_variance=density_cfg.get("fixed_likelihood_variance"),
    )
    load_encoder = bool(config.get("model_setting", {}).get("load_encoder", True))
    manifest = estimator.load_weights_from_run(
        checkpoint_dir=checkpoint_dir,
        epoch=epoch,
        egm_iter=egm_iter,
        load_encoder=load_encoder,
    )
    if str(config["hmc_settings"].get("initial_state", "encoder")) == "encoder":
        if "encoder_weights" not in manifest:
            raise FileNotFoundError(
                "HMC initial_state is 'encoder', but no encoder weights were loaded from checkpoint_dir."
            )
    return model, manifest


def make_estimator(model: BGM, config: Mapping[str, Any]) -> BGM_BridgeDensityEstimator:
    density_cfg = config["density_bridge_eval"]
    return BGM_BridgeDensityEstimator(
        model=model,
        likelihood=str(density_cfg.get("likelihood", "gaussian")),
        random_seed=int(density_cfg.get("seed", 42)),
        variance_floor=float(density_cfg.get("variance_floor", 1e-6)),
        fixed_likelihood_variance=density_cfg.get("fixed_likelihood_variance"),
    )


def save_partial_density(
    path: Path,
    v1: np.ndarray,
    v2: np.ndarray,
    grid: np.ndarray,
    log_px: np.ndarray,
    px: np.ndarray,
    log_px_sd: np.ndarray,
    proposal_is_ess_mean: np.ndarray,
    hmc_min_ess: np.ndarray,
    hmc_acceptance_rate: np.ndarray,
    bridge_iterations_mean: np.ndarray,
    bridge_convergence_rate_point: np.ndarray,
    bridge_abs_delta_mean: np.ndarray,
) -> None:
    np.savez(
        path,
        v1=v1,
        v2=v2,
        grid=grid,
        log_px=log_px,
        px=px,
        log_px_sd=log_px_sd,
        proposal_is_ess_mean=proposal_is_ess_mean,
        hmc_min_ess=hmc_min_ess,
        hmc_acceptance_rate=hmc_acceptance_rate,
        bridge_iterations_mean=bridge_iterations_mean,
        bridge_convergence_rate_point=bridge_convergence_rate_point,
        bridge_abs_delta_mean=bridge_abs_delta_mean,
    )


def load_existing_density(path: Path, n_points: int) -> Optional[Dict[str, np.ndarray]]:
    if not path.exists():
        return None
    with np.load(path) as data:
        if data["log_px"].shape[0] != n_points:
            return None
        return {key: data[key] for key in data.files}


def evaluate_grid_density(
    estimator: BGM_BridgeDensityEstimator,
    grid: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    config: Mapping[str, Any],
    output_path: Path,
) -> Dict[str, np.ndarray]:
    density_cfg = config["density_bridge_eval"]
    n_points = grid.shape[0]

    existing = load_existing_density(output_path, n_points)
    if existing is None:
        log_px = np.full(n_points, np.nan, dtype=np.float64)
        px = np.full(n_points, np.nan, dtype=np.float64)
        log_px_sd = np.full(n_points, np.nan, dtype=np.float64)
        proposal_is_ess_mean = np.full(n_points, np.nan, dtype=np.float64)
        hmc_min_ess = np.full(n_points, np.nan, dtype=np.float64)
        hmc_acceptance_rate = np.full(n_points, np.nan, dtype=np.float64)
        bridge_iterations_mean = np.full(n_points, np.nan, dtype=np.float64)
        bridge_convergence_rate_point = np.full(n_points, np.nan, dtype=np.float64)
        bridge_abs_delta_mean = np.full(n_points, np.nan, dtype=np.float64)
    else:
        log_px = existing["log_px"].copy()
        px = existing["px"].copy()
        log_px_sd = existing["log_px_sd"].copy()
        proposal_is_ess_mean = existing["proposal_is_ess_mean"].copy()
        hmc_min_ess = existing["hmc_min_ess"].copy()
        hmc_acceptance_rate = existing["hmc_acceptance_rate"].copy()
        bridge_iterations_mean = existing["bridge_iterations_mean"].copy()
        bridge_convergence_rate_point = existing["bridge_convergence_rate_point"].copy()
        bridge_abs_delta_mean = existing["bridge_abs_delta_mean"].copy()

    save_every = int(density_cfg.get("save_every", 25))
    completed = int(np.sum(np.isfinite(log_px)))
    start_time = time.time()

    with tqdm(
        total=n_points,
        initial=completed,
        desc="Bridge grid density",
        unit="pt",
        dynamic_ncols=True,
        ascii=True,
    ) as pbar:
        for idx, point in enumerate(grid):
            if np.isfinite(log_px[idx]):
                continue

            result = estimator.estimate(
                point.astype(np.float32),
                K=int(density_cfg.get("K", 5)),
                S=int(density_cfg["S"]),
                nu=float(density_cfg.get("nu", 3.0)),
                epsilon=float(density_cfg.get("epsilon", 0.05)),
                n_repeats=int(density_cfg["n_repeats"]),
                hmc_settings=config["hmc_settings"],
                eval_batch_size=int(density_cfg.get("eval_batch_size", 4096)),
                bridge_tol=float(density_cfg.get("tol", 1e-5)),
                bridge_max_iter=int(density_cfg.get("max_iter", 1000)),
                fit_fraction=float(density_cfg.get("fit_fraction", 0.5)),
                use_neff=bool(density_cfg.get("use_neff", True)),
                proposal_scale_multiplier=float(density_cfg.get("proposal_scale_multiplier", 1.0)),
                proposal_covariance_mode=str(density_cfg.get("proposal_covariance_mode", "fitted_full")),
                proposal_scale=(None if density_cfg.get("proposal_scale") is None else float(density_cfg["proposal_scale"])),
                proposal_center=str(density_cfg.get("proposal_center", "fitted_gmm")),
                proposal_scoring=str(density_cfg.get("proposal_scoring", "multivariate_t")),
                return_proposal=False,
            )

            diag = result["diagnostics"][0]
            bridge_iter = np.asarray(diag["bridge_iterations_repeats"], dtype=np.float64)
            bridge_conv = np.asarray(diag["bridge_converged_repeats"], dtype=bool)
            bridge_delta = np.asarray(diag["bridge_abs_delta_repeats"], dtype=np.float64)
            proposal_ess = np.asarray(diag["proposal_is_ess_repeats"], dtype=np.float64)

            log_px[idx] = float(result["log_px"])
            px[idx] = float(np.exp(log_px[idx])) if log_px[idx] < 700 else math.inf
            log_px_sd[idx] = float(result["log_px_repeat_sd"])
            proposal_is_ess_mean[idx] = float(np.mean(proposal_ess))
            hmc_min_ess[idx] = float(diag["hmc_min_ess"])
            hmc_acceptance_rate[idx] = float(diag["hmc_acceptance_rate"])
            bridge_iterations_mean[idx] = float(np.mean(bridge_iter))
            bridge_convergence_rate_point[idx] = float(np.mean(bridge_conv))
            bridge_abs_delta_mean[idx] = float(np.mean(bridge_delta))

            pbar.update(1)
            pbar.set_postfix(
                log_px=f"{log_px[idx]:.3f}",
                it=f"{bridge_iterations_mean[idx]:.1f}",
                conv=f"{bridge_convergence_rate_point[idx]:.2f}",
                ess=f"{proposal_is_ess_mean[idx]:.1f}",
            )

            done = int(np.sum(np.isfinite(log_px)))
            if done % save_every == 0 or done == n_points:
                elapsed = time.time() - start_time
                print(f"Evaluated {done}/{n_points} bridge grid points in {elapsed:.1f}s")
                save_partial_density(
                    output_path,
                    v1,
                    v2,
                    grid,
                    log_px,
                    px,
                    log_px_sd,
                    proposal_is_ess_mean,
                    hmc_min_ess,
                    hmc_acceptance_rate,
                    bridge_iterations_mean,
                    bridge_convergence_rate_point,
                    bridge_abs_delta_mean,
                )

    save_partial_density(
        output_path,
        v1,
        v2,
        grid,
        log_px,
        px,
        log_px_sd,
        proposal_is_ess_mean,
        hmc_min_ess,
        hmc_acceptance_rate,
        bridge_iterations_mean,
        bridge_convergence_rate_point,
        bridge_abs_delta_mean,
    )
    return {
        "log_px": log_px,
        "px": px,
        "log_px_sd": log_px_sd,
        "proposal_is_ess_mean": proposal_is_ess_mean,
        "hmc_min_ess": hmc_min_ess,
        "hmc_acceptance_rate": hmc_acceptance_rate,
        "bridge_iterations_mean": bridge_iterations_mean,
        "bridge_convergence_rate_point": bridge_convergence_rate_point,
        "bridge_abs_delta_mean": bridge_abs_delta_mean,
    }


def np_logmeanexp(values: np.ndarray, axis: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    max_values = np.nanmax(values, axis=axis, keepdims=True)
    shifted = np.exp(values - max_values)
    return np.squeeze(max_values, axis=axis) + np.log(np.nanmean(shifted, axis=axis))


def nanmean_or_none(values: np.ndarray) -> Optional[float]:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return None
    return float(np.mean(arr[finite]))


def nanmin_or_none(values: np.ndarray) -> Optional[float]:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return None
    return float(np.min(arr[finite]))


def summarize_repeat_log_likelihood(
    log_px_repeats: np.ndarray,
    finite_mask: np.ndarray,
    source: str,
) -> Dict[str, Any]:
    repeat_values = np.asarray(log_px_repeats, dtype=np.float64)[finite_mask]
    if repeat_values.ndim != 2 or repeat_values.shape[0] == 0:
        raise RuntimeError("No finite repeat log-likelihood values were available.")
    repeat_means = np.mean(repeat_values, axis=0)
    std = float(np.std(repeat_means, ddof=1)) if repeat_means.shape[0] > 1 else None
    return {
        "mean_log_px": float(np.mean(repeat_means)),
        "std_log_px": std,
        "std_log_px_status": "estimated" if repeat_means.shape[0] > 1 else "not_estimable_single_repeat",
        "n_repeats": int(repeat_means.shape[0]),
        "log_likelihood_source": source,
        "log_likelihood_points": int(repeat_values.shape[0]),
        "log_px_repeat_means": repeat_means.tolist(),
    }


def select_iid_test_points(
    x_test: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    test = np.asarray(x_test, dtype=np.float32)
    if test.ndim != 2:
        raise ValueError("x_test must have shape (n_test, x_dim).")
    density_cfg = config.get("density", {})
    requested = density_cfg.get("test_log_likelihood_points", min(256, len(test)))
    if isinstance(requested, str) and requested.lower() == "all":
        n_points = len(test)
    else:
        n_points = min(int(requested), len(test))
    if n_points < 1:
        raise ValueError("density.test_log_likelihood_points must be positive.")
    seed = int(density_cfg.get("test_log_likelihood_seed", int(config["data"]["seed"]) + 7919))
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(test), size=n_points, replace=False))
    return test[indices], indices.astype(np.int64)


def save_bridge_test_log_likelihood(
    path: Path,
    x_test: np.ndarray,
    test_indices: np.ndarray,
    sample_size: int,
    log_px_repeats: np.ndarray,
    log_px: np.ndarray,
    px: np.ndarray,
    log_px_sd: np.ndarray,
    proposal_is_ess_mean: np.ndarray,
    hmc_min_ess: np.ndarray,
    hmc_acceptance_rate: np.ndarray,
    bridge_iterations_mean: np.ndarray,
    bridge_convergence_rate_point: np.ndarray,
    bridge_abs_delta_mean: np.ndarray,
) -> None:
    np.savez(
        path,
        x_test=x_test,
        test_indices=test_indices,
        sample_size=int(sample_size),
        log_px_repeats=log_px_repeats,
        log_px=log_px,
        px=px,
        log_px_sd=log_px_sd,
        proposal_is_ess_mean=proposal_is_ess_mean,
        hmc_min_ess=hmc_min_ess,
        hmc_acceptance_rate=hmc_acceptance_rate,
        bridge_iterations_mean=bridge_iterations_mean,
        bridge_convergence_rate_point=bridge_convergence_rate_point,
        bridge_abs_delta_mean=bridge_abs_delta_mean,
    )


def load_existing_test_log_likelihood(
    path: Path,
    n_points: int,
    n_repeats: int,
    expected_x_test: Optional[np.ndarray] = None,
    expected_indices: Optional[np.ndarray] = None,
    expected_sample_size: Optional[int] = None,
) -> Optional[Dict[str, np.ndarray]]:
    if not path.exists():
        return None
    with np.load(path) as data:
        if "log_px_repeats" not in data.files:
            return None
        if data["log_px_repeats"].shape != (n_points, n_repeats):
            return None
        if expected_x_test is not None:
            if "x_test" not in data.files:
                return None
            x_test = data["x_test"]
            if x_test.shape != expected_x_test.shape or not np.allclose(x_test, expected_x_test):
                return None
        if expected_indices is not None:
            if "test_indices" not in data.files:
                return None
            if not np.array_equal(data["test_indices"], expected_indices):
                return None
        if expected_sample_size is not None:
            if "sample_size" not in data.files:
                return None
            if int(np.asarray(data["sample_size"]).item()) != int(expected_sample_size):
                return None
        return {key: data[key] for key in data.files}


def evaluate_iid_test_log_likelihood(
    new_estimator: Callable[[], BGM_BridgeDensityEstimator],
    x_test: np.ndarray,
    test_indices: np.ndarray,
    config: Mapping[str, Any],
    output_path: Path,
) -> Dict[str, np.ndarray]:
    density_cfg = config["density_bridge_eval"]
    n_points = x_test.shape[0]
    n_repeats = int(density_cfg["n_repeats"])
    sample_size = int(density_cfg["S"])

    existing = load_existing_test_log_likelihood(
        output_path,
        n_points,
        n_repeats,
        expected_x_test=x_test,
        expected_indices=test_indices,
        expected_sample_size=sample_size,
    )
    if existing is None:
        log_px_repeats = np.full((n_points, n_repeats), np.nan, dtype=np.float64)
        log_px = np.full(n_points, np.nan, dtype=np.float64)
        px = np.full(n_points, np.nan, dtype=np.float64)
        log_px_sd = np.full(n_points, np.nan, dtype=np.float64)
        proposal_is_ess_mean = np.full(n_points, np.nan, dtype=np.float64)
        hmc_min_ess = np.full(n_points, np.nan, dtype=np.float64)
        hmc_acceptance_rate = np.full(n_points, np.nan, dtype=np.float64)
        bridge_iterations_mean = np.full(n_points, np.nan, dtype=np.float64)
        bridge_convergence_rate_point = np.full(n_points, np.nan, dtype=np.float64)
        bridge_abs_delta_mean = np.full(n_points, np.nan, dtype=np.float64)
    else:
        log_px_repeats = existing["log_px_repeats"].copy()
        log_px = existing["log_px"].copy()
        px = existing["px"].copy()
        log_px_sd = existing["log_px_sd"].copy()
        proposal_is_ess_mean = existing["proposal_is_ess_mean"].copy()
        hmc_min_ess = existing["hmc_min_ess"].copy()
        hmc_acceptance_rate = existing["hmc_acceptance_rate"].copy()
        bridge_iterations_mean = existing["bridge_iterations_mean"].copy()
        bridge_convergence_rate_point = existing["bridge_convergence_rate_point"].copy()
        bridge_abs_delta_mean = existing["bridge_abs_delta_mean"].copy()

    block_size = int(density_cfg.get("rng_block_size") or n_points)
    for start in range(0, n_points, block_size):
        block = slice(start, start + block_size)
        if not np.all(np.isfinite(log_px[block])):
            for values in (
                log_px_repeats, log_px, px, log_px_sd, proposal_is_ess_mean, hmc_min_ess,
                hmc_acceptance_rate, bridge_iterations_mean, bridge_convergence_rate_point,
                bridge_abs_delta_mean,
            ):
                values[block] = np.nan

    save_every = int(density_cfg.get("test_log_likelihood_save_every", density_cfg.get("save_every", 25)))
    completed = int(np.sum(np.isfinite(log_px)))
    start_time = time.time()
    estimator = None

    with tqdm(
        total=n_points,
        initial=completed,
        desc="Bridge IID test density",
        unit="pt",
        dynamic_ncols=True,
        ascii=True,
    ) as pbar:
        for idx, point in enumerate(x_test):
            if np.isfinite(log_px[idx]) and np.all(np.isfinite(log_px_repeats[idx])):
                continue
            if estimator is None or idx % block_size == 0:
                estimator = new_estimator()

            result = estimator.estimate(
                point.astype(np.float32),
                K=int(density_cfg.get("K", 5)),
                S=sample_size,
                nu=float(density_cfg.get("nu", 3.0)),
                epsilon=float(density_cfg.get("epsilon", 0.05)),
                n_repeats=n_repeats,
                hmc_settings=config["hmc_settings"],
                eval_batch_size=int(density_cfg.get("eval_batch_size", 4096)),
                bridge_tol=float(density_cfg.get("tol", 1e-5)),
                bridge_max_iter=int(density_cfg.get("max_iter", 1000)),
                fit_fraction=float(density_cfg.get("fit_fraction", 0.5)),
                use_neff=bool(density_cfg.get("use_neff", True)),
                proposal_scale_multiplier=float(density_cfg.get("proposal_scale_multiplier", 1.0)),
                proposal_covariance_mode=str(density_cfg.get("proposal_covariance_mode", "fitted_full")),
                proposal_scale=(None if density_cfg.get("proposal_scale") is None else float(density_cfg["proposal_scale"])),
                proposal_center=str(density_cfg.get("proposal_center", "fitted_gmm")),
                proposal_scoring=str(density_cfg.get("proposal_scoring", "multivariate_t")),
                return_proposal=False,
            )

            diag = result["diagnostics"][0]
            repeats = np.asarray(result["log_px_repeats"], dtype=np.float64)
            if repeats.shape != (n_repeats,):
                raise RuntimeError(
                    f"Expected {n_repeats} repeat log-likelihoods for test index {idx}, "
                    f"got shape {repeats.shape}."
                )

            bridge_iter = np.asarray(diag["bridge_iterations_repeats"], dtype=np.float64)
            bridge_conv = np.asarray(diag["bridge_converged_repeats"], dtype=bool)
            bridge_delta = np.asarray(diag["bridge_abs_delta_repeats"], dtype=np.float64)
            proposal_ess = np.asarray(diag["proposal_is_ess_repeats"], dtype=np.float64)

            log_px_repeats[idx, :] = repeats
            log_px[idx] = float(np_logmeanexp(repeats, axis=0))
            px[idx] = float(np.exp(log_px[idx])) if log_px[idx] < 700 else math.inf
            log_px_sd[idx] = float(result["log_px_repeat_sd"])
            proposal_is_ess_mean[idx] = float(np.mean(proposal_ess))
            hmc_min_ess[idx] = float(diag["hmc_min_ess"])
            hmc_acceptance_rate[idx] = float(diag["hmc_acceptance_rate"])
            bridge_iterations_mean[idx] = float(np.mean(bridge_iter))
            bridge_convergence_rate_point[idx] = float(np.mean(bridge_conv))
            bridge_abs_delta_mean[idx] = float(np.mean(bridge_delta))

            pbar.update(1)
            pbar.set_postfix(
                log_px=f"{log_px[idx]:.3f}",
                it=f"{bridge_iterations_mean[idx]:.1f}",
                conv=f"{bridge_convergence_rate_point[idx]:.2f}",
                ess=f"{proposal_is_ess_mean[idx]:.1f}",
            )

            done = int(np.sum(np.isfinite(log_px)))
            if done % save_every == 0 or done == n_points:
                elapsed = time.time() - start_time
                print(f"Evaluated {done}/{n_points} bridge IID test points in {elapsed:.1f}s")
                save_bridge_test_log_likelihood(
                    output_path,
                    x_test,
                    test_indices,
                    sample_size,
                    log_px_repeats,
                    log_px,
                    px,
                    log_px_sd,
                    proposal_is_ess_mean,
                    hmc_min_ess,
                    hmc_acceptance_rate,
                    bridge_iterations_mean,
                    bridge_convergence_rate_point,
                    bridge_abs_delta_mean,
                )

    save_bridge_test_log_likelihood(
        output_path,
        x_test,
        test_indices,
        sample_size,
        log_px_repeats,
        log_px,
        px,
        log_px_sd,
        proposal_is_ess_mean,
        hmc_min_ess,
        hmc_acceptance_rate,
        bridge_iterations_mean,
        bridge_convergence_rate_point,
        bridge_abs_delta_mean,
    )
    return {
        "x_test": x_test,
        "test_indices": test_indices,
        "sample_size": np.asarray(sample_size, dtype=np.int64),
        "log_px_repeats": log_px_repeats,
        "log_px": log_px,
        "px": px,
        "log_px_sd": log_px_sd,
        "proposal_is_ess_mean": proposal_is_ess_mean,
        "hmc_min_ess": hmc_min_ess,
        "hmc_acceptance_rate": hmc_acceptance_rate,
        "bridge_iterations_mean": bridge_iterations_mean,
        "bridge_convergence_rate_point": bridge_convergence_rate_point,
        "bridge_abs_delta_mean": bridge_abs_delta_mean,
    }


def plot_heatmap(
    values: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    save_path: Path,
    title: str,
    cmap: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    arr = values.reshape(v1.shape)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        arr,
        extent=[float(v1.min()), float(v1.max()), float(v2.min()), float(v2.max())],
        origin="lower",
        cmap=cmap,
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_side_by_side(
    truth: np.ndarray,
    estimate: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    save_path: Path,
    cmap: str,
) -> None:
    truth_arr = truth.reshape(v1.shape)
    est_arr = estimate.reshape(v1.shape)
    finite = np.isfinite(truth_arr) & np.isfinite(est_arr)
    vmax = float(np.max([np.max(truth_arr[finite]), np.max(est_arr[finite])])) if np.any(finite) else None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ax, arr, title in zip(axes, [truth_arr, est_arr], ["Exact density", "BGM Bridge density"]):
        im = ax.imshow(
            arr,
            extent=[float(v1.min()), float(v1.max()), float(v2.min()), float(v2.max())],
            origin="lower",
            cmap=cmap,
            aspect="equal",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.savefig(save_path, dpi=220)
    plt.close(fig)


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


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    rx = average_rank(x)
    ry = average_rank(y)
    if np.std(rx) == 0.0 or np.std(ry) == 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def render_outputs(
    run_dir: Path,
    v1: np.ndarray,
    v2: np.ndarray,
    truth_px: np.ndarray,
    density: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> Dict[str, float]:
    out_cfg = config["outputs"]
    cmap_density = str(out_cfg.get("cmap_density", "Blues"))

    bridge_px = density["px"]
    log_px = density["log_px"]

    plot_heatmap(bridge_px, v1, v2, run_dir / "fig_bgm_density.png", "BGM Bridge density", cmap_density, vmin=0.0)
    plot_heatmap(
        log_px,
        v1,
        v2,
        run_dir / "fig_log_px_heatmap.png",
        "BGM Bridge log density",
        str(out_cfg.get("cmap_log_px", "viridis")),
    )
    plot_heatmap(
        density["proposal_is_ess_mean"],
        v1,
        v2,
        run_dir / "fig_is_ess_heatmap.png",
        "Bridge proposal IS ESS",
        str(out_cfg.get("cmap_ess", "cividis")),
        vmin=0.0,
    )
    plot_heatmap(
        density["bridge_iterations_mean"],
        v1,
        v2,
        run_dir / "fig_bridge_iterations_heatmap.png",
        "Bridge iterations",
        str(out_cfg.get("cmap_iter", "plasma")),
        vmin=0.0,
    )
    plot_heatmap(
        density["bridge_convergence_rate_point"],
        v1,
        v2,
        run_dir / "fig_bridge_convergence_heatmap.png",
        "Bridge convergence rate",
        str(out_cfg.get("cmap_converged", "Greens")),
        vmin=0.0,
        vmax=1.0,
    )
    dataset_name = str(config.get("data", {}).get("name", "data"))
    plot_heatmap(
        truth_px,
        v1,
        v2,
        run_dir / "fig_truth_density.png",
        f"Exact {dataset_name} density",
        cmap_density,
        vmin=0.0,
    )
    plot_side_by_side(truth_px, bridge_px, v1, v2, run_dir / "fig_bgm_vs_truth.png", cmap_density)
    finite = np.isfinite(bridge_px) & np.isfinite(truth_px) & np.isfinite(log_px)
    if not np.any(finite):
        raise RuntimeError("No finite bridge density estimates were produced.")

    mean_log_px = float(np.mean(log_px[finite]))
    metrics = density_calibration_metrics(
        truth_px=truth_px,
        estimate_px=bridge_px,
        estimate_log_px=log_px,
        truth_log_px=safe_log_density(truth_px),
        finite=finite,
        extra={
            "density_metric_source": f"grid_{v1.shape[0]}x{v1.shape[1]}",
            "grid_points": int(len(bridge_px)),
            "finite_grid_points": int(np.sum(finite)),
            "mean_log_px": mean_log_px,
            "mean_log_px_grid": mean_log_px,
            "mean_bridge_iterations": float(np.nanmean(density["bridge_iterations_mean"][finite])),
            "bridge_convergence_rate": float(np.nanmean(density["bridge_convergence_rate_point"][finite])),
            "mean_proposal_is_ess": float(np.nanmean(density["proposal_is_ess_mean"][finite])),
            "mean_hmc_min_ess": float(np.nanmean(density["hmc_min_ess"][finite])),
            "mean_hmc_acceptance_rate": float(np.nanmean(density["hmc_acceptance_rate"][finite])),
            "mean_bridge_abs_delta": float(np.nanmean(density["bridge_abs_delta_mean"][finite])),
            "truth_source": f"{dataset_name}.get_density",
        },
    )
    metrics["spearman_corr"] = metrics["density_spearman_corr"]
    return metrics


def render_highdim_outputs(
    truth_px: np.ndarray,
    test_log_likelihood: Mapping[str, np.ndarray],
    truth_log_px: np.ndarray | None = None,
) -> Dict[str, Any]:
    estimate_log_px = np.asarray(test_log_likelihood["log_px"], dtype=np.float64)
    estimate_px = np.asarray(test_log_likelihood["px"], dtype=np.float64)
    truth_px = np.asarray(truth_px, dtype=np.float64)
    truth_log_px = (
        safe_log_density(truth_px)
        if truth_log_px is None
        else np.asarray(truth_log_px, dtype=np.float64)
    )
    log_px_repeats = np.asarray(test_log_likelihood["log_px_repeats"], dtype=np.float64)

    finite = (
        np.isfinite(truth_log_px)
        & np.isfinite(estimate_log_px)
        & np.all(np.isfinite(log_px_repeats), axis=1)
    )
    if not np.any(finite):
        raise RuntimeError("No finite high-dimensional bridge density estimates were produced.")

    log_likelihood_summary = summarize_repeat_log_likelihood(log_px_repeats, finite, "iid_test")
    metrics: Dict[str, Any] = {
        "grid_points": None,
        "finite_grid_points": None,
        "density_points": int(len(estimate_px)),
        "finite_density_points": int(np.sum(finite)),
        "density_metric_source": "iid_test",
        "primary_density_scale": "log_density",
        "raw_density_cross_dimension_comparable": False,
        "raw_density_metric_note": "Density scale changes exponentially with dimension.",
        "density_spearman_corr": spearman_corr(estimate_log_px[finite], truth_log_px[finite]),
        "spearman_corr": spearman_corr(estimate_log_px[finite], truth_log_px[finite]),
        "mean_log_px": log_likelihood_summary["mean_log_px"],
        "std_log_px": log_likelihood_summary["std_log_px"],
        "log_likelihood_source": log_likelihood_summary["log_likelihood_source"],
        "log_likelihood_points": log_likelihood_summary["log_likelihood_points"],
        "log_px_repeat_means": log_likelihood_summary["log_px_repeat_means"],
        "mean_proposal_is_ess": nanmean_or_none(test_log_likelihood["proposal_is_ess_mean"][finite]),
        "mean_hmc_min_ess": nanmean_or_none(test_log_likelihood["hmc_min_ess"][finite]),
        "mean_hmc_acceptance_rate": nanmean_or_none(test_log_likelihood["hmc_acceptance_rate"][finite]),
        "mean_bridge_iterations": nanmean_or_none(test_log_likelihood["bridge_iterations_mean"][finite]),
        "bridge_convergence_rate": nanmean_or_none(
            test_log_likelihood["bridge_convergence_rate_point"][finite]
        ),
        "mean_bridge_abs_delta": nanmean_or_none(test_log_likelihood["bridge_abs_delta_mean"][finite]),
        "test_mean_proposal_is_ess": nanmean_or_none(
            test_log_likelihood["proposal_is_ess_mean"][finite]
        ),
        "test_min_proposal_is_ess": nanmin_or_none(
            test_log_likelihood["proposal_is_ess_mean"][finite]
        ),
        "test_mean_hmc_min_ess": nanmean_or_none(test_log_likelihood["hmc_min_ess"][finite]),
        "test_mean_hmc_acceptance_rate": nanmean_or_none(
            test_log_likelihood["hmc_acceptance_rate"][finite]
        ),
    }
    metrics.update(masked_spearman_metrics(truth_px, estimate_px, finite))
    stable_spearman = spearman_corr(estimate_log_px[finite], truth_log_px[finite])
    metrics["density_spearman_corr"] = stable_spearman
    metrics["spearman_corr"] = stable_spearman
    metrics["log_density_spearman_corr"] = stable_spearman
    metrics["primary_spearman_corr"] = stable_spearman
    metrics["log_density_spearman_bootstrap_ci"] = bootstrap_spearman_ci(
        truth_log_px[finite], estimate_log_px[finite]
    )
    metrics["truth_log_density_source"] = "stable_exact_log_density"
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bridge-sampling BGM density visualization on simulated data.")
    parser.add_argument(
        "--config",
        default="configs/bridge.yaml",
        help="Bridge estimator config with density_bridge_eval and hmc_settings.",
    )
    parser.add_argument(
        "--visual-config",
        default="configs/model.yaml",
        help="Visualization/data/grid config reused from the IS runner.",
    )
    parser.add_argument("--preset", choices=["smoke", "final"], default="final")
    parser.add_argument("--run-name", default=None, help="Optional output run name.")
    parser.add_argument("--checkpoint-dir", default=None, help="Pre-trained BGM checkpoint directory.")
    parser.add_argument("--epoch", type=int, default=None, help="Generator epoch to load.")
    parser.add_argument("--egm-iter", type=int, default=None, help="Encoder EGM iteration to load.")
    parser.add_argument(
        "--load-encoder",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Load encoder weights from the checkpoint (default: model_setting.load_encoder).",
    )
    parser.add_argument("--grid-n", type=int, default=None, help="Override grid resolution.")
    parser.add_argument("--K", type=int, default=None, help="Override density_bridge_eval.K.")
    parser.add_argument("--nu", type=float, default=None, help="Override density_bridge_eval.nu.")
    parser.add_argument("--epsilon", type=float, default=None, help="Override density_bridge_eval.epsilon.")
    parser.add_argument("--sample-size", type=int, default=None, help="Override density_bridge_eval.S.")
    parser.add_argument("--n-repeats", type=int, default=None, help="Override density_bridge_eval.n_repeats.")
    parser.add_argument("--hmc-M", type=int, default=None, help="Override hmc_settings.M.")
    parser.add_argument("--hmc-burn-in", type=int, default=None, help="Override hmc_settings.burn_in.")
    parser.add_argument("--hmc-step-size", type=float, default=None, help="Override hmc_settings.step_size.")
    parser.add_argument(
        "--num-leapfrog-steps",
        type=int,
        default=None,
        help="Override hmc_settings.num_leapfrog_steps.",
    )
    parser.add_argument(
        "--target-accept-prob",
        type=float,
        default=None,
        help="Override hmc_settings.target_accept_prob.",
    )
    parser.add_argument(
        "--initial-state-scale",
        type=float,
        default=None,
        help="Override hmc_settings.initial_state_scale.",
    )
    parser.add_argument(
        "--hmc-initial-state",
        choices=["encoder", "prior", "mixed_encoder_prior"],
        default=None,
        help="Override hmc_settings.initial_state.",
    )
    parser.add_argument(
        "--variance-floor",
        type=float,
        default=None,
        help="Override density_bridge_eval.variance_floor.",
    )
    parser.add_argument(
        "--fixed-likelihood-variance",
        type=float,
        default=None,
        help="Override learned decoder variance with a fixed Gaussian likelihood variance.",
    )
    parser.add_argument(
        "--proposal-covariance-floor",
        type=float,
        default=None,
        help="Override hmc_settings.proposal_covariance_floor.",
    )
    parser.add_argument(
        "--proposal-scale-multiplier",
        type=float,
        default=None,
        help="Override density_bridge_eval.proposal_scale_multiplier.",
    )
    parser.add_argument("--proposal-covariance-mode", choices=["fitted_full", "fixed_isotropic"], default=None)
    parser.add_argument("--proposal-scale", type=float, default=None)
    parser.add_argument("--proposal-center", choices=["fitted_gmm", "hmc_fit_mean"], default=None)
    parser.add_argument("--proposal-scoring", choices=["multivariate_t", "product_univariate_t"], default=None)
    parser.add_argument(
        "--min-effective-samples",
        type=int,
        default=None,
        help="Override hmc_settings.min_effective_samples.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Override hmc_settings.max_retries.",
    )
    parser.add_argument(
        "--bridge-max-iter",
        type=int,
        default=None,
        help="Override density_bridge_eval.max_iter.",
    )
    parser.add_argument(
        "--test-points",
        type=int,
        default=None,
        help="Override density.test_log_likelihood_points for high-dimensional evaluation.",
    )
    parser.add_argument(
        "--rng-block-size",
        type=int,
        default=None,
        help="Restart the estimator's random stream every this many test points.",
    )
    parser.add_argument(
        "--force-iid-test-eval",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override density.force_iid_test_eval. Use --no-force-iid-test-eval "
            "to restore the native 2D visualization-grid protocol when a reused "
            "training config was created for IID evaluation."
        ),
    )
    parser.add_argument(
        "--density-workers",
        type=int,
        default=None,
        help="Override data.density_workers used only for true-density calculations.",
    )
    parser.add_argument(
        "--test-indices",
        type=int,
        nargs="+",
        default=None,
        help="Use explicit indices into X_test for high-dimensional debug evaluation.",
    )
    parser.add_argument(
        "--test-indices-file",
        default=None,
        help=(
            "Load explicit evaluation indices from a .npy file or from the "
            "test_indices array in a .npz file. This is mutually exclusive "
            "with --test-indices."
        ),
    )
    parser.add_argument(
        "--test-shard-index",
        type=int,
        default=None,
        help="Zero-based shard of the explicit test indices to evaluate.",
    )
    parser.add_argument(
        "--test-shard-count",
        type=int,
        default=None,
        help="Number of deterministic, disjoint shards for explicit test indices.",
    )
    parser.add_argument(
        "--evaluation-split",
        choices=["validation", "test"],
        default="test",
        help="Select bridge-evaluation points from X_val or X_test (default: test).",
    )
    parser.add_argument(
        "--truth-cache",
        default=None,
        help="Optional cached IID truth density npz with x_test, test_indices, and px.",
    )
    parser.add_argument(
        "--generation-only",
        action="store_true",
        help="Only run BGM prior-generation diagnostics, then stop before bridge density evaluation.",
    )
    parser.add_argument(
        "--skip-generation-diagnostics",
        action="store_true",
        help="Skip prior-generation diagnostics and run only the requested bridge density evaluation.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Override outputs.root_dir.",
    )
    parser.add_argument(
        "--fixed-variance-head",
        action="store_true",
        help="Expect/load a checkpoint trained with a true fixed decoder variance head.",
    )
    parser.add_argument(
        "--fixed-variance-value",
        type=float,
        default=None,
        help="Variance value used when --fixed-variance-head is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint_dir:
        raise ValueError("Bridge visualization is inference-only: --checkpoint-dir is required.")
    if args.epoch is not None and args.egm_iter is not None:
        raise ValueError(
            "--epoch and --egm-iter select different generator checkpoint namespaces and "
            "cannot be supplied together. For an iterative generator with its EGM encoder, "
            "pass --epoch only; the latest weights_at_egm_init_* encoder is loaded automatically."
        )
    if args.generation_only and args.skip_generation_diagnostics:
        raise ValueError("--generation-only and --skip-generation-diagnostics cannot be used together.")

    bridge_config_path = resolve_path(args.config)
    visual_config_path = resolve_path(args.visual_config)
    bridge_cfg = load_yaml(bridge_config_path)
    visual_cfg = materialize_visual_config(load_yaml(visual_config_path), args.preset)
    config = merge_bridge_config(visual_cfg, bridge_cfg)
    if args.density_workers is not None:
        if int(args.density_workers) <= 0:
            raise ValueError("--density-workers must be positive.")
        config.setdefault("data", {})["density_workers"] = int(args.density_workers)
    validate_bridge_config(config)

    if args.grid_n is not None:
        config["grid"]["n"] = int(args.grid_n)
    if args.K is not None:
        config["density_bridge_eval"]["K"] = int(args.K)
    if args.nu is not None:
        config["density_bridge_eval"]["nu"] = float(args.nu)
    if args.epsilon is not None:
        config["density_bridge_eval"]["epsilon"] = float(args.epsilon)
    if args.sample_size is not None:
        config["density_bridge_eval"]["S"] = int(args.sample_size)
    if args.n_repeats is not None:
        config["density_bridge_eval"]["n_repeats"] = int(args.n_repeats)
    if args.hmc_M is not None:
        config["hmc_settings"]["M"] = int(args.hmc_M)
    if args.hmc_burn_in is not None:
        config["hmc_settings"]["burn_in"] = int(args.hmc_burn_in)
    if args.hmc_step_size is not None:
        config["hmc_settings"]["step_size"] = float(args.hmc_step_size)
    if args.num_leapfrog_steps is not None:
        config["hmc_settings"]["num_leapfrog_steps"] = int(args.num_leapfrog_steps)
    if args.target_accept_prob is not None:
        config["hmc_settings"]["target_accept_prob"] = float(args.target_accept_prob)
    if args.initial_state_scale is not None:
        config["hmc_settings"]["initial_state_scale"] = float(args.initial_state_scale)
    if args.hmc_initial_state is not None:
        config["hmc_settings"]["initial_state"] = str(args.hmc_initial_state)
    if args.load_encoder is not None:
        config.setdefault("model_setting", {})["load_encoder"] = bool(args.load_encoder)
    if args.variance_floor is not None:
        config["density_bridge_eval"]["variance_floor"] = float(args.variance_floor)
    if args.fixed_likelihood_variance is not None:
        config["density_bridge_eval"]["fixed_likelihood_variance"] = float(args.fixed_likelihood_variance)
    if args.proposal_covariance_floor is not None:
        config["hmc_settings"]["proposal_covariance_floor"] = float(args.proposal_covariance_floor)
    if args.proposal_scale_multiplier is not None:
        config["density_bridge_eval"]["proposal_scale_multiplier"] = float(args.proposal_scale_multiplier)
    if args.proposal_covariance_mode is not None:
        config["density_bridge_eval"]["proposal_covariance_mode"] = str(args.proposal_covariance_mode)
    if args.proposal_scale is not None:
        config["density_bridge_eval"]["proposal_scale"] = float(args.proposal_scale)
    if args.proposal_center is not None:
        config["density_bridge_eval"]["proposal_center"] = str(args.proposal_center)
    if args.proposal_scoring is not None:
        config["density_bridge_eval"]["proposal_scoring"] = str(args.proposal_scoring)
    if args.min_effective_samples is not None:
        config["hmc_settings"]["min_effective_samples"] = int(args.min_effective_samples)
    if args.max_retries is not None:
        config["hmc_settings"]["max_retries"] = int(args.max_retries)
    if args.bridge_max_iter is not None:
        config["density_bridge_eval"]["max_iter"] = int(args.bridge_max_iter)
    if args.test_points is not None:
        config.setdefault("density", {})["test_log_likelihood_points"] = int(args.test_points)
    if args.rng_block_size is not None:
        if int(args.rng_block_size) <= 0:
            raise ValueError("--rng-block-size must be positive.")
        config["density_bridge_eval"]["rng_block_size"] = int(args.rng_block_size)
    if args.force_iid_test_eval is not None:
        config.setdefault("density", {})["force_iid_test_eval"] = bool(args.force_iid_test_eval)
    if args.output_root is not None:
        config.setdefault("outputs", {})["root_dir"] = str(args.output_root)
    if args.fixed_variance_head:
        config.setdefault("model", {})["fixed_variance_head"] = True
    if args.fixed_variance_value is not None:
        config.setdefault("model", {})["fixed_variance_value"] = float(args.fixed_variance_value)
    config.setdefault("run_management", {})["evaluation_split"] = str(args.evaluation_split)

    validate_indep_gmm_dimensions(config)
    validate_bridge_config(config)

    require_gpu(config)
    print(
        "Bridge budget from YAML: "
        f"M={config['hmc_settings']['M']}, "
        f"burn_in={config['hmc_settings']['burn_in']}, "
        f"min_effective_samples={config['hmc_settings']['min_effective_samples']}, "
        f"fit_fraction={config['density_bridge_eval']['fit_fraction']}, "
        f"fit_budget={config['density_bridge_eval']['effective_fit_budget']}, "
        f"S={config['density_bridge_eval']['S']}, "
        f"n_repeats={config['density_bridge_eval']['n_repeats']}"
    )

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"bridge_{args.preset}_{timestamp}"
    run_root = resolve_path(config["outputs"]["root_dir"])
    run_dir = run_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    save_yaml(run_dir / "run_config_used.yaml", config)
    save_json(
        run_dir / "runtime_manifest.json",
        {
            "tensorflow_version": tf.__version__,
            "bridge_config_path": str(bridge_config_path),
            "visual_config_path": str(visual_config_path),
        },
    )

    sampler = build_sampler(config["data"])
    x_all, _ = sampler.load_all()
    evaluation_pool = sampler.X_val if args.evaluation_split == "validation" else sampler.X_test
    iid_test_points, iid_test_indices = select_iid_test_points(evaluation_pool, config)
    if args.test_indices is not None and args.test_indices_file is not None:
        raise ValueError("Use only one of --test-indices and --test-indices-file.")
    explicit_indices = None
    if args.test_indices_file is not None:
        indices_path = resolve_path(args.test_indices_file)
        loaded = np.load(indices_path)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            try:
                if "test_indices" not in loaded.files:
                    raise ValueError(f"{indices_path} has no test_indices array.")
                explicit_indices = np.asarray(loaded["test_indices"], dtype=np.int64)
            finally:
                loaded.close()
        else:
            explicit_indices = np.asarray(loaded, dtype=np.int64)
    elif args.test_indices is not None:
        explicit_indices = np.asarray(args.test_indices, dtype=np.int64)

    shard_args = (args.test_shard_index, args.test_shard_count)
    if (shard_args[0] is None) != (shard_args[1] is None):
        raise ValueError("--test-shard-index and --test-shard-count must be specified together.")
    if explicit_indices is None and shard_args[0] is not None:
        raise ValueError("Test sharding requires --test-indices or --test-indices-file.")
    if explicit_indices is not None:
        if explicit_indices.ndim != 1 or explicit_indices.size < 1:
            raise ValueError("Explicit test indices must be a non-empty one-dimensional array.")
        if len(np.unique(explicit_indices)) != len(explicit_indices):
            raise ValueError("Explicit test indices contain duplicates.")
        if shard_args[0] is not None:
            shard_index, shard_count = int(shard_args[0]), int(shard_args[1])
            if shard_count < 1 or shard_index < 0 or shard_index >= shard_count:
                raise ValueError("Require 0 <= --test-shard-index < --test-shard-count.")
            explicit_indices = np.array_split(explicit_indices, shard_count)[shard_index]
            if explicit_indices.size < 1:
                raise ValueError("Requested test shard is empty.")
        if np.any(explicit_indices < 0) or np.any(explicit_indices >= len(evaluation_pool)):
            raise ValueError(f"--test-indices contains an index outside {args.evaluation_split} split.")
        iid_test_indices = explicit_indices
        iid_test_points = np.asarray(evaluation_pool, dtype=np.float32)[iid_test_indices]
    save_dataset_fingerprint(
        run_dir,
        x_train=sampler.X_train,
        x_val=sampler.X_val,
        x_test=sampler.X_test,
        selected_test_indices=iid_test_indices,
        selected_test_points=iid_test_points,
    )
    data_dim = int(config["data"].get("dim", sampler.X_train.shape[1]))
    force_iid_test_eval = bool(config.get("density", {}).get("force_iid_test_eval", False))
    high_dimensional = data_dim > 2 or force_iid_test_eval

    if high_dimensional:
        remove_artifacts(
            run_dir,
            (
                "truth_density_grid.npz",
                "bridge_density_grid.npz",
                "fig_bgm_density.png",
                "fig_log_px_heatmap.png",
                "fig_is_ess_heatmap.png",
                "fig_bridge_iterations_heatmap.png",
                "fig_bridge_convergence_heatmap.png",
                "fig_truth_density.png",
                "fig_bgm_vs_truth.png",
            ),
        )
        np.savez(
            run_dir / f"data_{config['data']['name']}.npz",
            x_all=x_all,
            x_train=sampler.X_train,
            x_val=sampler.X_val,
            x_test=sampler.X_test,
            selected_test_indices=iid_test_indices,
            selected_test_points=iid_test_points,
        )
        v1 = v2 = grid = truth_px = None
    else:
        v1, v2, grid = create_2d_slice_grid(config["grid"], config["data"])
        truth_px = sampler.get_density(grid.astype(np.float64)).astype(np.float64)

        np.savez(
            run_dir / f"data_{config['data']['name']}.npz",
            x_all=x_all,
            x_train=sampler.X_train,
            x_val=sampler.X_val,
            x_test=sampler.X_test,
            selected_test_indices=iid_test_indices,
            selected_test_points=iid_test_points,
            v1=v1,
            v2=v2,
            grid=grid,
        )
        np.savez(run_dir / "truth_density_grid.npz", v1=v1, v2=v2, grid=grid, px=truth_px)

    checkpoint_dir = resolve_path(args.checkpoint_dir)
    validate_checkpoint_matches_config(checkpoint_dir, config)
    model, manifest = load_frozen_model(
        config=config,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        epoch=args.epoch,
        egm_iter=args.egm_iter,
    )
    save_yaml(run_dir / "weights_manifest.yaml", manifest)
    generation_diagnostics = None
    if not args.skip_generation_diagnostics:
        generation_diagnostics = run_bgm_generation_diagnostics(
            model=model,
            sampler=sampler,
            run_dir=run_dir,
            config=config,
            seed=int(
                config["density_bridge_eval"].get(
                    "generation_seed",
                    config["data"].get("seed", 1024) + 31415,
                )
            ),
        )
    if args.generation_only:
        metrics = {
            "estimator": "BGM_BridgeDensityEstimator",
            "mode": "generation_only",
            "preset": args.preset,
            "run_dir": str(run_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "bridge_config_path": str(bridge_config_path),
            "visual_config_path": str(visual_config_path),
            "dataset": config["data"]["name"],
            "grid_n": None if high_dimensional else int(config["grid"]["n"]),
            "bridge_S": int(config["density_bridge_eval"]["S"]),
            "bridge_K": int(config["density_bridge_eval"]["K"]),
            "bridge_n_repeats": int(config["density_bridge_eval"]["n_repeats"]),
            "fixed_likelihood_variance": config["density_bridge_eval"].get("fixed_likelihood_variance"),
            "bridge_epsilon": float(config["density_bridge_eval"].get("epsilon", 0.05)),
            "bridge_nu": float(config["density_bridge_eval"].get("nu", 3.0)),
            "proposal_scale_multiplier": float(
                config["density_bridge_eval"].get("proposal_scale_multiplier", 1.0)
            ),
            "bridge_hmc_M": int(config["hmc_settings"]["M"]),
            "batch_size": int(config["training"]["batch_size"]),
            "decorrelation_weight": float(config["model"].get("decorrelation_weight", 0.0)),
            "x_adv_use_mean": bool(config["model"].get("x_adv_use_mean", False)),
            "fixed_variance_head": bool(config["model"].get("fixed_variance_head", False)),
            "fixed_variance_value": (
                None
                if "fixed_variance_value" not in config["model"]
                else float(config["model"]["fixed_variance_value"])
            ),
            "selected_test_indices_path": str(run_dir / "selected_test_indices.npz"),
            "dataset_fingerprint_path": str(run_dir / "dataset_fingerprint.json"),
            "generated_samples_path": str(run_dir / "generated_samples.npz"),
            "generation_diagnostics_path": str(run_dir / "generation_diagnostics.json"),
            "generation_diagnostics_skipped": False,
            "train_mean_log_true_px": generation_diagnostics["train"]["density"]["mean_log_true_px"],
            "test_mean_log_true_px": generation_diagnostics["test"]["density"]["mean_log_true_px"],
            "generated_mean_log_true_px": generation_diagnostics["generated_mean"]["density"][
                "mean_log_true_px"
            ],
            "generated_sample_mean_log_true_px": generation_diagnostics["generated_sample"]["density"][
                "mean_log_true_px"
            ],
            "generated_mean_corr_frobenius_error": generation_diagnostics["generated_mean"][
                "corr_frobenius_error_to_independent"
            ],
            "generated_sample_corr_frobenius_error": generation_diagnostics["generated_sample"][
                "corr_frobenius_error_to_independent"
            ],
        }
        metrics.update(generation_metric_summary(generation_diagnostics["generated_mean"], "generated_mean"))
        metrics.update(generation_metric_summary(generation_diagnostics["generated_sample"], "generated_sample"))
        save_json(run_dir / "metrics.json", metrics)
        print("Generation diagnostics only. Outputs saved to:", run_dir)
        return

    if high_dimensional:
        test_ll_path = run_dir / "bridge_test_log_likelihood.npz"
        bind_density_cache_to_run(cache_path=test_ll_path, manifest=manifest, config=config)
        test_log_likelihood = evaluate_iid_test_log_likelihood(
            lambda: make_estimator(model, config),
            iid_test_points,
            iid_test_indices,
            config,
            test_ll_path,
        )
        truth_test_px = load_cached_truth_iid_test_points(
            run_dir / "truth_iid_test_points.npz",
            test_points=np.asarray(iid_test_points, dtype=np.float32),
            test_indices=np.asarray(iid_test_indices, dtype=np.int64),
        )
        if truth_test_px is None and args.truth_cache is not None:
            truth_test_px = load_cached_truth_iid_test_points(
                resolve_path(args.truth_cache),
                test_points=np.asarray(iid_test_points, dtype=np.float32),
                test_indices=np.asarray(iid_test_indices, dtype=np.int64),
            )
        if truth_test_px is None:
            truth_test_px = sampler.get_density(iid_test_points.astype(np.float64)).astype(np.float64)
        truth_test_log_px = (
            np.asarray(sampler.get_log_density(iid_test_points.astype(np.float64)), dtype=np.float64)
            if hasattr(sampler, "get_log_density")
            else safe_log_density(truth_test_px)
        )
        np.savez(
            run_dir / "truth_iid_test_points.npz",
            x_test=np.asarray(iid_test_points, dtype=np.float32),
            test_indices=np.asarray(iid_test_indices, dtype=np.int64),
            px=truth_test_px,
            log_px=truth_test_log_px,
        )
        metrics = render_highdim_outputs(
            truth_test_px,
            test_log_likelihood,
            truth_log_px=truth_test_log_px,
        )
    else:
        assert v1 is not None and v2 is not None and grid is not None and truth_px is not None
        density_path = run_dir / "bridge_density_grid.npz"
        bind_density_cache_to_run(cache_path=density_path, manifest=manifest, config=config)
        density = evaluate_grid_density(make_estimator(model, config), grid, v1, v2, config, density_path)
        metrics = render_outputs(run_dir, v1, v2, truth_px, density, config)
    metrics.update(
        {
            "estimator": "BGM_BridgeDensityEstimator",
            "preset": args.preset,
            "run_dir": str(run_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "bridge_config_path": str(bridge_config_path),
            "visual_config_path": str(visual_config_path),
            "dataset": config["data"]["name"],
            "grid_n": None if high_dimensional else int(config["grid"]["n"]),
            "bridge_S": int(config["density_bridge_eval"]["S"]),
            "bridge_K": int(config["density_bridge_eval"]["K"]),
            "bridge_n_repeats": int(config["density_bridge_eval"]["n_repeats"]),
            "fixed_likelihood_variance": config["density_bridge_eval"].get("fixed_likelihood_variance"),
            "bridge_epsilon": float(config["density_bridge_eval"].get("epsilon", 0.05)),
            "bridge_nu": float(config["density_bridge_eval"].get("nu", 3.0)),
            "proposal_scale_multiplier": float(
                config["density_bridge_eval"].get("proposal_scale_multiplier", 1.0)
            ),
            "bridge_hmc_M": int(config["hmc_settings"]["M"]),
            "bridge_burn_in": int(config["hmc_settings"]["burn_in"]),
            "bridge_min_effective_samples": int(config["hmc_settings"]["min_effective_samples"]),
            "batch_size": int(config["training"]["batch_size"]),
            "decorrelation_weight": float(config["model"].get("decorrelation_weight", 0.0)),
            "x_adv_use_mean": bool(config["model"].get("x_adv_use_mean", False)),
            "fixed_variance_head": bool(config["model"].get("fixed_variance_head", False)),
            "fixed_variance_value": (
                None
                if "fixed_variance_value" not in config["model"]
                else float(config["model"]["fixed_variance_value"])
            ),
            "fit_fraction": float(config["density_bridge_eval"]["fit_fraction"]),
            "effective_fit_budget": int(config["density_bridge_eval"]["effective_fit_budget"]),
            "selected_test_indices_path": str(run_dir / "selected_test_indices.npz"),
            "dataset_fingerprint_path": str(run_dir / "dataset_fingerprint.json"),
            "generated_samples_path": str(run_dir / "generated_samples.npz"),
            "generation_diagnostics_path": str(run_dir / "generation_diagnostics.json"),
            "generation_diagnostics_skipped": bool(args.skip_generation_diagnostics),
            "truth_cache_arg": None if args.truth_cache is None else str(resolve_path(args.truth_cache)),
        }
    )
    if generation_diagnostics is not None:
        metrics.update(
            {
                "generated_mean_log_true_px": generation_diagnostics["generated_mean"]["density"][
                    "mean_log_true_px"
                ],
                "generated_sample_mean_log_true_px": generation_diagnostics["generated_sample"]["density"][
                    "mean_log_true_px"
                ],
            }
        )
        metrics.update(generation_metric_summary(generation_diagnostics["generated_mean"], "generated_mean"))
        metrics.update(generation_metric_summary(generation_diagnostics["generated_sample"], "generated_sample"))
    save_json(run_dir / "metrics.json", metrics)
    print("Finished. Outputs saved to:", run_dir)


if __name__ == "__main__":
    main()
