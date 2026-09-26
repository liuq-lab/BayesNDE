#!/usr/bin/env python3
"""The BGM-BS protocol shared by every UCI data set."""
from __future__ import annotations

import copy
import csv
import importlib.util
import json
import math
import os
import random
import re
import sys
import time
import types
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import tensorflow as tf
from scipy.special import logsumexp
import yaml

from bayesgm.datasets import Base_sampler
from bayesgm.models import BGM
from bayesnde.diagnostics.generation import (
    run_bgm_generation_diagnostics as run_uci_bgm_generation_diagnostics,
)
from bayesnde.estimators.bridge import BGM_BridgeDensityEstimator
from bayesnde.estimators.pointwise import load_keras_weights_compat
from bayesnde.training.generation_optimized import build_bgm_model
from experiments.uci.data import (
    DATA_ROOT,
    PROCESSED_ROOT,
    RAW_ARCHIVES,
    REPO_ROOT,
    PaperSplit,
    append_log,
    download,
    json_default,
    load_json_if_exists,
    load_yaml,
    paper_split,
    prepare_dataset,
    resolve_path,
    save_json,
    save_yaml,
    sha256_array,
    sha256_file,
)

METHOD_ORDER = ("bgm1",)
TEST_POINTS = {"BANK": 4521}


HERE = Path(__file__).resolve().parent


EXISTING = HERE


SRC_DENSITY = HERE.parent


OUTPUTS = HERE / "outputs"


SCREEN_POINTS = 200


VALIDATION_BLOCK_POINTS = 400


LEGACY_PROPOSAL_CONVENTION = "reported_shared_chi2"


def load_shared(alias: str, filename: str) -> ModuleType:
    if alias in sys.modules:
        return sys.modules[alias]
    source = EXISTING / filename
    if not source.exists():
        raise FileNotFoundError(f"Missing shared implementation: {source}")
    spec = importlib.util.spec_from_file_location(alias, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {source} as {alias}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(alias, None)
        raise
    return module


def isolate_base(runner: ModuleType) -> None:
    original = runner.base
    runner.base = types.SimpleNamespace(
        **{name: getattr(original, name) for name in dir(original) if not name.startswith("__")}
    )


def _layout(val_x: np.ndarray, test_x: np.ndarray, n_val: int) -> dict[str, Any]:
    return {
        "order": "validation_then_test",
        "validation_points": int(n_val),
        "test_points": int(len(test_x)),
        "screen_points": int(min(SCREEN_POINTS, n_val)),
        "validation_sha256": sha256_array(val_x[:n_val]),
        "screen_sha256": sha256_array(val_x[: min(SCREEN_POINTS, n_val)]),
        "test_sha256": sha256_array(test_x),
    }


def split_block(values: np.ndarray, layout: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    n_val = int(layout["validation_points"])
    n_test = int(layout["test_points"])
    if values.shape != (n_val + n_test,):
        raise ValueError(f"Expected {n_val + n_test} values, got {values.shape}")
    return values[:n_val], values[n_val:]


def _validation_prefix(n_available: int, requested: int) -> int:
    if n_available < 1:
        raise ValueError("Split has no validation rows")
    return int(min(int(requested), int(n_available)))


def screen_validation(split: Any, points: int = SCREEN_POINTS):
    val_x = np.asarray(split.val_x)
    n = _validation_prefix(len(val_x), points)
    positions = np.arange(n, dtype=np.int64)
    val_y = getattr(split, "val_y", None)
    y = None if val_y is None else np.asarray(val_y)[:n]
    return val_x[:n], y, positions


def eval_block(split: Any, validation_points: int = VALIDATION_BLOCK_POINTS):
    if getattr(split, "val_y", None) is None:
        x_eval, layout = unconditional_eval_block(split, validation_points)
        return x_eval, None, layout
    return conditional_eval_block(split, validation_points)


def unconditional_eval_block(split: Any, validation_points: int = VALIDATION_BLOCK_POINTS):
    val_x = np.asarray(split.val_x)
    test_x = np.asarray(split.selected_test_x)
    n_val = _validation_prefix(len(val_x), validation_points)
    x_eval = np.concatenate([val_x[:n_val], test_x], axis=0)
    return x_eval, _layout(val_x, test_x, n_val)


def conditional_eval_block(split: Any, validation_points: int = VALIDATION_BLOCK_POINTS):
    val_x = np.asarray(split.val_x)
    val_y = np.asarray(split.val_y)
    test_x = np.asarray(split.test_x)
    test_y = np.asarray(split.test_y)
    n_val = _validation_prefix(len(val_x), validation_points)
    x_eval = np.concatenate([val_x[:n_val], test_x], axis=0)
    y_eval = np.concatenate([val_y[:n_val], test_y], axis=0)
    return x_eval, y_eval, _layout(val_x, test_x, n_val)


def _legacy_sample_student_t_mixture(proposal: Any, size: int, rng: np.random.Generator) -> np.ndarray:
    comp = rng.choice(len(proposal.weights), size=int(size), p=proposal.weights)
    dim = proposal.loc.shape[1]
    out = np.empty((int(size), dim), dtype=np.float64)
    for k in range(len(proposal.weights)):
        mask = comp == k
        if not np.any(mask):
            continue
        n = int(mask.sum())
        if proposal.scoring == "product_univariate_t":
            normal = rng.normal(size=(n, dim)) * np.diag(proposal.scale_tril[k]).reshape(1, -1)
            chi2 = rng.chisquare(proposal.df, size=(n, 1))
        else:
            normal = rng.normal(size=(n, dim)) @ proposal.scale_tril[k].T
            chi2 = rng.chisquare(proposal.df, size=(n, 1))
        out[mask] = proposal.loc[k] + normal / np.sqrt(chi2 / proposal.df)
    return out


def install_legacy_proposal() -> dict[str, Any]:
    from bayesnde.estimators import conditional as bridge

    original = bridge.sample_student_t_mixture
    if getattr(original, "_is_legacy_proposal", False):
        return {"proposal_convention": LEGACY_PROPOSAL_CONVENTION, "already_installed": True}
    _legacy_sample_student_t_mixture._is_legacy_proposal = True
    bridge.sample_student_t_mixture = _legacy_sample_student_t_mixture
    return {
        "proposal_convention": LEGACY_PROPOSAL_CONVENTION,
        "patched": "conditional_bgm_bridge.sample_student_t_mixture",
        "matches": ["the reported unconditional and conditional BayesNDE paths"],
        "proposal_sampling": "multivariate_student_t_shared_chi_square",
        "proposal_scoring": "product_univariate_t",
        "proposal_scoring_matches_sampling": False,
        "rationale": "exact reproduction of the frozen reported estimator",
    }


def deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def materialize_bgm1_config(
    cfg: Mapping[str, Any],
    split: PaperSplit,
    z_dim: int,
    out_dir: Path,
    variant_name: str | None = None,
) -> dict[str, Any]:
    config = bgm1_config_for_variant(cfg, variant_name)
    config.setdefault("data", {})
    config.setdefault("model", {})
    config.setdefault("training", {})
    config.setdefault("egm", {})
    config.setdefault("density_bridge_eval", {})
    config.setdefault("hmc_settings", {})
    config.setdefault("generation_diagnostics", {})

    density_bridge = dict(config.get("density_bridge_eval", {}))
    density_bridge.setdefault("tol", 1.0e-5)
    density_bridge.setdefault("max_iter", 1000)
    density_bridge.setdefault("fit_fraction", 0.5)
    density_bridge.setdefault("use_neff", True)
    density_bridge.setdefault("proposal_scale_multiplier", 1.0)
    density_bridge["test_log_likelihood_points"] = int(len(split.selected_test_x))
    config["density_bridge_eval"] = density_bridge

    config["data"].update(
        {
            "name": f"UCI_{split.dataset}",
            "seed": int(config.get("seed", 42)),
            "dim": int(split.train_x.shape[1]),
            "z_dim": int(z_dim),
            "n": int(len(split.train_x)),
            "source": "checksum-verified public UCI archive with the fixed paper split",
        }
    )
    config["model"] = deep_merge(dict(config["model"]), dict(config.get("egm", {})))
    config["model"].update(
        {
            "dataset": f"BGM1_UCI_{split.dataset}",
            "output_dir": str(out_dir),
            "x_dim": int(split.train_x.shape[1]),
            "z_dim": int(z_dim),
        }
    )
    config["outputs"] = dict(config.get("outputs", {}))
    config["outputs"]["root_dir"] = str(out_dir)
    config["experiment_variant"] = variant_name or str(config.get("default_variant", "default"))
    return config


def public_bgm1_run_config(config: Mapping[str, Any]) -> dict[str, Any]:
    public = copy.deepcopy(dict(config))
    model_cfg = dict(public.get("model", {}))
    injected: dict[str, Any] = {}
    for key in ("x_dim", "z_dim"):
        if key in model_cfg:
            injected[key] = model_cfg.pop(key)
    public["model"] = model_cfg
    if injected:
        public["runtime_injected_dimensions"] = {
            **injected,
            "reason": "BGM constructor requires these dimensions; values are derived from the UCI dataset split, not user-tuned model hyperparameters.",
        }
    return public


def _variant_payload(cfg: Mapping[str, Any], variant_name: str, stack: tuple[str, ...] = ()) -> dict[str, Any]:
    variants = cfg.get("variants", {})
    if not isinstance(variants, Mapping) or variant_name not in variants:
        raise ValueError(f"Unknown bgm1 variant {variant_name!r}")
    if variant_name in stack:
        raise ValueError(f"Recursive bgm1 variant inheritance: {' -> '.join((*stack, variant_name))}")
    raw = copy.deepcopy(dict(variants[variant_name]))
    parent = raw.pop("extends", None)
    raw.pop("description", None)
    if parent is None:
        return raw
    base = _variant_payload(cfg, str(parent), (*stack, variant_name))
    return deep_merge(base, raw)


def bgm1_config_for_variant(cfg: Mapping[str, Any], variant_name: str | None = None) -> dict[str, Any]:
    control_keys = {"variants", "default_variants", "default_variant", "description"}
    config = {key: copy.deepcopy(value) for key, value in cfg.items() if key not in control_keys}
    if variant_name:
        config = deep_merge(config, _variant_payload(cfg, variant_name))
    config["experiment_variant"] = variant_name or str(cfg.get("default_variant", "default"))
    return config


def bgm1_variant_names(cfg: Mapping[str, Any], requested: str | None) -> list[str | None]:
    if requested is None or not requested.strip():
        return [None]
    if requested.strip().lower() in {"default", "none"}:
        return [None]
    values = [item.strip() for item in requested.split(",") if item.strip()]
    variants = cfg.get("variants", {})
    invalid = [value for value in values if value not in variants]
    if invalid:
        raise ValueError(f"Unsupported bgm1 variants: {invalid}")
    return values


def bgm1_generation_summary(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    generated = diagnostics.get("generated_sample", {})
    train_metrics = generated.get("two_sample_vs_train", {}) if isinstance(generated, Mapping) else {}
    test_metrics = generated.get("two_sample_vs_test", {}) if isinstance(generated, Mapping) else {}
    sigma = diagnostics.get("generator_sigma_square_global", {})
    umap = diagnostics.get("umap", {})
    return {
        "generated_vs_train_wasserstein_mean": train_metrics.get("wasserstein_mean"),
        "generated_vs_train_mmd_rbf": train_metrics.get("mmd_rbf"),
        "generated_vs_train_lisi_normalized_mean": train_metrics.get("lisi_normalized_mean"),
        "generated_vs_test_wasserstein_mean": test_metrics.get("wasserstein_mean"),
        "generated_vs_test_mmd_rbf": test_metrics.get("mmd_rbf"),
        "generated_vs_test_lisi_normalized_mean": test_metrics.get("lisi_normalized_mean"),
        "generator_sigma_square_mean": sigma.get("mean") if isinstance(sigma, Mapping) else None,
        "embedding_method": umap.get("method") if isinstance(umap, Mapping) else None,
    }


def bgm1_weights_paths(out_dir: Path) -> dict[str, Path]:
    weights_dir = out_dir / "checkpoint" / "final_weights"
    return {
        "weights_dir": weights_dir,
        "generator_weights": weights_dir / "g_net_final.weights.h5",
        "encoder_weights": weights_dir / "e_net_final.weights.h5",
        "latent_discriminator_weights": weights_dir / "dz_net_final.weights.h5",
        "data_discriminator_weights": weights_dir / "dx_net_final.weights.h5",
    }


def build_bgm1_networks_for_weight_io(model: BGM) -> None:
    x_dim = int(model.params["x_dim"])
    z_dim = int(model.params["z_dim"])
    z0 = np.zeros((1, z_dim), dtype=np.float32)
    x0 = np.zeros((1, x_dim), dtype=np.float32)
    model.g_net(z0, training=False)
    model.e_net(x0, training=False)
    model.dz_net(z0, training=False)
    model.dx_net(x0, training=False)


def save_bgm1_weights(
    model: BGM,
    out_dir: Path,
    config: Mapping[str, Any],
    *,
    trained_in_run: bool = True,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    paths = bgm1_weights_paths(out_dir)
    paths["weights_dir"].mkdir(parents=True, exist_ok=True)
    build_bgm1_networks_for_weight_io(model)
    model.g_net.save_weights(str(paths["generator_weights"]))
    model.e_net.save_weights(str(paths["encoder_weights"]))
    model.dz_net.save_weights(str(paths["latent_discriminator_weights"]))
    model.dx_net.save_weights(str(paths["data_discriminator_weights"]))
    manifest = {
        "trained_in_run": bool(trained_in_run),
        "method": "BayesNDE",
        "method_variant": "bgm_bs",
        "experiment_variant": str(config.get("experiment_variant", "default")),
        "source_experiment_basis": str(config.get("source_experiment_basis", "clean_uci_bgm1")),
        "checkpoint_dir": str(paths["weights_dir"]),
        **{key: str(value) for key, value in paths.items() if key != "weights_dir"},
        "epochs": int(config["training"]["epochs"]),
        "egm_n_iter": int(
            config.get("runtime_resume", {}).get(
                "requested_total_egm_n_iter",
                config["training"].get("egm_n_iter", 0),
            )
        ),
        "effective_egm_n_iter_this_process": int(config["training"].get("egm_n_iter", 0)),
        "use_egm_init": bool(config["training"].get("use_egm_init", False)),
    }
    if extra:
        manifest.update(dict(extra))
    save_json(out_dir / "weights_manifest.json", manifest)
    save_yaml(out_dir / "weights_manifest.yaml", manifest)
    return manifest


def maybe_load_bgm1_weights(model: BGM, out_dir: Path) -> dict[str, Any] | None:
    manifest_path = out_dir / "weights_manifest.json"
    paths = bgm1_weights_paths(out_dir)
    required = (
        paths["generator_weights"],
        paths["encoder_weights"],
        paths["latent_discriminator_weights"],
        paths["data_discriminator_weights"],
    )
    if not manifest_path.exists() or not all(path.exists() for path in required):
        return None
    build_bgm1_networks_for_weight_io(model)
    load_keras_weights_compat(model.g_net, paths["generator_weights"])
    load_keras_weights_compat(model.e_net, paths["encoder_weights"])
    load_keras_weights_compat(model.dz_net, paths["latent_discriminator_weights"])
    load_keras_weights_compat(model.dx_net, paths["data_discriminator_weights"])
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def bgm1_partial_egm_state_path(out_dir: Path) -> Path:
    return out_dir / "partial_egm_resume_state.json"


def find_latest_bgm1_partial_egm_checkpoint(model: BGM, out_dir: Path) -> dict[str, Any] | None:
    checkpoint_root = out_dir / "checkpoints" / str(model.params["dataset"])
    if not checkpoint_root.exists():
        return None
    state = load_json_if_exists(bgm1_partial_egm_state_path(out_dir))
    run_offsets = {
        str(run_name): int(run_info.get("base_egm_iter", 0))
        for run_name, run_info in dict(state.get("runs", {})).items()
        if isinstance(run_info, Mapping)
    }
    candidates: list[dict[str, Any]] = []
    for run_dir in checkpoint_root.iterdir():
        if not run_dir.is_dir():
            continue
        base_iter = int(run_offsets.get(run_dir.name, 0))
        for generator_path in run_dir.glob("weights_at_egm_init_*_generator.weights.h5"):
            match = None
            name = generator_path.name
            if name.startswith("weights_at_egm_init_") and name.endswith("_generator.weights.h5"):
                raw = name.removeprefix("weights_at_egm_init_").removesuffix("_generator.weights.h5")
                if raw.isdigit():
                    match = int(raw)
            if match is None:
                continue
            encoder_path = run_dir / f"weights_at_egm_init_{match}_encoder.weights.h5"
            if not encoder_path.exists():
                continue
            candidates.append(
                {
                    "phase": "egm_init",
                    "run_dir": str(run_dir),
                    "run_name": run_dir.name,
                    "base_egm_iter": int(base_iter),
                    "raw_egm_iter": int(match),
                    "cumulative_egm_iter": int(base_iter + match),
                    "generator_weights": str(generator_path),
                    "encoder_weights": str(encoder_path),
                    "mtime": float(max(generator_path.stat().st_mtime, encoder_path.stat().st_mtime)),
                }
            )
    if not candidates:
        return None
    candidates.sort(key=lambda item: (int(item["cumulative_egm_iter"]), float(item["mtime"])))
    return candidates[-1]


def maybe_resume_bgm1_partial_egm(
    model: BGM,
    out_dir: Path,
    config: dict[str, Any],
    log_path: Path,
    split: PaperSplit,
) -> dict[str, Any] | None:
    train_cfg = config["training"]
    if not bool(train_cfg.get("use_egm_init", False)):
        return None
    requested_egm = int(train_cfg.get("egm_n_iter", 0))
    if requested_egm <= 0:
        return None
    latest = find_latest_bgm1_partial_egm_checkpoint(model, out_dir)
    if latest is None:
        return None
    cumulative = int(latest["cumulative_egm_iter"])
    if cumulative <= 0:
        return None
    build_bgm1_networks_for_weight_io(model)
    model.g_net.load_weights(str(latest["generator_weights"]))
    model.e_net.load_weights(str(latest["encoder_weights"]))
    remaining = max(0, requested_egm - cumulative)
    train_cfg["egm_n_iter"] = int(remaining)
    train_cfg["use_egm_init"] = bool(remaining > 0)
    resume_record = {
        "dataset": split.dataset,
        "experiment_variant": str(config.get("experiment_variant", "default")),
        "requested_total_egm_n_iter": int(requested_egm),
        "loaded_cumulative_egm_iter": int(cumulative),
        "remaining_egm_n_iter_for_this_run": int(remaining),
        "loaded_checkpoint": latest,
        "new_run_timestamp": str(model.timestamp),
        "note": "Generator/encoder EGM HDF5 checkpoint resume; discriminators are reinitialized because EGM snapshots only contain G/E.",
    }
    state_path = bgm1_partial_egm_state_path(out_dir)
    state = load_json_if_exists(state_path)
    runs = dict(state.get("runs", {}))
    runs[str(model.timestamp)] = {
        "base_egm_iter": int(cumulative),
        "planned_remaining_egm_n_iter": int(remaining),
        "source_run_name": str(latest["run_name"]),
        "source_raw_egm_iter": int(latest["raw_egm_iter"]),
        "source_cumulative_egm_iter": int(cumulative),
    }
    state.update(
        {
            "requested_total_egm_n_iter": int(requested_egm),
            "last_resume": resume_record,
            "runs": runs,
        }
    )
    save_json(state_path, state)
    append_log(
        log_path,
        f"resume partial EGM for {split.dataset} bgm1/{config['experiment_variant']}: "
        f"loaded {cumulative}/{requested_egm}, remaining={remaining}",
    )
    config["runtime_resume"] = resume_record
    return resume_record


def save_bgm1_likelihood(
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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            x_test=x_test,
            test_indices=test_indices,
            sample_size=np.asarray(sample_size, dtype=np.int64),
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
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_existing_bgm1_likelihood(
    path: Path,
    x_test: np.ndarray,
    test_indices: np.ndarray,
    n_repeats: int,
    sample_size: int,
) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    with np.load(path) as data:
        required = {
            "x_test",
            "test_indices",
            "sample_size",
            "log_px_repeats",
            "log_px",
            "px",
            "log_px_sd",
            "proposal_is_ess_mean",
            "hmc_min_ess",
            "hmc_acceptance_rate",
            "bridge_iterations_mean",
            "bridge_convergence_rate_point",
            "bridge_abs_delta_mean",
        }
        if not required.issubset(data.files):
            return None
        if data["log_px_repeats"].shape != (len(x_test), int(n_repeats)):
            return None
        if not np.array_equal(np.asarray(data["test_indices"], dtype=np.int64), np.asarray(test_indices, dtype=np.int64)):
            return None
        if not np.allclose(np.asarray(data["x_test"], dtype=np.float32), np.asarray(x_test, dtype=np.float32)):
            return None
        if int(np.asarray(data["sample_size"]).item()) != int(sample_size):
            return None
        return {key: np.asarray(data[key]) for key in data.files}


def evaluate_bgm1_bridge_log_likelihood(
    estimator: BGM_BridgeDensityEstimator,
    x_test: np.ndarray,
    test_indices: np.ndarray,
    config: Mapping[str, Any],
    output_path: Path,
    log_path: Path,
) -> dict[str, np.ndarray]:
    density_cfg = config["density_bridge_eval"]
    n_points = int(x_test.shape[0])
    n_repeats = int(density_cfg["n_repeats"])
    sample_size = int(density_cfg["S"])
    existing = load_existing_bgm1_likelihood(output_path, x_test, test_indices, n_repeats, sample_size)
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

    save_every = int(density_cfg.get("test_log_likelihood_save_every", density_cfg.get("save_every", 25)))
    for idx, point in enumerate(np.asarray(x_test, dtype=np.float32)):
        if np.isfinite(log_px[idx]) and np.all(np.isfinite(log_px_repeats[idx])):
            continue
        result = estimator.estimate(
            point.astype(np.float32),
            K=int(density_cfg.get("K", 5)),
            S=sample_size,
            nu=float(density_cfg.get("nu", 3.0)),
            epsilon=float(density_cfg.get("epsilon", 0.05)),
            n_repeats=n_repeats,
            hmc_settings=config["hmc_settings"],
            eval_batch_size=int(density_cfg.get("eval_batch_size", 4096)),
            bridge_tol=float(density_cfg.get("tol", density_cfg.get("bridge_tol", 1.0e-5))),
            bridge_max_iter=int(density_cfg.get("max_iter", density_cfg.get("bridge_max_iter", 1000))),
            fit_fraction=float(density_cfg.get("fit_fraction", 0.5)),
            use_neff=bool(density_cfg.get("use_neff", True)),
            proposal_scale_multiplier=float(density_cfg.get("proposal_scale_multiplier", 1.0)),
            proposal_covariance_mode=str(density_cfg.get("proposal_covariance_mode", "fitted")),
            proposal_scale=density_cfg.get("proposal_scale"),
            proposal_center=str(density_cfg.get("proposal_center", "gmm")),
            proposal_scoring=str(density_cfg.get("proposal_scoring", "multivariate_t")),
            return_proposal=False,
        )
        diag = result["diagnostics"][0]
        repeats = np.asarray(result["log_px_repeats"], dtype=np.float64)
        if repeats.shape != (n_repeats,):
            raise RuntimeError(f"BGM1 expected {n_repeats} repeats for point {idx}, got {repeats.shape}")
        bridge_iter = np.asarray(diag["bridge_iterations_repeats"], dtype=np.float64)
        bridge_conv = np.asarray(diag["bridge_converged_repeats"], dtype=bool)
        bridge_delta = np.asarray(diag["bridge_abs_delta_repeats"], dtype=np.float64)
        proposal_ess = np.asarray(diag["proposal_is_ess_repeats"], dtype=np.float64)
        log_px_repeats[idx, :] = repeats
        log_px[idx] = float(logsumexp(repeats) - math.log(len(repeats)))
        px[idx] = float(np.exp(log_px[idx])) if log_px[idx] < 700.0 else math.inf
        log_px_sd[idx] = float(result["log_px_repeat_sd"])
        proposal_is_ess_mean[idx] = float(np.mean(proposal_ess))
        hmc_min_ess[idx] = float(diag["hmc_min_ess"])
        hmc_acceptance_rate[idx] = float(diag["hmc_acceptance_rate"])
        bridge_iterations_mean[idx] = float(np.mean(bridge_iter))
        bridge_convergence_rate_point[idx] = float(np.mean(bridge_conv))
        bridge_abs_delta_mean[idx] = float(np.mean(bridge_delta))
        done = int(np.sum(np.isfinite(log_px)))
        if done % max(1, save_every) == 0 or done == n_points:
            save_bgm1_likelihood(
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
            append_log(log_path, f"BGM1 bridge {output_path.name}: {done}/{n_points} mean={np.nanmean(log_px):.6f}")
    save_bgm1_likelihood(
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


def parse_list(text: str, allowed: Iterable[str]) -> list[str]:
    allowed_set = set(allowed)
    values = [item.strip() for item in text.split(",") if item.strip()]
    invalid = [item for item in values if item not in allowed_set]
    if invalid:
        raise ValueError(f"Unsupported values: {invalid}")
    return values


def configure_tensorflow(cpu: bool) -> None:
    if cpu:
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            pass
        return
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def effective_config(config: Mapping[str, Any], *, smoke: bool) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    if not smoke:
        return result
    smoke_cfg = result["smoke"]
    if "bgm1" in result:
        bgm1 = result["bgm1"]
        training = bgm1.setdefault("training", {})
        training["batch_size"] = min(int(training.get("batch_size", 64)), 64)
        training["epochs"] = 1
        training["epochs_per_eval"] = 1
        training["egm_n_iter"] = 10
        training["egm_batches_per_eval"] = 5
        training["verbose"] = 0
        density_bridge = bgm1.setdefault("density_bridge_eval", {})
        density_bridge["K"] = 2
        density_bridge["S"] = 32
        density_bridge["n_repeats"] = 1
        density_bridge["eval_batch_size"] = 64
        density_bridge["save_every"] = 4
        hmc = bgm1.setdefault("hmc_settings", {})
        hmc["M"] = 8
        hmc["burn_in"] = 4
        hmc["num_chains"] = 2
        hmc["num_leapfrog_steps"] = 3
        hmc["min_effective_samples"] = 0
        hmc["max_retries"] = 0
        diag = bgm1.setdefault("generation_diagnostics", {})
        diag["n_samples"] = 64
        diag["reference_max_points"] = 64
        diag["umap_max_points_per_group"] = 32
        diag["umap_n_neighbors"] = 5
    return result


SEED = 1024


SAVE_EPOCHS = tuple(
    sorted({1, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 250, *range(300, 2001, 50)})
)


OUTPUT_ROOT = SRC_DENSITY / "outputs/uci_paper"


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


def materialize_config(dataset: str, x_dim: int, out_dir: Path) -> dict[str, Any]:
    return {
        "seed": SEED,
        "method": "BayesNDE",
        "data": {
            "name": dataset,
            "source": "checksum-verified public UCI archive with the fixed paper split",
            "seed": 42,
            "dim": int(x_dim),
            "z_dim": int(x_dim),
        },
        "model": {
            "dataset": f"BGM_BS_UCI_{dataset}",
            "output_dir": str(out_dir),
            "save_res": False,
            "save_model": True,
            "use_bnn": False,
            "x_dim": int(x_dim),
            "z_dim": int(x_dim),
            "random_seed": SEED,
            "g_network_type": "residual",
            "e_network_type": "residual",
            "g_units": [256] * 5,
            "e_units": [256] * 5,
            "dz_units": [256, 256, 128, 64],
            "dx_units": [256, 256, 128, 64],
            "lr": 1.0e-3,
            "lr_theta": 0.005,
            "lr_z": 0.005,
            "optimizer": "adam",
            "weight_decay": 0.0,
            "kl_weight": 5.0e-5,
            "g_d_freq": 1,
            "use_z_rec": True,
            "alpha": 0.0,
            "gamma": 0.0,
            "cycle_weight": 5.0,
            "x_cycle_weight": 3.0,
            "z_cycle_weight": 1.0,
            "g_adv_weight": 1.0,
            "e_adv_weight": 1.0,
            "marginal_moment_weight": 0.0,
            "covariance_weight": 0.0,
            "marginal_mmd_weight": 1000.0,
            "joint_mmd_weight": 100.0,
            "sliced_wasserstein_weight": 300.0,
            "sliced_wasserstein_projections": 64,
            "sliced_wasserstein_seed": 20260831,
            "decorrelation_weight": 0.3,
            "decorrelation_target": "train",
            "mmd_scales": [0.05, 0.1, 0.2, 0.5, 1.0],
            "x_cycle_use_mean": False,
            "z_cycle_use_mean": False,
            "variance_target": 0.01,
            "variance_target_weight": 0.0,
            "variance_log_target": 0.01,
            "variance_log_target_weight": 0.0,
            "variance_log_eps": 1.0e-8,
            "factorized_generator": False,
            "low_rank_generator": False,
            "iterative_empirical_corr_weight": 0.3,
            "iterative_prior_mmd_weight": 1000.0,
        },
        "training": {
            "batch_size": 256,
            "egm_n_iter": 1000,
            "egm_batches_per_eval": 50,
            "epochs": 2000,
            "epochs_per_eval": 50,
            "save_epochs": list(SAVE_EPOCHS),
        },
        "epoch_selection": {
            "split": "validation",
            "validation_points": 400,
            "epoch_zero_selectable": False,
            "decoder_prefilter_top_k": 5,
            "criterion": "max fixed-validation BGM-BS bridge mean log likelihood",
            "K": 5,
            "S": 2000,
            "nu": 3.0,
            "epsilon": 0.05,
            "n_repeats": 1,
            "M": 100,
            "burn_in": 100,
        },
        "density_bridge_eval": {
            "K": 5,
            "S": 20000,
            "nu": 3.0,
            "epsilon": 0.05,
            "n_repeats": 1,
            "eval_batch_size": 1024,
            "likelihood": "gaussian",
            "variance_floor": 0.01,
            "seed": 42,
            "save_every": 1,
            "test_log_likelihood_save_every": 1,
            "tol": 1.0e-5,
            "max_iter": 1000,
            "fit_fraction": 0.5,
            "use_neff": True,
            "proposal_scale_multiplier": 1.0,
            "proposal_covariance_mode": "fixed_isotropic",
            "proposal_scale": 0.005,
            "proposal_center": "fitted_gmm",
            "proposal_scoring": "product_univariate_t",
        },
        "hmc_settings": {
            "M": 1600,
            "burn_in": 800,
            "step_size": 0.003,
            "num_leapfrog_steps": 10,
            "target_accept_prob": 0.75,
            "num_chains": 4,
            "initial_state": "encoder",
            "initial_state_scale": 0.01,
            "min_effective_samples": 0,
            "max_retries": 0,
            "retry_multiplier": 1.5,
            "strict_ess": False,
            "proposal_covariance_floor": 0.001,
        },
        "protocol": {
            "source_reference_job": 24755657,
            "x_dim_equals_z_dim": True,
            "test_used_for_epoch_selection": False,
            "proposal_parameter_historical_name": "scale",
            "proposal_scale_tril_diagonal": 0.005,
            "proposal_covariance": "0.005^2 I",
            "proposal_scoring": "product_univariate_t",
        },
    }


def validation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(config))
    selection = config["epoch_selection"]
    result["density_bridge_eval"].update(
        K=int(selection["K"]),
        S=int(selection["S"]),
        nu=float(selection["nu"]),
        epsilon=float(selection["epsilon"]),
        n_repeats=int(selection["n_repeats"]),
        proposal_covariance_mode="fixed_isotropic",
        proposal_scale=0.005,
        proposal_center="fitted_gmm",
        proposal_scoring="product_univariate_t",
        save_every=1,
        test_log_likelihood_save_every=1,
    )
    result["hmc_settings"].update(M=int(selection["M"]), burn_in=int(selection["burn_in"]))
    return result


def run_root(dataset: str) -> Path:
    return OUTPUT_ROOT / dataset / "bgm-bs"


def save_protocol_inputs(dataset: str, split: Any, config: Mapping[str, Any]) -> None:
    root = run_root(dataset)
    root.mkdir(parents=True, exist_ok=True)
    (root / "resolved_config.yaml").write_text(
        yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8"
    )
    save_json(
        root / "dataset_fingerprint.json",
        {
            **split.metadata,
            "train_sha256": sha256_array(split.train_x),
            "validation_sha256": sha256_array(split.val_x),
            "test_sha256": sha256_array(split.test_x),
            "x_dim": int(split.train_x.shape[1]),
            "z_dim": int(split.train_x.shape[1]),
        },
    )
    n_validation = min(int(config["epoch_selection"]["validation_points"]), len(split.val_x))
    np.savez_compressed(
        root / "fixed_validation_subset.npz",
        x_validation=np.asarray(split.val_x[:n_validation], dtype=np.float32),
        validation_indices=np.asarray(split.val_indices[:n_validation], dtype=np.int64),
    )


def load_curve(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(json.loads(path.read_text(encoding="utf-8")).get("rows", []))


def save_curve(root: Path, rows: list[dict[str, Any]]) -> None:
    rows = sorted({int(row["epoch"]): row for row in rows}.values(), key=lambda row: int(row["epoch"]))
    atomic_json(root / "iterative_validation_curve.json", {"rows": rows})
    columns = [
        "epoch", "validation_decoder_mean_log_likelihood", "loss_x",
        "loss_z", "empirical_corr_loss", "generator_weights",
    ]
    temporary = root / f".iterative_validation_curve.{os.getpid()}.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in columns} for row in rows])
    os.replace(temporary, root / "iterative_validation_curve.csv")


def build_resume_checkpoint(model: Any, data_z: tf.Variable, root: Path):
    checkpoint = tf.train.Checkpoint(
        g_net=model.g_net,
        e_net=model.e_net,
        dz_net=model.dz_net,
        dx_net=model.dx_net,
        g_optimizer=model.g_optimizer,
        posterior_optimizer=model.posterior_optimizer,
        data_z=data_z,
    )
    manager = tf.train.CheckpointManager(checkpoint, str(root / "checkpoint/iterative_resume"), max_to_keep=1)
    return checkpoint, manager


def decoder_log_likelihood(model: Any, values: np.ndarray, batch_size: int = 1024) -> float:
    chunks: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        x = tf.convert_to_tensor(values[start : start + batch_size], tf.float32)
        z = model.e_net(x, training=False)
        mean, variance = model._decode_generator(z, training=False)
        variance = tf.maximum(variance, tf.cast(1.0e-6, variance.dtype))
        logp = -0.5 * tf.reduce_sum(
            tf.math.log(tf.cast(2.0 * math.pi, variance.dtype) * variance)
            + tf.square(x - mean) / variance,
            axis=1,
        )
        chunks.append(np.asarray(logp.numpy(), dtype=np.float64))
    return float(np.mean(np.concatenate(chunks)))


def newest_egm_checkpoint(root: Path) -> tuple[int, Path, Path] | None:
    pattern = re.compile(r"egm_step_(\d+)_generator\.weights\.h5$")
    found = []
    for generator in (root / "checkpoint/egm").glob("egm_step_*_generator.weights.h5"):
        match = pattern.search(generator.name)
        if match:
            step = int(match.group(1))
            encoder = generator.with_name(f"egm_step_{step:04d}_encoder.weights.h5")
            if encoder.exists():
                found.append((step, generator, encoder))
    return max(found, default=None, key=lambda value: value[0])


def train_egm(model: Any, train_x: np.ndarray, config: Mapping[str, Any], root: Path) -> None:
    target = int(config["training"]["egm_n_iter"])
    checkpoint = newest_egm_checkpoint(root)
    start = 0
    if checkpoint is not None:
        start, generator, encoder = checkpoint
        model.g_net.load_weights(str(generator))
        model.e_net.load_weights(str(encoder))
        if start >= target:
            return
    batch_size = int(config["training"]["batch_size"])
    sampler = Base_sampler(x=train_x, y=train_x, v=train_x, batch_size=batch_size, normalize=False)
    model.data_sampler = sampler
    every = int(config["training"]["egm_batches_per_eval"])
    for step in range(start + 1, target + 1):
        for _ in range(int(model.params["g_d_freq"])):
            batch_x, _, _ = sampler.next_batch()
            batch_z = model.z_sampler.get_batch(batch_size)
            model.train_disc_step(batch_z, batch_x)
        batch_x, _, _ = sampler.next_batch()
        batch_z = model.z_sampler.get_batch(batch_size)
        losses = model.train_gen_step(batch_z, batch_x)
        if step % every == 0 or step == target:
            prefix = root / "checkpoint/egm" / f"egm_step_{step:04d}"
            atomic_weights(model.g_net, Path(str(prefix) + "_generator.weights.h5"))
            atomic_weights(model.e_net, Path(str(prefix) + "_encoder.weights.h5"))
            append_log(root / "run.log", f"EGM step={step}/{target} total_loss={float(losses[-1]):.8g}")


def train_iterative(model: Any, split: Any, config: Mapping[str, Any], root: Path) -> None:
    train_x = np.asarray(split.train_x, dtype=np.float32)
    model.g_net.trainable = True
    model.e_net.trainable = False
    model.dz_net.trainable = False
    model.dx_net.trainable = False
    correlation = np.corrcoef(train_x.astype(np.float64), rowvar=False)
    if np.ndim(correlation) == 0:
        correlation = np.asarray([[1.0]])
    if not np.isfinite(correlation).all():
        raise ValueError(f"{split.dataset}: non-finite training correlation matrix")
    model.params["iterative_empirical_corr_target"] = correlation.tolist()

    encoded: list[np.ndarray] = []
    for start in range(0, len(train_x), 1024):
        encoded.append(np.asarray(model.e_net(train_x[start : start + 1024], training=False).numpy()))
    model.data_z = tf.Variable(np.vstack(encoded).astype(np.float32), name="Latent Variable", trainable=True)
    resume_checkpoint, resume_manager = build_resume_checkpoint(model, model.data_z, root)
    start_epoch = 0
    if resume_manager.latest_checkpoint:
        resume_checkpoint.restore(resume_manager.latest_checkpoint).expect_partial()
        match = re.search(r"-(\d+)$", resume_manager.latest_checkpoint)
        start_epoch = int(match.group(1)) if match else 0
        append_log(root / "run.log", f"resume iterative epoch={start_epoch}")

    rows = load_curve(root / "iterative_validation_curve.json")
    batch_size = int(config["training"]["batch_size"])
    max_epoch = int(config["training"]["epochs"])
    save_epochs = set(int(value) for value in config["training"]["save_epochs"])
    for epoch in range(start_epoch + 1, max_epoch + 1):
        order = np.random.default_rng(SEED + 700000 + epoch).permutation(len(train_x))
        losses_x: list[float] = []
        losses_z: list[float] = []
        losses_corr: list[float] = []
        stride = min(batch_size, len(train_x))
        for start in range(0, len(train_x) - stride + 1, stride):
            indices = order[start : start + stride]
            batch_z = tf.Variable(tf.gather(model.data_z, indices), trainable=True)
            batch_x = train_x[indices]
            loss_x = model.update_g_net(batch_z, batch_x)
            corr_weight = float(model.params.get("iterative_empirical_corr_weight", 0.0))
            if corr_weight > 0.0:
                if not hasattr(model, "update_iterative_empirical_corr"):
                    raise TypeError(
                        "iterative_empirical_corr_weight requires a model with "
                        "update_iterative_empirical_corr"
                    )
                losses_corr.append(float(model.update_iterative_empirical_corr(batch_x)))
            loss_z = model.update_latent_variable_sgd(batch_z, batch_x)
            model.data_z.scatter_nd_update(tf.expand_dims(indices, axis=1), batch_z)
            losses_x.append(float(loss_x))
            losses_z.append(float(loss_z))
        if epoch not in save_epochs and epoch != max_epoch:
            continue
        generator_path = root / "checkpoint/iterative" / f"weights_at_{epoch}_generator.weights.h5"
        atomic_weights(model.g_net, generator_path)
        score = decoder_log_likelihood(model, np.asarray(split.val_x, dtype=np.float32))
        row = {
            "epoch": epoch,
            "validation_decoder_mean_log_likelihood": score,
            "loss_x": float(np.mean(losses_x)),
            "loss_z": float(np.mean(losses_z)),
            "empirical_corr_loss": float(np.mean(losses_corr)) if losses_corr else 0.0,
            "generator_weights": str(generator_path),
        }
        rows.append(row)
        save_curve(root, rows)
        resume_manager.save(checkpoint_number=epoch)
        append_log(root / "run.log", f"iterative epoch={epoch}/{max_epoch} decoder_val_ll={score:.8g}")


def select_epoch(model: Any, split: Any, config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    frozen_path = root / "frozen_validation_selection.json"
    if frozen_path.exists():
        return json.loads(frozen_path.read_text(encoding="utf-8"))
    rows = [row for row in load_curve(root / "iterative_validation_curve.json") if int(row["epoch"]) > 0]
    if not rows:
        raise RuntimeError("No nonzero iterative checkpoints are available")
    top_k = int(config["epoch_selection"]["decoder_prefilter_top_k"])
    candidates = sorted(
        rows, key=lambda row: float(row["validation_decoder_mean_log_likelihood"]), reverse=True
    )[:top_k]
    fixed = np.load(root / "fixed_validation_subset.npz")
    x_validation = np.asarray(fixed["x_validation"], dtype=np.float32)
    validation_indices = np.asarray(fixed["validation_indices"], dtype=np.int64)
    bridge_cfg = validation_config(config)
    results: list[dict[str, Any]] = []
    for row in candidates:
        epoch = int(row["epoch"])
        model.g_net.load_weights(str(row["generator_weights"]))
        estimator = BGM_BridgeDensityEstimator(
            model=model,
            likelihood="gaussian",
            random_seed=SEED + 800000 + epoch,
            variance_floor=float(bridge_cfg["density_bridge_eval"]["variance_floor"]),
        )
        output = root / "validation" / f"epoch_{epoch:04d}" / "bridge_log_likelihood.npz"
        values = evaluate_bgm1_bridge_log_likelihood(
            estimator, x_validation, validation_indices, bridge_cfg, output, root / "run.log"
        )
        log_px = np.asarray(values["log_px"], dtype=np.float64)
        results.append(
            {
                **row,
                "validation_bridge_mean_log_likelihood": float(np.mean(log_px)),
                "validation_bridge_standard_deviation": float(np.std(log_px)),
                "validation_points": int(len(log_px)),
                "bridge_artifact": str(output),
            }
        )
    best = max(results, key=lambda row: float(row["validation_bridge_mean_log_likelihood"]))
    if int(best["epoch"]) <= 0:
        raise AssertionError("epoch zero must never be selected")
    model.g_net.load_weights(str(best["generator_weights"]))
    payload = {
        "criterion": "max fixed-validation BGM-BS bridge mean log likelihood",
        "decoder_prefilter_only": True,
        "decoder_prefilter_top_k": top_k,
        "selected": best,
        "bridge_candidates": results,
        "validation_indices_sha256": sha256_array(validation_indices),
        "test_points_used_for_selection": 0,
        "frozen_before_test": True,
    }
    atomic_json(frozen_path, payload)
    return payload


def train_and_select(dataset: str) -> dict[str, Any]:
    split = load_split(dataset)
    root = run_root(dataset)
    config = materialize_config(dataset, split.train_x.shape[1], root)
    save_protocol_inputs(dataset, split, config)
    if (root / "training_complete.json").exists() and maybe_load_selected_model(dataset, split, config)[1]:
        return json.loads((root / "training_complete.json").read_text(encoding="utf-8"))
    np.random.seed(SEED)
    random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    model = build_bgm_model(config["model"], timestamp="bgm_bs_best", random_seed=SEED)
    build_bgm1_networks_for_weight_io(model)
    train_egm(model, np.asarray(split.train_x, dtype=np.float32), config, root)
    final_egm = newest_egm_checkpoint(root)
    if final_egm is None or final_egm[0] != int(config["training"]["egm_n_iter"]):
        raise RuntimeError("EGM did not reach its fixed 1000-step checkpoint")
    model.g_net.load_weights(str(final_egm[1]))
    model.e_net.load_weights(str(final_egm[2]))
    train_iterative(model, split, config, root)
    selection = select_epoch(model, split, config, root)
    manifest = save_bgm1_weights(
        model,
        root,
        config,
        extra={
            "method": "BayesNDE",
            "selected_epoch": int(selection["selected"]["epoch"]),
            "selection_criterion": selection["criterion"],
            "test_used_for_selection": False,
            "proposal_scale": 0.005,
            "proposal_scoring": "product_univariate_t",
        },
    )
    complete = {
        "dataset": dataset,
        "method": "BayesNDE",
        "training_complete": True,
        "x_dim": int(split.train_x.shape[1]),
        "z_dim": int(split.train_x.shape[1]),
        "selected_epoch": int(selection["selected"]["epoch"]),
        "best_validation_mean_log_likelihood": float(
            selection["selected"]["validation_bridge_mean_log_likelihood"]
        ),
        "weights_manifest": manifest,
    }
    atomic_json(root / "training_complete.json", complete)
    return complete


def maybe_load_selected_model(dataset: str, split: Any, config: Mapping[str, Any]):
    params = dict(config["model"])
    params["save_model"] = False
    model = build_bgm_model(params, timestamp="selected_eval", random_seed=SEED)
    manifest = maybe_load_bgm1_weights(model, run_root(dataset))
    return model, manifest


def evaluate_test(dataset: str) -> dict[str, Any]:
    split = load_split(dataset)
    root = run_root(dataset)
    config = materialize_config(dataset, split.train_x.shape[1], root)
    if not (root / "frozen_validation_selection.json").exists():
        raise FileNotFoundError("Frozen validation selection is required before test evaluation")
    model, manifest = maybe_load_selected_model(dataset, split, config)
    if manifest is None:
        raise FileNotFoundError("Selected BGM-BS weights are unavailable")
    estimator = BGM_BridgeDensityEstimator(
        model=model,
        likelihood="gaussian",
        random_seed=SEED + 900000,
        variance_floor=float(config["density_bridge_eval"]["variance_floor"]),
    )
    artifact = root / "test" / "test_log_likelihood.npz"
    values = evaluate_bgm1_bridge_log_likelihood(
        estimator,
        np.asarray(split.selected_test_x, dtype=np.float32),
        np.asarray(split.selected_test_indices, dtype=np.int64),
        config,
        artifact,
        root / "run.log",
    )
    log_px = np.asarray(values["log_px"], dtype=np.float64)
    metrics = {
        "dataset": dataset,
        "method": "BayesNDE",
        "points": int(len(log_px)),
        "mean_log_likelihood": float(np.mean(log_px)),
        "standard_deviation": float(np.std(log_px)),
        "two_standard_errors": float(2.0 * np.std(log_px) / math.sqrt(len(log_px))),
        "selected_epoch": int(manifest["selected_epoch"]),
        "epoch_selection_split": "validation",
        "test_used_for_epoch_selection": False,
        "proposal_covariance_mode": "fixed_isotropic",
        "proposal_scale": 0.005,
        "proposal_covariance": "0.005^2 I",
        "proposal_scoring": "product_univariate_t",
        "test_artifact": str(artifact),
    }
    atomic_json(root / "test" / "metrics.json", metrics)
    return metrics


def validate_protocol(dataset: str) -> dict[str, Any]:
    split = load_split(dataset)
    config = materialize_config(dataset, split.train_x.shape[1], run_root(dataset))
    assert config["model"]["x_dim"] == config["model"]["z_dim"]
    assert config["model"]["g_units"] == [256] * 5
    assert config["model"]["g_network_type"] == "residual"
    assert config["model"]["marginal_mmd_weight"] == 1000.0
    assert config["model"]["joint_mmd_weight"] == 100.0
    assert config["model"]["sliced_wasserstein_weight"] == 300.0
    assert config["density_bridge_eval"]["proposal_scale"] == 0.005
    assert config["density_bridge_eval"]["proposal_scoring"] == "product_univariate_t"
    assert config["epoch_selection"]["epoch_zero_selectable"] is False
    return {
        "dataset": dataset,
        "valid": True,
        "train_shape": list(split.train_x.shape),
        "validation_shape": list(split.val_x.shape),
        "test_shape": list(split.test_x.shape),
        "x_dim": int(split.train_x.shape[1]),
        "z_dim": int(split.train_x.shape[1]),
    }


def load_split(dataset: str):
    return prepare_dataset(DATA_ROOT, dataset, TEST_POINTS[dataset])
