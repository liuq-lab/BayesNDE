#!/usr/bin/env python3
"""Validation-selected BayesNDE tuning with fixed architecture and learning rates."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from bayesnde.diagnostics.generation import generation_rank_features, rank_candidates
import tensorflow as tf
import tensorflow_probability as tfp
import yaml

HERE = Path(__file__).resolve().parent
SRC_DENSITY = HERE.parent
REPO_ROOT = SRC_DENSITY.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.uci import core as base
from bayesnde.estimators.bridge import BGM_BridgeDensityEstimator
from bayesnde.diagnostics.generation import _two_sample_metrics


tfd = tfp.distributions
SEED = 1024
DATASET = "BANK"
X_DIM = 17
Z_DIMS = (8,)
DEFAULT_TEST_EPSILON = 0.0
EGM_MAX_STEP = 22000
EGM_CHECKPOINT_EVERY = 1000
ITERATIVE_MAX_EPOCH = 2000
ITERATIVE_EPOCHS = base.SAVE_EPOCHS
TOP_K_PER_Z = 3
TEST_SHARDS = 48
PROPOSAL_MODES = ("legacy_product_q",)
OUTPUT_ROOT = HERE / "outputs" / DATASET / "bgm-bs-tuning-v2"

EGM_VARIANTS: dict[str, dict[str, float]] = {
    "plain": {
        "alpha": 0.0,
        "variance_log_target_weight": 0.0,
        "decorrelation_weight": 0.0,
        "marginal_moment_weight": 0.0,
        "covariance_weight": 0.0,
    },
    "variance": {
        "alpha": 0.1,
        "variance_log_target_weight": 0.01,
        "decorrelation_weight": 0.0,
        "marginal_moment_weight": 0.0,
        "covariance_weight": 0.0,
    },
    "correlation": {
        "alpha": 0.0,
        "variance_log_target_weight": 0.0,
        "decorrelation_weight": 0.3,
        "marginal_moment_weight": 0.0,
        "covariance_weight": 0.0,
    },
    "legacy_balanced": {
        "alpha": 0.1,
        "variance_log_target_weight": 0.01,
        "decorrelation_weight": 0.3,
        "marginal_moment_weight": 0.0,
        "covariance_weight": 0.0,
    },
    "moments": {
        "alpha": 0.1,
        "variance_log_target_weight": 0.01,
        "decorrelation_weight": 0.3,
        "marginal_moment_weight": 1.0,
        "covariance_weight": 1.0,
    },
}

ITERATIVE_VARIANTS: dict[str, dict[str, float]] = {
    "decoder_only": {
        "variance_log_target_weight": 0.0,
        "iterative_empirical_corr_weight": 0.0,
        "iterative_prior_moment_weight": 0.0,
        "iterative_prior_covariance_weight": 0.0,
        "iterative_prior_mmd_weight": 0.0,
    },
    "variance": {
        "variance_log_target_weight": 0.01,
        "iterative_empirical_corr_weight": 0.0,
        "iterative_prior_moment_weight": 0.0,
        "iterative_prior_covariance_weight": 0.0,
        "iterative_prior_mmd_weight": 0.0,
    },
    "correlation": {
        "variance_log_target_weight": 0.0,
        "iterative_empirical_corr_weight": 0.3,
        "iterative_prior_moment_weight": 0.0,
        "iterative_prior_covariance_weight": 0.0,
        "iterative_prior_mmd_weight": 0.0,
    },
    "moments": {
        "variance_log_target_weight": 0.0,
        "iterative_empirical_corr_weight": 0.0,
        "iterative_prior_moment_weight": 1.0,
        "iterative_prior_covariance_weight": 1.0,
        "iterative_prior_mmd_weight": 0.0,
    },
    "mild_mmd": {
        "variance_log_target_weight": 0.0,
        "iterative_empirical_corr_weight": 0.0,
        "iterative_prior_moment_weight": 0.0,
        "iterative_prior_covariance_weight": 0.0,
        "iterative_prior_mmd_weight": 100.0,
    },
    "strong_reference": {
        "variance_log_target_weight": 0.0,
        "iterative_empirical_corr_weight": 0.3,
        "iterative_prior_moment_weight": 0.0,
        "iterative_prior_covariance_weight": 0.0,
        "iterative_prior_mmd_weight": 1000.0,
    },
}


def egm_rank_features(metrics: Mapping[str, Any], dim: int) -> dict[str, float] | None:
    generated = metrics.get("per_dim", {}).get("std")
    reference = metrics.get("validation_per_dim_std")
    fraction = None
    if generated is not None and reference is not None:
        g = np.asarray(generated, dtype=np.float64)
        v = np.asarray(reference, dtype=np.float64)
        if g.shape == v.shape and g.size:
            ratio = g / np.maximum(v, 1.0e-12)
            fraction = float(np.mean((ratio >= 0.75) & (ratio <= 1.25)))
    return generation_rank_features(metrics, dim, fraction)


def set_seed() -> None:
    np.random.seed(SEED)
    random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)


def egm_jobs() -> list[tuple[int, str]]:
    return [(z_dim, name) for z_dim in Z_DIMS for name in EGM_VARIANTS]


def iterative_jobs() -> list[tuple[int, str]]:
    return [(z_dim, name) for z_dim in Z_DIMS for name in ITERATIVE_VARIANTS]


def egm_root(z_dim: int, variant: str) -> Path:
    return OUTPUT_ROOT / "egm" / f"zdim_{z_dim}" / variant


def iterative_root(z_dim: int, variant: str) -> Path:
    return OUTPUT_ROOT / "iterative" / f"zdim_{z_dim}" / variant


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    return dict(value)


def save_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")
    os.replace(temporary, path)


def base_config(z_dim: int, out_dir: Path) -> dict[str, Any]:
    config = base.materialize_config(DATASET, X_DIM, out_dir)
    model = config["model"]
    model.update(
        output_dir=str(out_dir),
        x_dim=X_DIM,
        z_dim=int(z_dim),
        random_seed=SEED,
        g_network_type="residual",
        e_network_type="residual",
        g_units=[256] * 5,
        e_units=[256] * 5,
        lr=1.0e-3,
        lr_theta=0.005,
        lr_z=0.005,
        marginal_mmd_weight=0.0,
        joint_mmd_weight=0.0,
        sliced_wasserstein_weight=0.0,
        iterative_empirical_corr_weight=0.0,
        iterative_prior_moment_weight=0.0,
        iterative_prior_covariance_weight=0.0,
        iterative_prior_mmd_weight=0.0,
    )
    config["data"]["z_dim"] = int(z_dim)
    config["training"].update(
        batch_size=256,
        egm_n_iter=EGM_MAX_STEP,
        egm_batches_per_eval=EGM_CHECKPOINT_EVERY,
        epochs=ITERATIVE_MAX_EPOCH,
        save_epochs=list(ITERATIVE_EPOCHS),
    )
    config["density_bridge_eval"].update(
        K=5,
        S=20000,
        nu=3.0,
        epsilon=DEFAULT_TEST_EPSILON,
        n_repeats=1,
        eval_batch_size=1024,
        variance_floor=1.0e-6,
        proposal_covariance_mode="fixed_isotropic",
        proposal_scale=0.005,
        proposal_center="fitted_gmm",
        proposal_scoring="product_univariate_t",
        save_every=1,
        test_log_likelihood_save_every=1,
    )
    config["hmc_settings"].update(
        M=1600,
        burn_in=800,
        step_size=0.003,
        num_leapfrog_steps=10,
        target_accept_prob=0.75,
        num_chains=4,
        initial_state="encoder",
        initial_state_scale=0.01,
        min_effective_samples=0,
        max_retries=0,
        proposal_covariance_floor=0.001,
    )
    config["tuning_protocol"] = {
        "dataset": DATASET,
        "architecture_locked": "residual generator/encoder, five hidden layers of width 256",
        "egm_learning_rate_locked": 1.0e-3,
        "iterative_learning_rates_locked": {"lr_theta": 0.005, "lr_z": 0.005},
        "epsilon_locked": DEFAULT_TEST_EPSILON,
        "egm_step_candidates": list(range(EGM_CHECKPOINT_EVERY, EGM_MAX_STEP + 1, EGM_CHECKPOINT_EVERY)),
        "iterative_epoch_candidates": list(ITERATIVE_EPOCHS),
        "selection": "validation-only generation-quality rank sum",
        "test_points_used_for_selection": 0,
        "top_k_test_candidates_per_z_dim": TOP_K_PER_Z,
    }
    return config


def materialize_egm_config(z_dim: int, variant: str) -> dict[str, Any]:
    root = egm_root(z_dim, variant)
    config = base_config(z_dim, root)
    config["model"].update(EGM_VARIANTS[variant])
    config["experiment_variant"] = f"egm_{variant}_zdim_{z_dim}"
    config["egm_loss_variant"] = {"name": variant, **EGM_VARIANTS[variant]}
    return config


def materialize_iterative_config(z_dim: int, variant: str, selected: Mapping[str, Any]) -> dict[str, Any]:
    root = iterative_root(z_dim, variant)
    config = load_yaml(Path(str(selected["config_path"])))
    config = deepcopy(config)
    config["model"]["output_dir"] = str(root)
    config["model"].update(ITERATIVE_VARIANTS[variant])
    config["training"].update(epochs=ITERATIVE_MAX_EPOCH, save_epochs=list(ITERATIVE_EPOCHS))
    config["experiment_variant"] = f"iterative_{variant}_zdim_{z_dim}"
    config["iterative_loss_variant"] = {"name": variant, **ITERATIVE_VARIANTS[variant]}
    config["selected_egm"] = dict(selected)
    return config


def save_dataset_fingerprint(split: Any) -> None:
    path = OUTPUT_ROOT / "dataset_fingerprint.json"
    if path.exists():
        return
    base.atomic_json(
        path,
        {
            **split.metadata,
            "train_shape": list(split.train_x.shape),
            "validation_shape": list(split.val_x.shape),
            "test_shape": list(split.test_x.shape),
            "train_sha256": base.sha256_array(split.train_x),
            "validation_sha256": base.sha256_array(split.val_x),
            "test_sha256": base.sha256_array(split.test_x),
        },
    )


def build_model(config: Mapping[str, Any], timestamp: str):
    params = dict(config["model"])
    params["save_model"] = False
    model = base.build_bgm_model(params, timestamp=timestamp, random_seed=SEED)
    base.build_bgm1_networks_for_weight_io(model)
    return model


def generate_from_prior(model: Any, n_samples: int = 5000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED + 61000 + int(model.params["z_dim"]))
    z = rng.normal(size=(n_samples, int(model.params["z_dim"]))).astype(np.float32)
    noise = rng.normal(size=(n_samples, int(model.params["x_dim"]))).astype(np.float32)
    means: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    for start in range(0, n_samples, 1024):
        mean, variance = model._decode_generator(z[start : start + 1024], training=False)
        means.append(np.asarray(mean.numpy(), dtype=np.float32))
        variances.append(np.asarray(variance.numpy(), dtype=np.float32))
    mean = np.vstack(means)
    variance = np.vstack(variances)
    generated = mean + np.sqrt(np.maximum(variance, 1.0e-12)) * noise
    return generated.astype(np.float32), mean.astype(np.float32), variance.astype(np.float32)


def generation_row(model: Any, validation_x: np.ndarray) -> dict[str, Any]:
    generated, mean, variance = generate_from_prior(model)
    metrics = _two_sample_metrics(np.asarray(validation_x, dtype=np.float32), generated)
    return {
        "validation_generation_metrics": metrics,
        "validation_decoder_mean_log_likelihood": base.decoder_log_likelihood(
            model, np.asarray(validation_x, dtype=np.float32)
        ),
        "generated_sample_mean_abs": float(np.mean(np.abs(generated))),
        "generated_mean_mean_abs": float(np.mean(np.abs(mean))),
        "decoder_variance_mean": float(np.mean(variance)),
        "decoder_variance_min": float(np.min(variance)),
        "decoder_variance_max": float(np.max(variance)),
        "generation_samples": int(len(generated)),
        "selection_split": "validation",
        "test_points_used": 0,
    }


def newest_egm_checkpoint(root: Path) -> tuple[int, Path, Path] | None:
    return base.newest_egm_checkpoint(root)


def run_egm(task: int) -> dict[str, Any]:
    jobs = egm_jobs()
    if not 0 <= task < len(jobs):
        raise ValueError(f"EGM task must be in [0,{len(jobs) - 1}]")
    z_dim, variant = jobs[task]
    root = egm_root(z_dim, variant)
    config = materialize_egm_config(z_dim, variant)
    split = base.load_split(DATASET)
    save_dataset_fingerprint(split)
    save_yaml(root / "resolved_config.yaml", config)
    set_seed()
    model = build_model(config, f"egm_{z_dim}_{variant}")
    base.train_egm(model, np.asarray(split.train_x, dtype=np.float32), config, root)

    rows: list[dict[str, Any]] = []
    for step in range(EGM_CHECKPOINT_EVERY, EGM_MAX_STEP + 1, EGM_CHECKPOINT_EVERY):
        prefix = root / "checkpoint" / "egm" / f"egm_step_{step:04d}"
        generator = Path(str(prefix) + "_generator.weights.h5")
        encoder = Path(str(prefix) + "_encoder.weights.h5")
        if not generator.exists() or not encoder.exists():
            raise FileNotFoundError(f"Missing EGM checkpoint {prefix}")
        metric_path = root / "generation_metrics" / f"step_{step:05d}.json"
        if metric_path.exists():
            row = json.loads(metric_path.read_text(encoding="utf-8"))
        else:
            model.g_net.load_weights(str(generator))
            model.e_net.load_weights(str(encoder))
            row = {
                "stage": "egm",
                "z_dim": z_dim,
                "variant": variant,
                "step": step,
                "generator_weights": str(generator),
                "encoder_weights": str(encoder),
                "config_path": str(root / "resolved_config.yaml"),
                **generation_row(model, split.val_x),
            }
            base.atomic_json(metric_path, row)
        rows.append(row)
        base.atomic_json(root / "generation_curve.json", {"rows": rows})
        base.append_log(root / "run.log", f"generation diagnostics step={step}/{EGM_MAX_STEP}")
    result = {"task": task, "z_dim": z_dim, "variant": variant, "rows": len(rows), "complete": True}
    base.atomic_json(root / "complete.json", result)
    return result


def add_rank_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows
    features = []
    for row in rows:
        feature = egm_rank_features(row["validation_generation_metrics"], int(row["z_dim"]))
        if feature is None:
            raise RuntimeError(
                f"Candidate {row.get('variant')}/{row.get('step')} is missing a generation "
                f"diagnostic required by the shared selection rule."
            )
        features.append(feature)
    scores = rank_candidates(features)
    return [{**row, "generation_rank_components": feature, "generation_rank_sum": float(score)}
            for row, feature, score in zip(rows, features, scores)]


def select_egm() -> dict[str, Any]:
    frozen = OUTPUT_ROOT / "frozen_egm_selection.json"
    if frozen.exists():
        return json.loads(frozen.read_text(encoding="utf-8"))
    selected: dict[str, Any] = {}
    ranked_all: dict[str, Any] = {}
    for z_dim in Z_DIMS:
        rows: list[dict[str, Any]] = []
        for variant in EGM_VARIANTS:
            path = egm_root(z_dim, variant) / "generation_curve.json"
            if not path.exists():
                raise FileNotFoundError(path)
            rows.extend(json.loads(path.read_text(encoding="utf-8"))["rows"])
        ranked = sorted(add_rank_scores(rows), key=lambda row: (row["generation_rank_sum"], row["step"]))
        selected[str(z_dim)] = ranked[0]
        ranked_all[str(z_dim)] = ranked
    payload = {
        "criterion": "minimum equal-weight average percentile rank over the shared "
                     "validation-only generation diagnostics",
        "selected_by_z_dim": selected,
        "all_ranked_candidates": ranked_all,
        "test_points_used_for_selection": 0,
        "frozen_before_iterative_and_test": True,
    }
    base.atomic_json(frozen, payload)
    return payload


def run_iterative(task: int) -> dict[str, Any]:
    jobs = iterative_jobs()
    if not 0 <= task < len(jobs):
        raise ValueError(f"iterative task must be in [0,{len(jobs) - 1}]")
    selection = select_egm()
    z_dim, variant = jobs[task]
    selected = selection["selected_by_z_dim"][str(z_dim)]
    root = iterative_root(z_dim, variant)
    config = materialize_iterative_config(z_dim, variant, selected)
    save_yaml(root / "resolved_config.yaml", config)
    split = base.load_split(DATASET)
    set_seed()
    model = build_model(config, f"iterative_{z_dim}_{variant}")
    model.g_net.load_weights(str(selected["generator_weights"]))
    model.e_net.load_weights(str(selected["encoder_weights"]))
    base.train_iterative(model, split, config, root)

    rows: list[dict[str, Any]] = []
    train_rows = {int(row["epoch"]): row for row in base.load_curve(root / "iterative_validation_curve.json")}
    for epoch in ITERATIVE_EPOCHS:
        generator = root / "checkpoint" / "iterative" / f"weights_at_{epoch}_generator.weights.h5"
        if not generator.exists():
            raise FileNotFoundError(generator)
        metric_path = root / "generation_metrics" / f"epoch_{epoch:04d}.json"
        if metric_path.exists():
            row = json.loads(metric_path.read_text(encoding="utf-8"))
        else:
            model.g_net.load_weights(str(generator))
            row = {
                "stage": "iterative",
                "z_dim": z_dim,
                "variant": variant,
                "epoch": int(epoch),
                "generator_weights": str(generator),
                "egm_generator_weights": str(selected["generator_weights"]),
                "egm_encoder_weights": str(selected["encoder_weights"]),
                "egm_step": int(selected["step"]),
                "egm_variant": str(selected["variant"]),
                "config_path": str(root / "resolved_config.yaml"),
                "training_losses": train_rows.get(int(epoch), {}),
                **generation_row(model, split.val_x),
            }
            base.atomic_json(metric_path, row)
        rows.append(row)
        base.atomic_json(root / "generation_curve.json", {"rows": rows})
        base.append_log(root / "run.log", f"generation diagnostics epoch={epoch}/{ITERATIVE_MAX_EPOCH}")
    result = {"task": task, "z_dim": z_dim, "variant": variant, "rows": len(rows), "complete": True}
    base.atomic_json(root / "complete.json", result)
    return result


def select_iterative() -> dict[str, Any]:
    frozen = OUTPUT_ROOT / "frozen_iterative_top3.json"
    if frozen.exists():
        return json.loads(frozen.read_text(encoding="utf-8"))
    selected: dict[str, list[dict[str, Any]]] = {}
    ranked_all: dict[str, Any] = {}
    for z_dim in Z_DIMS:
        rows: list[dict[str, Any]] = []
        for variant in ITERATIVE_VARIANTS:
            path = iterative_root(z_dim, variant) / "generation_curve.json"
            if not path.exists():
                raise FileNotFoundError(path)
            rows.extend(json.loads(path.read_text(encoding="utf-8"))["rows"])
        ranked = sorted(add_rank_scores(rows), key=lambda row: (row["generation_rank_sum"], row["epoch"]))
        selected[str(z_dim)] = [{**row, "rank_within_z_dim": rank + 1} for rank, row in enumerate(ranked[:TOP_K_PER_Z])]
        ranked_all[str(z_dim)] = ranked
    payload = {
        "criterion": "top three minimum shared generation-rank scores of validation-only "
                     "generation diagnostics per z_dim",
        "selected_top3_by_z_dim": selected,
        "all_ranked_candidates": ranked_all,
        "test_points_used_for_selection": 0,
        "frozen_before_test": True,
    }
    base.atomic_json(frozen, payload)
    return payload


class LegacyProductQBridgeEstimator(BGM_BridgeDensityEstimator):
    def sample_defensive_proposal(self, proposal: Any, sample_size: int) -> tf.Tensor:
        weights = tf.convert_to_tensor(proposal.weights, dtype=self.dtype)
        loc = tf.convert_to_tensor(proposal.loc, dtype=self.dtype)
        scale_tril = tf.convert_to_tensor(proposal.scale_tril, dtype=self.dtype)
        df = tf.cast(proposal.df, self.dtype)
        n = int(sample_size)
        z_dim = int(loc.shape[-1])
        component_ids = tf.squeeze(
            tf.random.categorical(tf.math.log(weights)[None, :], n, seed=self._next_seed()), axis=0
        )
        component_loc = tf.gather(loc, component_ids, axis=0)
        component_scale = tf.gather(scale_tril, component_ids, axis=0)
        gaussian = tf.random.normal((n, z_dim), dtype=self.dtype, seed=self._next_seed())
        gaussian_scaled = tf.linalg.matvec(component_scale, gaussian)
        chi2 = tfd.Chi2(df=df).sample(sample_shape=(n,), seed=self._next_seed())
        t_samples = component_loc + gaussian_scaled / tf.sqrt(chi2[:, None] / df)
        epsilon = float(proposal.defensive_weight)
        if epsilon <= 0.0:
            return t_samples
        prior_samples = tf.random.normal((n, z_dim), dtype=self.dtype, seed=self._next_seed())
        if epsilon >= 1.0:
            return prior_samples
        defensive_mask = (
            tf.random.uniform((n, 1), dtype=self.dtype, seed=self._next_seed()) < epsilon
        )
        return tf.where(defensive_mask, prior_samples, t_samples)

    def estimate_point(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        result = super().estimate_point(*args, **kwargs)
        result["diagnostics"]["proposal_sampling"] = "multivariate_student_t_shared_chi_square"
        result["diagnostics"]["proposal_scoring"] = "product_univariate_t"
        result["diagnostics"]["proposal_scoring_matches_sampling"] = False
        return result


def test_task_count() -> int:
    return len(Z_DIMS) * TOP_K_PER_Z * len(PROPOSAL_MODES) * TEST_SHARDS


def decode_test_task(task: int) -> tuple[int, int, str, int]:
    if not 0 <= task < test_task_count():
        raise ValueError(f"test task must be in [0,{test_task_count() - 1}]")
    shard = task % TEST_SHARDS
    remainder = task // TEST_SHARDS
    mode = PROPOSAL_MODES[remainder % len(PROPOSAL_MODES)]
    remainder //= len(PROPOSAL_MODES)
    rank_index = remainder % TOP_K_PER_Z
    z_dim = Z_DIMS[remainder // TOP_K_PER_Z]
    return z_dim, rank_index, mode, shard


def candidate_slug(candidate: Mapping[str, Any]) -> str:
    raw = f"z{candidate['z_dim']}_rank{candidate['rank_within_z_dim']}_{candidate['variant']}_epoch{int(candidate['epoch']):04d}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def run_test_shard(task: int) -> dict[str, Any]:
    selection = select_iterative()
    z_dim, rank_index, mode, shard = decode_test_task(task)
    candidate = selection["selected_top3_by_z_dim"][str(z_dim)][rank_index]
    split = base.load_split(DATASET)
    positions = np.array_split(np.arange(len(split.selected_test_x), dtype=np.int64), TEST_SHARDS)[shard]
    epsilon = DEFAULT_TEST_EPSILON
    root = OUTPUT_ROOT / "test" / candidate_slug(candidate) / mode / f"shard_{shard:03d}"
    config = load_yaml(Path(str(candidate["config_path"])))
    config["density_bridge_eval"].update(
        epsilon=epsilon,
        proposal_covariance_mode="fixed_isotropic",
        proposal_scale=0.005,
        proposal_center="fitted_gmm",
        proposal_scoring="product_univariate_t",
        S=20000,
        n_repeats=1,
        variance_floor=1.0e-6,
        test_log_likelihood_save_every=1,
    )
    config["hmc_settings"].update(M=1600, burn_in=800)
    save_yaml(root / "resolved_config.yaml", config)
    set_seed()
    model = build_model(config, f"test_{z_dim}_{rank_index}_{shard}")
    model.e_net.load_weights(str(candidate["egm_encoder_weights"]))
    model.g_net.load_weights(str(candidate["generator_weights"]))
    estimator_cls = LegacyProductQBridgeEstimator
    estimator = estimator_cls(
        model=model,
        likelihood="gaussian",
        random_seed=SEED + 900000 + z_dim * 10000 + rank_index * 1000 + shard,
        variance_floor=1.0e-6,
    )
    artifact = root / "test_log_likelihood.npz"
    values = base.evaluate_bgm1_bridge_log_likelihood(
        estimator,
        np.asarray(split.selected_test_x[positions], dtype=np.float32),
        np.asarray(split.selected_test_indices[positions], dtype=np.int64),
        config,
        artifact,
        root / "run.log",
    )
    log_px = np.asarray(values["log_px"], dtype=np.float64)
    payload = {
        "task": task,
        "candidate": candidate_slug(candidate),
        "z_dim": z_dim,
        "rank_within_z_dim": rank_index + 1,
        "mode": mode,
        "shard": shard,
        "positions_start": int(positions[0]),
        "positions_stop": int(positions[-1]) + 1,
        "points": int(len(positions)),
        "mean_log_likelihood": float(np.mean(log_px)),
        "artifact": str(artifact),
        "proposal_sampling": "multivariate_student_t_shared_chi_square",
        "proposal_scoring": "product_univariate_t",
        "proposal_scoring_matches_sampling": False,
        "epsilon": epsilon,
    }
    base.atomic_json(root / "complete.json", payload)
    return payload


def merge_single_test() -> dict[str, Any]:
    z_dim = int(os.environ["TARGET_Z_DIM"])
    rank = int(os.environ["TARGET_RANK"])
    mode = "legacy_product_q"
    if z_dim not in Z_DIMS or not 1 <= rank <= TOP_K_PER_Z or mode not in PROPOSAL_MODES:
        raise ValueError("Invalid TARGET_Z_DIM, TARGET_RANK, or TARGET_MODE")
    epsilon = DEFAULT_TEST_EPSILON
    candidate = select_iterative()["selected_top3_by_z_dim"][str(z_dim)][rank - 1]
    out = OUTPUT_ROOT / "test" / candidate_slug(candidate) / mode
    parts: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for shard in range(TEST_SHARDS):
        root = out / f"shard_{shard:03d}"
        if not (root / "complete.json").exists() or not (root / "test_log_likelihood.npz").exists():
            raise FileNotFoundError(f"Incomplete test shard: {root}")
        with np.load(root / "test_log_likelihood.npz") as values:
            parts.append(np.asarray(values["log_px"], dtype=np.float64))
            indices.append(np.asarray(values["test_indices"], dtype=np.int64))
    log_px = np.concatenate(parts)
    test_indices = np.concatenate(indices)
    expected_points = len(base.load_split(DATASET).selected_test_x)
    if len(log_px) != expected_points or not np.isfinite(log_px).all():
        raise RuntimeError(f"Invalid merged result: {log_px.shape}")
    np.savez_compressed(out / "full_test_log_likelihood.npz", log_px=log_px, test_indices=test_indices)
    mean_ll = float(np.mean(log_px))
    payload = {
        "candidate": candidate_slug(candidate),
        "z_dim": z_dim,
        "rank_within_z_dim": rank,
        "egm_variant": candidate["egm_variant"],
        "egm_step": int(candidate["egm_step"]),
        "iterative_variant": candidate["variant"],
        "iterative_epoch": int(candidate["epoch"]),
        "validation_generation_rank_sum": float(candidate["generation_rank_sum"]),
        "proposal_mode": mode,
        "proposal_sampling": "multivariate_student_t_shared_chi_square",
        "proposal_scoring": "product_univariate_t",
        "proposal_scoring_matches_sampling": False,
        "epsilon": epsilon,
        "points": int(len(log_px)),
        "mean_log_likelihood": mean_ll,
        "standard_deviation": float(np.std(log_px)),
        "two_standard_errors": float(2.0 * np.std(log_px) / math.sqrt(len(log_px))),
    }
    base.atomic_json(out / "metrics.json", payload)
    return payload


def merge_test() -> dict[str, Any]:
    selection = select_iterative()
    epsilon = DEFAULT_TEST_EPSILON
    expected_points = len(base.load_split(DATASET).selected_test_x)
    rows: list[dict[str, Any]] = []
    arrays: dict[tuple[str, str], np.ndarray] = {}
    for z_dim in Z_DIMS:
        for candidate in selection["selected_top3_by_z_dim"][str(z_dim)]:
            slug = candidate_slug(candidate)
            for mode in PROPOSAL_MODES:
                parts: list[np.ndarray] = []
                indices: list[np.ndarray] = []
                for shard in range(TEST_SHARDS):
                    root = OUTPUT_ROOT / "test" / slug / mode / f"shard_{shard:03d}"
                    complete = root / "complete.json"
                    artifact = root / "test_log_likelihood.npz"
                    if not complete.exists() or not artifact.exists():
                        raise FileNotFoundError(f"Incomplete test shard: {root}")
                    with np.load(artifact) as values:
                        parts.append(np.asarray(values["log_px"], dtype=np.float64))
                        indices.append(np.asarray(values["test_indices"], dtype=np.int64))
                log_px = np.concatenate(parts)
                test_indices = np.concatenate(indices)
                if len(log_px) != expected_points or not np.isfinite(log_px).all():
                    raise RuntimeError(f"Invalid merged result {slug}/{mode}: {log_px.shape}")
                arrays[(slug, mode)] = log_px
                out = OUTPUT_ROOT / "test" / slug / mode
                np.savez_compressed(out / "full_test_log_likelihood.npz", log_px=log_px, test_indices=test_indices)
                mean_ll = float(np.mean(log_px))
                row = {
                    "candidate": slug,
                    "z_dim": z_dim,
                    "rank_within_z_dim": int(candidate["rank_within_z_dim"]),
                    "egm_variant": candidate["egm_variant"],
                    "egm_step": int(candidate["egm_step"]),
                    "iterative_variant": candidate["variant"],
                    "iterative_epoch": int(candidate["epoch"]),
                    "validation_generation_rank_sum": float(candidate["generation_rank_sum"]),
                    "proposal_mode": mode,
                    "proposal_sampling": "multivariate_student_t_shared_chi_square",
                    "proposal_scoring": "product_univariate_t",
                    "proposal_scoring_matches_sampling": False,
                    "epsilon": epsilon,
                    "points": int(len(log_px)),
                    "mean_log_likelihood": mean_ll,
                    "standard_deviation": float(np.std(log_px)),
                    "two_standard_errors": float(2.0 * np.std(log_px) / math.sqrt(len(log_px))),
                }
                base.atomic_json(out / "metrics.json", row)
                rows.append(row)
    payload = {
        "dataset": DATASET,
        "selection_split": "validation",
        "test_points_used_for_selection": 0,
        "architecture": "residual 5x256",
        "fixed_learning_rates": {"egm": 0.001, "iterative_theta": 0.005, "iterative_z": 0.005},
        "epsilon": epsilon,
        "results": sorted(rows, key=lambda row: row["mean_log_likelihood"], reverse=True),
    }
    base.atomic_json(OUTPUT_ROOT / "final_summary.json", payload)
    return payload


def validate() -> dict[str, Any]:
    split = base.load_split(DATASET)
    checks = []
    for z_dim, variant in egm_jobs():
        config = materialize_egm_config(z_dim, variant)
        model = config["model"]
        assert model["g_units"] == [256] * 5 and model["e_units"] == [256] * 5
        assert model["lr"] == 1.0e-3 and model["lr_theta"] == 0.005 and model["lr_z"] == 0.005
        assert model["marginal_mmd_weight"] == 0.0
        assert model["joint_mmd_weight"] == 0.0
        assert model["sliced_wasserstein_weight"] == 0.0
        assert config["density_bridge_eval"]["epsilon"] == DEFAULT_TEST_EPSILON
        checks.append({"z_dim": z_dim, "variant": variant})
    mapping = [decode_test_task(index) for index in range(test_task_count())]
    assert len(set(mapping)) == test_task_count()
    return {
        "valid": True,
        "dataset": DATASET,
        "train_shape": list(split.train_x.shape),
        "validation_shape": list(split.val_x.shape),
        "test_shape": list(split.test_x.shape),
        "egm_jobs": len(egm_jobs()),
        "iterative_jobs": len(iterative_jobs()),
        "test_jobs": test_task_count(),
        "egm_variants": list(EGM_VARIANTS),
        "iterative_variants": list(ITERATIVE_VARIANTS),
        "proposal_modes": list(PROPOSAL_MODES),
        "output_root": str(OUTPUT_ROOT),
        "locked_config_checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("validate", "egm", "select-egm", "select-iterative", "iterative", "test-shard", "merge-single-test", "merge-test"),
    )
    parser.add_argument("--task", type=int)
    args = parser.parse_args()
    if args.command in {"egm", "iterative", "test-shard"} and args.task is None:
        parser.error(f"{args.command} requires --task")
    if args.command == "validate":
        result = validate()
    elif args.command == "egm":
        result = run_egm(int(args.task))
    elif args.command == "select-egm":
        result = select_egm()
    elif args.command == "iterative":
        result = run_iterative(int(args.task))
    elif args.command == "select-iterative":
        result = select_iterative()
    elif args.command == "test-shard":
        result = run_test_shard(int(args.task))
    elif args.command == "merge-single-test":
        result = merge_single_test()
    else:
        result = merge_test()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
