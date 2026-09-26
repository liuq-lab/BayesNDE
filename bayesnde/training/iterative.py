"""Iterative updating of a BGM from an EGM checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import tensorflow as tf
import yaml
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]

from bayesgm.models import BGM
from bayesnde.training.generation_optimized import build_bgm_model
from bayesnde.estimators.pointwise import BGM_PointwiseDensityEstimator
from bayesnde.diagnostics.generation import run_bgm_generation_diagnostics
from bayesnde.data.samplers import build_sampler as build_simulation_sampler
from bayesnde.evaluation.metrics import generation_metric_summary
from bayesnde.diagnostics.generation import load_candidates, select_pooled


DEFAULT_CONFIG = None
DEFAULT_CHECKPOINT = None
DEFAULT_PHASE4_OUT_DIR = "outputs/iterative"
FALLBACK_PHASE4_OUT_DIR = "outputs/iterative"


def load_yaml(path: Path) -> dict[str, Any]:
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


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def make_run_dir(out_dir: str | Path, run_name: str) -> tuple[Path, dict[str, Any]]:
    requested_root = resolve_path(out_dir)
    metadata: dict[str, Any] = {"requested_out_dir": str(requested_root), "used_fallback": False}
    try:
        run_dir = requested_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        metadata["out_dir"] = str(requested_root)
        return run_dir, metadata
    except OSError as exc:
        fallback_root = resolve_path(FALLBACK_PHASE4_OUT_DIR)
        run_dir = fallback_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        metadata.update(
            {
                "out_dir": str(fallback_root),
                "used_fallback": True,
                "fallback_reason": str(exc),
            }
        )
        return run_dir, metadata


def current_learning_rates(
    *,
    epoch: int,
    lr_theta0: float,
    lr_z0: float,
    schedule: str,
    power: float,
    timescale: float,
) -> tuple[float, float]:
    if schedule == "constant":
        return float(lr_theta0), float(lr_z0)
    if schedule == "rm_decay":
        factor = (1.0 + float(epoch) / max(float(timescale), 1.0e-12)) ** (-float(power))
        return float(lr_theta0 * factor), float(lr_z0 * factor)
    raise ValueError(f"Unsupported lr schedule: {schedule!r}")


def set_optimizer_lr(optimizer: tf.keras.optimizers.Optimizer, value: float) -> None:
    lr = optimizer.learning_rate
    if hasattr(lr, "assign"):
        lr.assign(float(value))
    else:
        tf.keras.backend.set_value(lr, float(value))


def build_sampler(config: Mapping[str, Any]) -> Any:
    return build_simulation_sampler(config["data"])


def configure_tensorflow_gpu(config: Mapping[str, Any]) -> None:
    gpu_cfg = config.get("gpu", {})
    gpus = tf.config.list_physical_devices("GPU")
    if bool(gpu_cfg.get("required", False)) and not gpus:
        raise RuntimeError("GPU is required by config, but TensorFlow sees no GPU.")
    if bool(gpu_cfg.get("memory_growth", True)):
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass


def latest_checkpoint_dir(path: Path) -> Path:
    if path.exists() and any(path.glob("*.weights.h5")):
        return path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    children = [p for p in path.iterdir() if p.is_dir() and any(p.glob("*.weights.h5"))]
    if not children:
        raise FileNotFoundError(f"No weight files found under checkpoint path: {path}")
    return sorted(children, key=lambda p: p.name)[-1]


def build_model_from_checkpoint(
    *,
    config: Mapping[str, Any],
    run_dir: Path,
    checkpoint_dir: Path,
    epoch: int | None,
    egm_iter: int | None,
) -> tuple[BGM, dict[str, Any]]:
    params = dict(config["model"])
    params["use_bnn"] = False
    params["save_model"] = True
    params["save_res"] = False
    params["output_dir"] = str(run_dir)
    model = build_bgm_model(
        params=params,
        random_seed=int(config.get("model", {}).get("random_seed", config["data"].get("seed", 1024))),
    )
    estimator = BGM_PointwiseDensityEstimator(
        model=model,
        likelihood=str(config.get("density", {}).get("likelihood", "gaussian")),
        random_seed=int(config.get("density", {}).get("seed", 42)),
        variance_floor=float(config.get("density", {}).get("variance_floor", 1.0e-6)),
    )
    manifest = estimator.load_weights_from_run(
        checkpoint_dir=checkpoint_dir,
        epoch=epoch,
        egm_iter=egm_iter,
        load_encoder=True,
    )
    if "encoder_weights" not in manifest:
        raise FileNotFoundError("Iterative update requires encoder weights for data_z initialization.")
    model.g_net.trainable = True
    model.e_net.trainable = False
    model.dz_net.trainable = False
    model.dx_net.trainable = False
    manifest.update(
        {
            "init_mode": "checkpoint",
            "egm_init_ran_in_this_run": False,
            "initialized_from_egm_checkpoint": egm_iter is not None,
        }
    )
    return model, manifest


def build_model_with_egm_init(
    *,
    config: Mapping[str, Any],
    run_dir: Path,
    train_data: np.ndarray,
    egm_n_iter: int,
    egm_batches_per_eval: int,
    batch_size: int,
    verbose: int,
) -> tuple[BGM, dict[str, Any]]:
    params = dict(config["model"])
    params["use_bnn"] = False
    params["save_model"] = True
    params["save_res"] = False
    params["output_dir"] = str(run_dir)
    model = build_bgm_model(
        params=params,
        random_seed=int(config.get("model", {}).get("random_seed", config["data"].get("seed", 1024))),
    )
    model.egm_init(
        train_data,
        egm_n_iter=int(egm_n_iter),
        batch_size=int(batch_size),
        egm_batches_per_eval=int(egm_batches_per_eval),
        verbose=int(verbose),
    )
    model.g_net.trainable = True
    model.e_net.trainable = False
    model.dz_net.trainable = False
    model.dx_net.trainable = False
    manifest = {
        "init_mode": "egm",
        "egm_init_ran_in_this_run": True,
        "initialized_from_egm_checkpoint": False,
        "checkpoint_dir": str(model.checkpoint_path),
        "egm_iter": int(egm_n_iter),
        "encoder_weights": str(Path(model.checkpoint_path) / f"weights_at_egm_init_{int(egm_n_iter)}_encoder.weights.h5"),
        "generator_weights": str(Path(model.checkpoint_path) / f"weights_at_egm_init_{int(egm_n_iter)}_generator.weights.h5"),
    }
    return model, manifest


def encode_data_batched(
    *,
    model: BGM,
    data: np.ndarray,
    batch_size: int = 1024,
) -> np.ndarray:
    chunks = []
    data_arr = np.asarray(data, dtype=np.float32)
    for start in range(0, len(data_arr), int(batch_size)):
        end = min(start + int(batch_size), len(data_arr))
        z_batch = model.e_net(tf.convert_to_tensor(data_arr[start:end], dtype=tf.float32), training=False)
        chunks.append(np.asarray(z_batch.numpy(), dtype=np.float32))
    return np.vstack(chunks)


def evaluate_decoder_log_likelihood_batched(model: BGM, data: np.ndarray, batch_size: int = 1024) -> float:
    values: list[np.ndarray] = []
    data_arr = np.asarray(data, dtype=np.float32)
    for start in range(0, len(data_arr), int(batch_size)):
        x = tf.convert_to_tensor(data_arr[start : start + int(batch_size)], dtype=tf.float32)
        z = model.e_net(x, training=False)
        mu, variance = model._decode_generator(z, training=False)
        variance = tf.maximum(variance, tf.cast(1.0e-6, variance.dtype))
        logp = -0.5 * tf.reduce_sum(
            tf.math.log(tf.cast(2.0 * math.pi, variance.dtype) * variance) + tf.square(x - mu) / variance,
            axis=1,
        )
        values.append(np.asarray(logp.numpy(), dtype=np.float64))
    return float(np.mean(np.concatenate(values)))


def save_checkpoint(model: BGM, epoch: int) -> dict[str, str]:
    checkpoint_path = Path(model.checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    if not any(checkpoint_path.glob("weights_at_egm_init_*_encoder.weights.h5")):
        encoder_base = checkpoint_path / "weights_at_egm_init_0"
        model.e_net.save_weights(str(encoder_base) + "_encoder.weights.h5")
    if epoch == 0:
        egm_base = checkpoint_path / "weights_at_egm_init_0"
        model.e_net.save_weights(str(egm_base) + "_encoder.weights.h5")
        model.g_net.save_weights(str(egm_base) + "_generator.weights.h5")
    base = checkpoint_path / f"weights_at_{epoch}"
    model.g_net.save_weights(str(base) + "_generator.weights.h5")
    return {
        "checkpoint_dir": str(checkpoint_path),
        "generator_weights": str(base) + "_generator.weights.h5",
    }


def flatten_epoch_metrics(
    *,
    epoch: int,
    stage_dir: Path,
    diagnostics: Mapping[str, Any],
    lr_theta: float,
    lr_z: float,
    lr_schedule: str,
    train_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sigma_global = diagnostics.get("generator_sigma_square_global", {}) or {}
    sigma_quantiles = sigma_global.get("quantiles") or [None] * 7
    row: dict[str, Any] = {
        "epoch": int(epoch),
        "stage_dir": str(stage_dir),
        "lr_theta": float(lr_theta),
        "lr_z": float(lr_z),
        "lr_schedule": str(lr_schedule),
        "train_mean_log_true_px": diagnostics["train"]["density"]["mean_log_true_px"],
        "val_mean_log_true_px": diagnostics.get("validation", diagnostics["test"])["density"]["mean_log_true_px"],
        "test_mean_log_true_px": diagnostics["test"]["density"]["mean_log_true_px"],
        "generator_sigma_square_mean": sigma_global.get("mean"),
        "generator_sigma_square_min": sigma_global.get("min"),
        "generator_sigma_square_max": sigma_global.get("max"),
        "generator_sigma_square_q10": sigma_quantiles[1],
        "generator_sigma_square_q50": sigma_quantiles[3],
        "generator_sigma_square_q90": sigma_quantiles[5],
    }
    if train_metrics:
        row.update(train_metrics)
    row.update(generation_metric_summary(diagnostics["generated_mean"], "generated_mean"))
    row.update(generation_metric_summary(diagnostics["generated_sample"], "generated_sample"))
    return row


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[Mapping[str, Any]]) -> None:
    columns = [
        "epoch",
        "lr_theta",
        "lr_z",
        "generated_sample_mean_log_true_px",
        "generated_sample_wasserstein_mean",
        "generated_sample_ks_stat_mean",
        "generated_sample_sym_kl_mean",
        "generated_sample_mmd_rbf",
        "generated_sample_lisi_normalized_mean",
        "generated_sample_corr_frobenius_error",
        "generated_sample_max_abs_offdiag_corr",
        "generator_sigma_square_q50",
    ]
    lines = ["# Iterative updating curve", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col)
            if isinstance(value, float):
                values.append(f"{value:.6g}" if math.isfinite(value) else "nan")
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_metric_curves(run_dir: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.asarray([row.get("epoch") for row in rows], dtype=np.float64)
    lower_better = [
        "generated_sample_wasserstein_mean",
        "generated_sample_ks_stat_mean",
        "generated_sample_sym_kl_mean",
        "generated_sample_mmd_rbf",
        "generated_sample_corr_frobenius_error",
        "generator_sigma_square_q50",
    ]
    higher_better = [
        "generated_sample_lisi_normalized_mean",
        "generated_sample_mean_log_true_px",
        "generated_sample_mean_fraction_within_1sd",
    ]

    def values_for(metric: str) -> np.ndarray:
        vals = []
        for row in rows:
            value = row.get(metric)
            vals.append(float(value) if value is not None and np.isfinite(float(value)) else np.nan)
        return np.asarray(vals, dtype=np.float64)

    for group_name, metrics in (
        ("lower_better", lower_better),
        ("higher_better", higher_better),
    ):
        fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 2.2 * len(metrics)), squeeze=False)
        for ax, metric in zip(axes[:, 0], metrics):
            vals = values_for(metric)
            ax.plot(epochs, vals, marker="o", linewidth=1.5, label="iterative update")
            baseline_mask = epochs == 0
            if np.any(baseline_mask):
                baseline_value = vals[np.where(baseline_mask)[0][0]]
                if np.isfinite(float(baseline_value)):
                    ax.axhline(
                        baseline_value,
                        color="tab:red",
                        linestyle="--",
                        linewidth=1.2,
                        alpha=0.9,
                        label="epoch 0 before update",
                    )
            epoch10_mask = epochs == 10
            if np.any(epoch10_mask):
                epoch10_value = vals[np.where(epoch10_mask)[0][0]]
                if np.isfinite(float(epoch10_value)):
                    ax.axhline(
                        epoch10_value,
                        color="tab:green",
                        linestyle="--",
                        linewidth=1.2,
                        alpha=0.9,
                        label="epoch 10",
                    )
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7, loc="best")
        axes[-1, 0].set_xlabel("epoch")
        fig.suptitle(f"Iterative updating metrics ({group_name})")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(run_dir / f"fig_iterative_metrics_{group_name}.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(epochs, values_for("lr_theta"), marker="o", label="lr_theta")
    ax.plot(epochs, values_for("lr_z"), marker="o", label="lr_z")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "fig_iterative_learning_rates.png", dpi=180)
    plt.close(fig)


def write_curve_outputs(run_dir: Path, rows: list[Mapping[str, Any]]) -> None:
    write_csv(run_dir / "iterative_curve.csv", rows)
    save_json(run_dir / "iterative_curve.json", {"rows": rows})
    write_markdown(run_dir / "iterative_curve.md", rows)
    plot_metric_curves(run_dir, rows)


def run_generation_stage(
    *,
    model: BGM,
    sampler: Any,
    run_dir: Path,
    config: Mapping[str, Any],
    epoch: int,
    lr_theta: float,
    lr_z: float,
    lr_schedule: str,
    train_data: np.ndarray,
    train_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    save_checkpoint(model, epoch)
    stage_dir = run_dir / f"stage_epoch_{epoch:03d}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    generation_seed_offset = (
        0
        if bool(config.get("generation_diagnostics", {}).get("fixed_seed_across_epochs", False))
        else epoch
    )
    diagnostics = run_bgm_generation_diagnostics(
        model=model,
        sampler=sampler,
        run_dir=stage_dir,
        config=config,
        seed=(
            int(config.get("density", {}).get("generation_seed", int(config["data"]["seed"]) + 31415))
            + generation_seed_offset
        ),
    )
    row = flatten_epoch_metrics(
        epoch=epoch,
        stage_dir=stage_dir,
        diagnostics=diagnostics,
        lr_theta=lr_theta,
        lr_z=lr_z,
        lr_schedule=lr_schedule,
        train_metrics=train_metrics,
    )
    if hasattr(sampler, "X_val") and len(sampler.X_val):
        row["val_mean_decoder_log_likelihood"] = evaluate_decoder_log_likelihood_batched(
            model,
            np.asarray(sampler.X_val, dtype=np.float32),
            int(config.get("density", {}).get("eval_batch_size", 1024)),
        )
    save_json(stage_dir / "metrics.json", row)
    return row


def parse_epochs(text: str) -> list[int]:
    values = sorted({int(v.strip()) for v in text.split(",") if v.strip()})
    if not values or values[0] < 0:
        raise ValueError("--save-epochs must contain nonnegative integers.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run iterative updating from an EGM checkpoint.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, required=True,
                        help="Model/data config YAML (e.g. a resolved_config.yaml from the EGM stage).")
    parser.add_argument("--init-mode", choices=["checkpoint", "egm"], default="checkpoint")
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--egm-iter", type=int, default=None)
    parser.add_argument("--out-dir", default=DEFAULT_PHASE4_OUT_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--z-dim", type=int, default=None, help="Override model.z_dim before model construction.")
    parser.add_argument("--lr-theta", type=float, default=1.0e-4)
    parser.add_argument("--lr-z", type=float, default=1.0e-4)
    parser.add_argument("--lr-schedule", choices=["constant", "rm_decay"], default="constant")
    parser.add_argument("--lr-decay-power", type=float, default=0.6)
    parser.add_argument("--lr-decay-timescale", type=float, default=10.0)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--marginal-mmd-weight", type=float, default=None)
    parser.add_argument(
        "--iterative-prior-mmd-weight",
        type=float,
        default=None,
        help=(
            "During iterative updating, match fresh prior-generated samples "
            "to each empirical batch with a generic multiscale MMD penalty."
        ),
    )
    parser.add_argument("--mmd-scales", type=float, nargs="+", default=None)
    parser.add_argument("--variance-log-target", type=float, default=None)
    parser.add_argument("--variance-log-target-weight", type=float, default=None)
    parser.add_argument("--variance-log-eps", type=float, default=None)
    parser.add_argument("--variance-lower-bound", type=float, default=None)
    parser.add_argument("--variance-lower-bound-weight", type=float, default=None)
    parser.add_argument("--low-rank-generator", action="store_true")
    parser.add_argument("--low-rank-rank", type=int, default=None)
    parser.add_argument("--low-rank-init-std", type=float, default=None)
    parser.add_argument("--low-rank-l2-weight", type=float, default=None)
    parser.add_argument("--low-rank-variance-floor", type=float, default=None)
    parser.add_argument("--low-rank-woodbury-jitter", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epoch", type=int, default=50)
    parser.add_argument("--save-epochs", default="0,5,10,20,50")
    parser.add_argument("--egm-n-iter", type=int, default=None)
    parser.add_argument("--egm-batches-per-eval", type=int, default=None)
    parser.add_argument("--generation-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--model-seed", type=int, default=None)
    parser.add_argument(
        "--iterative-decorrelation-target",
        choices=["identity", "train"],
        default=None,
        help="Apply this decorrelation target only after EGM initialization.",
    )
    parser.add_argument(
        "--iterative-variance-log-target-weight",
        type=float,
        default=None,
        help="Apply this soft-log-variance weight only during iterative updating.",
    )
    parser.add_argument(
        "--iterative-empirical-corr-weight",
        type=float,
        default=None,
        help=(
            "Weight for an additional prior-generation step matching the "
            "generated and empirical batch correlation matrices."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    checkpoint_dir = latest_checkpoint_dir(resolve_path(args.checkpoint_dir)) if args.init_mode == "checkpoint" else None
    schedule_tag = args.lr_schedule if args.lr_schedule != "constant" else "const"
    run_name = args.run_name or f"{schedule_tag}_lrtheta{args.lr_theta:g}_lrz{args.lr_z:g}".replace(".", "p")
    run_dir, output_metadata = make_run_dir(args.out_dir, run_name)

    config = load_yaml(config_path)
    configure_tensorflow_gpu(config)
    if args.model_seed is not None:
        config["model"]["random_seed"] = int(args.model_seed)
    if args.z_dim is not None:
        config["model"]["z_dim"] = int(args.z_dim)
    config["model"]["lr_theta"] = float(args.lr_theta)
    config["model"]["lr_z"] = float(args.lr_z)
    if args.iterative_empirical_corr_weight is not None:
        if args.iterative_empirical_corr_weight < 0.0:
            raise ValueError("--iterative-empirical-corr-weight must be non-negative.")
        config["model"]["iterative_empirical_corr_weight"] = float(
            args.iterative_empirical_corr_weight
        )
    config["model"]["lr_schedule"] = str(args.lr_schedule)
    config["model"]["lr_decay_power"] = float(args.lr_decay_power)
    config["model"]["lr_decay_timescale"] = float(args.lr_decay_timescale)
    if args.optimizer is not None:
        config["model"]["optimizer"] = str(args.optimizer)
    config["model"]["weight_decay"] = float(args.weight_decay)
    if args.marginal_mmd_weight is not None:
        config["model"]["marginal_mmd_weight"] = float(args.marginal_mmd_weight)
    if args.iterative_prior_mmd_weight is not None:
        if args.iterative_prior_mmd_weight < 0.0:
            raise ValueError("--iterative-prior-mmd-weight must be non-negative.")
        config["model"]["iterative_prior_mmd_weight"] = float(
            args.iterative_prior_mmd_weight
        )
    if args.mmd_scales is not None:
        config["model"]["mmd_scales"] = [float(value) for value in args.mmd_scales]
    if args.variance_log_target is not None:
        config["model"]["variance_log_target"] = float(args.variance_log_target)
    if args.variance_log_target_weight is not None:
        config["model"]["variance_log_target_weight"] = float(args.variance_log_target_weight)
    if args.variance_log_eps is not None:
        config["model"]["variance_log_eps"] = float(args.variance_log_eps)
    if args.variance_lower_bound is not None:
        config["model"]["variance_lower_bound"] = float(args.variance_lower_bound)
    if args.variance_lower_bound_weight is not None:
        config["model"]["variance_lower_bound_weight"] = float(args.variance_lower_bound_weight)
    if args.low_rank_generator:
        config["model"]["low_rank_generator"] = True
    if args.low_rank_rank is not None:
        config["model"]["low_rank_rank"] = int(args.low_rank_rank)
    if args.low_rank_init_std is not None:
        config["model"]["low_rank_init_std"] = float(args.low_rank_init_std)
    if args.low_rank_l2_weight is not None:
        config["model"]["low_rank_l2_weight"] = float(args.low_rank_l2_weight)
    if args.low_rank_variance_floor is not None:
        config["model"]["low_rank_variance_floor"] = float(args.low_rank_variance_floor)
    if args.low_rank_woodbury_jitter is not None:
        config["model"]["low_rank_woodbury_jitter"] = float(args.low_rank_woodbury_jitter)
    config["training"]["batch_size"] = int(args.batch_size)
    config["training"]["epochs"] = int(args.max_epoch)
    config["training"]["epochs_per_eval"] = 1
    initialized_from_egm_checkpoint = args.init_mode == "checkpoint" and args.egm_iter is not None
    config["training"]["init_mode"] = str(args.init_mode)
    config["training"]["use_egm_init"] = bool(
        args.init_mode == "egm" or initialized_from_egm_checkpoint
    )
    config["training"]["egm_init_ran_in_this_run"] = args.init_mode == "egm"
    config["training"]["initialized_from_egm_checkpoint"] = initialized_from_egm_checkpoint
    config["training"]["init_checkpoint_dir"] = (
        str(checkpoint_dir) if args.init_mode == "checkpoint" else None
    )
    config["training"]["init_checkpoint_epoch"] = (
        int(args.epoch) if args.init_mode == "checkpoint" and args.epoch is not None else None
    )
    config["training"]["init_checkpoint_egm_iter"] = (
        int(args.egm_iter) if initialized_from_egm_checkpoint else None
    )
    config["training"]["egm_only"] = False
    if args.generation_samples is not None:
        config.setdefault("generation_diagnostics", {})["n_samples"] = int(args.generation_samples)
    sampler = build_sampler(config)
    train_data = np.asarray(sampler.X_train, dtype=np.float32)
    if float(config["model"].get("iterative_empirical_corr_weight", 0.0)) > 0.0:
        empirical_corr = np.corrcoef(train_data.astype(np.float64), rowvar=False)
        if not np.all(np.isfinite(empirical_corr)):
            raise ValueError("Training correlation matrix contains non-finite values.")
        config["model"]["iterative_empirical_corr_target"] = empirical_corr.tolist()
    if args.init_mode == "checkpoint":
        model, manifest = build_model_from_checkpoint(
            config=config,
            run_dir=run_dir,
            checkpoint_dir=checkpoint_dir,
            epoch=args.epoch,
            egm_iter=args.egm_iter,
        )
    else:
        train_cfg = config.get("training", {})
        model, manifest = build_model_with_egm_init(
            config=config,
            run_dir=run_dir,
            train_data=train_data,
            egm_n_iter=int(args.egm_n_iter if args.egm_n_iter is not None else train_cfg.get("egm_n_iter", 10000)),
            egm_batches_per_eval=int(
                args.egm_batches_per_eval
                if args.egm_batches_per_eval is not None
                else train_cfg.get("egm_batches_per_eval", 500)
            ),
            batch_size=int(args.batch_size),
            verbose=int(train_cfg.get("verbose", 1)),
        )

    iterative_overrides: dict[str, Any] = {}
    if args.iterative_decorrelation_target is not None:
        iterative_overrides["decorrelation_target"] = str(args.iterative_decorrelation_target)
    if args.iterative_variance_log_target_weight is not None:
        iterative_overrides["variance_log_target_weight"] = float(
            args.iterative_variance_log_target_weight
        )
    egm_objective = {
        "decorrelation_target": config["model"].get("decorrelation_target", "identity"),
        "variance_log_target_weight": float(config["model"].get("variance_log_target_weight", 0.0)),
    }
    for key, value in iterative_overrides.items():
        config["model"][key] = value
        model.params[key] = value
    config["training"]["stage_objectives"] = {
        "egm": egm_objective,
        "iterative": {
            "decorrelation_target": config["model"].get("decorrelation_target", "identity"),
            "variance_log_target_weight": float(config["model"].get("variance_log_target_weight", 0.0)),
            "iterative_empirical_corr_weight": float(
                config["model"].get("iterative_empirical_corr_weight", 0.0)
            ),
            "iterative_prior_mmd_weight": float(
                config["model"].get("iterative_prior_mmd_weight", 0.0)
            ),
        },
    }
    save_yaml(run_dir / "run_config_used.yaml", config)
    save_yaml(run_dir / "weights_manifest.yaml", manifest)
    save_json(run_dir / "output_location.json", output_metadata)
    init_z = encode_data_batched(
        model=model,
        data=train_data,
        batch_size=int(config.get("density", {}).get("eval_batch_size", 1024)),
    )
    model.data_z = tf.Variable(init_z, name="Latent Variable", trainable=True)

    save_epochs = parse_epochs(args.save_epochs)
    max_epoch = max(int(args.max_epoch), max(save_epochs))
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(args.seed))
    lr_theta, lr_z = current_learning_rates(
        epoch=0,
        lr_theta0=float(args.lr_theta),
        lr_z0=float(args.lr_z),
        schedule=str(args.lr_schedule),
        power=float(args.lr_decay_power),
        timescale=float(args.lr_decay_timescale),
    )
    set_optimizer_lr(model.g_optimizer, lr_theta)
    set_optimizer_lr(model.posterior_optimizer, lr_z)
    if 0 in save_epochs:
        rows.append(
            run_generation_stage(
                model=model,
                sampler=sampler,
                run_dir=run_dir,
                config=config,
                epoch=0,
                lr_theta=lr_theta,
                lr_z=lr_z,
                lr_schedule=str(args.lr_schedule),
                train_data=train_data,
                train_metrics={
                    "epoch_loss_x_mean": None,
                    "epoch_loss_z_mean": None,
                    "epoch_empirical_corr_loss_mean": None,
                },
            )
        )
        write_curve_outputs(run_dir, rows)

    batch_size = int(args.batch_size)
    for epoch in range(1, max_epoch + 1):
        lr_theta, lr_z = current_learning_rates(
            epoch=epoch,
            lr_theta0=float(args.lr_theta),
            lr_z0=float(args.lr_z),
            schedule=str(args.lr_schedule),
            power=float(args.lr_decay_power),
            timescale=float(args.lr_decay_timescale),
        )
        set_optimizer_lr(model.g_optimizer, lr_theta)
        set_optimizer_lr(model.posterior_optimizer, lr_z)
        sample_idx = rng.permutation(len(train_data))
        total_batches = len(train_data) // batch_size
        loss_x_values: list[float] = []
        loss_z_values: list[float] = []
        corr_loss_values: list[float] = []
        with tqdm(total=total_batches, desc=f"Iter epoch {epoch}/{max_epoch}", unit="batch") as pbar:
            for start in range(0, len(train_data) - batch_size + 1, batch_size):
                batch_idx = sample_idx[start : start + batch_size]
                batch_z = tf.Variable(tf.gather(model.data_z, batch_idx, axis=0), name="batch_z", trainable=True)
                batch_x = train_data[batch_idx, :]
                loss_x = model.update_g_net(batch_z, batch_x)
                corr_weight = float(model.params.get("iterative_empirical_corr_weight", 0.0))
                if corr_weight > 0.0:
                    if not hasattr(model, "update_iterative_empirical_corr"):
                        raise TypeError(
                            "iterative_empirical_corr_weight requires GenerationOptimizedBGM."
                        )
                    corr_loss_values.append(
                        float(model.update_iterative_empirical_corr(batch_x))
                    )
                loss_z = model.update_latent_variable_sgd(batch_z, batch_x)
                loss_x_value = float(loss_x)
                loss_z_value = float(loss_z)
                loss_x_values.append(loss_x_value)
                loss_z_values.append(loss_z_value)
                model.data_z.scatter_nd_update(
                    indices=tf.expand_dims(batch_idx, axis=1),
                    updates=batch_z,
                )
                pbar.set_postfix(
                    lr=f"{lr_theta:.2e}/{lr_z:.2e}",
                    loss_x=f"{loss_x_value:.4f}",
                    loss_z=f"{loss_z_value:.4f}",
                )
                pbar.update(1)

        train_metrics = {
            "epoch_loss_x_mean": float(np.mean(loss_x_values)) if loss_x_values else float("nan"),
            "epoch_loss_z_mean": float(np.mean(loss_z_values)) if loss_z_values else float("nan"),
            "epoch_empirical_corr_loss_mean": (
                float(np.mean(corr_loss_values)) if corr_loss_values else None
            ),
        }
        if not all(
            value is None or np.isfinite(float(value)) for value in train_metrics.values()
        ):
            raise FloatingPointError(f"Non-finite BGM iterative loss at epoch {epoch}: {train_metrics}")

        print(
            f"epoch={epoch}/{max_epoch} lr_theta={lr_theta:.6g} lr_z={lr_z:.6g} "
            f"loss_x={train_metrics['epoch_loss_x_mean']:.6g} "
            f"loss_z={train_metrics['epoch_loss_z_mean']:.6g}",
            flush=True,
        )
        if epoch in save_epochs:
            rows.append(
                run_generation_stage(
                    model=model,
                    sampler=sampler,
                    run_dir=run_dir,
                    config=config,
                    epoch=epoch,
                    lr_theta=lr_theta,
                    lr_z=lr_z,
                    lr_schedule=str(args.lr_schedule),
                    train_data=train_data,
                    train_metrics=train_metrics,
                )
            )
            write_curve_outputs(run_dir, rows)

    if not rows or int(rows[-1].get("epoch", -1)) != max_epoch:
        rows.append(
            run_generation_stage(
                model=model,
                sampler=sampler,
                run_dir=run_dir,
                config=config,
                epoch=max_epoch,
                lr_theta=lr_theta,
                lr_z=lr_z,
                lr_schedule=str(args.lr_schedule),
                train_data=train_data,
                train_metrics=train_metrics,
            )
        )
        write_curve_outputs(run_dir, rows)

    stage_dirs = [Path(row["stage_dir"]) for row in rows
                  if row.get("stage_dir") and int(row.get("epoch", 0)) > 0]
    candidates = load_candidates(stage_dirs, "epoch")
    if candidates:
        payload = select_pooled(candidates, int(config["model"]["x_dim"]), key="epoch")
        payload["checkpoint_dir"] = str(model.checkpoint_path)
        save_json(run_dir / "selected_epoch.json", payload)
        print(f"Selected epoch {payload['epoch']} (generation rank {payload['score']:.4f})", flush=True)

    print(f"Iterative-update outputs: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
