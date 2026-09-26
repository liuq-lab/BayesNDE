#!/usr/bin/env python3
"""Independent Gaussian mixture simulations."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from bayesnde.data.samplers import (
    BASE_CONFIG,
    assert_disjoint_splits,
    build_sampler,
    dataset_fingerprint,
)
from bayesnde.training.generation_optimized import (
    build_bgm_model,
    select_checkpoint,
    train_egm_with_diagnostics,
)

SRC = Path(__file__).resolve().parent
REPO = SRC.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BRIDGE = REPO / "configs/bridge.yaml"
REPRODUCTION = REPO / "configs/reproduction.yaml"
SEED = 1024
ITERATIVE_SEED = 711024

VISUAL_ROOT = REPO / "outputs/simulation/independent_gmm_visualization"
LOW_ROOT = REPO / "outputs/simulation/independent_gmm_dimension_scaling"
HIGH_ROOT = REPO / "outputs/simulation/independent_gmm_highdim"

VISUAL_EGM_LOSS = {"x_cycle_weight": 1.0}
LOW_EGM_LOSS = {"marginal_mmd_weight": 100.0}
LOW_EGM_STEPS = 22000
LOW_SAVE_EPOCHS = (0, 1, 10, 25, 50, 100, 200, 400, 700, 1000, 1500, 2000, 2500, 3000)
LOW_MAX_EPOCH = 3000

VISUAL_EPOCH = 2000
LOW_EPOCH = {2: 700, 5: 2500, 10: 1500}
HIGH_EPOCH = {15: 1700, 20: 1600, 25: 1900}

HIGH_EGM_STEPS = 3000
HIGH_EGM_EVERY = 50
HIGH_SAVE_EPOCHS = (0, 1, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 250, *range(300, 2001, 50))
HIGH_MAX_EPOCH = 2000

LOW_RNG_BLOCK = 400
HIGH_RNG_BLOCK = 100


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)),
        encoding="utf-8",
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO, check=True)


def find_checkpoint_dir(run_dir: Path, epoch: int) -> Path:
    matches = sorted({path.parent for path in (run_dir / "checkpoints").rglob(
        f"weights_at_{epoch}_generator.weights.h5")})
    if len(matches) != 1:
        raise RuntimeError(f"Expected one checkpoint directory for epoch {epoch} below {run_dir}; found {matches}")
    return matches[0]


def low_config(dim: int, loss: Mapping[str, float], run_dir: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    cfg["data"].update({"name": "indep_gmm", "seed": SEED, "n": 20000, "dim": dim,
                        "sd": 0.1, "n_components": 3, "bound": 1.0})
    cfg["model"].update({
        "dataset": "BGM_indep_gmm", "output_dir": str(run_dir.resolve()), "save_model": True,
        "save_res": False, "use_bnn": False, "x_dim": dim, "z_dim": dim, "random_seed": SEED,
        "g_network_type": "mlp", "e_network_type": "mlp", "g_units": [256] * 5,
        "e_units": [256] * 5, "dx_units": [256, 256, 128, 64], "dz_units": [256, 256, 128, 64],
        "lr": 1.0e-3, "cycle_weight": 5.0, "x_cycle_weight": 3.0,
        "z_cycle_weight": 1.0, "g_adv_weight": 1.0, "e_adv_weight": 1.0,
        "marginal_mmd_weight": 0.0, "joint_mmd_weight": 0.0,
        "sliced_wasserstein_weight": 0.0, "sliced_wasserstein_projections": 64,
        "decorrelation_weight": 0.3, "decorrelation_target": "train",
        "variance_log_target": 0.01, "variance_log_target_weight": 0.01,
        "factorized_generator": False, "low_rank_generator": False,
    })
    cfg["model"].update(loss)
    cfg["training"].update({"batch_size": 256, "egm_n_iter": LOW_EGM_STEPS, "egm_batches_per_eval": 1000,
                            "fresh_training_required": True, "external_checkpoint_reuse_allowed": False,
                            "early_stop": None})
    cfg.setdefault("generation_diagnostics", {}).update({
        "n_samples": 5000, "plots_enabled": False, "umap_enabled": False,
        "marginal_plot_dims_per_page": 10, "plot_reference_split": "validation"})
    cfg.setdefault("density", {}).update({
        "test_log_likelihood_points": 2000, "test_log_likelihood_seed": 8943,
        "n_repeats": 1, "variance_floor": 0.01, "force_iid_test_eval": True})
    cfg["experiment_protocol"] = {
        "name": "indep_gmm_lowdim", "single_seed": SEED,
        "test_used_for_model_selection": False, "fresh_initialization": True,
    }
    return cfg


def high_config(dim: int, run_dir: Path) -> dict[str, Any]:
    cfg = deepcopy(yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8")))
    cfg["data"].update({"name": "indep_gmm", "seed": SEED, "n": 20000, "dim": dim,
                        "sd": 0.1, "n_components": 3, "bound": 1.0})
    cfg["model"].update({
        "dataset": "BGM_indep_gmm", "output_dir": str(run_dir.resolve()), "save_model": True,
        "save_res": False, "use_bnn": False, "x_dim": dim, "z_dim": dim, "random_seed": SEED,
        "g_network_type": "residual", "e_network_type": "residual",
        "g_units": [256] * 5, "e_units": [256] * 5,
        "dx_units": [256, 256, 128, 64], "dz_units": [256, 256, 128, 64],
        "lr": 1.0e-3, "lr_theta": 0.005, "lr_z": 0.005,
        "cycle_weight": 5.0, "x_cycle_weight": 3.0, "z_cycle_weight": 1.0,
        "g_adv_weight": 1.0, "e_adv_weight": 1.0,
        "marginal_mmd_weight": 1000.0, "joint_mmd_weight": 100.0,
        "sliced_wasserstein_weight": 300.0, "sliced_wasserstein_projections": 64,
        "sliced_wasserstein_seed": 20260831,
        "decorrelation_weight": 0.3, "decorrelation_target": "train",
        "mmd_scales": [0.05, 0.1, 0.2, 0.5, 1.0],
        "variance_log_target": 0.01, "variance_log_target_weight": 0.0,
        "factorized_generator": False, "low_rank_generator": False,
    })
    cfg["training"].update({
        "batch_size": 256, "egm_n_iter": HIGH_EGM_STEPS, "egm_batches_per_eval": HIGH_EGM_EVERY,
        "epochs": HIGH_MAX_EPOCH, "epochs_per_eval": 50,
        "fresh_training_required": True, "external_checkpoint_reuse_allowed": False, "early_stop": None,
    })
    cfg.setdefault("generation_diagnostics", {}).update({
        "n_samples": 5000, "plots_enabled": True, "umap_enabled": False,
        "marginal_plot_dims_per_page": 10, "plot_reference_split": "validation"})
    cfg.setdefault("density", {}).update({
        "test_log_likelihood_points": 2000, "test_log_likelihood_seed": 8943,
        "n_repeats": 1, "variance_floor": 0.01, "force_iid_test_eval": True})
    cfg.setdefault("protocol", {}).update({
        "name": "indep_gmm_highdim", "x_dim_equals_z_dim": True, "factorized_generator": False,
        "validation_points": 400, "validation_checkpoint_selection_only": True, "test_points": 2000,
    })
    return cfg


def train_egm(cfg: dict[str, Any], run_dir: Path, steps: int, every: int, **kwargs: Any) -> list[dict[str, Any]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    sampler = build_sampler(cfg["data"])
    assert_disjoint_splits(sampler.X_train, sampler.X_val, sampler.X_test)
    save_json(run_dir / "dataset_fingerprint.json", dataset_fingerprint(sampler))
    model = build_bgm_model(cfg["model"], timestamp="fresh", random_seed=SEED)
    return train_egm_with_diagnostics(
        model=model, train_data=np.asarray(sampler.X_train, dtype=np.float32), sampler=sampler,
        config=cfg, diagnostics_root=run_dir / "egm_diagnostics", max_steps=steps, batch_size=256,
        every=every, seed=SEED, **kwargs)


def select_gated_egm_step(rows: Sequence[Mapping[str, Any]], run_dir: Path) -> None:
    try:
        selected, status = select_checkpoint(rows, True), "passed_generation_gate"
    except RuntimeError:
        selected, status = select_checkpoint(rows, False), "failed_generation_gate"
    save_json(run_dir / "selected_egm_step.json", {"step": int(selected["step"]), "status": status,
                                                   "selected": dict(selected)})


def select_ranked_egm_step(rows: Sequence[Mapping[str, Any]], dim: int, run_dir: Path) -> None:
    lower_metrics = ("mmd_rbf", "sym_kl_mean", "sliced_wasserstein", "wasserstein_mean", "ks_stat_mean")
    metric_names = [*lower_metrics, "corr_error_per_sqrt_dim", "one_minus_lisi", "one_minus_std_ratio_fraction"]
    candidates: list[dict[str, Any]] = []
    for row in rows:
        step = int(row.get("step", 0))
        if step <= 0:
            continue
        diagnostics_path = Path(str(row["diagnostics_dir"])) / "generation_diagnostics.json"
        metrics = load_json(diagnostics_path)["generated_sample"]["two_sample_vs_validation"]
        record = {
            "step": step,
            "generator_weights": row["generator_weights"],
            "encoder_weights": row["encoder_weights"],
            **{name: float(metrics[name]) for name in lower_metrics},
            "corr_error_per_sqrt_dim": float(metrics["corr_matrix_frobenius_error"]) / np.sqrt(float(dim)),
            "one_minus_lisi": 1.0 - float(metrics["lisi_normalized_mean"]),
            "one_minus_std_ratio_fraction": 1.0 - float(row["sample_gate"]["std_ratio_fraction_in_0p75_1p25"]),
            "finite_generated_sample_fraction": float(row["finite_generated_sample_fraction"]),
        }
        if np.isfinite([record[name] for name in metric_names]).all() and record["finite_generated_sample_fraction"] == 1.0:
            candidates.append(record)
    if not candidates:
        raise RuntimeError(f"Dimension {dim}: no finite nonzero EGM checkpoint is selectable")

    values = np.asarray([[record[name] for name in metric_names] for record in candidates], dtype=np.float64)
    ranks = np.empty_like(values)
    for column in range(values.shape[1]):
        order = np.argsort(values[:, column], kind="mergesort")
        ordered = values[order, column]
        begin = 0
        while begin < len(order):
            end = begin + 1
            while end < len(order) and ordered[end] == ordered[begin]:
                end += 1
            ranks[order[begin:end], column] = 0.5 * float(begin + end - 1)
            begin = end
    percentile_ranks = ranks / float(max(len(candidates) - 1, 1))
    for index, record in enumerate(candidates):
        record["generation_rank_score"] = float(np.mean(percentile_ranks[index]))
    selected = min(candidates, key=lambda record: (record["generation_rank_score"], -record["step"]))
    save_json(run_dir / "selected_egm_step.json", {
        "step": int(selected["step"]), "metrics": metric_names, "selected": selected, "candidates": candidates})
    with (run_dir / "egm_generation_ranking.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "generation_rank_score", *metric_names], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(candidates, key=lambda record: record["step"]))


def train_iterative(egm_dir: Path, out_dir: Path, options: Sequence[str]) -> None:
    egm = load_json(egm_dir / "selected_egm_step.json")
    run_command([
        sys.executable, "-m", "bayesnde.training.iterative",
        "--config", str(egm_dir / "resolved_config.yaml"),
        "--init-mode", "checkpoint", "--checkpoint-dir", str(Path(egm["selected"]["generator_weights"]).parent),
        "--egm-iter", str(int(egm["step"])), "--out-dir", str(out_dir), "--run-name", "seed_1024",
        "--model-seed", str(SEED), "--seed", str(ITERATIVE_SEED), "--lr-theta", "0.005", "--lr-z", "0.005",
        "--batch-size", "256", "--iterative-decorrelation-target", "train",
        "--iterative-empirical-corr-weight", "0.3", "--iterative-prior-mmd-weight", "1000",
        *options,
    ])


def evaluate(run_dir: Path, epoch: int, output: Path, options: Sequence[str]) -> None:
    run_command([
        sys.executable, "-m", "bayesnde.evaluation.bridge_eval",
        "--config", str(BRIDGE), "--visual-config", str(run_dir / "run_config_used.yaml"),
        "--preset", "final", "--run-name", "result",
        "--checkpoint-dir", str(find_checkpoint_dir(run_dir, epoch)), "--epoch", str(epoch), "--load-encoder",
        "--evaluation-split", "test", "--test-points", "2000",
        "--sample-size", "20000", "--n-repeats", "1", "--K", "5", "--nu", "3", "--epsilon", "0.05",
        "--hmc-M", "1600", "--hmc-burn-in", "800", "--hmc-step-size", "0.003",
        "--num-leapfrog-steps", "10", "--target-accept-prob", "0.75",
        "--initial-state-scale", "0.01", "--hmc-initial-state", "encoder",
        "--min-effective-samples", "0", "--max-retries", "0",
        "--proposal-covariance-floor", "0.001", "--proposal-covariance-mode", "fitted_full",
        "--output-root", str(output), *options,
    ])


def visualization(command: str) -> None:
    egm_dir = VISUAL_ROOT / "egm"
    if command in ("reproduce", "train-egm"):
        rows = train_egm(low_config(2, VISUAL_EGM_LOSS, egm_dir), egm_dir, LOW_EGM_STEPS, 1000,
                         earliest_stop=LOW_EGM_STEPS + 1, patience=LOW_EGM_STEPS + 1)
        select_gated_egm_step(rows, egm_dir)
    if command in ("reproduce", "iterative"):
        train_iterative(egm_dir, VISUAL_ROOT / "iterative", [
            "--lr-schedule", "constant", "--optimizer", "adamw", "--weight-decay", "1e-5",
            "--max-epoch", str(LOW_MAX_EPOCH), "--save-epochs", ",".join(map(str, LOW_SAVE_EPOCHS))])
    if command in ("reproduce", "test"):
        evaluate(VISUAL_ROOT / "iterative" / "seed_1024", VISUAL_EPOCH, VISUAL_ROOT / "test", [
            "--no-force-iid-test-eval", "--grid-n", "50", "--variance-floor", "1e-6"])


def lowdim(command: str, dim: int) -> None:
    root = LOW_ROOT / f"dim_{dim:03d}"
    if command in ("reproduce", "train-egm"):
        rows = train_egm(low_config(dim, LOW_EGM_LOSS, root / "egm"), root / "egm", LOW_EGM_STEPS, 1000,
                         earliest_stop=LOW_EGM_STEPS + 1, patience=LOW_EGM_STEPS + 1)
        select_gated_egm_step(rows, root / "egm")
    if command in ("reproduce", "iterative"):
        train_iterative(root / "egm", root / "iterative", [
            "--lr-schedule", "rm_decay", "--optimizer", "adam",
            "--max-epoch", str(LOW_MAX_EPOCH), "--save-epochs", ",".join(map(str, LOW_SAVE_EPOCHS))])
    if command in ("reproduce", "test"):
        evaluate(root / "iterative" / "seed_1024", LOW_EPOCH[dim], root / "test", [
            "--force-iid-test-eval", "--variance-floor", "1e-6" if dim == 2 else "0.01",
            "--rng-block-size", str(LOW_RNG_BLOCK), "--skip-generation-diagnostics"])


def highdim(command: str, dim: int) -> None:
    root = HIGH_ROOT / f"dim_{dim:03d}"
    egm_dir = root / "egm" / "seed_1024"
    if command in ("reproduce", "train-egm"):
        rows = train_egm(high_config(dim, egm_dir), egm_dir, HIGH_EGM_STEPS, HIGH_EGM_EVERY,
                         earliest_stop=HIGH_EGM_STEPS + 1, patience=4, min_relative_improvement=0.01)
        select_ranked_egm_step(rows, dim, egm_dir)
    if command in ("reproduce", "iterative"):
        train_iterative(egm_dir, root / "iterative", [
            "--lr-schedule", "constant", "--optimizer", "adam",
            "--max-epoch", str(HIGH_MAX_EPOCH), "--save-epochs", ",".join(map(str, HIGH_SAVE_EPOCHS))])
    if command in ("reproduce", "test"):
        evaluate(root / "iterative" / "seed_1024", HIGH_EPOCH[dim], root / "test", [
            "--variance-floor", "0.01", "--rng-block-size", str(HIGH_RNG_BLOCK),
            "--skip-generation-diagnostics"])


def configured_dimensions(config_path: Path, group: str) -> tuple[int, ...]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    key = "low_dimensions" if group == "lowdim" else "high_dimensions"
    values = tuple(int(value) for value in payload["simulation"]["independent_gmm"][key])
    allowed = set(LOW_EPOCH if group == "lowdim" else HIGH_EPOCH)
    if not values or not set(values).issubset(allowed):
        raise ValueError(f"Invalid {key} in {config_path}: {values}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("group", choices=("visualization", "lowdim", "highdim"))
    parser.add_argument("command", choices=("reproduce", "train-egm", "iterative", "test"),
                        help="reproduce runs the three stages in order.")
    parser.add_argument("--dim", type=int, help="Run a single dimension instead of those in --config.")
    parser.add_argument("--config", type=Path, default=REPRODUCTION,
                        help="Experiment-set configuration listing the dimensions.")
    args = parser.parse_args()

    if args.group == "visualization":
        visualization(args.command)
        return
    dims = (args.dim,) if args.dim is not None else configured_dimensions(args.config, args.group)
    allowed = LOW_EPOCH if args.group == "lowdim" else HIGH_EPOCH
    if any(dim not in allowed for dim in dims):
        parser.error(f"{args.group} supports --dim in {tuple(allowed)}")
    for dim in dims:
        (lowdim if args.group == "lowdim" else highdim)(args.command, dim)


if __name__ == "__main__":
    main()
