#!/usr/bin/env python3
"""The involute (Swiss-roll) simulation."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bayesnde.data.samplers import (
    BASE_CONFIG,
    assert_disjoint_splits,
    build_sampler,
    dataset_fingerprint,
)
from bayesnde.diagnostics.generation import load_candidates, select_pooled
from bayesnde.training.generation_optimized import (
    build_bgm_model,
    train_egm_with_diagnostics,
)

SRC = Path(__file__).resolve().parent
REPO = SRC.parents[1]
OUTPUT = REPO / "outputs/simulation/involute"
BRIDGE = REPO / "configs/bridge.yaml"
SEED = 1024

EGM_SPEC = {"width": 256, "depth": 5, "lr": 1e-3, "loss": {"x_cycle_weight": 1.0}}
EGM_STEPS = 22000
EGM_EVERY = 1000
ITERATIVE = {"optimizer": "adamw", "schedule": "constant",
             "weight_decay": 1e-5, "corr": 0.3, "prior": 1000.0}
MAX_EPOCH = 3000
SAVE_EPOCHS = (0, 1, 10, 25, 50, 100, 200, 400, 700, 1000, 1500, 2000, 2500, 3000)

LR_VARIANTS = {
    "lr1e-6": (1e-6, 1e-7),
    "lr1e-5": (1e-5, 1e-6),
    "lr1e-4": (1e-4, 1e-5),
    "lr1e-3": (1e-3, 1e-4),
}

DATA = {"name": "involute", "seed": SEED, "n": 20000, "dim": 2,
        "theta": 2.0 * np.pi, "scale": 2.0, "sigma": 0.4}
GRID = {"x1_min": -6.0, "x1_max": 5.0, "x2_min": -5.0, "x2_max": 5.0, "n": 50}

EGM_DIR = OUTPUT / "egm"


def iter_dir(tag: str) -> Path:
    return OUTPUT / "iterative" / tag / "seed_1024"


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(REPO), check=True)


def build_config() -> dict[str, Any]:
    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    width, depth = EGM_SPEC["width"], EGM_SPEC["depth"]
    cfg["data"] = dict(DATA)
    cfg["grid"] = dict(GRID)
    cfg["model"].update({
        "dataset": "BGM_involute", "output_dir": str(EGM_DIR.resolve()), "save_model": True,
        "save_res": False, "use_bnn": False, "x_dim": 2, "z_dim": 2, "random_seed": SEED,
        "g_network_type": "mlp", "e_network_type": "mlp",
        "g_units": [width] * depth, "e_units": [width] * depth,
        "dx_units": [256, 256, 128, 64], "dz_units": [256, 256, 128, 64],
        "lr": float(EGM_SPEC["lr"]), "cycle_weight": 5.0, "x_cycle_weight": 3.0,
        "z_cycle_weight": 1.0, "g_adv_weight": 1.0, "e_adv_weight": 1.0,
        "marginal_mmd_weight": 0.0, "joint_mmd_weight": 0.0,
        "sliced_wasserstein_weight": 0.0, "sliced_wasserstein_projections": 64,
        "decorrelation_weight": 0.3, "decorrelation_target": "train",
        "variance_log_target": 0.01, "variance_log_target_weight": 0.01,
    })
    cfg["model"].update(EGM_SPEC["loss"])
    cfg["training"].update({"batch_size": 256, "egm_n_iter": EGM_STEPS,
                            "egm_batches_per_eval": EGM_EVERY,
                            "fresh_training_required": True,
                            "external_checkpoint_reuse_allowed": False, "early_stop": None})
    cfg.setdefault("generation_diagnostics", {}).update({
        "n_samples": 5000, "plots_enabled": False, "umap_enabled": False,
        "marginal_plot_dims_per_page": 10, "plot_reference_split": "validation"})
    cfg.setdefault("density", {}).update({
        "test_log_likelihood_points": 2000, "test_log_likelihood_seed": 8943,
        "n_repeats": 1, "variance_floor": 1e-6, "force_iid_test_eval": False})
    cfg["experiment_protocol"] = {
        "name": "involute_v1", "single_seed": SEED,
        "recipe_inherited_from": "the frozen 2D independent-GMM candidate",
        "selection_rule": "pooled generation rank, both stages",
        "test_used_for_model_selection": False, "fresh_initialization": True,
    }
    return cfg


def stage_egm() -> None:
    EGM_DIR.mkdir(parents=True, exist_ok=True)
    cfg = build_config()
    (EGM_DIR / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    sampler = build_sampler(cfg["data"])
    assert_disjoint_splits(sampler.X_train, sampler.X_val, sampler.X_test)
    save_json(EGM_DIR / "dataset_fingerprint.json", dataset_fingerprint(sampler))
    model = build_bgm_model(cfg["model"], timestamp="fresh", random_seed=SEED)
    train_egm_with_diagnostics(
        model=model, train_data=np.asarray(sampler.X_train, np.float32), sampler=sampler,
        config=cfg, diagnostics_root=EGM_DIR / "egm_diagnostics", max_steps=EGM_STEPS,
        batch_size=256, every=EGM_EVERY, earliest_stop=EGM_STEPS + 1,
        patience=EGM_STEPS + 1, seed=SEED)

    candidates = load_candidates((EGM_DIR / "egm_diagnostics").glob("step_*"), "step")
    selected = select_pooled(candidates, 2, key="step")
    save_json(EGM_DIR / "selected_egm_step.json", selected)
    print(f"EGM step selected by generation rank: {selected['step']}")


def stage_iterative(tag: str) -> None:
    lr_theta, lr_z = LR_VARIANTS[tag]
    step = int(json.loads((EGM_DIR / "selected_egm_step.json").read_text(encoding="utf-8"))["step"])
    hits = sorted({p.parent for p in (EGM_DIR / "checkpoints").rglob(
        f"weights_at_egm_init_{step}_generator.weights.h5")})
    if len(hits) != 1:
        raise RuntimeError(f"Expected one checkpoint directory for EGM step {step}, found {hits}")
    run([sys.executable, "-m", "bayesnde.training.iterative",
         "--config", str(EGM_DIR / "resolved_config.yaml"),
         "--init-mode", "checkpoint", "--checkpoint-dir", str(hits[0]),
         "--egm-iter", str(step), "--out-dir", str(iter_dir(tag).parent),
         "--run-name", "seed_1024", "--model-seed", str(SEED), "--seed", "711024",
         "--lr-theta", str(lr_theta), "--lr-z", str(lr_z),
         "--lr-schedule", ITERATIVE["schedule"], "--optimizer", ITERATIVE["optimizer"],
         "--weight-decay", str(ITERATIVE["weight_decay"]),
         "--max-epoch", str(MAX_EPOCH), "--save-epochs", ",".join(map(str, SAVE_EPOCHS)),
         "--batch-size", "256", "--iterative-decorrelation-target", "train",
         "--iterative-empirical-corr-weight", str(ITERATIVE["corr"]),
         "--iterative-prior-mmd-weight", str(ITERATIVE["prior"])])


def stage_select() -> None:
    pooled: list[dict[str, Any]] = []
    for tag in LR_VARIANTS:
        directory = iter_dir(tag)
        if directory.exists():
            pooled.extend(load_candidates(directory.glob("stage_epoch_*"), "epoch", variant=tag))
    if not pooled:
        raise RuntimeError("No iterative variant produced usable diagnostics.")
    payload = select_pooled(pooled, 2, key="epoch")
    winner = payload["selected"]
    payload["lr_theta"], payload["lr_z"] = LR_VARIANTS[winner["variant"]]
    save_json(OUTPUT / "selected_iterative_epoch.json", payload)
    print(f"selected {winner['variant']} epoch {winner['epoch']} (score {winner['score']:.4f})")


def stage_eval() -> None:
    chosen = json.loads((OUTPUT / "selected_iterative_epoch.json").read_text(encoding="utf-8"))
    winner = chosen["selected"]
    epoch, variant = int(winner["epoch"]), str(winner["variant"])
    directory = iter_dir(variant)
    hits = sorted({p.parent for p in (directory / "checkpoints").rglob(
        f"weights_at_{epoch}_generator.weights.h5")})
    if len(hits) != 1:
        raise RuntimeError(f"Expected one checkpoint directory for epoch {epoch}, found {hits}")
    run([sys.executable, "-m", "bayesnde.evaluation.bridge_eval",
         "--config", str(BRIDGE), "--visual-config", str(directory / "run_config_used.yaml"),
         "--preset", "final", "--run-name", "result",
         "--checkpoint-dir", str(hits[0]), "--epoch", str(epoch), "--load-encoder",
         "--evaluation-split", "test", "--test-points", "2000",
         "--no-force-iid-test-eval", "--grid-n", str(GRID["n"]),
         "--sample-size", "20000", "--n-repeats", "1", "--K", "5", "--nu", "3",
         "--epsilon", "0.05", "--hmc-M", "1600", "--hmc-burn-in", "800",
         "--hmc-step-size", "0.003", "--num-leapfrog-steps", "10",
         "--target-accept-prob", "0.75", "--initial-state-scale", "0.01",
         "--hmc-initial-state", "encoder", "--min-effective-samples", "0",
         "--max-retries", "0", "--proposal-covariance-floor", "0.001",
         "--proposal-covariance-mode", "fitted_full", "--variance-floor", "1e-6",
         "--output-root", str(OUTPUT / "test" / f"{variant}_e{epoch}")])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=("egm", "iterative", "select", "eval", "reproduce"))
    parser.add_argument("--variant", choices=tuple(LR_VARIANTS),
                        help="Iterative learning-rate variant (required by --stage iterative).")
    args = parser.parse_args()
    if args.stage == "egm":
        stage_egm()
    elif args.stage == "iterative":
        if args.variant is None:
            parser.error("--stage iterative requires --variant")
        stage_iterative(args.variant)
    elif args.stage == "select":
        stage_select()
    elif args.stage == "eval":
        stage_eval()
    else:
        stage_egm()
        for tag in LR_VARIANTS:
            stage_iterative(tag)
        stage_select()
        stage_eval()


if __name__ == "__main__":
    main()
