#!/usr/bin/env python3
"""Shuttle anomaly detection with BayesNDE."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DATASET = "shuttle"
SEED = 0
RESULTS_ROOT = REPO / "outputs" / "shuttle"
SHARED = RESULTS_ROOT / "seed_0"
OUT = SHARED

BASE_CONFIG = REPO / "configs" / "model.yaml"
REPRODUCTION_CONFIG = REPO / "configs" / "reproduction.yaml"

EGM_STEPS, EGM_EVERY, MAX_EPOCH = 22000, 1000, 3000
SAVE_EPOCHS = (1, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450,
               500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1100, 1200,
               1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2200, 2400, 2600,
               2800, 3000)
DIAG_SAMPLES = 5000


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def materialize(x_dim: int) -> dict[str, Any]:
    import yaml

    source = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    model = dict(source["model"])
    model.update({"dataset": f"BGM_BS_ODDS_{DATASET}_transfer10d_seed{SEED}",
                  "output_dir": str(OUT), "save_res": False, "save_model": True,
                  "use_bnn": False, "x_dim": int(x_dim), "z_dim": int(x_dim),
                  "random_seed": SEED, "g_network_type": "mlp",
                  "e_network_type": "mlp", "g_units": [256] * 5,
                  "e_units": [256] * 5, "dx_units": [256, 256, 128, 64],
                  "dz_units": [256, 256, 128, 64], "lr": 1.0e-3,
                  "lr_theta": 0.005, "lr_z": 0.005,
                  "marginal_mmd_weight": 100.0, "joint_mmd_weight": 0.0,
                  "sliced_wasserstein_weight": 0.0, "sliced_wasserstein_projections": 64,
                  "decorrelation_weight": 0.3,
                  "variance_log_target": 0.01, "variance_log_target_weight": 0.01,
                  "iterative_empirical_corr_weight": 0.3,
                  "iterative_prior_mmd_weight": 1000.0,
                  "lr_schedule": "rm_decay", "lr_decay_power": 0.6,
                  "lr_decay_timescale": 10.0, "optimizer": "adam"})
    return {
        "seed": SEED, "method": "BayesNDE",
        "provenance": {
            "recipe": "frozen 10-dimensional independent-GMM protocol",
            "recipe_study": "indep_gmm_bgmbs_lowdim_tuned_v1, dimension 10",
            "tuned_on_shuttle": False,
            "egm_selection": "pooled generation rank (simulation rule)",
            "iterative_selection": "pooled generation rank (simulation rule)",
        },
        "model": model,
        "training": {"batch_size": int(source["training"]["batch_size"]),
                     "egm_n_iter": EGM_STEPS, "egm_batches_per_eval": EGM_EVERY,
                     "epochs": MAX_EPOCH, "save_epochs": list(SAVE_EPOCHS)},
        "generation_diagnostics": {"n_samples": DIAG_SAMPLES, "plots_enabled": False,
                                   "umap_enabled": False, "reference_max_points": 5000},
        "egm_selection": {"criterion": "pooled generation rank", "tie_break": "earliest step"},
        "density_bridge_eval": {
            "K": 5, "S": 20000, "nu": 3.0, "epsilon": 0.05, "n_repeats": 1,
            "likelihood": "gaussian", "variance_floor": 1.0e-6, "tol": 1.0e-5,
            "max_iter": 1000, "fit_fraction": 0.5, "use_neff": True,
            "eval_batch_size": 1024,
            "proposal_scale_multiplier": 1.0, "proposal_covariance_mode": "fitted_full",
            "proposal_scale": None, "proposal_center": "fitted_gmm",
            "proposal_scoring": "product_univariate_t"},
        "hmc_settings": {"M": 1600, "burn_in": 800, "step_size": 0.003,
                         "num_leapfrog_steps": 10, "target_accept_prob": 0.75,
                         "num_chains": 4, "initial_state": "encoder",
                         "initial_state_scale": 0.01, "min_effective_samples": 0,
                         "max_retries": 0, "retry_multiplier": 1.5, "strict_ess": False,
                         "proposal_covariance_floor": 0.001},
    }


def build(config: Any, timestamp: str):
    from bayesnde.training.generation_optimized import build_bgm_model

    params = dict(config["model"])
    params["save_model"] = False
    return build_bgm_model(params, timestamp=timestamp, random_seed=SEED)


def diagnose(model, data, config, run_dir: Path) -> None:
    from bayesnde.diagnostics.generation import run_uci_generation_diagnostics_from_arrays

    n = int(config["generation_diagnostics"]["n_samples"])
    samples, _variance = model.generate(nb_samples=n, use_x_sd=True)
    generated = np.asarray(samples, dtype=np.float64)
    if not np.isfinite(generated).all():
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "diverged.json").write_text(json.dumps({
            "finite_rows": int(np.isfinite(generated).all(axis=1).sum()),
            "rows": int(generated.shape[0]),
            "reason": "generator produced non-finite samples",
        }, indent=2), encoding="utf-8")
        return
    run_uci_generation_diagnostics_from_arrays(
        generated_x=generated, train_x=data["train_x"], validation_x=data["val_x"],
        test_x=data["val_x"], run_dir=run_dir, config=config, seed=20260803)


def _scheduled_rates(params, epoch: int) -> tuple[float, float]:
    lr_theta0, lr_z0 = float(params["lr_theta"]), float(params["lr_z"])
    schedule = str(params.get("lr_schedule", "constant"))
    if schedule == "constant":
        return lr_theta0, lr_z0
    if schedule == "rm_decay":
        factor = (1.0 + float(epoch) / max(float(params.get("lr_decay_timescale", 10.0)), 1e-12)) \
            ** (-float(params.get("lr_decay_power", 0.6)))
        return lr_theta0 * factor, lr_z0 * factor
    raise ValueError(f"Unsupported lr schedule: {schedule!r}")


def _set_lr(optimizer, value: float) -> None:
    import tensorflow as tf

    lr = optimizer.learning_rate
    if hasattr(lr, "assign"):
        lr.assign(float(value))
    else:
        tf.keras.backend.set_value(lr, float(value))


def train_iterative_scheduled(model: Any, train_x: np.ndarray, val_x: np.ndarray,
                              config: Mapping[str, Any], root: Path, seed: int) -> None:
    import re
    import tensorflow as tf
    from experiments.application_odds.run_bgm_bs import (
        append_log, atomic_weights, decoder_log_likelihood, load_curve, save_curve)

    model.g_net.trainable = True; model.e_net.trainable = False
    model.dz_net.trainable = False; model.dx_net.trainable = False
    correlation = np.corrcoef(train_x.astype(np.float64), rowvar=False)
    if np.ndim(correlation) == 0: correlation = np.asarray([[1.0]])
    if not np.isfinite(correlation).all(): raise ValueError("Non-finite training correlation matrix")
    model.params["iterative_empirical_corr_target"] = correlation.tolist()
    encoded = [np.asarray(model.e_net(train_x[start:start + 1024], training=False).numpy())
               for start in range(0, len(train_x), 1024)]
    model.data_z = tf.Variable(np.vstack(encoded).astype(np.float32), name="Latent Variable", trainable=True)
    resume = tf.train.Checkpoint(g_net=model.g_net, e_net=model.e_net, dz_net=model.dz_net,
                                 dx_net=model.dx_net, g_optimizer=model.g_optimizer,
                                 posterior_optimizer=model.posterior_optimizer, data_z=model.data_z)
    manager = tf.train.CheckpointManager(resume, str(root / "checkpoint/iterative_resume"), max_to_keep=1)
    start_epoch = 0
    if manager.latest_checkpoint:
        resume.restore(manager.latest_checkpoint).expect_partial()
        match = re.search(r"-(\d+)$", manager.latest_checkpoint); start_epoch = int(match.group(1)) if match else 0
    rows, batch_size = load_curve(root), int(config["training"]["batch_size"])
    max_epoch, save_epochs = int(config["training"]["epochs"]), set(config["training"]["save_epochs"])
    for epoch in range(start_epoch + 1, max_epoch + 1):
        lr_theta, lr_z = _scheduled_rates(model.params, epoch)
        _set_lr(model.g_optimizer, lr_theta)
        _set_lr(model.posterior_optimizer, lr_z)
        order = np.random.default_rng(seed + 700000 + epoch).permutation(len(train_x))
        lx, lz, lcorr = [], [], []
        for start in range(0, len(train_x) - batch_size + 1, batch_size):
            indices = order[start:start + batch_size]; batch_z = tf.Variable(tf.gather(model.data_z, indices), trainable=True)
            batch_x = train_x[indices]; a = model.update_g_net(batch_z, batch_x)
            lcorr.append(float(model.update_iterative_empirical_corr(batch_x)))
            c = model.update_latent_variable_sgd(batch_z, batch_x)
            model.data_z.scatter_nd_update(tf.expand_dims(indices, axis=1), batch_z)
            lx.append(float(a)); lz.append(float(c))
        if epoch not in save_epochs and epoch != max_epoch: continue
        path = root / "checkpoint/iterative" / f"weights_at_{epoch}_generator.weights.h5"
        atomic_weights(model.g_net, path); score = decoder_log_likelihood(model, val_x)
        rows.append({"epoch": epoch, "validation_decoder_mean_log_likelihood": score,
                     "loss_x": float(np.mean(lx)), "loss_z": float(np.mean(lz)),
                     "empirical_corr_loss": float(np.mean(lcorr)), "generator_weights": str(path)})
        save_curve(root, rows); manager.save(checkpoint_number=epoch)
        append_log(root / "run.log", f"iterative epoch={epoch}/{max_epoch} decoder_val_ll={score:.8g}")


def stage_egm(data, config) -> None:
    from experiments.application_odds import run_bgm_bs as app

    model = build(config, "transfer10d_egm")
    app.train_egm(model, data["train_x"], data["val_x"], config, SHARED, data["val_y"])
    for step in range(EGM_EVERY, EGM_STEPS + 1, EGM_EVERY):
        prefix = SHARED / "checkpoint/egm" / f"egm_step_{step:04d}"
        generator = Path(str(prefix) + "_generator.weights.h5")
        if not generator.exists():
            continue
        target = SHARED / "egm_diagnostics" / f"step_{step:05d}"
        if (target / "generation_diagnostics.json").exists():
            continue
        model.g_net.load_weights(str(generator))
        model.e_net.load_weights(str(Path(str(prefix) + "_encoder.weights.h5")))
        diagnose(model, data, config, target)
        print(f"diagnostics written for EGM step {step}", flush=True)


def stage_select_egm(data, config, rule: str) -> None:
    if rule == "genrank":
        from bayesnde.diagnostics.generation import load_candidates, select_pooled

        candidates = load_candidates((SHARED / "egm_diagnostics").glob("step_*"), "step")
        payload = select_pooled(candidates, int(config["model"]["x_dim"]), key="step")
        step = int(payload["step"])
    else:
        rows = json.loads((SHARED / "egm_validation_curve.json").read_text(encoding="utf-8"))
        rows = rows["rows"] if isinstance(rows, dict) and "rows" in rows else rows
        finite = [r for r in rows
                  if float(r["validation_decoder_mean_log_likelihood"]) ==
                  float(r["validation_decoder_mean_log_likelihood"])]
        best = max(finite, key=lambda r: (float(r["validation_decoder_mean_log_likelihood"]),
                                          -int(r["step"])))
        step = int(best["step"])
        payload = {"criterion": "maximum validation decoder mean log likelihood",
                   "step": step, "selected": best, "pooled_candidates": len(finite)}
    prefix = SHARED / "checkpoint/egm" / f"egm_step_{step:04d}"
    payload["generator_weights"] = str(Path(str(prefix) + "_generator.weights.h5"))
    payload["encoder_weights"] = str(Path(str(prefix) + "_encoder.weights.h5"))
    payload["rule"] = rule
    save_json(OUT / "selected_egm_step.json", payload)
    print(f"EGM step by {rule}: {step}  (pool {payload['pooled_candidates']})")


def stage_iterative(data, config, with_diagnostics: bool = True) -> None:
    from experiments.application_odds import run_bgm_bs as app

    chosen = json.loads((OUT / "selected_egm_step.json").read_text(encoding="utf-8"))
    model = build(config, "transfer10d_iterative")
    model.g_net.load_weights(chosen["generator_weights"])
    model.e_net.load_weights(chosen["encoder_weights"])
    train_iterative_scheduled(model, data["train_x"], data["val_x"], config, OUT, SEED)
    if not with_diagnostics:
        return
    for epoch in SAVE_EPOCHS:
        weights = OUT / "checkpoint/iterative" / f"weights_at_{epoch}_generator.weights.h5"
        if not weights.exists():
            continue
        target = OUT / "iterative_diagnostics" / f"stage_epoch_{epoch:05d}"
        if (target / "generation_diagnostics.json").exists():
            continue
        model.g_net.load_weights(str(weights))
        diagnose(model, data, config, target)
        print(f"diagnostics written for epoch {epoch}", flush=True)


def stage_select_iterative(data, config, rule: str) -> None:
    if rule == "genrank":
        from bayesnde.diagnostics.generation import load_candidates, select_pooled

        candidates = load_candidates((OUT / "iterative_diagnostics").glob("stage_epoch_*"), "epoch")
        payload = select_pooled(candidates, int(config["model"]["x_dim"]), key="epoch")
        epoch = int(payload["epoch"])
    else:
        rows = json.loads((OUT / "iterative_validation_curve.json").read_text(encoding="utf-8"))
        rows = rows["rows"] if isinstance(rows, dict) and "rows" in rows else rows
        finite = [r for r in rows
                  if float(r["validation_decoder_mean_log_likelihood"]) ==
                  float(r["validation_decoder_mean_log_likelihood"])]
        best = max(finite, key=lambda r: (float(r["validation_decoder_mean_log_likelihood"]),
                                          -int(r["epoch"])))
        epoch = int(best["epoch"])
        payload = {"criterion": "maximum validation decoder mean log likelihood",
                   "epoch": epoch, "selected": best, "pooled_candidates": len(finite)}
    payload["generator_weights"] = str(
        OUT / "checkpoint/iterative" / f"weights_at_{epoch}_generator.weights.h5")
    payload["rule"] = rule
    save_json(OUT / f"selected_iterative_epoch_{rule}.json", payload)
    save_json(OUT / "selected_iterative_epoch.json", payload)
    print(f"iterative epoch by {rule}: {epoch}  (pool {payload['pooled_candidates']})")


def stage_test(data, config, shard_index: int, num_shards: int, rule: str) -> None:
    from experiments.application_odds import run_bgm_bs as app
    from bayesnde.estimators.bridge import BGM_BridgeDensityEstimator

    chosen = json.loads((OUT / f"selected_iterative_epoch_{rule}.json").read_text(encoding="utf-8"))
    egm = json.loads((OUT / "selected_egm_step.json").read_text(encoding="utf-8"))
    model = build(config, "transfer10d_test")
    model.g_net.load_weights(chosen["generator_weights"])
    model.e_net.load_weights(egm["encoder_weights"])
    test_x = data["test_x"]
    bounds = np.array_split(np.arange(len(test_x)), num_shards)[shard_index]
    cfg = config["density_bridge_eval"]
    out = OUT / f"test_{rule}" / f"shard_{shard_index:03d}"
    out.mkdir(parents=True, exist_ok=True)
    artifact = out / "log_px.npz"
    values = np.full(len(bounds), np.nan)
    if artifact.exists():
        with np.load(artifact) as raw:
            if np.array_equal(np.asarray(raw["indices"]), bounds):
                values = np.asarray(raw["log_px"], dtype=np.float64)

    for position, index in enumerate(bounds):
        if np.isfinite(values[position]):
            continue
        estimator = BGM_BridgeDensityEstimator(
            model=model, likelihood="gaussian",
            random_seed=app.point_seed(SEED, int(index)),
            variance_floor=float(cfg["variance_floor"]))
        output = estimator.estimate(
            test_x[int(index)], K=int(cfg["K"]), S=int(cfg["S"]), nu=float(cfg["nu"]),
            epsilon=float(cfg["epsilon"]), n_repeats=int(cfg["n_repeats"]),
            hmc_settings=config["hmc_settings"],
            eval_batch_size=int(cfg["eval_batch_size"]),
            bridge_tol=float(cfg["tol"]), bridge_max_iter=int(cfg["max_iter"]),
            fit_fraction=float(cfg["fit_fraction"]), use_neff=bool(cfg["use_neff"]),
            proposal_scale_multiplier=float(cfg["proposal_scale_multiplier"]),
            proposal_covariance_mode=str(cfg["proposal_covariance_mode"]),
            proposal_scale=cfg["proposal_scale"],
            proposal_center=str(cfg["proposal_center"]),
            proposal_scoring=str(cfg["proposal_scoring"]), return_proposal=False)
        values[position] = float(output["log_px"])
        if (position + 1) % 25 == 0 or position + 1 == len(bounds):
            np.savez(artifact, indices=bounds, log_px=values)
            print(f"shard {shard_index}: {position + 1}/{len(bounds)}", flush=True)
    np.savez(artifact, indices=bounds, log_px=values)


def stage_merge(data, config, rule: str) -> None:
    from experiments.application_odds import run_bgm_bs as app

    shards = sorted((OUT / f"test_{rule}").glob("shard_*/log_px.npz"))
    if not shards:
        raise FileNotFoundError("No test shards to merge")
    log_px = np.full(len(data["test_x"]), np.nan)
    for path in shards:
        with np.load(path) as raw:
            log_px[np.asarray(raw["indices"])] = np.asarray(raw["log_px"])
    missing = int(np.sum(~np.isfinite(log_px)))
    if missing:
        raise RuntimeError(f"{missing} test points are still unscored")
    precision = app.precision_at_k(log_px, data["test_y"])
    payload = {
        "dataset": DATASET, "arm": "transfer_from_10d_simulation", "seed": SEED,
        "precision_at_k": precision,
        "k": int(np.sum(data["test_y"] == 1)), "points": int(len(log_px)),
        "selected_egm_step": json.loads((OUT / "selected_egm_step.json").read_text())["step"],
        "egm_rule": json.loads((OUT / "selected_egm_step.json").read_text())["rule"],
        "epoch_rule": rule,
        "selected_iterative_epoch": json.loads(
            (OUT / f"selected_iterative_epoch_{rule}.json").read_text())["epoch"],
        "provenance": config["provenance"],
        "mean_log_px": float(np.mean(log_px)),
    }
    np.savez(OUT / f"test_log_px_{rule}.npz", log_px=log_px, labels=data["test_y"])
    save_json(OUT / f"metrics_{rule}.json", payload)
    print(json.dumps(payload, indent=2, default=str))


def run_once(args: argparse.Namespace, seed: int) -> None:
    from experiments.application_odds import run_bgm_bs as app

    global OUT, SHARED, SEED
    SEED = int(seed)
    SHARED = RESULTS_ROOT / f"seed_{SEED}"
    OUT = SHARED / "arms" / args.arm
    import random
    import tensorflow as tf
    np.random.seed(SEED)
    random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    data = app.load_data(DATASET, args.data_root)
    config = materialize(data["train_x"].shape[1])
    config["provenance"]["arm"] = args.arm
    config["provenance"]["egm_selection"] = args.egm_rule
    config["provenance"]["iterative_selection"] = args.epoch_rule
    OUT.mkdir(parents=True, exist_ok=True)
    save_json(OUT / "resolved_config.json", config)

    if args.stage == "egm":
        stage_egm(data, config)
    elif args.stage == "select-egm":
        stage_select_egm(data, config, args.egm_rule)
    elif args.stage == "iterative":
        stage_iterative(data, config, with_diagnostics=not args.skip_iterative_diagnostics)
    elif args.stage == "select-iterative":
        stage_select_iterative(data, config, args.epoch_rule)
    elif args.stage == "test":
        stage_test(data, config, args.shard_index, args.num_shards, args.epoch_rule)
    elif args.stage == "merge":
        stage_merge(data, config, args.epoch_rule)
    else:
        stage_egm(data, config)
        stage_select_egm(data, config, "genrank")
        stage_iterative(data, config, with_diagnostics=False)
        stage_select_iterative(data, config, "decoderll")
        stage_test(data, config, 0, 1, "decoderll")
        stage_merge(data, config, "decoderll")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="reproduce",
                        choices=("egm", "select-egm", "iterative", "select-iterative",
                                 "test", "merge", "reproduce"))
    parser.add_argument("--seed", type=int,
                        help="Optional single-seed override for a lower-level run.")
    parser.add_argument("--config", type=Path, default=REPRODUCTION_CONFIG)
    parser.add_argument("--skip-iterative-diagnostics", action="store_true")
    parser.add_argument("--arm", default="egm_genrank_rm",
                        help="Output subdirectory; arms share the EGM checkpoints.")
    parser.add_argument("--egm-rule", choices=("genrank", "decoderll"), default="genrank")
    parser.add_argument("--epoch-rule", choices=("genrank", "decoderll"), default="decoderll")
    parser.add_argument("--data-root", type=Path, default=HERE / "data/processed")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    import yaml
    if args.stage == "reproduce" and args.seed is None:
        from experiments.application_odds.prepare_data import prepare
        from experiments.application_odds.summarize import main as summarize

        processed = args.data_root / f"{DATASET}.npz"
        if not processed.exists():
            prepare(args.data_root.parent)
        configured = yaml.safe_load(args.config.read_text(encoding="utf-8"))["shuttle"]["seeds"]
        for seed in configured:
            run_once(args, int(seed))
        summarize(args.config)
    else:
        if args.seed is None:
            parser.error("a lower-level stage requires --seed")
        run_once(args, args.seed)


if __name__ == "__main__":
    main()
