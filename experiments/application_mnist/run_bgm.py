#!/usr/bin/env python3
"""Train, select and evaluate the conditional MNIST BGM-BS classifier."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

APP = Path(__file__).resolve().parent
REPO = APP.parents[1]

from experiments.application_mnist.model import ConditionalMNISTBGM
from bayesnde.estimators.conditional import estimate_point_bridge, one_hot


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def sha256_array(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def data_path(config: dict) -> Path:
    return APP / "data" / "mnist_protocol.npz"


def prepare(config: dict, force: bool = False) -> None:
    target = data_path(config)
    if target.exists() and not force:
        print(target)
        return
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train.astype(np.float32) / 255.0).reshape(-1, 784)
    x_test = (x_test.astype(np.float32) / 255.0).reshape(-1, 784)
    indices = np.arange(len(x_train))
    train_idx, val_idx = train_test_split(
        indices, test_size=int(config["data"]["validation_size"]),
        random_state=int(config["data"]["split_seed"]), stratify=y_train)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, train_x=x_train[train_idx], train_y=y_train[train_idx],
                        val_x=x_train[val_idx], val_y=y_train[val_idx],
                        test_x=x_test, test_y=y_test,
                        train_indices=train_idx, validation_indices=val_idx)
    manifest = {"source": "tf.keras.datasets.mnist", "pixel_scale": "[0,1] grayscale",
                "split_seed": int(config["data"]["split_seed"]), "arrays": {}}
    for name, value in (("train_x", x_train[train_idx]), ("train_y", y_train[train_idx]),
                        ("val_x", x_train[val_idx]), ("val_y", y_train[val_idx]),
                        ("test_x", x_test), ("test_y", y_test)):
        manifest["arrays"][name] = {"shape": list(value.shape), "sha256": sha256_array(value)}
    atomic_json(target.with_suffix(".manifest.json"), manifest)


def load_data(config: dict) -> dict[str, np.ndarray]:
    path = data_path(config)
    if not path.exists():
        raise FileNotFoundError(f"Run prepare first: {path}")
    with np.load(path) as raw:
        return {key: np.asarray(raw[key]) for key in raw.files}


def model_params(config: dict, smoke: bool = False) -> dict:
    params = dict(config["model"])
    params.update({"x_dim": 784, "y_dim": 10, "use_bnn": False,
                   "save_model": True, "save_res": False,
                   "dz_units": [256, 256, 128, 64],
                   "dx_units": [256, 256, 128, 64], "gamma": 0.0,
                   "cycle_weight": 10.0, "x_cycle_weight": 10.0,
                   "z_cycle_weight": 10.0, "x_adv_use_mean": False,
                   "x_cycle_use_mean": False, "z_cycle_use_mean": False})
    if smoke:
        params["filters"] = 4
        params["dz_units"] = [16, 8]
        params["dx_units"] = [16, 8]
    return params


def run_root(seed: int, smoke: bool = False) -> Path:
    if smoke:
        return REPO / "outputs" / "mnist_smoke" / f"seed_{seed}"
    return REPO / "outputs" / "mnist_v2" / "bernoulli_logit" / f"seed_{seed}"


def checkpoint_dir(seed: int, epoch: int, smoke: bool = False) -> Path:
    return run_root(seed, smoke) / "checkpoints" / f"epoch_{epoch:04d}"


def train(config: dict, seed: int, smoke: bool) -> None:
    random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed)
    data = load_data(config)
    train_x, train_y = data["train_x"].astype(np.float32), data["train_y"].astype(np.int64)
    val_x, val_y = data["val_x"].astype(np.float32), data["val_y"].astype(np.int64)
    if smoke:
        train_x, train_y, val_x, val_y = train_x[:128], train_y[:128], val_x[:32], val_y[:32]
    model = ConditionalMNISTBGM(model_params(config, smoke)); model.build()
    atomic_json(run_root(seed, smoke) / "runtime_manifest.json", {
        "tensorflow_version": tf.__version__,
        "smoke": bool(smoke),
    })
    batch_size = 16 if smoke else int(config["training"]["batch_size"])
    egm_iterations = 2 if smoke else int(config["training"]["egm_iterations"])
    epochs = 2 if smoke else int(config["training"]["iterative_epochs"])
    save_every = 1 if smoke else int(config["training"]["checkpoint_every"])
    y_onehot = one_hot(train_y, 10)
    history, started = [], time.time()
    for step in range(1, egm_iterations + 1):
        idx = np.random.randint(0, len(train_x), size=batch_size)
        z = tf.random.normal((batch_size, 10))
        model.train_disc_step(z, train_x[idx], y_onehot[idx])
        losses = model.train_gen_step(z, train_x[idx], y_onehot[idx])
        if step == 1 or step % max(1, egm_iterations // 10) == 0:
            print(f"EGM {step}/{egm_iterations} loss={float(losses[-1]):.6f}", flush=True)
    latents = tf.Variable(model.encode(train_x, y_onehot, training=False))
    for epoch in range(1, epochs + 1):
        order = np.random.permutation(len(train_x))
        epoch_losses = []
        for start in range(0, len(order) - batch_size + 1, batch_size):
            idx = order[start:start + batch_size]
            batch_z = tf.Variable(tf.gather(latents, idx))
            loss_x = model.update_g_net(batch_z, train_x[idx], y_onehot[idx])
            model.update_latent_variable_sgd(batch_z, train_x[idx], y_onehot[idx])
            latents.scatter_nd_update(idx[:, None], batch_z)
            epoch_losses.append(float(loss_x))
        row = {"epoch": epoch, "train_nll": float(np.mean(epoch_losses)),
               "elapsed_seconds": time.time() - started}
        history.append(row); print(row, flush=True)
        if epoch % save_every == 0 or epoch == epochs:
            model.save(checkpoint_dir(seed, epoch, smoke))
    atomic_json(run_root(seed, smoke) / "training_history.json", history)
    select_checkpoint(config, seed, val_x, val_y, smoke)


def select_checkpoint(config: dict, seed: int, val_x: np.ndarray, val_y: np.ndarray,
                      smoke: bool) -> None:
    rows = []
    for path in sorted((run_root(seed, smoke) / "checkpoints").glob("epoch_*")):
        epoch = int(path.name.split("_")[-1])
        model = ConditionalMNISTBGM(model_params(config, smoke)); model.restore(path)
        values = []
        for start in range(0, len(val_x), 256):
            x = tf.convert_to_tensor(val_x[start:start + 256])
            y = tf.convert_to_tensor(one_hot(val_y[start:start + 256], 10))
            z = model.encode(x, y, training=False)
            mean, var, _ = model._decode_generator(z, y, training=False)
            values.extend((-model.observation_log_likelihood(x, mean, var)).numpy().tolist())
        rows.append({"epoch": epoch, "validation_encoder_nll": float(np.mean(values))})
    if smoke:
        best = min(rows, key=lambda row: (row["validation_encoder_nll"], row["epoch"]))
        criterion = f"minimum validation {ConditionalMNISTBGM.likelihood_name} encoder NLL (smoke only)"
    else:
        selection = config["selection"]
        candidate_count = int(selection["encoder_nll_candidates"])
        candidates = sorted(rows, key=lambda row: (row["validation_encoder_nll"], row["epoch"]))[:candidate_count]
        per_class = int(selection["validation_points_per_class"])
        validation_indices = np.concatenate([
            np.flatnonzero(val_y == label)[:per_class] for label in range(10)
        ])
        hmc, kwargs = bridge_settings(config, smoke=False)
        hmc.update({
            "num_chains": int(selection["hmc_chains"]),
            "burn_in": int(selection["hmc_burn_in"]),
            "M": int(selection["hmc_posterior_samples"]),
        })
        kwargs["hmc_settings"] = hmc
        kwargs["S"] = int(selection["proposal_samples"])
        for row in candidates:
            model = ConditionalMNISTBGM(model_params(config, False))
            model.restore(checkpoint_dir(seed, int(row["epoch"]), False))
            scores, acceptance, bridge_ess = [], [], []
            for index in validation_indices:
                point_seed = seed * 10_000_019 + int(row["epoch"]) * 100_003 + int(index) * 1009 + int(val_y[index]) * 97
                result = estimate_point_bridge(
                    model, val_x[index], int(val_y[index]), seed=point_seed, **kwargs)
                scores.append(float(result["log_px_std"]))
                acceptance.append(float(result["diagnostics"]["acceptance_rate"]))
                bridge_ess.append(float(result["diagnostics"]["bridge_ess"]))
            row.update({
                "bridge_candidate": True,
                "validation_bridge_points": int(len(validation_indices)),
                "validation_bridge_mean_log_density": float(np.mean(scores)),
                "validation_bridge_mean_acceptance": float(np.mean(acceptance)),
                "validation_bridge_median_ess": float(np.median(bridge_ess)),
            })
        candidate_epochs = {int(row["epoch"]) for row in candidates}
        for row in rows:
            row.setdefault("bridge_candidate", int(row["epoch"]) in candidate_epochs)
        best = max(candidates, key=lambda row: (row["validation_bridge_mean_log_density"], -row["epoch"]))
        criterion = "maximum BGM-BS mean log p(x|true_label) among three encoder-NLL candidates"
    atomic_json(run_root(seed, smoke) / "selection_manifest.json",
                {"criterion": criterion, "test_used": False, "rows": rows,
                 "selected_epoch": best["epoch"],
                 "variant": "bernoulli_logit",
                 "likelihood": ConditionalMNISTBGM.likelihood_name})


def bridge_settings(config: dict, smoke: bool) -> tuple[dict, dict]:
    bridge = config["bridge"]
    hmc = {"num_chains": 2 if smoke else int(bridge["hmc_chains"]),
           "burn_in": 2 if smoke else int(bridge["hmc_burn_in"]),
           "M": 4 if smoke else int(bridge["hmc_posterior_samples"]),
           "step_size": float(bridge["hmc_step_size"]),
           "num_leapfrog_steps": 2 if smoke else int(bridge["hmc_leapfrog_steps"]),
           "target_accept_prob": float(bridge["target_acceptance"]),
           "initial_state": "encoder", "initial_state_scale": 0.01,
           "min_effective_samples": 0, "max_retries": 0, "use_neff": True}
    args = {"K": int(bridge["components"]),
            "S": 16 if smoke else int(bridge["proposal_samples"]),
            "nu": float(bridge["student_t_df"]),
            "epsilon": float(bridge["defensive_prior_weight"]),
            "hmc_settings": hmc, "eval_batch_size": 256,
            "fit_fraction": float(bridge["fit_fraction"]),
            "bridge_tol": float(bridge["tolerance"]),
            "bridge_max_iter": int(bridge["max_iterations"]),
            "covariance_floor": 1e-3, "proposal_covariance_mode": "fitted_full",
            "proposal_scale": None, "proposal_center": "fitted_gmm",
            "proposal_scoring": "multivariate_t"}
    return hmc, args


def evaluate_shard(config: dict, seed: int, shard: int, shard_size: int, smoke: bool) -> None:
    data = load_data(config); x, labels = data["test_x"], data["test_y"].astype(np.int64)
    if smoke: x, labels = x[:4], labels[:4]
    start, stop = shard * shard_size, min(len(x), (shard + 1) * shard_size)
    if start >= stop: raise ValueError("empty shard")
    selection = json.loads((run_root(seed, smoke) / "selection_manifest.json").read_text())
    model = ConditionalMNISTBGM(model_params(config, smoke))
    model.restore(checkpoint_dir(seed, int(selection["selected_epoch"]), smoke))
    _, kwargs = bridge_settings(config, smoke)
    scores = np.empty((stop - start, 10)); diagnostics = []
    for local, global_idx in enumerate(range(start, stop)):
        for label in range(10):
            point_seed = seed * 10_000_019 + global_idx * 1009 + label * 97 + 17
            result = estimate_point_bridge(model, x[global_idx], label, seed=point_seed, **kwargs)
            scores[local, label] = result["log_px_std"]
            diagnostics.append({"index": global_idx, "label": label, **result["diagnostics"]})
        print(f"test {global_idx}: true={labels[global_idx]} pred={scores[local].argmax()}", flush=True)
    out = run_root(seed, smoke) / "shards"; out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"shard_{shard:04d}.npz", indices=np.arange(start, stop),
                        labels=labels[start:stop], log_density=scores)
    atomic_json(out / f"shard_{shard:04d}.diagnostics.json", diagnostics)


def merge(config: dict, seed: int, smoke: bool) -> None:
    expected = 4 if smoke else int(config["data"]["test_size"])
    chunks = []
    for path in sorted((run_root(seed, smoke) / "shards").glob("shard_*.npz")):
        with np.load(path) as raw:
            chunks.append(tuple(np.asarray(raw[k]) for k in ("indices", "labels", "log_density")))
    if not chunks: raise FileNotFoundError("no shards")
    indices = np.concatenate([x[0] for x in chunks]); order = np.argsort(indices)
    if not np.array_equal(indices[order], np.arange(expected)): raise RuntimeError("shards are incomplete or duplicated")
    labels = np.concatenate([x[1] for x in chunks])[order]
    scores = np.concatenate([x[2] for x in chunks])[order]
    pred = scores.argmax(axis=1); accuracy = float(np.mean(pred == labels))
    metrics = {"method": "BayesNDE", "seed": seed, "n_test": expected,
               "accuracy": accuracy, "correct": int(np.sum(pred == labels)),
               "confusion_matrix": confusion_matrix(labels, pred, labels=np.arange(10)).tolist(),
               "likelihood": ConditionalMNISTBGM.likelihood_name,
               "variant": "bernoulli_logit",
               "z_dim": 10, "test_used_for_selection": False}
    np.savez_compressed(run_root(seed, smoke) / "predictions.npz", labels=labels,
                        predictions=pred, log_density=scores)
    atomic_json(run_root(seed, smoke) / "metrics.json", metrics); print(json.dumps(metrics, indent=2))


def evaluate(config: dict, seed: int, smoke: bool, chunk_size: int = 25) -> None:
    total = 4 if smoke else int(config["data"]["test_size"])
    for index in range((total + chunk_size - 1) // chunk_size):
        evaluate_shard(config, seed, index, chunk_size, smoke)
    merge(config, seed, smoke)


def reproduce(config: dict, seed: int, smoke: bool, force: bool = False) -> None:
    prepare(config, force=force)
    train(config, seed, smoke)
    evaluate(config, seed, smoke)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "train", "evaluate", "reproduce"])
    parser.add_argument("--config", type=Path, default=APP / "config.yaml")
    parser.add_argument("--seed", type=int, default=0); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); config = load_config(args.config)
    if args.command == "prepare": prepare(config, args.force)
    elif args.command == "train": train(config, args.seed, args.smoke)
    elif args.command == "evaluate": evaluate(config, args.seed, args.smoke)
    else: reproduce(config, args.seed, args.smoke, args.force)


if __name__ == "__main__": main()
