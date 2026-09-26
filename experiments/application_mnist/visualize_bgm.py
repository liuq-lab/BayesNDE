#!/usr/bin/env python3
"""Reproducible conditional-generation samples from the selected MNIST BGM."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from experiments.application_mnist.run_bgm import APP, checkpoint_dir, load_config, model_params, run_root
from experiments.application_mnist.model import ConditionalMNISTBGM


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def save_grid(values: np.ndarray, path: Path, title: str, *, cmap: str = "gray",
              vmin: float | None = 0.0, vmax: float | None = 1.0) -> None:
    canvas = values.reshape(10, 10, 28, 28).transpose(0, 2, 1, 3).reshape(280, 280)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(canvas, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks(14 + 28 * np.arange(10))
    ax.set_yticklabels(np.arange(10))
    ax.set_ylabel("Conditioning label")
    ax.set_title(title)
    for boundary in 28 * np.arange(1, 10):
        ax.axhline(boundary - 0.5, color="white", linewidth=0.25)
        ax.axvline(boundary - 0.5, color="white", linewidth=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=APP / "config.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=20260916)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    root = run_root(args.seed, args.smoke)
    selection = json.loads((root / "selection_manifest.json").read_text())
    epoch = int(selection["selected_epoch"])
    model = ConditionalMNISTBGM(model_params(config, args.smoke))
    model.restore(checkpoint_dir(args.seed, epoch, args.smoke))

    rng = np.random.default_rng(args.sample_seed)
    z = rng.normal(size=(100, 10)).astype(np.float32)
    labels = np.repeat(np.arange(10, dtype=np.int64), 10)
    onehot = np.eye(10, dtype=np.float32)[labels]
    first, variance, _ = model._decode_generator(
        tf.convert_to_tensor(z), tf.convert_to_tensor(onehot), training=False)
    first = np.asarray(first.numpy(), dtype=np.float32)
    variance = np.asarray(variance.numpy(), dtype=np.float32)

    noise = rng.normal(size=first.shape).astype(np.float32)
    effective_logits = first / np.sqrt(1.0 + np.pi * variance / 8.0)
    display_mean = 1.0 / (1.0 + np.exp(-effective_logits))
    observation_sample = 1.0 / (1.0 + np.exp(-(first + noise * np.sqrt(variance))))

    out = root / "visualizations"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "conditional_samples.npz", z=z, labels=labels,
        decoder_first=first, variance=variance, display_mean=display_mean,
        observation_sample=observation_sample)
    save_grid(display_mean, out / "decoder_mean_grid.png",
              "Observation means — rows are labels 0-9")
    save_grid(np.clip(observation_sample, 0.0, 1.0), out / "observation_sample_grid.png",
              "Logistic-normal probability draws — rows are labels 0-9")
    variance_max = float(max(np.percentile(variance, 99.0), 1.0e-8))
    save_grid(variance, out / "variance_grid.png",
              "Decoder logit variance — common scale", cmap="magma",
              vmin=0.0, vmax=variance_max)
    atomic_json(out / "visualization_manifest.json", {
        "likelihood": ConditionalMNISTBGM.likelihood_name,
        "seed": args.seed,
        "sample_seed": args.sample_seed,
        "selected_epoch": epoch,
        "z_dim": 10,
        "samples_per_class": 10,
        "decoder_first_parameter": "logit_mean",
        "display_clip": [0.0, 1.0],
        "variance_color_vmax_p99": variance_max,
        "display_mean_range": [float(display_mean.min()), float(display_mean.max())],
        "variance_range": [float(variance.min()), float(variance.max())],
    })
    print(out)


if __name__ == "__main__":
    main()
