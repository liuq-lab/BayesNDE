#!/usr/bin/env python3
"""The shared conditional BGM-BS implementation for the p(x | y) data sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from bayesnde.diagnostics.generation import generation_rank_features, rank_candidates
import tensorflow as tf
import yaml

HERE = Path(__file__).resolve().parent
SRC_DENSITY = HERE.parent
REPO_ROOT = SRC_DENSITY.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.uci import core as base
from bayesnde.estimators.conditional import (
    ConditionalBGM,
    ProcessedSplit,
    estimate_point_bridge,
    one_hot,
)
from bayesnde.diagnostics.generation import _two_sample_metrics
from bayesnde.data.uci_conditional import (
    load_processed_split,
    split_indices,
    standardize_from_train,
)
from bayesgm.models.bgm.base import correlation_alignment_loss


SEED = 1024
DATA_SEED = 42
EGM_MAX_STEP = 22000
EGM_CHECKPOINT_EVERY = 1000
ITERATIVE_MAX_EPOCH = 2000
ITERATIVE_EPOCHS = base.SAVE_EPOCHS
TOP_K = 3

SCORE_KEYS = (
    "mmd_rbf",
    "sliced_wasserstein",
    "wasserstein_mean",
    "ks_stat_mean",
    "corr_matrix_frobenius_error",
    "sym_kl_mean",
    "lisi_normalized_mean",
)

DATASET_SPECS: dict[str, dict[str, Any]] = {}

EGM_VARIANTS: dict[str, dict[str, float]] = {
    "plain": {
        "alpha": 0.0,
        "variance_log_target_weight": 0.0,
        "decorrelation_weight": 0.0,
        "marginal_moment_weight": 0.0,
        "covariance_weight": 0.0,
        "marginal_mmd_weight": 0.0,
    },
    "variance": {
        "alpha": 0.1,
        "variance_log_target_weight": 0.01,
        "decorrelation_weight": 0.0,
        "marginal_moment_weight": 0.0,
        "covariance_weight": 0.0,
        "marginal_mmd_weight": 0.0,
    },
    "correlation": {
        "alpha": 0.0,
        "variance_log_target_weight": 0.0,
        "decorrelation_weight": 0.3,
        "marginal_moment_weight": 0.0,
        "covariance_weight": 0.0,
        "marginal_mmd_weight": 0.0,
    },
    "moments": {
        "alpha": 0.1,
        "variance_log_target_weight": 0.01,
        "decorrelation_weight": 0.3,
        "marginal_moment_weight": 1.0,
        "covariance_weight": 1.0,
        "marginal_mmd_weight": 0.0,
    },
    "mild_mmd": {
        "alpha": 0.0,
        "variance_log_target_weight": 0.0,
        "decorrelation_weight": 0.0,
        "marginal_moment_weight": 0.0,
        "covariance_weight": 0.0,
        "marginal_mmd_weight": 10.0,
    },
    "strong_reference": {
        "alpha": 0.0,
        "variance_log_target_weight": 0.0,
        "decorrelation_weight": 0.3,
        "marginal_moment_weight": 0.0,
        "covariance_weight": 0.0,
        "marginal_mmd_weight": 1000.0,
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


def canonical_dataset(value: str) -> str:
    name = value.strip()
    if name not in DATASET_SPECS:
        raise ValueError(f"dataset must be one of {sorted(DATASET_SPECS)}, got {value!r}")
    return name


def output_root(dataset: str) -> Path:
    return HERE / "outputs" / dataset / "conditional-bgmbs-v1"


def set_seed(offset: int = 0) -> None:
    value = SEED + int(offset)
    np.random.seed(value)
    random.seed(value)
    tf.keras.utils.set_random_seed(value)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def save_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_weights(model: tf.keras.Model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.weights.h5")
    model.save_weights(str(temporary))
    os.replace(temporary, path)


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def load_split(dataset: str) -> ProcessedSplit:
    raise NotImplementedError(
        "This module carries the protocol only; the wrapper that owns the data sets "
        "installs its own load_split."
    )


class ConditionalResidualBlock(tf.keras.layers.Layer):
    def __init__(self, width: int, name: str):
        super().__init__(name=name)
        self.dense1 = tf.keras.layers.Dense(int(width))
        self.act1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1.0e-5)
        self.dense2 = tf.keras.layers.Dense(int(width))
        self.act2 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1.0e-5)

    def call(self, x: tf.Tensor, labels: tf.Tensor, training: bool = True) -> tf.Tensor:
        del training
        residual = x
        x = tf.concat([x, labels], axis=-1)
        x = self.norm1(self.act1(self.dense1(x)))
        x = self.norm2(self.act2(self.dense2(x)))
        return (residual + x) / tf.cast(math.sqrt(2.0), x.dtype)


class ConditionalResidualGenerator(tf.keras.Model):
    def __init__(self, z_dim: int, label_dim: int, x_dim: int, width: int, blocks: int, variance_floor: float):
        super().__init__(name="conditional_residual_g_net")
        self.z_dim = int(z_dim)
        self.label_dim = int(label_dim)
        self.x_dim = int(x_dim)
        self.variance_floor = float(variance_floor)
        self.z_projection_layer = None
        self.z_feature_dim = self.z_dim
        self.input_norm = tf.keras.layers.BatchNormalization(name="input_standardization")
        self.projection = tf.keras.layers.Dense(int(width), name="input_projection")
        self.blocks = [ConditionalResidualBlock(width, f"generator_block_{idx}") for idx in range(int(blocks))]
        self.mean_layer = tf.keras.layers.Dense(self.x_dim, name="mean")
        self.var_layer = tf.keras.layers.Dense(self.x_dim, name="variance_logits")

    def call(self, inputs: Sequence[tf.Tensor], training: bool = True):
        z, labels = inputs
        first = tf.concat([tf.cast(z, tf.float32), tf.cast(labels, tf.float32)], axis=-1)
        x = self.projection(self.input_norm(first, training=training))
        for block in self.blocks:
            x = block(x, labels, training=training)
        output = tf.concat([x, labels], axis=-1)
        mean = self.mean_layer(output)
        variance = tf.nn.softplus(self.var_layer(output)) + tf.cast(self.variance_floor, mean.dtype)
        return mean, variance

    @staticmethod
    def reparameterize(mean: tf.Tensor, variance: tf.Tensor) -> tf.Tensor:
        return mean + tf.random.normal(tf.shape(mean), dtype=mean.dtype) * tf.sqrt(variance)


class ConditionalResidualEncoder(tf.keras.Model):
    def __init__(self, x_dim: int, label_dim: int, z_dim: int, width: int, blocks: int):
        super().__init__(name="conditional_residual_e_net")
        self.input_norm = tf.keras.layers.BatchNormalization(name="input_standardization")
        self.projection = tf.keras.layers.Dense(int(width), name="input_projection")
        self.blocks = [ConditionalResidualBlock(width, f"encoder_block_{idx}") for idx in range(int(blocks))]
        self.output_layer = tf.keras.layers.Dense(int(z_dim), name="latent")

    def call(self, x: tf.Tensor, labels: tf.Tensor, training: bool = True) -> tf.Tensor:
        first = tf.concat([tf.cast(x, tf.float32), tf.cast(labels, tf.float32)], axis=-1)
        h = self.projection(self.input_norm(first, training=training))
        for block in self.blocks:
            h = block(h, labels, training=training)
        return self.output_layer(tf.concat([h, labels], axis=-1))


class EveryLayerMLPEncoder(tf.keras.Model):
    def __init__(self, x_dim: int, label_dim: int, z_dim: int, hidden: Sequence[int]):
        super().__init__(name="conditional_every_layer_e_net")
        self.norm = tf.keras.layers.BatchNormalization()
        self.hidden = [tf.keras.layers.Dense(int(width)) for width in hidden]
        self.output_layer = tf.keras.layers.Dense(int(z_dim))

    def call(self, x: tf.Tensor, labels: tf.Tensor, training: bool = True) -> tf.Tensor:
        h = self.norm(tf.cast(x, tf.float32), training=training)
        for layer in self.hidden:
            h = tf.keras.layers.LeakyReLU(alpha=0.2)(layer(tf.concat([h, labels], axis=-1)))
        return self.output_layer(tf.concat([h, labels], axis=-1))


class TunedConditionalBGM(ConditionalBGM):
    def __init__(self, x_dim: int, y_dim: int, z_dim: int, params: Mapping[str, Any]):
        hidden = list(params["g_units"])
        encoder_hidden = list(params["e_units"])
        variance_floor = float(params["variance_floor"])
        super().__init__(x_dim, y_dim, z_dim, hidden, encoder_hidden, variance_floor, params)
        network_type = str(params.get("g_network_type", "mlp"))
        if network_type == "residual":
            if len(set(hidden)) != 1 or len(set(encoder_hidden)) != 1:
                raise ValueError("Residual template requires constant widths")
            self.g_net = ConditionalResidualGenerator(z_dim, y_dim, x_dim, hidden[0], len(hidden), variance_floor)
            self.e_net = ConditionalResidualEncoder(x_dim, y_dim, z_dim, encoder_hidden[0], len(encoder_hidden))
        elif network_type == "mlp":
            self.e_net = EveryLayerMLPEncoder(x_dim, y_dim, z_dim, encoder_hidden)
        else:
            raise ValueError(f"Unsupported network type: {network_type}")
        self.ckpt = tf.train.Checkpoint(
            g_net=self.g_net,
            e_net=self.e_net,
            dz_net=self.dz_net,
            dx_net=self.dx_net,
            g_pre_optimizer=self.g_pre_optimizer,
            d_pre_optimizer=self.d_pre_optimizer,
            g_optimizer=self.g_optimizer,
            posterior_optimizer=self.posterior_optimizer,
        )
        self.build()

    @tf.function
    def update_g_net(self, data_z: tf.Tensor, data_x: tf.Tensor, data_y: tf.Tensor):
        with tf.GradientTape() as tape:
            mean, variance, low_rank = self._decode_generator(data_z, data_y, training=True)
            decoder_loss = tf.reduce_mean(self._decoder_nll_from_parts(data_x, mean, variance, low_rank))
            regularization = tf.cast(self.params.get("variance_log_target_weight", 0.0), decoder_loss.dtype) * self._variance_log_target_loss(variance)
            moment_weight = float(self.params.get("iterative_prior_moment_weight", 0.0))
            covariance_weight = float(self.params.get("iterative_prior_covariance_weight", 0.0))
            mmd_weight = float(self.params.get("iterative_prior_mmd_weight", 0.0))
            if moment_weight > 0.0 or covariance_weight > 0.0 or mmd_weight > 0.0:
                prior_z = tf.random.normal(tf.shape(data_z), dtype=data_z.dtype)
                prior_mean, prior_variance, prior_low_rank = self._decode_generator(prior_z, data_y, training=True)
                prior_x = self._sample_decoder(prior_mean, prior_variance, prior_low_rank)
                real_mean = tf.reduce_mean(data_x, axis=0)
                generated_mean = tf.reduce_mean(prior_x, axis=0)
                real_var = tf.math.reduce_variance(data_x, axis=0)
                generated_var = tf.math.reduce_variance(prior_x, axis=0)
                if moment_weight > 0.0:
                    value = tf.reduce_mean(tf.square(real_mean - generated_mean)) + tf.reduce_mean(tf.square(real_var - generated_var))
                    regularization += tf.cast(moment_weight, decoder_loss.dtype) * value
                if covariance_weight > 0.0:
                    denominator = tf.cast(tf.maximum(tf.shape(data_x)[0] - 1, 1), data_x.dtype)
                    real_centered = data_x - real_mean
                    generated_centered = prior_x - generated_mean
                    real_cov = tf.matmul(real_centered, real_centered, transpose_a=True) / denominator
                    generated_cov = tf.matmul(generated_centered, generated_centered, transpose_a=True) / denominator
                    regularization += tf.cast(covariance_weight, decoder_loss.dtype) * tf.reduce_mean(tf.square(real_cov - generated_cov))
                if mmd_weight > 0.0:
                    value = tf.constant(0.0, dtype=data_x.dtype)
                    real_pairwise = data_x[None, :, :] - data_x[:, None, :]
                    generated_pairwise = prior_x[None, :, :] - prior_x[:, None, :]
                    cross_pairwise = data_x[:, None, :] - prior_x[None, :, :]
                    scales = tf.constant(self.params.get("mmd_scales", [0.05, 0.1, 0.2, 0.5, 1.0]), dtype=data_x.dtype)
                    for scale in tf.unstack(scales):
                        denominator = 2.0 * tf.square(tf.maximum(scale, 1.0e-6))
                        value += tf.reduce_mean(tf.exp(-tf.square(real_pairwise) / denominator) + tf.exp(-tf.square(generated_pairwise) / denominator) - 2.0 * tf.exp(-tf.square(cross_pairwise) / denominator))
                    value /= tf.cast(tf.size(scales), data_x.dtype)
                    regularization += tf.cast(mmd_weight, decoder_loss.dtype) * value
            loss_x = decoder_loss + regularization
        variables = self._decoder_trainable_variables()
        gradients = tape.gradient(loss_x, variables)
        self.g_optimizer.apply_gradients([(grad, var) for grad, var in zip(gradients, variables) if grad is not None])
        return loss_x

    @tf.function
    def update_iterative_empirical_corr(self, data_x: tf.Tensor, data_y: tf.Tensor) -> tf.Tensor:
        weight = tf.cast(self.params.get("iterative_empirical_corr_weight", 0.0), data_x.dtype)
        with tf.GradientTape() as tape:
            prior_z = tf.random.normal([tf.shape(data_x)[0], self.z_dim], dtype=data_x.dtype)
            mean, variance, low_rank = self._decode_generator(prior_z, data_y, training=True)
            generated = self._sample_decoder(mean, variance, low_rank)
            corr_loss = correlation_alignment_loss(generated, data_x)
            weighted = weight * corr_loss
        variables = self._decoder_trainable_variables()
        gradients = tape.gradient(weighted, variables)
        self.g_optimizer.apply_gradients([(grad, var) for grad, var in zip(gradients, variables) if grad is not None])
        return corr_loss


def model_params(dataset: str, z_dim: int, out_dir: Path) -> dict[str, Any]:
    spec = DATASET_SPECS[dataset]
    return {
        "dataset": f"conditional_{dataset}",
        "output_dir": str(out_dir),
        "x_dim": int(spec["x_dim"]),
        "y_dim": 2,
        "z_dim": int(z_dim),
        "g_network_type": str(spec["network_type"]),
        "e_network_type": str(spec["network_type"]),
        "g_units": [256] * 5,
        "e_units": [256] * 5,
        "dz_units": [256, 256, 128, 64],
        "dx_units": [256, 256, 128, 64],
        "label_conditioning": "one_hot",
        "decoder_conditioning_mode": "every_layer",
        "encoder_conditioning_mode": "every_layer",
        "variance_floor": 1.0e-6,
        "lr": 1.0e-3,
        "lr_theta": 0.005,
        "lr_z": 0.005,
        "optimizer": "adam",
        "weight_decay": 0.0,
        "use_bnn": False,
        "kl_weight": 5.0e-5,
        "g_d_freq": 1,
        "alpha": 0.0,
        "gamma": 0.0,
        "cycle_weight": 5.0,
        "x_cycle_weight": 3.0,
        "z_cycle_weight": 1.0,
        "g_adv_weight": 1.0,
        "e_adv_weight": 1.0,
        "marginal_moment_weight": 0.0,
        "covariance_weight": 0.0,
        "marginal_mmd_weight": 0.0,
        "decorrelation_weight": 0.0,
        "decorrelation_target": "train",
        "mmd_scales": [0.05, 0.1, 0.2, 0.5, 1.0],
        "x_cycle_use_mean": False,
        "x_adv_use_mean": False,
        "z_cycle_use_mean": False,
        "variance_target": 0.01,
        "variance_target_weight": 0.0,
        "variance_log_target": 0.01,
        "variance_log_target_weight": 0.0,
        "variance_log_eps": 1.0e-8,
        "iterative_empirical_corr_weight": 0.0,
        "iterative_prior_moment_weight": 0.0,
        "iterative_prior_covariance_weight": 0.0,
        "iterative_prior_mmd_weight": 0.0,
    }


def base_config(dataset: str, z_dim: int, epsilon: float, out_dir: Path) -> dict[str, Any]:
    spec = DATASET_SPECS[dataset]
    return {
        "seed": SEED,
        "dataset": dataset,
        "model": model_params(dataset, z_dim, out_dir),
        "training": {
            "batch_size": 256,
            "egm_n_iter": EGM_MAX_STEP,
            "egm_batches_per_eval": EGM_CHECKPOINT_EVERY,
            "epochs": ITERATIVE_MAX_EPOCH,
            "save_epochs": list(ITERATIVE_EPOCHS),
        },
        "density": {
            "K": 5,
            "S": 20000,
            "nu": 3.0,
            "epsilon": float(epsilon),
            "fit_fraction": 0.5,
            "bridge_tol": 1.0e-5,
            "bridge_max_iter": 1000,
            "eval_batch_size": 1024,
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
            "proposal_covariance_floor": 0.001,
        },
        "protocol": {
            "conditional_target": "p(X|Y)",
            "y_injection": "one-hot Y at every generator and encoder hidden layer; generator output also conditioned",
            "architecture_locked": f"{spec['network_type']} 5x256 generator/encoder",
            "egm_learning_rate_locked": 1.0e-3,
            "iterative_learning_rates_locked": {"lr_theta": 0.005, "lr_z": 0.005},
            "source_network": str(spec["source_network"]),
            "selection_split": "validation",
            "selection_metric": "conditional generation-quality rank sum",
            "test_points_used_for_selection": 0,
            "proposal_q": "product of coordinate-wise Student-t densities",
        },
    }


def setting_for_z(dataset: str, z_dim: int) -> tuple[int, float]:
    for candidate_z, epsilon in DATASET_SPECS[dataset]["z_epsilon"]:
        if int(candidate_z) == int(z_dim):
            return int(candidate_z), float(epsilon)
    raise ValueError(f"Invalid z_dim={z_dim} for {dataset}")


def egm_jobs(dataset: str) -> list[tuple[int, float, str]]:
    return [(int(z), float(epsilon), variant) for z, epsilon in DATASET_SPECS[dataset]["z_epsilon"] for variant in EGM_VARIANTS]


def iterative_jobs(dataset: str) -> list[tuple[int, float, str]]:
    return [(int(z), float(epsilon), variant) for z, epsilon in DATASET_SPECS[dataset]["z_epsilon"] for variant in ITERATIVE_VARIANTS]


def egm_root(dataset: str, z_dim: int, variant: str) -> Path:
    return output_root(dataset) / "egm" / f"zdim_{z_dim}" / variant


def iterative_root(dataset: str, z_dim: int, variant: str) -> Path:
    return output_root(dataset) / "iterative" / f"zdim_{z_dim}" / variant


def build_model(config: Mapping[str, Any]) -> TunedConditionalBGM:
    params = dict(config["model"])
    return TunedConditionalBGM(int(params["x_dim"]), int(params["y_dim"]), int(params["z_dim"]), params)


def conditional_generation_metrics(
    model: TunedConditionalBGM,
    split: ProcessedSplit,
    *,
    seed: int,
) -> dict[str, Any]:
    reference = np.asarray(split.val_x, dtype=np.float32)
    labels = np.asarray(split.val_y, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    generated = np.empty_like(reference)
    for start in range(0, len(reference), 1024):
        end = min(start + 1024, len(reference))
        z = rng.normal(size=(end - start, model.z_dim)).astype(np.float32)
        y = one_hot(labels[start:end], model.y_dim)
        mean, variance, _ = model._decode_generator(z, y, training=False)
        noise = rng.normal(size=(end - start, model.x_dim)).astype(np.float32)
        generated[start:end] = mean.numpy() + noise * np.sqrt(variance.numpy())
    rows: list[dict[str, Any]] = []
    aggregate = {key: 0.0 for key in SCORE_KEYS}
    for label in sorted(int(value) for value in np.unique(labels)):
        mask = labels == label
        label_reference = reference[mask]
        label_generated = generated[mask]
        if np.isfinite(label_generated).all():
            metrics = _two_sample_metrics(label_reference, label_generated)
            centered_reference = label_reference - np.mean(label_reference, axis=0, keepdims=True)
            centered_generated = label_generated - np.mean(label_generated, axis=0, keepdims=True)
            reference_scale = np.std(centered_reference, axis=0, keepdims=True)
            generated_scale = np.std(centered_generated, axis=0, keepdims=True)
            normalized_reference = np.divide(
                centered_reference,
                reference_scale,
                out=np.zeros_like(centered_reference),
                where=reference_scale > 1.0e-12,
            )
            normalized_generated = np.divide(
                centered_generated,
                generated_scale,
                out=np.zeros_like(centered_generated),
                where=generated_scale > 1.0e-12,
            )
            reference_corr = normalized_reference.T @ normalized_reference / max(len(label_reference), 1)
            generated_corr = normalized_generated.T @ normalized_generated / max(len(label_generated), 1)
            metrics["corr_matrix_frobenius_error"] = float(np.linalg.norm(reference_corr - generated_corr, ord="fro"))
        else:
            metrics = {key: float("inf") for key in SCORE_KEYS}
        row = {"label": label, "points": int(np.sum(mask)), **{key: float(metrics[key]) for key in SCORE_KEYS}}
        rows.append(row)
        weight = float(np.mean(mask))
        for key in SCORE_KEYS:
            aggregate[key] += weight * float(metrics[key])
    return {
        **aggregate,
        "points": int(len(reference)),
        "labels": rows,
        "conditional_weighting": "validation label-frequency weighted mean",
    }


def save_dataset_fingerprint(dataset: str, split: ProcessedSplit) -> None:
    root = output_root(dataset)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "dataset_fingerprint.json"
    if not path.exists():
        atomic_json(
            path,
            {
                **split.metadata,
                "train_shape": list(split.train_x.shape),
                "validation_shape": list(split.val_x.shape),
                "test_shape": list(split.test_x.shape),
                "train_sha256": sha256_array(split.train_x),
                "validation_sha256": sha256_array(split.val_x),
                "test_sha256": sha256_array(split.test_x),
                "train_y_sha256": sha256_array(split.train_y),
                "validation_y_sha256": sha256_array(split.val_y),
                "test_y_sha256": sha256_array(split.test_y),
            },
        )


def run_egm(dataset: str, task: int) -> dict[str, Any]:
    jobs = egm_jobs(dataset)
    if not 0 <= task < len(jobs):
        raise ValueError(f"EGM task must be in [0,{len(jobs)-1}]")
    z_dim, epsilon, variant = jobs[task]
    split = load_split(dataset)
    save_dataset_fingerprint(dataset, split)
    root = egm_root(dataset, z_dim, variant)
    config = base_config(dataset, z_dim, epsilon, root)
    config["model"].update(EGM_VARIANTS[variant])
    config["egm_loss_variant"] = {"name": variant, **EGM_VARIANTS[variant]}
    save_yaml(root / "resolved_config.yaml", config)
    set_seed(task * 1000)
    model = build_model(config)
    start_step = 0
    existing = sorted((root / "checkpoint").glob("egm_step_*_generator.weights.h5"))
    if existing:
        match = re.search(r"egm_step_(\d+)_generator", existing[-1].name)
        if match:
            start_step = int(match.group(1))
            model.g_net.load_weights(str(existing[-1]))
            model.e_net.load_weights(str(Path(str(existing[-1]).replace("_generator.weights.h5", "_encoder.weights.h5"))))
    rows_path = root / "generation_curve.json"
    rows = json.loads(rows_path.read_text())["rows"] if rows_path.exists() else []
    train_x = tf.convert_to_tensor(split.train_x, tf.float32)
    train_y = tf.convert_to_tensor(one_hot(split.train_y, split.y_dim), tf.float32)
    rng = np.random.default_rng(SEED + task * 1000)
    batch_size = int(config["training"]["batch_size"])
    for step in range(start_step + 1, EGM_MAX_STEP + 1):
        for _ in range(int(model.params.get("g_d_freq", 1))):
            indices = rng.integers(0, len(split.train_x), size=batch_size)
            batch_x = tf.gather(train_x, indices)
            batch_y = tf.gather(train_y, indices)
            batch_z = tf.random.normal((batch_size, z_dim), dtype=tf.float32)
            model.train_disc_step(batch_z, batch_x, batch_y)
        indices = rng.integers(0, len(split.train_x), size=batch_size)
        batch_x = tf.gather(train_x, indices)
        batch_y = tf.gather(train_y, indices)
        batch_z = tf.random.normal((batch_size, z_dim), dtype=tf.float32)
        losses = model.train_gen_step(batch_z, batch_x, batch_y)
        if step % EGM_CHECKPOINT_EVERY != 0 and step != EGM_MAX_STEP:
            continue
        prefix = root / "checkpoint" / f"egm_step_{step:05d}"
        generator_path = Path(str(prefix) + "_generator.weights.h5")
        encoder_path = Path(str(prefix) + "_encoder.weights.h5")
        atomic_weights(model.g_net, generator_path)
        atomic_weights(model.e_net, encoder_path)
        metrics = conditional_generation_metrics(model, split, seed=SEED + 61000 + z_dim)
        row = {
            "z_dim": z_dim,
            "epsilon": epsilon,
            "variant": variant,
            "step": step,
            "g_e_loss": float(losses[-1]),
            **metrics,
            "generator_weights": str(generator_path),
            "encoder_weights": str(encoder_path),
            "config_path": str(root / "resolved_config.yaml"),
        }
        rows = [old for old in rows if int(old["step"]) != step] + [row]
        rows.sort(key=lambda value: int(value["step"]))
        atomic_json(rows_path, {"rows": rows})
    payload = {"task": task, "z_dim": z_dim, "epsilon": epsilon, "variant": variant, "rows": len(rows), "complete": True}
    atomic_json(root / "complete.json", payload)
    return payload


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    features = []
    for row in rows:
        feature = generation_rank_features(row, int(row["z_dim"]))
        if feature is None:
            missing = [k for k in ("mmd_rbf", "sym_kl_mean", "sliced_wasserstein",
                                   "wasserstein_mean", "ks_stat_mean",
                                   "corr_matrix_frobenius_error", "lisi_normalized_mean")
                       if row.get(k) is None]
            raise RuntimeError(
                f"Candidate {row.get('variant')}/{row.get('step', row.get('epoch'))} cannot be "
                f"scored by the shared selection rule; missing diagnostics: {missing}. "
                f"Recompute the generation diagnostics for this run."
            )
        features.append(feature)
    scores = rank_candidates(features)
    scored = [{**row, "generation_rank_score": float(score)} for row, score in zip(rows, scores)]
    return sorted(scored, key=lambda row: (row["generation_rank_score"],
                                           int(row.get("step", row.get("epoch", 0)))))


def select_egm(dataset: str) -> dict[str, Any]:
    frozen = output_root(dataset) / "frozen_egm_selection.json"
    if frozen.exists():
        return json.loads(frozen.read_text())
    selected: dict[str, Any] = {}
    all_ranked: dict[str, Any] = {}
    for z_dim, epsilon in DATASET_SPECS[dataset]["z_epsilon"]:
        rows: list[dict[str, Any]] = []
        for variant in EGM_VARIANTS:
            root = egm_root(dataset, int(z_dim), variant)
            if not (root / "complete.json").exists():
                raise FileNotFoundError(f"Incomplete EGM run: {root}")
            rows.extend(json.loads((root / "generation_curve.json").read_text())["rows"])
        ranked = rank_rows(rows)
        selected[str(z_dim)] = ranked[0]
        all_ranked[str(z_dim)] = ranked
    payload = {
        "dataset": dataset,
        "criterion": "minimum validation conditional-generation rank sum",
        "score_keys": list(SCORE_KEYS),
        "selected_by_z_dim": selected,
        "ranked_by_z_dim": all_ranked,
        "test_points_used_for_selection": 0,
        "frozen_before_iterative_and_test": True,
    }
    atomic_json(frozen, payload)
    return payload


def materialize_iterative_config(dataset: str, z_dim: int, epsilon: float, variant: str, selected: Mapping[str, Any]) -> dict[str, Any]:
    root = iterative_root(dataset, z_dim, variant)
    config = yaml.safe_load(Path(str(selected["config_path"])).read_text())
    config = deepcopy(config)
    config["model"]["output_dir"] = str(root)
    config["model"].update(ITERATIVE_VARIANTS[variant])
    config["density"]["epsilon"] = float(epsilon)
    config["iterative_loss_variant"] = {"name": variant, **ITERATIVE_VARIANTS[variant]}
    config["selected_egm"] = dict(selected)
    return config


def encode_batches(model: TunedConditionalBGM, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    values: list[np.ndarray] = []
    for start in range(0, len(x), 1024):
        end = min(start + 1024, len(x))
        values.append(model.encode(x[start:end], one_hot(y[start:end], model.y_dim), training=False).numpy())
    return np.concatenate(values).astype(np.float32)


def run_iterative(dataset: str, task: int) -> dict[str, Any]:
    jobs = iterative_jobs(dataset)
    if not 0 <= task < len(jobs):
        raise ValueError(f"iterative task must be in [0,{len(jobs)-1}]")
    z_dim, epsilon, variant = jobs[task]
    selected = select_egm(dataset)["selected_by_z_dim"][str(z_dim)]
    split = load_split(dataset)
    root = iterative_root(dataset, z_dim, variant)
    config = materialize_iterative_config(dataset, z_dim, epsilon, variant, selected)
    save_yaml(root / "resolved_config.yaml", config)
    set_seed(50000 + task * 1000)
    model = build_model(config)
    model.g_net.load_weights(str(selected["generator_weights"]))
    model.e_net.load_weights(str(selected["encoder_weights"]))
    model.g_net.trainable = True
    model.e_net.trainable = False
    model.dz_net.trainable = False
    model.dx_net.trainable = False
    model.data_z = tf.Variable(encode_batches(model, split.train_x, split.train_y), name="Latent Variable", trainable=True)
    resume = tf.train.Checkpoint(g_net=model.g_net, g_optimizer=model.g_optimizer, posterior_optimizer=model.posterior_optimizer, data_z=model.data_z)
    manager = tf.train.CheckpointManager(resume, str(root / "checkpoint/resume"), max_to_keep=1)
    start_epoch = 0
    if manager.latest_checkpoint:
        resume.restore(manager.latest_checkpoint).expect_partial()
        match = re.search(r"-(\d+)$", manager.latest_checkpoint)
        start_epoch = int(match.group(1)) if match else 0
    rows_path = root / "generation_curve.json"
    rows = json.loads(rows_path.read_text())["rows"] if rows_path.exists() else []
    train_x = tf.convert_to_tensor(split.train_x, tf.float32)
    train_y = tf.convert_to_tensor(one_hot(split.train_y, split.y_dim), tf.float32)
    batch_size = int(config["training"]["batch_size"])
    save_epochs = set(int(value) for value in ITERATIVE_EPOCHS)
    for epoch in range(start_epoch + 1, ITERATIVE_MAX_EPOCH + 1):
        order = np.random.default_rng(SEED + 700000 + task * 10000 + epoch).permutation(len(split.train_x))
        losses_x: list[float] = []
        losses_z: list[float] = []
        losses_corr: list[float] = []
        for start in range(0, len(order) - batch_size + 1, batch_size):
            indices = order[start : start + batch_size]
            batch_z = tf.Variable(tf.gather(model.data_z, indices), trainable=True)
            batch_x = tf.gather(train_x, indices)
            batch_y = tf.gather(train_y, indices)
            loss_x = model.update_g_net(batch_z, batch_x, batch_y)
            if float(model.params.get("iterative_empirical_corr_weight", 0.0)) > 0.0:
                losses_corr.append(float(model.update_iterative_empirical_corr(batch_x, batch_y)))
            loss_z = model.update_latent_variable_sgd(batch_z, batch_x, batch_y)
            model.data_z.scatter_nd_update(tf.expand_dims(indices, axis=1), batch_z)
            losses_x.append(float(loss_x))
            losses_z.append(float(loss_z))
        if epoch not in save_epochs and epoch != ITERATIVE_MAX_EPOCH:
            continue
        generator_path = root / "checkpoint" / f"weights_at_{epoch:04d}_generator.weights.h5"
        atomic_weights(model.g_net, generator_path)
        metrics = conditional_generation_metrics(model, split, seed=SEED + 61000 + z_dim)
        row = {
            "z_dim": z_dim,
            "epsilon": epsilon,
            "variant": variant,
            "epoch": epoch,
            "loss_x": float(np.mean(losses_x)),
            "loss_z": float(np.mean(losses_z)),
            "empirical_corr_loss": float(np.mean(losses_corr)) if losses_corr else 0.0,
            **metrics,
            "generator_weights": str(generator_path),
            "encoder_weights": str(selected["encoder_weights"]),
            "egm_generator_weights": str(selected["generator_weights"]),
            "egm_variant": str(selected["variant"]),
            "egm_step": int(selected["step"]),
            "config_path": str(root / "resolved_config.yaml"),
        }
        rows = [old for old in rows if int(old["epoch"]) != epoch] + [row]
        rows.sort(key=lambda value: int(value["epoch"]))
        atomic_json(rows_path, {"rows": rows})
        manager.save(checkpoint_number=epoch)
    payload = {"task": task, "z_dim": z_dim, "epsilon": epsilon, "variant": variant, "rows": len(rows), "complete": True}
    atomic_json(root / "complete.json", payload)
    return payload


def select_iterative(dataset: str) -> dict[str, Any]:
    frozen = output_root(dataset) / "frozen_iterative_top3.json"
    if frozen.exists():
        return json.loads(frozen.read_text())
    selected: dict[str, Any] = {}
    ranked_all: dict[str, Any] = {}
    for z_dim, epsilon in DATASET_SPECS[dataset]["z_epsilon"]:
        rows: list[dict[str, Any]] = []
        for variant in ITERATIVE_VARIANTS:
            root = iterative_root(dataset, int(z_dim), variant)
            if not (root / "complete.json").exists():
                raise FileNotFoundError(f"Incomplete iterative run: {root}")
            rows.extend(json.loads((root / "generation_curve.json").read_text())["rows"])
        ranked = rank_rows(rows)
        top = []
        for rank, row in enumerate(ranked[:TOP_K], start=1):
            top.append({**row, "rank_within_setting": rank})
        selected[str(z_dim)] = top
        ranked_all[str(z_dim)] = ranked
    payload = {
        "dataset": dataset,
        "criterion": "minimum validation conditional-generation rank sum",
        "score_keys": list(SCORE_KEYS),
        "selected_top3_by_z_dim": selected,
        "ranked_by_z_dim": ranked_all,
        "test_points_used_for_selection": 0,
        "frozen_before_test": True,
    }
    atomic_json(frozen, payload)
    return payload


def candidate_slug(candidate: Mapping[str, Any]) -> str:
    return f"z{int(candidate['z_dim'])}_rank{int(candidate['rank_within_setting'])}_{candidate['variant']}_epoch{int(candidate['epoch']):04d}"


def test_task_count(dataset: str) -> int:
    return len(DATASET_SPECS[dataset]["z_epsilon"]) * TOP_K * int(DATASET_SPECS[dataset]["test_shards"])


def decode_test_task(dataset: str, task: int) -> tuple[int, int, int]:
    shards = int(DATASET_SPECS[dataset]["test_shards"])
    if not 0 <= task < test_task_count(dataset):
        raise ValueError(f"test task must be in [0,{test_task_count(dataset)-1}]")
    shard = task % shards
    remainder = task // shards
    rank_index = remainder % TOP_K
    setting_index = remainder // TOP_K
    return setting_index, rank_index, shard


def test_subset(split: ProcessedSplit, dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_points = DATASET_SPECS[dataset]["test_points"]
    count = len(split.test_x) if max_points is None else min(int(max_points), len(split.test_x))
    return split.test_x[:count], split.test_y[:count], np.arange(count, dtype=np.int64)


def run_test_shard(dataset: str, task: int) -> dict[str, Any]:
    setting_index, rank_index, shard = decode_test_task(dataset, task)
    z_dim, epsilon = DATASET_SPECS[dataset]["z_epsilon"][setting_index]
    candidate = select_iterative(dataset)["selected_top3_by_z_dim"][str(z_dim)][rank_index]
    split = load_split(dataset)
    test_x, test_y, original_positions = test_subset(split, dataset)
    subset_positions = np.arange(len(test_x), dtype=np.int64)
    shard_subset_positions = np.array_split(subset_positions, int(DATASET_SPECS[dataset]["test_shards"]))[shard]
    shard_positions = original_positions[shard_subset_positions]
    config = yaml.safe_load(Path(str(candidate["config_path"])).read_text())
    config["density"]["epsilon"] = float(epsilon)
    set_seed(900000 + task)
    model = build_model(config)
    model.e_net.load_weights(str(candidate["encoder_weights"]))
    model.g_net.load_weights(str(candidate["generator_weights"]))
    root = output_root(dataset) / "test" / candidate_slug(candidate) / f"epsilon_{str(epsilon).replace('.', 'p')}" / f"shard_{shard:03d}"
    root.mkdir(parents=True, exist_ok=True)
    save_yaml(root / "resolved_config.yaml", config)
    log_std = np.empty(len(shard_subset_positions), dtype=np.float64)
    diagnostics: list[dict[str, Any]] = []
    density = config["density"]
    hmc = config["hmc_settings"]
    for local, (subset_position, position) in enumerate(zip(shard_subset_positions, shard_positions)):
        result = estimate_point_bridge(
            model,
            test_x[subset_position],
            int(test_y[subset_position]),
            K=int(density["K"]),
            S=int(density["S"]),
            nu=float(density["nu"]),
            epsilon=float(epsilon),
            hmc_settings=hmc,
            eval_batch_size=int(density["eval_batch_size"]),
            fit_fraction=float(density["fit_fraction"]),
            bridge_tol=float(density["bridge_tol"]),
            bridge_max_iter=int(density["bridge_max_iter"]),
            covariance_floor=float(hmc["proposal_covariance_floor"]),
            proposal_covariance_mode="fixed_isotropic",
            proposal_scale=0.005,
            proposal_center="fitted_gmm",
            proposal_scoring="product_univariate_t",
            seed=SEED + 900000 + int(z_dim) * 10000 + rank_index * 1000 + int(position),
        )
        log_std[local] = float(result["log_px_std"])
        diagnostics.append(result["diagnostics"])
    log_obs = log_std + float(split.jacobian_correction)
    np.savez_compressed(
        root / "test_log_likelihood.npz",
        log_px_std=log_std,
        log_px_obs=log_obs,
        test_positions=shard_positions,
        labels=test_y[shard_subset_positions],
    )
    payload = {
        "task": task,
        "dataset": dataset,
        "candidate": candidate_slug(candidate),
        "z_dim": int(z_dim),
        "epsilon": float(epsilon),
        "rank": rank_index + 1,
        "shard": shard,
        "points": int(len(shard_positions)),
        "mean_log_likelihood_obs": float(np.mean(log_obs)),
        "proposal_scoring": "product_univariate_t",
        "diagnostics": diagnostics,
    }
    atomic_json(root / "complete.json", payload)
    return {key: value for key, value in payload.items() if key != "diagnostics"}


def merge_test(dataset: str) -> dict[str, Any]:
    selection = select_iterative(dataset)
    rows: list[dict[str, Any]] = []
    shards = int(DATASET_SPECS[dataset]["test_shards"])
    reference_positions: np.ndarray | None = None
    for z_dim, epsilon in DATASET_SPECS[dataset]["z_epsilon"]:
        for candidate in selection["selected_top3_by_z_dim"][str(z_dim)]:
            root = output_root(dataset) / "test" / candidate_slug(candidate) / f"epsilon_{str(float(epsilon)).replace('.', 'p')}"
            log_parts: list[np.ndarray] = []
            position_parts: list[np.ndarray] = []
            label_parts: list[np.ndarray] = []
            for shard in range(shards):
                part = root / f"shard_{shard:03d}"
                if not (part / "complete.json").exists():
                    raise FileNotFoundError(f"Incomplete test shard: {part}")
                with np.load(part / "test_log_likelihood.npz") as values:
                    log_parts.append(np.asarray(values["log_px_obs"], dtype=np.float64))
                    position_parts.append(np.asarray(values["test_positions"], dtype=np.int64))
                    label_parts.append(np.asarray(values["labels"], dtype=np.int64))
            log_px = np.concatenate(log_parts)
            positions = np.concatenate(position_parts)
            labels = np.concatenate(label_parts)
            if reference_positions is None:
                reference_positions = positions
            elif not np.array_equal(positions, reference_positions):
                raise RuntimeError("Test positions differ between settings/candidates")
            np.savez_compressed(root / "full_test_log_likelihood.npz", log_px_obs=log_px, test_positions=positions, labels=labels)
            mean = float(np.mean(log_px))
            row = {
                "dataset": dataset,
                "candidate": candidate_slug(candidate),
                "z_dim": int(z_dim),
                "epsilon": float(epsilon),
                "rank": int(candidate["rank_within_setting"]),
                "egm_variant": candidate["egm_variant"],
                "egm_step": int(candidate["egm_step"]),
                "iterative_variant": candidate["variant"],
                "iterative_epoch": int(candidate["epoch"]),
                "points": int(len(log_px)),
                "mean_test_log_likelihood_obs": mean,
                "standard_deviation": float(np.std(log_px)),
                "two_standard_errors": float(2.0 * np.std(log_px) / math.sqrt(len(log_px))),
            }
            atomic_json(root / "metrics.json", row)
            rows.append(row)
    payload = {
        "dataset": dataset,
        "target": "conditional p(X|Y)",
        "proposal_scoring": "product_univariate_t",
        "identical_test_positions_verified": True,
        "test_points_used_for_selection": 0,
        "rows": rows,
    }
    atomic_json(output_root(dataset) / "final_summary.json", payload)
    return payload


def validate(dataset: str) -> dict[str, Any]:
    split = load_split(dataset)
    save_dataset_fingerprint(dataset, split)
    checks: list[dict[str, Any]] = []
    for z_dim, epsilon in DATASET_SPECS[dataset]["z_epsilon"]:
        config = base_config(dataset, int(z_dim), float(epsilon), output_root(dataset) / "validation_model")
        model = build_model(config)
        y = tf.convert_to_tensor(one_hot(np.asarray([0]), split.y_dim))
        model._decode_generator(tf.zeros((1, int(z_dim))), y, training=False)
        model.encode(tf.zeros((1, split.x_dim)), y, training=False)
        checks.append({
            "z_dim": int(z_dim),
            "epsilon": float(epsilon),
            "network_type": config["model"]["g_network_type"],
            "g_units": config["model"]["g_units"],
            "e_units": config["model"]["e_units"],
            "egm_lr": config["model"]["lr"],
            "iterative_lr_theta": config["model"]["lr_theta"],
            "iterative_lr_z": config["model"]["lr_z"],
            "decoder_conditioning_mode": config["model"]["decoder_conditioning_mode"],
            "encoder_conditioning_mode": config["model"]["encoder_conditioning_mode"],
        })
    payload = {
        "dataset": dataset,
        "x_dim": split.x_dim,
        "y_dim": split.y_dim,
        "train_shape": list(split.train_x.shape),
        "validation_shape": list(split.val_x.shape),
        "test_shape": list(split.test_x.shape),
        "egm_jobs": len(egm_jobs(dataset)),
        "iterative_jobs": len(iterative_jobs(dataset)),
        "test_jobs": test_task_count(dataset),
        "checks": checks,
    }
    atomic_json(output_root(dataset) / "validation.json", payload)
    return payload


def smoke_test(dataset: str) -> dict[str, Any]:
    split = load_split(dataset)
    z_dim, epsilon = DATASET_SPECS[dataset]["z_epsilon"][0]
    config = base_config(dataset, int(z_dim), float(epsilon), output_root(dataset) / "smoke_model")
    config["model"].update(EGM_VARIANTS["moments"])
    config["model"].update(ITERATIVE_VARIANTS["moments"])
    set_seed(990000)
    model = build_model(config)
    count = min(32, len(split.train_x))
    batch_x = tf.convert_to_tensor(split.train_x[:count], tf.float32)
    batch_y = tf.convert_to_tensor(one_hot(split.train_y[:count], split.y_dim), tf.float32)
    batch_z = tf.random.normal((count, int(z_dim)), dtype=tf.float32)
    disc = model.train_disc_step(batch_z, batch_x, batch_y)
    egm = model.train_gen_step(batch_z, batch_x, batch_y)
    latent = tf.Variable(model.encode(batch_x, batch_y, training=False), trainable=True)
    iterative = model.update_g_net(latent, batch_x, batch_y)
    posterior = model.update_latent_variable_sgd(latent, batch_x, batch_y)
    values = [float(value) for value in (*disc, *egm, iterative, posterior)]
    if not np.isfinite(values).all():
        raise FloatingPointError(f"Non-finite smoke-test values: {values}")
    return {"dataset": dataset, "z_dim": int(z_dim), "batch": count, "finite": True, "values": values}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "smoke", "egm", "select-egm", "iterative", "select-iterative", "test-shard", "merge-test"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", type=int)
    args = parser.parse_args()
    dataset = canonical_dataset(args.dataset)
    if args.command in {"egm", "iterative", "test-shard"} and args.task is None:
        parser.error(f"{args.command} requires --task")
    if args.command == "validate":
        result = validate(dataset)
    elif args.command == "smoke":
        result = smoke_test(dataset)
    elif args.command == "egm":
        result = run_egm(dataset, int(args.task))
    elif args.command == "select-egm":
        result = select_egm(dataset)
    elif args.command == "iterative":
        result = run_iterative(dataset, int(args.task))
    elif args.command == "select-iterative":
        result = select_iterative(dataset)
    elif args.command == "test-shard":
        result = run_test_shard(dataset, int(args.task))
    else:
        result = merge_test(dataset)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
