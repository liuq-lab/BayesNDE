"""Isolated, general-purpose BGM generation improvements."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import tensorflow as tf

from bayesgm.models import BGM


class ResidualDenseBlock(tf.keras.layers.Layer):
    def __init__(self, width: int, name: str):
        super().__init__(name=name)
        self.dense1 = tf.keras.layers.Dense(int(width))
        self.act1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1.0e-5)
        self.dense2 = tf.keras.layers.Dense(int(width))
        self.act2 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1.0e-5)

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
        del training
        x = self.norm1(self.act1(self.dense1(inputs)))
        x = self.norm2(self.act2(self.dense2(x)))
        return (inputs + x) * tf.cast(1.0 / math.sqrt(2.0), x.dtype)


class ResidualEncoder(tf.keras.Model):
    def __init__(self, input_dim: int, output_dim: int, width: int, blocks: int):
        super().__init__(name="e_net")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.input_norm = tf.keras.layers.BatchNormalization(name="input_standardization")
        self.projection = tf.keras.layers.Dense(int(width), name="input_projection")
        self.blocks = [ResidualDenseBlock(width, f"encoder_block_{i}") for i in range(int(blocks))]
        self.output_layer = tf.keras.layers.Dense(self.output_dim, name="latent")

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
        x = self.projection(self.input_norm(inputs, training=training))
        for block in self.blocks:
            x = block(x, training=training)
        return self.output_layer(x)


class ResidualGenerator(tf.keras.Model):
    def __init__(self, input_dim: int, output_dim: int, width: int, blocks: int):
        super().__init__(name="g_net")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.input_norm = tf.keras.layers.BatchNormalization(name="input_standardization")
        self.projection = tf.keras.layers.Dense(int(width), name="input_projection")
        self.blocks = [ResidualDenseBlock(width, f"generator_block_{i}") for i in range(int(blocks))]
        self.mean_layer = tf.keras.layers.Dense(self.output_dim, name="mean")
        self.var_layer = tf.keras.layers.Dense(self.output_dim, name="variance_logits")

    def call(
        self, inputs: tf.Tensor, eps: float = 1.0e-6, training: bool = True
    ) -> tuple[tf.Tensor, tf.Tensor]:
        x = self.projection(self.input_norm(inputs, training=training))
        for block in self.blocks:
            x = block(x, training=training)
        mean = self.mean_layer(x)
        variance = tf.nn.softplus(self.var_layer(x)) + tf.cast(eps, mean.dtype)
        return mean, variance

    @staticmethod
    def reparameterize(mean: tf.Tensor, variance: tf.Tensor) -> tf.Tensor:
        return mean + tf.random.normal(tf.shape(mean), dtype=mean.dtype) * tf.sqrt(variance)


def _standardize_joint(
    reference: tf.Tensor, generated: tf.Tensor, eps: float = 1.0e-6
) -> tuple[tf.Tensor, tf.Tensor]:
    reference = tf.convert_to_tensor(reference)
    generated = tf.cast(tf.convert_to_tensor(generated), reference.dtype)
    combined = tf.concat([reference, generated], axis=0)
    mean = tf.stop_gradient(tf.reduce_mean(combined, axis=0, keepdims=True))
    variance = tf.stop_gradient(tf.math.reduce_variance(combined, axis=0, keepdims=True))
    scale = tf.sqrt(variance + tf.cast(eps, reference.dtype))
    return (reference - mean) / scale, (generated - mean) / scale


def _squared_distances(left: tf.Tensor, right: tf.Tensor) -> tf.Tensor:
    left2 = tf.reduce_sum(tf.square(left), axis=1, keepdims=True)
    right2 = tf.transpose(tf.reduce_sum(tf.square(right), axis=1, keepdims=True))
    return tf.maximum(left2 + right2 - 2.0 * tf.matmul(left, right, transpose_b=True), 0.0)


def joint_rbf_mmd_loss(
    reference: tf.Tensor,
    generated: tf.Tensor,
    scales: Sequence[float] = (0.5, 1.0, 2.0),
) -> tf.Tensor:
    reference, generated = _standardize_joint(reference, generated)
    dxx = _squared_distances(reference, reference)
    dyy = _squared_distances(generated, generated)
    dxy = _squared_distances(reference, generated)
    dimension = tf.cast(tf.shape(reference)[1], reference.dtype)
    results = []
    for scale in tf.unstack(tf.cast(tf.convert_to_tensor(scales), reference.dtype)):
        bandwidth2 = tf.maximum(scale * dimension, tf.cast(1.0e-6, reference.dtype))
        results.append(
            tf.reduce_mean(tf.exp(-dxx / (2.0 * bandwidth2)))
            + tf.reduce_mean(tf.exp(-dyy / (2.0 * bandwidth2)))
            - 2.0 * tf.reduce_mean(tf.exp(-dxy / (2.0 * bandwidth2)))
        )
    return tf.maximum(tf.reduce_mean(tf.stack(results)), 0.0)


def sliced_wasserstein_loss(
    reference: tf.Tensor,
    generated: tf.Tensor,
    n_projections: int = 64,
    seed: int = 20260831,
) -> tf.Tensor:
    reference, generated = _standardize_joint(reference, generated)
    dimension_static = reference.shape[-1]
    if dimension_static is None:
        raise ValueError("sliced_wasserstein_loss requires a statically known feature dimension.")
    dimension = int(dimension_static)
    random_count = max(int(n_projections) - dimension, 0)
    random_projections = tf.random.stateless_normal(
        [dimension, random_count],
        seed=[int(seed), int(seed) ^ 0x5A17],
        dtype=reference.dtype,
    )
    random_projections /= tf.maximum(tf.norm(random_projections, axis=0, keepdims=True), 1.0e-6)
    projections = tf.concat([tf.eye(dimension, dtype=reference.dtype), random_projections], axis=1)
    ref_projected = tf.sort(tf.matmul(reference, projections), axis=0)
    gen_projected = tf.sort(tf.matmul(generated, projections), axis=0)
    return tf.reduce_mean(tf.square(ref_projected - gen_projected))


def _batch_corr(values: tf.Tensor, eps: float = 1.0e-6) -> tf.Tensor:
    centered = values - tf.reduce_mean(values, axis=0, keepdims=True)
    standardized = centered / tf.sqrt(tf.math.reduce_variance(values, axis=0) + eps)
    return tf.matmul(standardized, standardized, transpose_a=True) / tf.cast(tf.shape(values)[0], values.dtype)


def empirical_correlation_loss(reference: tf.Tensor, generated: tf.Tensor) -> tf.Tensor:
    reference_corr = tf.stop_gradient(_batch_corr(reference))
    generated_corr = _batch_corr(generated)
    return tf.reduce_mean(tf.square(generated_corr - reference_corr))


class GenerationOptimizedBGM(BGM):
    def __init__(self, params: Mapping[str, Any], timestamp: str | None = None, random_seed: int | None = None):
        params_copy = dict(params)
        network_type = str(params_copy.get("g_network_type", "mlp")).lower()
        encoder_type = str(params_copy.get("e_network_type", network_type)).lower()
        if network_type not in {"mlp", "residual"} or encoder_type not in {"mlp", "residual"}:
            raise ValueError("g_network_type/e_network_type must be 'mlp' or 'residual'.")
        if network_type != encoder_type:
            raise ValueError("The controlled experiment requires matching generator/encoder types.")
        if params_copy.get("factorized_generator"):
            raise ValueError("factorized_generator is forbidden in the general 30D experiment.")
        super().__init__(params_copy, timestamp=timestamp, random_seed=random_seed)
        if network_type == "residual":
            g_units = list(params_copy.get("g_units", [256] * 5))
            e_units = list(params_copy.get("e_units", g_units))
            if len(set(g_units)) != 1 or len(set(e_units)) != 1:
                raise ValueError("Residual configurations require constant-width unit lists.")
            self.g_net = ResidualGenerator(
                params_copy["z_dim"], params_copy["x_dim"], g_units[0], len(g_units)
            )
            self.e_net = ResidualEncoder(
                params_copy["x_dim"], params_copy["z_dim"], e_units[0], len(e_units)
            )
            self.g_net(tf.zeros([1, int(params_copy["z_dim"])], tf.float32), training=False)
            self.e_net(tf.zeros([1, int(params_copy["x_dim"])], tf.float32), training=False)
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
            self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=100)

    @tf.function
    def train_gen_step(self, data_z: tf.Tensor, data_x: tf.Tensor):
        with tf.GradientTape() as tape:
            mu, variance = self._decode_generator(data_z)
            sampled = self._sample_decoder(mu, variance)
            generated = mu if self.params.get("x_adv_use_mean", False) else sampled
            encoded = self.e_net(data_x)
            cycle_input = mu if self.params.get("z_cycle_use_mean", False) else sampled
            latent_cycle = self.e_net(cycle_input)
            recon_mu, recon_var = self._decode_generator(encoded)
            reconstructed = self._sample_decoder(recon_mu, recon_var)
            if self.params.get("x_cycle_use_mean", False):
                reconstructed = recon_mu

            g_adv = tf.reduce_mean(tf.square(0.9 - self.dx_net(generated)))
            e_adv = tf.reduce_mean(tf.square(0.9 - self.dz_net(encoded)))
            l2_z = tf.reduce_mean(tf.square(data_z - latent_cycle))
            l2_x = tf.reduce_mean(tf.square(data_x - reconstructed))
            reg = tf.reduce_mean(tf.square(variance))

            real_mean = tf.reduce_mean(data_x, axis=0)
            gen_mean = tf.reduce_mean(generated, axis=0)
            real_var = tf.math.reduce_variance(data_x, axis=0)
            gen_var = tf.math.reduce_variance(generated, axis=0)
            moment = tf.reduce_mean(tf.square(real_mean - gen_mean)) + tf.reduce_mean(tf.square(real_var - gen_var))
            denom = tf.cast(tf.maximum(tf.shape(data_x)[0] - 1, 1), data_x.dtype)
            real_centered = data_x - real_mean
            gen_centered = generated - gen_mean
            covariance = tf.reduce_mean(tf.square(
                tf.matmul(real_centered, real_centered, transpose_a=True) / denom
                - tf.matmul(gen_centered, gen_centered, transpose_a=True) / denom
            ))

            marginal_mmd = tf.constant(0.0, data_x.dtype)
            for scale in tf.unstack(tf.constant(self.params.get("mmd_scales", [0.05, 0.1, 0.2, 0.5, 1.0]), data_x.dtype)):
                bandwidth = tf.maximum(scale, tf.cast(1.0e-6, data_x.dtype))
                real_delta = data_x[None, :, :] - data_x[:, None, :]
                gen_delta = generated[None, :, :] - generated[:, None, :]
                cross_delta = data_x[:, None, :] - generated[None, :, :]
                marginal_mmd += tf.reduce_mean(
                    tf.exp(-tf.square(real_delta) / (2.0 * bandwidth**2))
                    + tf.exp(-tf.square(gen_delta) / (2.0 * bandwidth**2))
                    - 2.0 * tf.exp(-tf.square(cross_delta) / (2.0 * bandwidth**2))
                )
            marginal_mmd /= tf.cast(len(self.params.get("mmd_scales", [0.05, 0.1, 0.2, 0.5, 1.0])), data_x.dtype)

            target = str(self.params.get("decorrelation_target", "identity")).lower()
            gen_corr = _batch_corr(generated)
            corr_loss = (
                empirical_correlation_loss(data_x, generated)
                if target in {"train", "batch", "real", "data"}
                else tf.reduce_mean(tf.square(gen_corr - tf.eye(tf.shape(gen_corr)[0])))
            )
            variance_target = tf.cast(self.params.get("variance_target", 0.0), data_x.dtype)
            variance_target_loss = tf.reduce_mean(tf.square(variance - variance_target))
            variance_log_loss = self._variance_log_target_loss(variance)
            joint_mmd = (
                joint_rbf_mmd_loss(data_x, generated, self.params.get("joint_mmd_scales", [0.5, 1.0, 2.0]))
                if float(self.params.get("joint_mmd_weight", 0.0)) > 0.0
                else tf.constant(0.0, data_x.dtype)
            )
            swd = (
                sliced_wasserstein_loss(
                    data_x,
                    generated,
                    int(self.params.get("sliced_wasserstein_projections", 64)),
                    int(self.params.get("sliced_wasserstein_seed", 20260831)),
                )
                if float(self.params.get("sliced_wasserstein_weight", 0.0)) > 0.0
                else tf.constant(0.0, data_x.dtype)
            )

            cycle_weight = tf.cast(self.params.get("cycle_weight", 10.0), data_x.dtype)
            loss = (
                tf.cast(self.params.get("g_adv_weight", 1.0), data_x.dtype) * g_adv
                + tf.cast(self.params.get("e_adv_weight", 1.0), data_x.dtype) * e_adv
                + tf.cast(self.params.get("x_cycle_weight", cycle_weight), data_x.dtype) * l2_x
                + tf.cast(self.params.get("z_cycle_weight", cycle_weight), data_x.dtype) * l2_z
                + tf.cast(self.params.get("alpha", 0.0), data_x.dtype) * reg
                + tf.cast(self.params.get("variance_target_weight", 0.0), data_x.dtype) * variance_target_loss
                + tf.cast(self.params.get("variance_log_target_weight", 0.0), data_x.dtype) * variance_log_loss
                + tf.cast(self.params.get("marginal_moment_weight", 0.0), data_x.dtype) * moment
                + tf.cast(self.params.get("covariance_weight", 0.0), data_x.dtype) * covariance
                + tf.cast(self.params.get("marginal_mmd_weight", 0.0), data_x.dtype) * marginal_mmd
                + tf.cast(self.params.get("decorrelation_weight", 0.0), data_x.dtype) * corr_loss
                + tf.cast(self.params.get("joint_mmd_weight", 0.0), data_x.dtype) * joint_mmd
                + tf.cast(self.params.get("sliced_wasserstein_weight", 0.0), data_x.dtype) * swd
            )
        variables = self.g_net.trainable_variables + self.e_net.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.g_pre_optimizer.apply_gradients(zip(gradients, variables))
        return g_adv, e_adv, l2_z, l2_x, reg, loss

    @tf.function
    def update_iterative_empirical_corr(self, data_x: tf.Tensor) -> tf.Tensor:
        weight = tf.cast(
            self.params.get("iterative_empirical_corr_weight", 0.0), data_x.dtype
        )
        with tf.GradientTape() as tape:
            prior_z = tf.random.normal(
                [tf.shape(data_x)[0], int(self.params["z_dim"])], dtype=data_x.dtype
            )
            mean, variance = self._decode_generator(prior_z)
            generated = self._sample_decoder(mean, variance)
            generated_corr = _batch_corr(generated)
            target_value = self.params.get("iterative_empirical_corr_target")
            empirical_corr = (
                tf.stop_gradient(_batch_corr(data_x))
                if target_value is None
                else tf.constant(target_value, dtype=data_x.dtype)
            )
            tf.debugging.assert_equal(
                tf.shape(generated_corr),
                tf.shape(empirical_corr),
                message="Empirical correlation target has the wrong shape.",
            )
            corr_loss = tf.reduce_mean(tf.square(generated_corr - empirical_corr))
            weighted_loss = weight * corr_loss
        gradients = tape.gradient(weighted_loss, self.g_net.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gradients, self.g_net.trainable_variables))
        return corr_loss


def requires_local_bgm(params: Mapping[str, Any]) -> bool:
    return (
        str(params.get("g_network_type", "mlp")).lower() != "mlp"
        or str(params.get("e_network_type", "mlp")).lower() != "mlp"
        or float(params.get("joint_mmd_weight", 0.0)) != 0.0
        or float(params.get("sliced_wasserstein_weight", 0.0)) != 0.0
        or float(params.get("iterative_empirical_corr_weight", 0.0)) != 0.0
    )


def build_bgm_model(
    params: Mapping[str, Any], timestamp: str | None = None, random_seed: int | None = None
) -> BGM:
    cls = GenerationOptimizedBGM if requires_local_bgm(params) else BGM
    return cls(params=dict(params), timestamp=timestamp, random_seed=random_seed)


def _checkpoint_paths(model: BGM, step: int) -> dict[str, str]:
    root = Path(model.checkpoint_path)
    root.mkdir(parents=True, exist_ok=True)
    prefix = root / f"weights_at_egm_init_{int(step)}"
    encoder = f"{prefix}_encoder.weights.h5"
    generator = f"{prefix}_generator.weights.h5"
    model.e_net.save_weights(encoder)
    model.g_net.save_weights(generator)
    return {"encoder_weights": encoder, "generator_weights": generator}


def generation_gate(section: Mapping[str, Any], validation_std: Sequence[float]) -> dict[str, Any]:
    metrics = section["two_sample_vs_validation"]
    generated_std = np.asarray(section["per_dim"]["std"], dtype=np.float64)
    validation_std_array = np.asarray(validation_std, dtype=np.float64)
    ratios = generated_std / np.maximum(validation_std_array, 1.0e-12)
    std_fraction = float(np.mean((ratios >= 0.75) & (ratios <= 1.25)))
    values = {
        "mmd_rbf": metrics.get("mmd_rbf"),
        "sym_kl_mean": metrics.get("sym_kl_mean"),
        "lisi_normalized_mean": metrics.get("lisi_normalized_mean"),
        "ks_stat_mean": metrics.get("ks_stat_mean"),
        "peak_coverage_mean": metrics.get("peak_coverage_mean"),
        "peak_full_coverage_dim_fraction": metrics.get("peak_full_coverage_dim_fraction"),
        "marginal_histogram_correlation_mean": metrics.get("marginal_histogram_correlation_mean"),
        "generated_peak_valley_ratio_mean": metrics.get("generated_peak_valley_ratio_mean"),
        "peak_valley_ratio_error_mean": metrics.get("peak_valley_ratio_error_mean"),
        "peak_valley_separated_fraction": metrics.get("peak_valley_separated_fraction"),
        "std_ratio_fraction_in_0p75_1p25": std_fraction,
        "std_ratio_per_dim": ratios.tolist(),
    }
    finite = all(values[key] is not None and np.isfinite(float(values[key])) for key in (
        "mmd_rbf", "sym_kl_mean", "lisi_normalized_mean", "ks_stat_mean"
    ))
    values["passed"] = bool(
        finite
        and float(values["mmd_rbf"]) <= 0.05
        and float(values["sym_kl_mean"]) <= 5.0
        and float(values["lisi_normalized_mean"]) >= 0.5
        and float(values["ks_stat_mean"]) <= 0.2
        and std_fraction >= 0.9
        and float(values.get("peak_coverage_mean") or 0.0) >= 0.9
        and float(values.get("peak_full_coverage_dim_fraction") or 0.0) >= 0.8
    )
    if finite:
        values["selection_score"] = float(
            float(values["mmd_rbf"]) / 0.05
            + float(values["sym_kl_mean"]) / 5.0
            + max(0.0, 1.0 - float(values["lisi_normalized_mean"])) / 0.5
            + float(values["ks_stat_mean"]) / 0.2
            + max(0.0, 0.9 - std_fraction) / 0.1
            + max(0.0, 0.9 - float(values.get("peak_coverage_mean") or 0.0)) / 0.1
            + max(0.0, float(values.get("peak_valley_ratio_error_mean") or 1.0) - 0.2) / 0.2
            + max(0.0, 0.8 - float(values.get("peak_valley_separated_fraction") or 0.0)) / 0.2
        )
    else:
        values["selection_score"] = None
    return values


def train_egm_with_diagnostics(
    *,
    model: BGM,
    train_data: np.ndarray,
    sampler: Any,
    config: Mapping[str, Any],
    diagnostics_root: Path,
    max_steps: int,
    batch_size: int = 256,
    every: int = 50,
    earliest_stop: int = 400,
    patience: int = 4,
    min_relative_improvement: float = 0.01,
    seed: int = 1024,
    callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    from bayesgm.datasets import Base_sampler
    from bayesnde.diagnostics.generation import run_bgm_generation_diagnostics

    if max_steps < 1 or every < 1:
        raise ValueError("max_steps and every must be positive.")
    data = np.asarray(train_data, dtype=np.float32)
    model.data_sampler = Base_sampler(x=data, y=data, v=data, batch_size=int(batch_size), normalize=False)
    diagnostics_root = Path(diagnostics_root)
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    best_score = float("inf")
    stale = 0
    validation_std = np.std(np.asarray(sampler.X_val), axis=0)

    for step in range(1, int(max_steps) + 1):
        for _ in range(int(model.params["g_d_freq"])):
            batch_x, _, _ = model.data_sampler.next_batch()
            batch_z = model.z_sampler.get_batch(int(batch_size))
            dz_loss, dx_loss, d_loss = model.train_disc_step(batch_z, batch_x)
        batch_x, _, _ = model.data_sampler.next_batch()
        batch_z = model.z_sampler.get_batch(int(batch_size))
        losses = model.train_gen_step(batch_z, batch_x)
        if step % int(every) != 0 and step != int(max_steps):
            continue

        weights = _checkpoint_paths(model, step)
        step_dir = diagnostics_root / f"step_{step:05d}"
        diag_config = dict(config)
        diag_config["generation_diagnostics"] = dict(config.get("generation_diagnostics", {}))
        diag_config["generation_diagnostics"].setdefault("plots_enabled", True)
        diag = run_bgm_generation_diagnostics(
            model=model, sampler=sampler, run_dir=step_dir, config=diag_config, seed=int(seed) + step
        )
        mean_gate = generation_gate(diag["generated_mean"], validation_std)
        sample_gate = generation_gate(diag["generated_sample"], validation_std)
        score = sample_gate["selection_score"]
        row = {
            "step": int(step),
            "selectable": bool(step > 0),
            **weights,
            "diagnostics_dir": str(step_dir),
            "mean_gate": mean_gate,
            "sample_gate": sample_gate,
            "losses": {
                "g_adv": float(losses[0]), "e_adv": float(losses[1]),
                "l2_z": float(losses[2]), "l2_x": float(losses[3]),
                "variance_regularization": float(losses[4]), "total": float(losses[5]),
                "dz": float(dz_loss), "dx": float(dx_loss), "discriminator": float(d_loss),
            },
            "finite_generated_mean_fraction": float(np.mean(np.isfinite(np.load(step_dir / "generated_samples.npz")["x_gen_mean"]))),
            "finite_generated_sample_fraction": float(np.mean(np.isfinite(np.load(step_dir / "generated_samples.npz")["x_gen_sample"]))),
        }
        rows.append(row)
        with open(diagnostics_root / "checkpoint_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
        flat = {
            "step": step,
            "sample_passed": sample_gate["passed"],
            "sample_score": score,
            "sample_mmd": sample_gate["mmd_rbf"],
            "sample_sym_kl": sample_gate["sym_kl_mean"],
            "sample_lisi": sample_gate["lisi_normalized_mean"],
            "sample_ks": sample_gate["ks_stat_mean"],
            "sample_std_fraction": sample_gate["std_ratio_fraction_in_0p75_1p25"],
            "mean_passed": mean_gate["passed"],
        }
        with open(diagnostics_root / "checkpoint_metrics.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat))
            writer.writeheader()
            for saved in rows:
                sg = saved["sample_gate"]
                writer.writerow({
                    "step": saved["step"], "sample_passed": sg["passed"],
                    "sample_score": sg["selection_score"], "sample_mmd": sg["mmd_rbf"],
                    "sample_sym_kl": sg["sym_kl_mean"], "sample_lisi": sg["lisi_normalized_mean"],
                    "sample_ks": sg["ks_stat_mean"],
                    "sample_std_fraction": sg["std_ratio_fraction_in_0p75_1p25"],
                    "mean_passed": saved["mean_gate"]["passed"],
                })
        if callback is not None:
            callback(row)

        if score is not None and np.isfinite(float(score)) and float(score) <= best_score * (1.0 - min_relative_improvement):
            best_score = float(score)
            stale = 0
        else:
            stale += 1
        if step >= int(earliest_stop) and stale >= int(patience):
            break
    return rows


def select_checkpoint(rows: Sequence[Mapping[str, Any]], require_gate: bool = True) -> Mapping[str, Any]:
    candidates = [row for row in rows if int(row.get("step", 0)) > 0 and row.get("selectable", True)]
    if require_gate:
        candidates = [row for row in candidates if bool(row["sample_gate"]["passed"])]
    if not candidates:
        raise RuntimeError("No nonzero EGM checkpoint satisfies the requested selection rule.")
    return min(candidates, key=lambda row: (float(row["sample_gate"]["selection_score"]), int(row["step"])))
