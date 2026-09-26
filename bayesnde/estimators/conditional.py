"""Conditional BGM and its bridge-sampling estimator of ``p(x | y)``."""

from __future__ import annotations

import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.mixture import GaussianMixture

from bayesgm.models.bgm.base import correlation_alignment_loss, make_bgm_optimizer, offdiag_correlation_loss
from bayesgm.models.networks import BaseFullyConnectedNet, BaseVariationalNet, Discriminator

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

from bayesnde.data.uci_conditional import ProcessedSplit


tfd = tfp.distributions


V1_BGM_DEFAULTS: dict[str, Any] = {
    "use_bnn": False,
    "save_model": True,
    "save_res": False,
    "lr_theta": 1.0e-6,
    "lr_z": 1.0e-7,
    "lr": 1.0e-3,
    "optimizer": "adamw",
    "weight_decay": 0.0,
    "g_d_freq": 1,
    "kl_weight": 5.0e-5,
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
    "decorrelation_weight": 0.3,
    "decorrelation_target": "train",
    "mmd_scales": [0.05, 0.1, 0.2, 0.5, 1.0],
    "x_cycle_use_mean": False,
    "x_adv_use_mean": False,
    "z_cycle_use_mean": False,
    "variance_target": 0.01,
    "variance_target_weight": 0.0,
    "variance_log_target": 0.01,
    "variance_log_target_weight": 0.01,
    "variance_log_eps": 1.0e-8,
    "variance_floor": 1.0e-6,
    "factorized_generator": False,
    "low_rank_generator": False,
    "low_rank_rank": 1,
    "low_rank_init_std": 1.0e-4,
    "low_rank_l2_weight": 0.0,
    "low_rank_variance_floor": 0.01,
    "low_rank_woodbury_jitter": 1.0e-7,
    "dz_units": [256, 256, 128, 64],
    "dx_units": [256, 256, 128, 64],
    "label_conditioning": "one_hot",
    "label_embedding_dim": 8,
    "z_projection_dim": 0,
    "z_projection_activation": "swish",
    "decoder_conditioning_mode": "input_only",
    "egm_n_iter": 22000,
    "egm_batches_per_eval": 1000,
    "use_egm_init": True,
    "iterative_epoch_selection": False,
    "selection_every": 50,
    "selection_bridge_enabled": False,
    "selection_bridge_points": 0,
    "selection_bridge_candidate_top_k": 3,
    "selection_bridge_S": 2048,
    "selection_bridge_K": 3,
    "selection_bridge_M": 256,
    "selection_bridge_burn_in": 128,
    "selection_bridge_num_chains": 2,
}


def configure_tensorflow_gpu_memory_growth() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


configure_tensorflow_gpu_memory_growth()


def one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    out = np.zeros((labels.size, int(n_classes)), dtype=np.float32)
    out[np.arange(labels.size), labels] = 1.0
    return out


class ConditionalV1Generator(tf.keras.Model):
    def __init__(
        self,
        z_dim: int,
        y_dim: int,
        x_dim: int,
        hidden: Sequence[int],
        variance_floor: float,
        *,
        label_feature_dim: int | None = None,
        z_projection_dim: int = 0,
        z_projection_activation: str = "swish",
        conditioning_mode: str = "input_only",
    ):
        super().__init__()
        self.z_dim = int(z_dim)
        self.y_dim = int(y_dim)
        self.x_dim = int(x_dim)
        self.variance_floor = float(variance_floor)
        self.label_feature_dim = int(label_feature_dim if label_feature_dim is not None else y_dim)
        self.z_projection_dim = int(z_projection_dim or 0)
        self.z_feature_dim = self.z_projection_dim if self.z_projection_dim > 0 else self.z_dim
        self.z_projection_activation = str(z_projection_activation)
        self.conditioning_mode = str(conditioning_mode)
        if self.conditioning_mode not in {"input_only", "every_layer"}:
            raise ValueError("conditioning_mode must be input_only or every_layer.")
        self.z_projection_layer: tf.keras.layers.Dense | None = None
        if self.z_projection_dim > 0:
            self.z_projection_layer = tf.keras.layers.Dense(
                self.z_projection_dim,
                activation=self.z_projection_activation,
                name="z_projection",
            )
        self.base = BaseVariationalNet(
            input_dim=self.z_feature_dim + self.label_feature_dim,
            output_dim=self.x_dim,
            model_name="conditional_g_net",
            nb_units=[int(width) for width in hidden],
        )

    @property
    def hidden_layers(self) -> list[Any]:
        return list(self.base.all_layers)

    @property
    def mean_layer(self) -> tf.keras.layers.Layer:
        return self.base.mean_layer

    @property
    def logvar_layer(self) -> tf.keras.layers.Layer:
        return self.base.var_layer

    def call(self, inputs: tf.Tensor | Sequence[tf.Tensor], training: bool = True):
        if isinstance(inputs, (tuple, list)):
            z, label_features = inputs
            z = tf.cast(z, tf.float32)
            if self.z_projection_layer is not None:
                z = self.z_projection_layer(z, training=training)
            label_features = tf.cast(label_features, tf.float32)
            h = tf.concat([z, label_features], axis=-1)
        else:
            h = tf.cast(inputs, tf.float32)
            label_features = None
        if self.conditioning_mode == "input_only" or label_features is None:
            return self.base(h, eps=float(self.variance_floor), training=training)

        x = self.base.norm_layer(h, training=training)
        for layer_index, dense_layer in enumerate(self.base.all_layers):
            if layer_index > 0:
                x = tf.concat([x, label_features], axis=-1)
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(dense_layer(x))
        output_features = tf.concat([x, label_features], axis=-1)
        mean = self.base.mean_layer(output_features)
        variance = tf.nn.softplus(self.base.var_layer(output_features)) + float(self.variance_floor)
        return mean, variance

    def reparameterize(self, mean: tf.Tensor, var: tf.Tensor) -> tf.Tensor:
        return self.base.reparameterize(mean, var)


class ConditionalV1Encoder(tf.keras.Model):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        z_dim: int,
        hidden: Sequence[int],
        *,
        label_feature_dim: int | None = None,
    ):
        super().__init__()
        self.x_dim = int(x_dim)
        self.y_dim = int(y_dim)
        self.z_dim = int(z_dim)
        self.label_feature_dim = int(label_feature_dim if label_feature_dim is not None else y_dim)
        self.base = BaseFullyConnectedNet(
            input_dim=self.x_dim + self.label_feature_dim,
            output_dim=self.z_dim,
            model_name="conditional_e_net",
            nb_units=[int(width) for width in hidden],
        )

    @property
    def layers_(self) -> list[Any]:
        return [pair[0] for pair in self.base.all_layers[:-1]]

    @property
    def out_layer(self) -> tf.keras.layers.Layer:
        return self.base.all_layers[-1][0]

    def call(self, x: tf.Tensor, label_features: tf.Tensor | None = None, training: bool = True) -> tf.Tensor:
        if label_features is None:
            h = tf.cast(x, tf.float32)
        else:
            h = tf.concat([tf.cast(x, tf.float32), tf.cast(label_features, tf.float32)], axis=-1)
        return self.base(h, training=training)


class ConditionalBGM(tf.Module):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        z_dim: int,
        hidden: Sequence[int],
        encoder_hidden: Sequence[int],
        variance_floor: float,
        params: Mapping[str, Any],
    ):
        super().__init__()
        self.x_dim = int(x_dim)
        self.y_dim = int(y_dim)
        self.z_dim = int(z_dim)
        self.dtype = tf.float32
        self.params = dict(params)
        self.params["x_dim"] = self.x_dim
        self.params["y_dim"] = self.y_dim
        self.params["z_dim"] = self.z_dim
        self.params["use_bnn"] = bool(self.params.get("use_bnn", False))
        self.params["alpha"] = float(self.params.get("alpha", 0.0))
        self.params["gamma"] = float(self.params.get("gamma", 0.0))
        self.params["kl_weight"] = float(self.params.get("kl_weight", 5.0e-5))
        self.params["g_d_freq"] = int(self.params.get("g_d_freq", 1))
        self.label_conditioning = str(self.params.get("label_conditioning", "one_hot")).lower()
        if self.label_conditioning not in {"one_hot", "embedding"}:
            raise ValueError("label_conditioning must be either 'one_hot' or 'embedding'.")
        self.label_embedding_dim = int(self.params.get("label_embedding_dim", 8))
        if self.label_conditioning == "embedding" and self.label_embedding_dim <= 0:
            raise ValueError("label_embedding_dim must be positive when label_conditioning='embedding'.")
        self.z_projection_dim = int(self.params.get("z_projection_dim", 0) or 0)
        if self.z_projection_dim < 0:
            raise ValueError("z_projection_dim must be non-negative.")
        self.z_projection_activation = str(self.params.get("z_projection_activation", "swish"))
        self.decoder_conditioning_mode = str(self.params.get("decoder_conditioning_mode", "input_only"))
        if self.decoder_conditioning_mode not in {"input_only", "every_layer"}:
            raise ValueError("decoder_conditioning_mode must be input_only or every_layer.")
        self.label_feature_dim = self.label_embedding_dim if self.label_conditioning == "embedding" else self.y_dim
        self.label_embedding: tf.keras.layers.Embedding | None = None
        if self.label_conditioning == "embedding":
            self.label_embedding = tf.keras.layers.Embedding(
                input_dim=self.y_dim,
                output_dim=self.label_embedding_dim,
                name="shared_label_embedding",
            )

        self.g_net = ConditionalV1Generator(
            z_dim,
            y_dim,
            x_dim,
            hidden,
            variance_floor,
            label_feature_dim=self.label_feature_dim,
            z_projection_dim=self.z_projection_dim,
            z_projection_activation=self.z_projection_activation,
            conditioning_mode=self.decoder_conditioning_mode,
        )
        self.e_net = ConditionalV1Encoder(
            x_dim,
            y_dim,
            z_dim,
            encoder_hidden,
            label_feature_dim=self.label_feature_dim,
        )
        self.dz_net = Discriminator(
            input_dim=self.z_dim,
            model_name="conditional_dz_net",
            nb_units=[int(width) for width in self.params.get("dz_units", [256, 256, 128, 64])],
        )
        self.dx_net = Discriminator(
            input_dim=self.x_dim,
            model_name="conditional_dx_net",
            nb_units=[int(width) for width in self.params.get("dx_units", [256, 256, 128, 64])],
        )
        self.g_pre_optimizer = make_bgm_optimizer(self.params, float(self.params.get("lr", 1.0e-3)), beta_1=0.5, beta_2=0.9)
        self.d_pre_optimizer = make_bgm_optimizer(self.params, float(self.params.get("lr", 1.0e-3)), beta_1=0.5, beta_2=0.9)
        self.g_optimizer = make_bgm_optimizer(self.params, float(self.params.get("lr_theta", 1.0e-6)), beta_1=0.9, beta_2=0.99)
        self.posterior_optimizer = make_bgm_optimizer(self.params, float(self.params.get("lr_z", 1.0e-7)), beta_1=0.9, beta_2=0.99)
        self.data_z: tf.Variable | None = None
        checkpoint_items: dict[str, Any] = {
            "g_net": self.g_net,
            "e_net": self.e_net,
            "dz_net": self.dz_net,
            "dx_net": self.dx_net,
            "g_pre_optimizer": self.g_pre_optimizer,
            "d_pre_optimizer": self.d_pre_optimizer,
            "g_optimizer": self.g_optimizer,
            "posterior_optimizer": self.posterior_optimizer,
        }
        if self.label_embedding is not None:
            checkpoint_items["label_embedding"] = self.label_embedding
        self.ckpt = tf.train.Checkpoint(**checkpoint_items)

    def build(self) -> None:
        z = tf.zeros((1, self.z_dim), dtype=tf.float32)
        y = tf.zeros((1, self.y_dim), dtype=tf.float32)
        x = tf.zeros((1, self.x_dim), dtype=tf.float32)
        label_features = self._label_features(y, training=False)
        self.g_net((z, label_features), training=False)
        self.e_net(x, label_features, training=False)
        self.dz_net(z, training=False)
        self.dx_net(x, training=False)

    def _label_ids(self, data_y: tf.Tensor) -> tf.Tensor:
        data_y = tf.convert_to_tensor(data_y)
        if data_y.shape.rank == 1:
            return tf.cast(tf.reshape(data_y, [-1]), tf.int32)
        return tf.cast(tf.argmax(data_y, axis=-1), tf.int32)

    def _label_features(self, data_y: tf.Tensor, training: bool = True) -> tf.Tensor:
        data_y = tf.convert_to_tensor(data_y)
        if self.label_conditioning == "one_hot":
            if data_y.shape.rank == 1:
                return tf.one_hot(tf.cast(tf.reshape(data_y, [-1]), tf.int32), self.y_dim, dtype=tf.float32)
            return tf.cast(data_y, tf.float32)
        if self.label_embedding is None:
            raise RuntimeError("label_embedding is not initialized.")
        del training
        return self.label_embedding(self._label_ids(data_y))

    def encode(self, data_x: tf.Tensor, data_y: tf.Tensor, training: bool = True) -> tf.Tensor:
        return self.e_net(data_x, self._label_features(data_y, training=training), training=training)

    def _decode_generator(self, data_z: tf.Tensor, data_y: tf.Tensor, training: bool = True):
        outputs = self.g_net((data_z, self._label_features(data_y, training=training)), training=training)
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
            raise ValueError("Conditional V1 generator must return (mu, var).")
        mu_x, sigma_square_x = outputs
        return mu_x, sigma_square_x, None

    def _conditioning_trainable_variables(self) -> list[tf.Variable]:
        if self.label_embedding is None:
            return []
        return list(self.label_embedding.trainable_variables)

    @staticmethod
    def _unique_variables(variables: Sequence[tf.Variable]) -> list[tf.Variable]:
        out: list[tf.Variable] = []
        seen: set[Any] = set()
        for variable in variables:
            key = variable.ref() if hasattr(variable, "ref") else id(variable)
            if key in seen:
                continue
            seen.add(key)
            out.append(variable)
        return out

    def _decoder_trainable_variables(self) -> list[tf.Variable]:
        return self._unique_variables(list(self.g_net.trainable_variables) + self._conditioning_trainable_variables())

    def _joint_generator_encoder_variables(self) -> list[tf.Variable]:
        return self._unique_variables(
            list(self.g_net.trainable_variables)
            + list(self.e_net.trainable_variables)
            + self._conditioning_trainable_variables()
        )

    def get_decoder_selection_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {"g_net_weights": [np.asarray(w).copy() for w in self.g_net.get_weights()]}
        if self.label_embedding is not None:
            state["label_embedding_weights"] = [np.asarray(w).copy() for w in self.label_embedding.get_weights()]
        return state

    def set_decoder_selection_state(self, state: Mapping[str, Any]) -> None:
        self.g_net.set_weights(list(state["g_net_weights"]))
        if self.label_embedding is not None and "label_embedding_weights" in state:
            self.label_embedding.set_weights(list(state["label_embedding_weights"]))

    def _sample_decoder(self, mu_x: tf.Tensor, sigma_square_x: tf.Tensor, low_rank_u: tf.Tensor | None = None) -> tf.Tensor:
        del low_rank_u
        return self.g_net.reparameterize(mu_x, sigma_square_x)

    def _decoder_nll_from_parts(
        self,
        data_x: tf.Tensor,
        mu_x: tf.Tensor,
        sigma_square_x: tf.Tensor,
        low_rank_u: tf.Tensor | None = None,
    ) -> tf.Tensor:
        del low_rank_u
        safe_var = tf.maximum(sigma_square_x, tf.cast(float(self.params.get("variance_floor", 1.0e-6)), sigma_square_x.dtype))
        ll_term = tf.square(data_x - mu_x) / (2.0 * safe_var) + 0.5 * tf.math.log(safe_var)
        return tf.reduce_sum(ll_term, axis=1)

    def _decoder_nll(self, data_z: tf.Tensor, data_x: tf.Tensor, data_y: tf.Tensor, training: bool = True) -> tf.Tensor:
        mu_x, sigma_square_x, low_rank_u = self._decode_generator(data_z, data_y, training=training)
        return self._decoder_nll_from_parts(data_x, mu_x, sigma_square_x, low_rank_u)

    def _variance_log_target_loss(self, sigma_square_x: tf.Tensor) -> tf.Tensor:
        weight = float(self.params.get("variance_log_target_weight", 0.0))
        if weight <= 0.0:
            return tf.constant(0.0, dtype=sigma_square_x.dtype)
        eps = tf.cast(float(self.params.get("variance_log_eps", 1.0e-8)), sigma_square_x.dtype)
        target = tf.cast(float(self.params.get("variance_log_target", self.params.get("variance_target", 1.0e-2))), sigma_square_x.dtype)
        safe_var = tf.maximum(sigma_square_x, eps)
        safe_target = tf.maximum(target, eps)
        return tf.reduce_mean(tf.square(tf.math.log(safe_var + eps) - tf.math.log(safe_target)))

    @tf.function
    def update_g_net(self, data_z: tf.Tensor, data_x: tf.Tensor, data_y: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as gen_tape:
            mu_x, sigma_square_x, low_rank_u = self._decode_generator(data_z, data_y, training=True)
            loss_x = tf.reduce_mean(self._decoder_nll_from_parts(data_x, mu_x, sigma_square_x, low_rank_u))
            loss_x += tf.cast(self.params.get("variance_log_target_weight", 0.0), loss_x.dtype) * self._variance_log_target_loss(sigma_square_x)
            variance_lower_bound = float(self.params.get("variance_lower_bound", 0.0))
            variance_lower_bound_weight = float(self.params.get("variance_lower_bound_weight", 0.0))
            if variance_lower_bound > 0.0 and variance_lower_bound_weight > 0.0:
                lower_bound = tf.cast(variance_lower_bound, sigma_square_x.dtype)
                safe_sigma = tf.maximum(sigma_square_x, tf.cast(1.0e-12, sigma_square_x.dtype))
                log_gap = tf.nn.relu(tf.math.log(lower_bound) - tf.math.log(safe_sigma))
                loss_x += tf.cast(variance_lower_bound_weight, loss_x.dtype) * tf.reduce_mean(tf.square(log_gap))
        variables = self._decoder_trainable_variables()
        gradients = gen_tape.gradient(loss_x, variables)
        self.g_optimizer.apply_gradients([(grad, var) for grad, var in zip(gradients, variables) if grad is not None])
        return loss_x

    @tf.function
    def update_latent_variable_sgd(self, data_z: tf.Variable, data_x: tf.Tensor, data_y: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            loss_px_z = tf.reduce_mean(self._decoder_nll(data_z, data_x, data_y, training=True))
            loss_prior_z = tf.reduce_mean(tf.reduce_sum(tf.square(data_z), axis=1) / 2.0)
            loss_posterior_z = loss_px_z + loss_prior_z
        gradients = tape.gradient(loss_posterior_z, [data_z])
        self.posterior_optimizer.apply_gradients(zip(gradients, [data_z]))
        return loss_posterior_z

    @tf.function
    def train_disc_step(self, data_z: tf.Tensor, data_x: tf.Tensor, data_y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        epsilon_z = tf.random.uniform([], minval=0.0, maxval=1.0)
        epsilon_x = tf.random.uniform([], minval=0.0, maxval=1.0)
        with tf.GradientTape(persistent=True) as disc_tape:
            with tf.GradientTape() as gpz_tape:
                data_z_ = self.encode(data_x, data_y, training=True)
                data_z_hat = data_z * epsilon_z + data_z_ * (1.0 - epsilon_z)
                data_dz_hat = self.dz_net(data_z_hat, training=True)
            with tf.GradientTape() as gpx_tape:
                mu_x_, sigma_square_x_, low_rank_u_ = self._decode_generator(data_z, data_y, training=True)
                data_x_sample_ = self._sample_decoder(mu_x_, sigma_square_x_, low_rank_u_)
                data_x_ = mu_x_ if self.params.get("x_adv_use_mean", False) else data_x_sample_
                data_x_hat = data_x * epsilon_x + data_x_ * (1.0 - epsilon_x)
                data_dx_hat = self.dx_net(data_x_hat, training=True)
            data_dx_ = self.dx_net(data_x_, training=True)
            data_dz_ = self.dz_net(data_z_, training=True)
            data_dx = self.dx_net(data_x, training=True)
            data_dz = self.dz_net(data_z, training=True)
            dz_loss = (tf.reduce_mean(tf.square(0.9 * tf.ones_like(data_dz) - data_dz)) + tf.reduce_mean(tf.square(0.1 * tf.ones_like(data_dz_) - data_dz_))) / 2.0
            dx_loss = (tf.reduce_mean(tf.square(0.9 * tf.ones_like(data_dx) - data_dx)) + tf.reduce_mean(tf.square(0.1 * tf.ones_like(data_dx_) - data_dx_))) / 2.0
            grad_z = gpz_tape.gradient(data_dz_hat, data_z_hat)
            grad_x = gpx_tape.gradient(data_dx_hat, data_x_hat)
            gpz_loss = tf.reduce_mean(tf.square(tf.sqrt(tf.reduce_sum(tf.square(grad_z), axis=1)) - 1.0))
            gpx_loss = tf.reduce_mean(tf.square(tf.sqrt(tf.reduce_sum(tf.square(grad_x), axis=1)) - 1.0))
            d_loss = dx_loss + dz_loss + tf.cast(self.params.get("gamma", 0.0), dx_loss.dtype) * (gpz_loss + gpx_loss)
        variables = self.dz_net.trainable_variables + self.dx_net.trainable_variables
        gradients = disc_tape.gradient(d_loss, variables)
        self.d_pre_optimizer.apply_gradients(zip(gradients, variables))
        return dz_loss, dx_loss, d_loss

    @tf.function
    def train_gen_step(self, data_z: tf.Tensor, data_x: tf.Tensor, data_y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.GradientTape(persistent=True) as gen_tape:
            mu_x_, sigma_square_x_, low_rank_u_ = self._decode_generator(data_z, data_y, training=True)
            data_x_sample_ = self._sample_decoder(mu_x_, sigma_square_x_, low_rank_u_)
            data_x_ = mu_x_ if self.params.get("x_adv_use_mean", False) else data_x_sample_
            reg_loss = tf.reduce_mean(tf.square(sigma_square_x_))
            data_z_ = self.encode(data_x, data_y, training=True)
            z_cycle_input = mu_x_ if self.params.get("z_cycle_use_mean", False) else data_x_sample_
            data_z__ = self.encode(z_cycle_input, data_y, training=True)
            mu_x__, sigma_square_x__, low_rank_u__ = self._decode_generator(data_z_, data_y, training=True)
            data_x__ = self._sample_decoder(mu_x__, sigma_square_x__, low_rank_u__)
            x_cycle_reconstruction = mu_x__ if self.params.get("x_cycle_use_mean", False) else data_x__
            data_dx_ = self.dx_net(data_x_, training=True)
            data_dz_ = self.dz_net(data_z_, training=True)
            l2_loss_x = tf.reduce_mean(tf.square(data_x - x_cycle_reconstruction))
            l2_loss_z = tf.reduce_mean(tf.square(data_z - data_z__))
            g_loss_adv = tf.reduce_mean(tf.square(0.9 * tf.ones_like(data_dx_) - data_dx_))
            e_loss_adv = tf.reduce_mean(tf.square(0.9 * tf.ones_like(data_dz_) - data_dz_))
            cycle_weight = tf.cast(self.params.get("cycle_weight", 10.0), g_loss_adv.dtype)
            x_cycle_weight = tf.cast(self.params.get("x_cycle_weight", cycle_weight), g_loss_adv.dtype)
            z_cycle_weight = tf.cast(self.params.get("z_cycle_weight", cycle_weight), g_loss_adv.dtype)
            variance_target_weight = tf.cast(self.params.get("variance_target_weight", 0.0), g_loss_adv.dtype)
            variance_target = tf.cast(self.params.get("variance_target", 0.0), g_loss_adv.dtype)
            variance_target_loss = tf.reduce_mean(tf.square(sigma_square_x_ - variance_target))
            variance_log_target_weight = tf.cast(self.params.get("variance_log_target_weight", 0.0), g_loss_adv.dtype)
            variance_log_target_loss = self._variance_log_target_loss(sigma_square_x_)
            g_adv_weight = tf.cast(self.params.get("g_adv_weight", 1.0), g_loss_adv.dtype)
            e_adv_weight = tf.cast(self.params.get("e_adv_weight", 1.0), g_loss_adv.dtype)
            marginal_moment_weight = tf.cast(self.params.get("marginal_moment_weight", 0.0), g_loss_adv.dtype)
            covariance_weight = tf.cast(self.params.get("covariance_weight", 0.0), g_loss_adv.dtype)
            marginal_mmd_weight = tf.cast(self.params.get("marginal_mmd_weight", 0.0), g_loss_adv.dtype)
            decorrelation_weight = tf.cast(self.params.get("decorrelation_weight", 0.0), g_loss_adv.dtype)
            real_mean = tf.reduce_mean(data_x, axis=0)
            gen_mean = tf.reduce_mean(data_x_, axis=0)
            real_var = tf.math.reduce_variance(data_x, axis=0)
            gen_var = tf.math.reduce_variance(data_x_, axis=0)
            marginal_moment_loss = tf.reduce_mean(tf.square(real_mean - gen_mean)) + tf.reduce_mean(tf.square(real_var - gen_var))
            denom = tf.maximum(tf.cast(tf.shape(data_x)[0] - 1, data_x.dtype), tf.cast(1.0, data_x.dtype))
            real_centered = data_x - real_mean
            gen_centered = data_x_ - gen_mean
            real_cov = tf.matmul(real_centered, real_centered, transpose_a=True) / denom
            gen_cov = tf.matmul(gen_centered, gen_centered, transpose_a=True) / denom
            covariance_loss = tf.reduce_mean(tf.square(real_cov - gen_cov))
            mmd_scales = tf.constant(self.params.get("mmd_scales", [0.05, 0.1, 0.2, 0.5, 1.0]), dtype=data_x.dtype)
            real_pairwise = tf.expand_dims(data_x, axis=0) - tf.expand_dims(data_x, axis=1)
            gen_pairwise = tf.expand_dims(data_x_, axis=0) - tf.expand_dims(data_x_, axis=1)
            cross_pairwise = tf.expand_dims(data_x, axis=1) - tf.expand_dims(data_x_, axis=0)
            marginal_mmd_loss = tf.constant(0.0, dtype=data_x.dtype)
            for scale in tf.unstack(mmd_scales):
                bandwidth = tf.maximum(scale, tf.constant(1.0e-6, dtype=data_x.dtype))
                kernel_scale = 2.0 * tf.square(bandwidth)
                k_real = tf.exp(-tf.square(real_pairwise) / kernel_scale)
                k_gen = tf.exp(-tf.square(gen_pairwise) / kernel_scale)
                k_cross = tf.exp(-tf.square(cross_pairwise) / kernel_scale)
                marginal_mmd_loss += tf.reduce_mean(k_real + k_gen - 2.0 * k_cross)
            marginal_mmd_loss = marginal_mmd_loss / tf.cast(tf.size(mmd_scales), data_x.dtype)
            decorrelation_target = str(self.params.get("decorrelation_target", "identity")).lower()
            if decorrelation_target in {"train", "batch", "real", "data"}:
                decorrelation_loss = correlation_alignment_loss(data_x_, data_x)
            elif decorrelation_target == "identity":
                decorrelation_loss = offdiag_correlation_loss(data_x_)
            else:
                raise ValueError("decorrelation_target must be one of 'identity', 'train', 'batch', 'real', or 'data'.")
            g_e_loss = (
                g_adv_weight * g_loss_adv
                + e_adv_weight * e_loss_adv
                + x_cycle_weight * l2_loss_x
                + z_cycle_weight * l2_loss_z
                + tf.cast(self.params.get("alpha", 0.0), g_loss_adv.dtype) * reg_loss
                + variance_target_weight * variance_target_loss
                + variance_log_target_weight * variance_log_target_loss
                + marginal_moment_weight * marginal_moment_loss
                + covariance_weight * covariance_loss
                + marginal_mmd_weight * marginal_mmd_loss
                + decorrelation_weight * decorrelation_loss
            )
        variables = self._joint_generator_encoder_variables()
        gradients = gen_tape.gradient(g_e_loss, variables)
        self.g_pre_optimizer.apply_gradients([(grad, var) for grad, var in zip(gradients, variables) if grad is not None])
        return g_loss_adv, e_loss_adv, l2_loss_z, l2_loss_x, reg_loss, g_e_loss

    @tf.function(reduce_retracing=True)
    def _run_hmc_chain(
        self,
        x_tf: tf.Tensor,
        y_tf: tf.Tensor,
        initial_state: tf.Tensor,
        num_results: tf.Tensor,
        burn_in: tf.Tensor,
        step_size: tf.Tensor,
        num_leapfrog_steps: tf.Tensor,
        target_accept_prob: tf.Tensor,
        seed: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        def target_log_prob_fn(z_state: tf.Tensor) -> tf.Tensor:
            chain_count = tf.shape(z_state)[0]
            x_rep = tf.repeat(x_tf, repeats=chain_count, axis=0)
            y_rep = tf.repeat(y_tf, repeats=chain_count, axis=0)
            return log_joint(self, z_state, x_rep, y_rep)

        hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=tf.cast(step_size, self.dtype),
            num_leapfrog_steps=tf.cast(num_leapfrog_steps, tf.int32),
        )
        adaptive_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=hmc_kernel,
            num_adaptation_steps=tf.cast(tf.maximum(tf.cast(burn_in, tf.int32) * 8 // 10, 1), tf.int32),
            target_accept_prob=tf.cast(target_accept_prob, self.dtype),
        )
        states, is_accepted = tfp.mcmc.sample_chain(
            num_results=tf.cast(num_results, tf.int32),
            num_burnin_steps=tf.cast(burn_in, tf.int32),
            current_state=initial_state,
            kernel=adaptive_kernel,
            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
            seed=tf.cast(seed, tf.int32),
        )
        return states, is_accepted

    @tf.function(reduce_retracing=True)
    def _evaluate_log_joint_batch(self, z: tf.Tensor, x_tf: tf.Tensor, y_tf: tf.Tensor) -> tf.Tensor:
        return log_joint(self, z, x_tf, y_tf)

    def save(self, checkpoint_dir: Path) -> str:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.build()
        manager = tf.train.CheckpointManager(self.ckpt, str(checkpoint_dir), max_to_keep=3)
        return manager.save()

    def restore(self, checkpoint_dir: Path) -> None:
        self.build()
        manager = tf.train.CheckpointManager(self.ckpt, str(checkpoint_dir), max_to_keep=3)
        if manager.latest_checkpoint is None:
            raise FileNotFoundError(f"No TensorFlow checkpoint found in {checkpoint_dir}")
        self.ckpt.restore(manager.latest_checkpoint).expect_partial()


def gaussian_log_likelihood(x: tf.Tensor, mean: tf.Tensor, var: tf.Tensor) -> tf.Tensor:
    return -0.5 * tf.reduce_sum(
        tf.math.log(2.0 * math.pi * var) + tf.square(x - mean) / var,
        axis=-1,
    )


def prior_log_prob(z: tf.Tensor) -> tf.Tensor:
    return -0.5 * tf.reduce_sum(tf.square(z) + math.log(2.0 * math.pi), axis=-1)


@tf.function(reduce_retracing=True)
def log_joint(model: ConditionalBGM, z: tf.Tensor, x: tf.Tensor, y_onehot: tf.Tensor) -> tf.Tensor:
    z_count = tf.shape(z)[0]
    x = tf.broadcast_to(x, tf.stack([z_count, tf.shape(x)[1]]))
    y_onehot = tf.broadcast_to(y_onehot, tf.stack([z_count, tf.shape(y_onehot)[1]]))
    mean, var, _ = model._decode_generator(z, y_onehot, training=False)
    observation = getattr(model, "observation_log_likelihood", None)
    if observation is not None:
        return observation(x, mean, var) + prior_log_prob(z)
    return gaussian_log_likelihood(x, mean, var) + prior_log_prob(z)


def logsumexp_np(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    max_value = np.max(values, axis=axis, keepdims=True)
    stable = max_value + np.log(np.sum(np.exp(values - max_value), axis=axis, keepdims=True))
    if axis is None:
        return np.asarray(stable).reshape(())
    return np.squeeze(stable, axis=axis)


def logmeanexp_np(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    n = values.size if axis is None else values.shape[axis]
    return logsumexp_np(values, axis=axis) - math.log(float(n))


@dataclass(frozen=True)
class StudentTMixture:
    weights: np.ndarray
    loc: np.ndarray
    scale_tril: np.ndarray
    df: float
    epsilon: float
    covariance_mode: str = "fitted_full"
    center_mode: str = "fitted_gmm"
    scoring: str = "multivariate_t"
    requested_scale: float | None = None


def fit_student_t_mixture(
    samples: np.ndarray,
    *,
    n_components: int,
    df: float,
    covariance_floor: float,
    random_seed: int,
    covariance_mode: str = "fitted_full",
    fixed_scale: float | None = None,
    center_mode: str = "fitted_gmm",
    scoring: str = "multivariate_t",
) -> StudentTMixture:
    samples = np.asarray(samples, dtype=np.float64)
    if covariance_mode not in {"fitted_full", "fixed_isotropic"}:
        raise ValueError("covariance_mode must be fitted_full or fixed_isotropic.")
    if center_mode not in {"fitted_gmm", "hmc_fit_mean"}:
        raise ValueError("center_mode must be fitted_gmm or hmc_fit_mean.")
    if scoring not in {"multivariate_t", "product_univariate_t"}:
        raise ValueError("scoring must be multivariate_t or product_univariate_t.")
    n_components = 1 if center_mode == "hmc_fit_mean" else int(min(max(1, n_components), samples.shape[0]))
    if center_mode == "hmc_fit_mean":
        weights = np.ones(1, dtype=np.float64)
        means = np.mean(samples, axis=0, keepdims=True)
        covariances = np.cov(samples, rowvar=False, ddof=1).reshape(1, samples.shape[1], samples.shape[1])
    else:
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            reg_covar=float(covariance_floor),
            random_state=int(random_seed),
        )
        gmm.fit(samples)
        weights = np.asarray(gmm.weights_, dtype=np.float64)
        means = np.asarray(gmm.means_, dtype=np.float64)
        covariances = np.asarray(gmm.covariances_, dtype=np.float64)
    scale_tril = []
    eye = np.eye(samples.shape[1], dtype=np.float64)
    for cov in covariances:
        if covariance_mode == "fixed_isotropic":
            if fixed_scale is None or float(fixed_scale) <= 0.0:
                raise ValueError("fixed_isotropic requires a positive fixed_scale.")
            scale_tril.append(eye * float(fixed_scale))
        else:
            cov = np.asarray(cov, dtype=np.float64) + float(covariance_floor) * eye
            scale_tril.append(np.linalg.cholesky(cov))
    return StudentTMixture(
        weights=weights,
        loc=means,
        scale_tril=np.asarray(scale_tril, dtype=np.float64),
        df=float(df),
        epsilon=0.0,
        covariance_mode=str(covariance_mode),
        center_mode=str(center_mode),
        scoring=str(scoring),
        requested_scale=None if fixed_scale is None else float(fixed_scale),
    )


def sample_student_t_mixture(proposal: StudentTMixture, size: int, rng: np.random.Generator) -> np.ndarray:
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
            chi2 = rng.chisquare(proposal.df, size=(n, dim))
        else:
            normal = rng.normal(size=(n, dim)) @ proposal.scale_tril[k].T
            chi2 = rng.chisquare(proposal.df, size=(n, 1))
        out[mask] = proposal.loc[k] + normal / np.sqrt(chi2 / proposal.df)
    return out


def student_t_logpdf(x: np.ndarray, loc: np.ndarray, scale_tril: np.ndarray, df: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    loc = np.asarray(loc, dtype=np.float64)
    dim = loc.size
    diff = x - loc.reshape(1, -1)
    solve = np.linalg.solve(scale_tril, diff.T).T
    quad = np.sum(solve * solve, axis=1)
    logdet = np.sum(np.log(np.diag(scale_tril)))
    return (
        math.lgamma((df + dim) / 2.0)
        - math.lgamma(df / 2.0)
        - 0.5 * dim * math.log(df * math.pi)
        - logdet
        - 0.5 * (df + dim) * np.log1p(quad / df)
    )


def proposal_logpdf(proposal: StudentTMixture, z: np.ndarray, *, epsilon: float) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    logs = []
    for weight, loc, scale_tril in zip(proposal.weights, proposal.loc, proposal.scale_tril):
        if proposal.scoring == "product_univariate_t":
            diagonal = np.diag(scale_tril)
            standardized = (z - loc.reshape(1, -1)) / diagonal.reshape(1, -1)
            df = float(proposal.df)
            log_norm = math.lgamma((df + 1.0) / 2.0) - math.lgamma(df / 2.0) - 0.5 * math.log(df * math.pi)
            component = np.sum(log_norm - np.log(diagonal) - 0.5 * (df + 1.0) * np.log1p(np.square(standardized) / df), axis=1)
        else:
            component = student_t_logpdf(z, loc, scale_tril, proposal.df)
        logs.append(math.log(float(weight)) + component)
    mix_log = logsumexp_np(np.stack(logs, axis=1), axis=1)
    prior = -0.5 * np.sum(z * z + math.log(2.0 * math.pi), axis=1)
    if epsilon <= 0:
        return mix_log
    return logsumexp_np(
        np.stack([math.log1p(-float(epsilon)) + mix_log, math.log(float(epsilon)) + prior], axis=1),
        axis=1,
    )


def sample_defensive_proposal(proposal: StudentTMixture, size: int, *, epsilon: float, rng: np.random.Generator) -> np.ndarray:
    dim = proposal.loc.shape[1]
    choose_prior = rng.uniform(size=int(size)) < float(epsilon)
    out = np.empty((int(size), dim), dtype=np.float64)
    if np.any(~choose_prior):
        out[~choose_prior] = sample_student_t_mixture(proposal, int((~choose_prior).sum()), rng)
    if np.any(choose_prior):
        out[choose_prior] = rng.normal(size=(int(choose_prior.sum()), dim))
    return out


def effective_sample_size_from_log_weights(log_weights: np.ndarray) -> float:
    log_weights = np.asarray(log_weights, dtype=np.float64)
    log_norm = log_weights - logsumexp_np(log_weights)
    weights = np.exp(log_norm)
    denom = float(np.sum(weights * weights))
    return float(1.0 / denom) if denom > 0 else 0.0


def effective_sample_size_from_chain(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    try:
        ess = tfp.mcmc.effective_sample_size(tf.convert_to_tensor(samples, dtype=tf.float32)).numpy()
        ess = np.asarray(ess, dtype=np.float64)
        if ess.ndim == 2:
            ess = np.sum(ess, axis=0)
        elif ess.ndim > 2:
            ess = np.sum(ess.reshape((-1, ess.shape[-1])), axis=0)
        return np.maximum(ess, 1.0)
    except Exception as exc:
        warnings.warn(f"TFP effective_sample_size failed ({exc}); using retained sample count.", RuntimeWarning)
        return np.full(samples.shape[-1], samples.reshape((-1, samples.shape[-1])).shape[0], dtype=np.float64)


def bridge_effective_sample_size(d_bridge: np.ndarray, *, fallback_ess: float, use_neff: bool) -> float:
    d_bridge = np.asarray(d_bridge, dtype=np.float64)
    n_bridge = int(d_bridge.shape[0])
    if n_bridge < 1:
        raise ValueError("At least one held-out posterior bridge sample is required.")
    if not bool(use_neff):
        return float(n_bridge)
    try:
        ess_per_dim = effective_sample_size_from_chain(d_bridge[:, None, :])
        ess_bridge = float(np.min(ess_per_dim))
    except Exception as exc:
        warnings.warn(f"Held-out posterior ESS failed ({exc}); using HMC fallback ESS.", RuntimeWarning)
        ess_bridge = float(fallback_ess)
    if not np.isfinite(ess_bridge) or ess_bridge <= 0.0:
        warnings.warn("Held-out posterior ESS was unavailable; using physical bridge count.", RuntimeWarning)
        return float(n_bridge)
    return float(min(max(ess_bridge, 1.0), float(n_bridge)))


def bridge_update(log_ratio_q: np.ndarray, log_ratio_p: np.ndarray, *, ess_bridge: float, proposal_n: int, tol: float, max_iter: int) -> tuple[float, int, bool, float]:
    log_z = float(logmeanexp_np(log_ratio_q))
    s_p = float(ess_bridge) / float(ess_bridge + proposal_n)
    s_q = float(proposal_n) / float(ess_bridge + proposal_n)
    log_s_p = math.log(max(s_p, 1.0e-300))
    log_s_q = math.log(max(s_q, 1.0e-300))
    last_delta = float("inf")
    for iteration in range(1, int(max_iter) + 1):
        den_q = logsumexp_np(np.stack([log_s_p + log_ratio_q, log_s_q + log_z + np.zeros_like(log_ratio_q)], axis=1), axis=1)
        den_p = logsumexp_np(np.stack([log_s_p + log_ratio_p, log_s_q + log_z + np.zeros_like(log_ratio_p)], axis=1), axis=1)
        next_log_z = float(logmeanexp_np(log_ratio_q - den_q) - logmeanexp_np(-den_p))
        last_delta = abs(next_log_z - log_z)
        log_z = next_log_z
        if last_delta < float(tol):
            return log_z, iteration, True, last_delta
    return log_z, int(max_iter), False, last_delta


def sample_hmc(
    model: ConditionalBGM,
    x: np.ndarray,
    y_label: int,
    *,
    settings: Mapping[str, Any],
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    tf.random.set_seed(int(seed))
    x_tf = tf.convert_to_tensor(np.asarray(x, dtype=np.float32).reshape(1, -1))
    y_tf = tf.convert_to_tensor(one_hot(np.asarray([y_label]), model.y_dim))
    num_chains = int(settings.get("num_chains", 4))
    initial_state_name = str(settings.get("initial_state", "encoder")).lower()
    initial_state_scale = float(settings.get("initial_state_scale", 0.1))

    burn_in = int(settings.get("burn_in", 800))
    requested_m = int(settings.get("M", 1600))
    step_size = float(settings.get("step_size", 0.003))
    num_leapfrog_steps = int(settings.get("num_leapfrog_steps", 10))
    target_accept_prob = float(settings.get("target_accept_prob", 0.75))
    min_effective_samples = float(settings.get("min_effective_samples", 0))
    max_retries = int(settings.get("max_retries", 0))
    retry_multiplier = float(settings.get("retry_multiplier", 1.5))
    strict_ess = bool(settings.get("strict_ess", False))
    if requested_m < 1:
        raise ValueError("hmc_settings.M must be positive.")
    if num_chains < 1:
        raise ValueError("hmc_settings.num_chains must be positive.")

    def make_initial_state(attempt_seed: int) -> tf.Tensor:
        if initial_state_name == "encoder":
            z0_mean = model.encode(x_tf, y_tf, training=False)
            z0_mean = tf.reshape(tf.cast(z0_mean, tf.float32), [1, model.z_dim])
            noise = initial_state_scale * tf.random.normal((num_chains, model.z_dim), dtype=tf.float32, seed=attempt_seed)
            return tf.repeat(z0_mean, repeats=num_chains, axis=0) + noise
        if initial_state_name == "zero":
            return initial_state_scale * tf.random.normal((num_chains, model.z_dim), dtype=tf.float32, seed=attempt_seed)
        return tf.random.normal((num_chains, model.z_dim), dtype=tf.float32, seed=attempt_seed)

    current_m = requested_m
    current_burn_in = burn_in
    attempts = 0
    last: tuple[np.ndarray, dict[str, Any]] | None = None
    while attempts <= max_retries:
        attempts += 1
        num_results = int(math.ceil(current_m / num_chains))
        initial_state = make_initial_state(int(seed) + 1009 * attempts)
        hmc_start = time.perf_counter()
        states, trace = model._run_hmc_chain(
            x_tf=x_tf,
            y_tf=y_tf,
            initial_state=initial_state,
            num_results=tf.constant(num_results, dtype=tf.int32),
            burn_in=tf.constant(current_burn_in, dtype=tf.int32),
            step_size=tf.constant(step_size, dtype=tf.float32),
            num_leapfrog_steps=tf.constant(num_leapfrog_steps, dtype=tf.int32),
            target_accept_prob=tf.constant(target_accept_prob, dtype=tf.float32),
            seed=tf.constant(int(seed) + attempts, dtype=tf.int32),
        )
        hmc_elapsed = time.perf_counter() - hmc_start
        states_np_chain = np.asarray(states.numpy(), dtype=np.float64)
        if not np.all(np.isfinite(states_np_chain)):
            raise FloatingPointError("Posterior HMC produced non-finite latent samples.")
        states_np = states_np_chain.reshape(-1, model.z_dim)
        accepted = np.asarray(trace.numpy(), dtype=bool)
        ess_per_dim = effective_sample_size_from_chain(states_np_chain)
        min_ess = float(np.min(ess_per_dim))
        mean_ess = float(np.mean(ess_per_dim))
        diagnostics = {
            "acceptance_rate": float(accepted.mean()),
            "samples": int(states_np.shape[0]),
            "hmc_min_ess": float(min_ess),
            "hmc_mean_ess": float(mean_ess),
            "hmc_ess_per_dim": ess_per_dim.astype(float).tolist(),
            "hmc_requested_samples": int(requested_m),
            "hmc_retained_samples": int(states_np.shape[0]),
            "hmc_num_chains": int(num_chains),
            "hmc_burn_in": int(current_burn_in),
            "hmc_step_size": float(step_size),
            "hmc_num_leapfrog_steps": int(num_leapfrog_steps),
            "hmc_attempts": int(attempts),
            "hmc_elapsed_seconds": float(hmc_elapsed),
        }
        last = (states_np, diagnostics)
        if min_effective_samples <= 0 or min_ess >= min_effective_samples:
            return states_np, diagnostics
        if attempts <= max_retries:
            current_m = int(math.ceil(current_m * retry_multiplier))
            current_burn_in = int(math.ceil(current_burn_in * retry_multiplier))
    if last is None:
        raise RuntimeError("HMC did not produce samples.")
    message = (
        "Posterior HMC effective sample size is below the configured threshold: "
        f"min_ess={last[1]['hmc_min_ess']:.2f}, threshold={min_effective_samples:.2f}."
    )
    if strict_ess:
        raise RuntimeError(message)
    warnings.warn(message, RuntimeWarning)
    return last


def evaluate_log_joint_np(model: ConditionalBGM, z: np.ndarray, x: np.ndarray, y_label: int, batch_size: int) -> np.ndarray:
    out = []
    x_np = np.asarray(x, dtype=np.float32).reshape(1, -1)
    z_np = np.asarray(z, dtype=np.float32)
    for start in range(0, len(z), int(batch_size)):
        z_part = z_np[start : start + int(batch_size)]
        batch_count = int(z_part.shape[0])
        z_batch = tf.convert_to_tensor(z_part, dtype=tf.float32)
        y_batch = tf.convert_to_tensor(one_hot(np.full(batch_count, y_label, dtype=np.int64), model.y_dim), dtype=tf.float32)
        x_batch = tf.convert_to_tensor(np.repeat(x_np, batch_count, axis=0), dtype=tf.float32)
        out.append(model._evaluate_log_joint_batch(z_batch, x_batch, y_batch).numpy())
    return np.concatenate(out, axis=0).astype(np.float64)


def estimate_point_bridge(
    model: ConditionalBGM,
    x: np.ndarray,
    y_label: int,
    *,
    K: int,
    S: int,
    nu: float,
    epsilon: float,
    hmc_settings: Mapping[str, Any],
    eval_batch_size: int,
    fit_fraction: float,
    bridge_tol: float,
    bridge_max_iter: int,
    covariance_floor: float,
    proposal_covariance_mode: str = "fitted_full",
    proposal_scale: float | None = None,
    proposal_center: str = "fitted_gmm",
    proposal_scoring: str = "multivariate_t",
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    posterior, hmc_diag = sample_hmc(model, x, y_label, settings=hmc_settings, seed=seed)
    if posterior.ndim != 2 or posterior.shape[0] < 2:
        raise ValueError("HMC posterior must have shape (M, z_dim) with at least two samples.")
    if not np.all(np.isfinite(posterior)):
        raise FloatingPointError("HMC posterior contains non-finite values before proposal fitting.")
    perm = rng.permutation(posterior.shape[0])
    posterior = posterior[perm]
    if not (0.0 < float(fit_fraction) < 1.0):
        raise ValueError("fit_fraction must be strictly between 0 and 1.")
    if posterior.shape[0] < 3:
        raise ValueError("At least three HMC posterior samples are required for fit/bridge splitting.")
    n_fit = max(2, min(posterior.shape[0] - 1, int(round(float(fit_fraction) * posterior.shape[0]))))
    fit_samples = posterior[:n_fit]
    bridge_samples = posterior[n_fit:]
    proposal = fit_student_t_mixture(
        fit_samples,
        n_components=int(K),
        df=float(nu),
        covariance_floor=float(covariance_floor),
        random_seed=int(seed),
        covariance_mode=str(proposal_covariance_mode),
        fixed_scale=proposal_scale,
        center_mode=str(proposal_center),
        scoring=str(proposal_scoring),
    )
    proposal_samples = sample_defensive_proposal(proposal, int(S), epsilon=float(epsilon), rng=rng)
    log_pi_q = evaluate_log_joint_np(model, proposal_samples, x, y_label, eval_batch_size)
    log_q_q = proposal_logpdf(proposal, proposal_samples, epsilon=float(epsilon))
    log_ratio_q = log_pi_q - log_q_q
    log_pi_p = evaluate_log_joint_np(model, bridge_samples, x, y_label, eval_batch_size)
    log_q_p = proposal_logpdf(proposal, bridge_samples, epsilon=float(epsilon))
    log_ratio_p = log_pi_p - log_q_p
    for name, values in (
        ("proposal conditional log ratios", log_ratio_q),
        ("held-out posterior conditional log ratios", log_ratio_p),
    ):
        if not np.all(np.isfinite(values)):
            raise FloatingPointError(f"Non-finite {name} in bridge estimation.")
    ess = bridge_effective_sample_size(
        bridge_samples,
        fallback_ess=float(hmc_diag.get("hmc_min_ess", bridge_samples.shape[0])),
        use_neff=bool(hmc_settings.get("use_neff", True)),
    )
    log_z, iterations, converged, delta = bridge_update(
        log_ratio_q,
        log_ratio_p,
        ess_bridge=max(ess, 1.0),
        proposal_n=int(S),
        tol=float(bridge_tol),
        max_iter=int(bridge_max_iter),
    )
    return {
        "log_px_std": float(log_z),
        "diagnostics": {
            **hmc_diag,
            "posterior_fit_samples": int(fit_samples.shape[0]),
            "posterior_bridge_samples": int(bridge_samples.shape[0]),
            "proposal_samples": int(S),
            "proposal_covariance_mode": proposal.covariance_mode,
            "proposal_center": proposal.center_mode,
            "proposal_scoring": proposal.scoring,
            "proposal_scale": proposal.requested_scale,
            "proposal_sampling": "multivariate_student_t_shared_chi_square",
            "proposal_scoring_matches_sampling": proposal.scoring == "multivariate_t",
            "proposal_ess_is": effective_sample_size_from_log_weights(log_ratio_q),
            "max_log_ratio_proposal": float(np.max(log_ratio_q)),
            "bridge_ess": float(ess),
            "bridge_iterations": int(iterations),
            "bridge_converged": bool(converged),
            "bridge_abs_delta": float(delta),
        },
    }
