"""Conditional convolutional MNIST BGM with a logistic-normal Bernoulli observation."""
from __future__ import annotations

import math
from typing import Any, Mapping

import tensorflow as tf

from bayesnde.estimators.conditional import ConditionalBGM


class ConditionalImageDiscriminator(tf.keras.Model):
    def __init__(self, n_classes: int = 10, filters: int = 32):
        super().__init__(name="conditional_image_discriminator_v2")
        self.n_classes = int(n_classes)
        self.conv1 = tf.keras.layers.Conv2D(filters, 5, 2, "same")
        self.conv2 = tf.keras.layers.Conv2D(filters * 2, 5, 2, "same")
        self.conv3 = tf.keras.layers.Conv2D(filters * 4, 3, 2, "same")
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(128)
        self.out = tf.keras.layers.Dense(1)

    def call(self, inputs, training: bool = True):
        del training
        x, y = inputs
        x = tf.reshape(tf.cast(x, tf.float32), [-1, 28, 28, 1])
        y = tf.cast(y, tf.float32)
        label_map = tf.tile(y[:, None, None, :], [1, 28, 28, 1])
        h = tf.concat([x, label_map], axis=-1)
        h = tf.nn.leaky_relu(self.conv1(h), alpha=0.2)
        h = tf.nn.leaky_relu(self.conv2(h), alpha=0.2)
        h = tf.nn.leaky_relu(self.conv3(h), alpha=0.2)
        h = tf.concat([self.flatten(h), y], axis=-1)
        return self.out(tf.nn.leaky_relu(self.hidden(h), alpha=0.2))


class LogisticNormalGenerator(tf.keras.Model):
    def __init__(self, z_dim: int = 10, n_classes: int = 10, filters: int = 32,
                 variance_floor: float = 1.0e-4, variance_ceiling: float = 4.0):
        super().__init__(name="logistic_normal_generator")
        self.z_dim = int(z_dim)
        self.n_classes = int(n_classes)
        self.variance_floor = float(variance_floor)
        self.variance_ceiling = float(variance_ceiling)
        self.fc = tf.keras.layers.Dense(7 * 7 * filters * 4)
        self.reshape = tf.keras.layers.Reshape((7, 7, filters * 4))
        self.up1 = tf.keras.layers.Conv2DTranspose(filters * 2, 3, 2, "same", use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.up2 = tf.keras.layers.Conv2DTranspose(filters, 3, 2, "same", use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.refine = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.mean_head = tf.keras.layers.Conv2D(1, 1, padding="same", name="logit_mean_head")
        self.variance_head = tf.keras.layers.Conv2D(1, 1, padding="same", name="logit_variance_raw")

    def call(self, inputs, training: bool = True):
        z, y = inputs
        y = tf.cast(y, tf.float32)
        h = tf.nn.leaky_relu(self.fc(tf.concat([tf.cast(z, tf.float32), y], axis=-1)), alpha=0.2)
        h = self.reshape(h)
        h = tf.concat([h, tf.tile(y[:, None, None, :], [1, 7, 7, 1])], axis=-1)
        h = tf.nn.leaky_relu(self.bn1(self.up1(h), training=training), alpha=0.2)
        h = tf.nn.leaky_relu(self.bn2(self.up2(h), training=training), alpha=0.2)
        h = tf.nn.leaky_relu(self.bn3(self.refine(h), training=training), alpha=0.2)
        logits = tf.reshape(self.mean_head(h), [-1, 784])
        raw = tf.reshape(self.variance_head(h), [-1, 784])
        unit_scale = 1.0 - tf.exp(-tf.nn.softplus(raw))
        variance = self.variance_floor + (self.variance_ceiling - self.variance_floor) * unit_scale
        return logits, variance

    @staticmethod
    def reparameterize(mean, var):
        logits = mean + tf.random.normal(tf.shape(mean), dtype=mean.dtype) * tf.sqrt(var)
        return tf.nn.sigmoid(logits)


class ConditionalMNISTEncoder(tf.keras.Model):
    def __init__(self, z_dim: int = 10, n_classes: int = 10, filters: int = 32):
        super().__init__(name="conditional_mnist_encoder")
        self.n_classes = int(n_classes)
        self.conv1 = tf.keras.layers.Conv2D(filters, 3, 2, "same", use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters * 2, 3, 2, "same", use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters * 4, 3, padding="same", use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(256)
        self.out = tf.keras.layers.Dense(z_dim)

    def call(self, x, label_features=None, training: bool = True):
        x = tf.reshape(tf.cast(x, tf.float32), [-1, 28, 28, 1])
        y = tf.cast(label_features, tf.float32)
        ymap = tf.tile(y[:, None, None, :], [1, 28, 28, 1])
        h = tf.concat([x, ymap], axis=-1)
        h = tf.nn.leaky_relu(self.bn1(self.conv1(h), training=training), alpha=0.2)
        h = tf.nn.leaky_relu(self.bn2(self.conv2(h), training=training), alpha=0.2)
        h = tf.nn.leaky_relu(self.bn3(self.conv3(h), training=training), alpha=0.2)
        h = tf.concat([self.flatten(h), y], axis=-1)
        return self.out(tf.nn.leaky_relu(self.hidden(h), alpha=0.2))


class ConditionalMNISTBGM(ConditionalBGM):
    likelihood_name = "logistic_normal_bernoulli_approx"

    def __init__(self, params: Mapping[str, Any]):
        values = dict(params)
        super().__init__(784, 10, int(values.get("z_dim", 10)), [32], [32],
                         float(values.get("variance_floor", 1.0e-4)), values)
        self.g_net = LogisticNormalGenerator(
            self.z_dim, self.y_dim, int(values.get("filters", 32)),
            float(values.get("variance_floor", 1.0e-4)),
            float(values.get("logit_variance_ceiling", 4.0)))
        self.e_net = ConditionalMNISTEncoder(
            self.z_dim, self.y_dim, int(values.get("filters", 32)))
        self.dx_net = ConditionalImageDiscriminator(
            self.y_dim, int(values.get("discriminator_filters", 32)))
        self._rebuild_checkpoint()

    def _rebuild_checkpoint(self):
        self.ckpt = tf.train.Checkpoint(
            g_net=self.g_net, e_net=self.e_net, dz_net=self.dz_net, dx_net=self.dx_net,
            g_pre_optimizer=self.g_pre_optimizer, d_pre_optimizer=self.d_pre_optimizer,
            g_optimizer=self.g_optimizer, posterior_optimizer=self.posterior_optimizer)

    def build(self) -> None:
        z = tf.zeros((1, self.z_dim), tf.float32)
        y = tf.zeros((1, self.y_dim), tf.float32)
        x = tf.zeros((1, self.x_dim), tf.float32)
        label_features = self._label_features(y, training=False)
        self.g_net((z, label_features), training=False)
        self.e_net(x, label_features, training=False)
        self.dz_net(z, training=False)
        self.dx_net((x, y), training=False)

    def _decoder_nll_from_parts(self, data_x, mu_x, sigma_square_x, low_rank_u=None):
        if low_rank_u is not None:
            raise ValueError("The MNIST observation model is diagonal, not low-rank.")
        return -self.observation_log_likelihood(data_x, mu_x, sigma_square_x)

    @staticmethod
    def effective_logits(mean, variance):
        return tf.cast(mean, tf.float32) / tf.sqrt(1.0 + math.pi * tf.cast(variance, tf.float32) / 8.0)

    def observation_log_likelihood(self, x, mean, variance):
        logits = self.effective_logits(mean, variance)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(x, tf.float32), logits=logits)
        return -tf.reduce_sum(loss, axis=-1)

    def _display_mean(self, first, variance):
        return tf.nn.sigmoid(self.effective_logits(first, variance))

    @tf.function
    def update_g_net(self, data_z, data_x, data_y):
        with tf.GradientTape() as tape:
            first, variance, _ = self._decode_generator(data_z, data_y, training=True)
            per_pixel_nll = tf.reduce_mean(self._decoder_nll_from_parts(data_x, first, variance)) / 784.0
            target = tf.cast(float(self.params.get("variance_target", 0.01)), variance.dtype)
            variance_penalty = tf.reduce_mean(tf.square(tf.math.log(variance) - tf.math.log(target)))
            prior_z = tf.random.normal(tf.shape(data_z), dtype=data_z.dtype)
            prior_first, prior_variance, _ = self._decode_generator(prior_z, data_y, training=True)
            prior_x = self._display_mean(prior_first, prior_variance)
            prior_score = self.dx_net((prior_x, data_y), training=False)
            prior_adv = tf.reduce_mean(tf.square(0.9 - prior_score))
            recovered_z = self.encode(prior_x, data_y, training=False)
            prior_cycle = tf.reduce_mean(tf.square(prior_z - recovered_z))
            loss = (per_pixel_nll
                    + tf.cast(self.params.get("variance_penalty_weight", 0.1), variance.dtype) * variance_penalty
                    + tf.cast(self.params.get("prior_adv_weight", 0.1), variance.dtype) * prior_adv
                    + tf.cast(self.params.get("prior_cycle_weight", 0.1), variance.dtype) * prior_cycle)
        variables = self._decoder_trainable_variables()
        gradients = tape.gradient(loss, variables)
        self.g_optimizer.apply_gradients([(g, v) for g, v in zip(gradients, variables) if g is not None])
        return loss

    @tf.function
    def train_disc_step(self, data_z, data_x, data_y):
        with tf.GradientTape(persistent=True) as disc_tape:
            with tf.GradientTape() as gpz_tape:
                encoded = self.encode(data_x, data_y, training=True)
                mixed_z = 0.5 * data_z + 0.5 * encoded
                mixed_z_score = self.dz_net(mixed_z, training=True)
            with tf.GradientTape() as gpx_tape:
                first, variance, _ = self._decode_generator(data_z, data_y, training=True)
                generated = self._sample_decoder(first, variance)
                mixed_x = 0.5 * tf.cast(data_x, generated.dtype) + 0.5 * generated
                mixed_x_score = self.dx_net((mixed_x, data_y), training=True)
            real_z_score = self.dz_net(data_z, training=True)
            encoded_score = self.dz_net(encoded, training=True)
            real_x_score = self.dx_net((data_x, data_y), training=True)
            generated_score = self.dx_net((generated, data_y), training=True)
            dz_loss = 0.5 * (tf.reduce_mean(tf.square(0.9 - real_z_score)) + tf.reduce_mean(tf.square(0.1 - encoded_score)))
            dx_loss = 0.5 * (tf.reduce_mean(tf.square(0.9 - real_x_score)) + tf.reduce_mean(tf.square(0.1 - generated_score)))
            grad_z = gpz_tape.gradient(mixed_z_score, mixed_z)
            grad_x = gpx_tape.gradient(mixed_x_score, mixed_x)
            gpz = tf.reduce_mean(tf.square(tf.norm(grad_z, axis=1) - 1.0))
            gpx = tf.reduce_mean(tf.square(tf.norm(tf.reshape(grad_x, [tf.shape(grad_x)[0], -1]), axis=1) - 1.0))
            loss = dx_loss + dz_loss + tf.cast(self.params.get("gamma", 0.0), dx_loss.dtype) * (gpz + gpx)
        variables = self.dz_net.trainable_variables + self.dx_net.trainable_variables
        gradients = disc_tape.gradient(loss, variables)
        self.d_pre_optimizer.apply_gradients([(g, v) for g, v in zip(gradients, variables) if g is not None])
        return dz_loss, dx_loss, loss

    @tf.function
    def train_gen_step(self, data_z, data_x, data_y):
        with tf.GradientTape() as tape:
            first, variance, _ = self._decode_generator(data_z, data_y, training=True)
            generated = self._sample_decoder(first, variance)
            encoded = self.encode(data_x, data_y, training=True)
            recovered_z = self.encode(generated, data_y, training=True)
            rec_first, rec_variance, _ = self._decode_generator(encoded, data_y, training=True)
            reconstructed = self._display_mean(rec_first, rec_variance)
            x_adv = tf.reduce_mean(tf.square(0.9 - self.dx_net((generated, data_y), training=False)))
            z_adv = tf.reduce_mean(tf.square(0.9 - self.dz_net(encoded, training=False)))
            x_cycle = tf.reduce_mean(tf.square(tf.cast(data_x, reconstructed.dtype) - reconstructed))
            z_cycle = tf.reduce_mean(tf.square(data_z - recovered_z))
            variance_reg = tf.reduce_mean(tf.square(variance))
            loss = x_adv + z_adv + 10.0 * x_cycle + 10.0 * z_cycle
            loss += tf.cast(self.params.get("egm_variance_weight", 0.01), loss.dtype) * variance_reg
        variables = self._joint_generator_encoder_variables()
        gradients = tape.gradient(loss, variables)
        self.g_pre_optimizer.apply_gradients([(g, v) for g, v in zip(gradients, variables) if g is not None])
        return x_adv, z_adv, z_cycle, x_cycle, variance_reg, loss
