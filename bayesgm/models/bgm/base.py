import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfm = tfp.mcmc

from ..networks import (
    BaseFullyConnectedNet,
    Discriminator,
    BayesianVariationalNet,
    BaseVariationalNet,
)
import numpy as np
from bayesgm.datasets import Gaussian_sampler, Base_sampler
import dateutil.tz
import datetime
import os
import inspect
from tqdm import tqdm


def batch_correlation_matrix(values, eps=1.0e-6):
    values = tf.convert_to_tensor(values)
    dtype = values.dtype
    centered = values - tf.reduce_mean(values, axis=0, keepdims=True)
    variance = tf.reduce_mean(tf.square(centered), axis=0)
    standardized = centered / tf.sqrt(variance + tf.cast(eps, dtype))
    batch_size = tf.cast(tf.maximum(tf.shape(values)[0], 1), dtype)
    return tf.matmul(standardized, standardized, transpose_a=True) / batch_size


def offdiag_correlation_loss(values, eps=1.0e-6):
    corr = batch_correlation_matrix(values, eps=eps)
    dtype = corr.dtype
    dim = tf.shape(corr)[0]
    offdiag = corr * (1.0 - tf.eye(dim, dtype=dtype))
    return tf.reduce_sum(tf.square(offdiag))


def correlation_alignment_loss(generated_values, target_values, eps=1.0e-6):
    generated_corr = batch_correlation_matrix(generated_values, eps=eps)
    target_corr = batch_correlation_matrix(target_values, eps=eps)
    return tf.reduce_mean(tf.square(generated_corr - tf.stop_gradient(target_corr)))


def make_bgm_optimizer(params, learning_rate, *, beta_1, beta_2):
    optimizer_name = str(params.get("optimizer", "adam")).lower()
    weight_decay = float(params.get("weight_decay", 0.0))
    if optimizer_name == "adam":
        return tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
        )
    if optimizer_name == "adamw":
        adamw_cls = getattr(tf.keras.optimizers, "AdamW", None)
        if adamw_cls is None and hasattr(tf.keras.optimizers, "experimental"):
            adamw_cls = getattr(tf.keras.optimizers.experimental, "AdamW", None)
        if adamw_cls is None:
            raise ValueError("optimizer='adamw' requires AdamW in the active TensorFlow build.")
        kwargs = {}
        if "jit_compile" in inspect.signature(adamw_cls.__init__).parameters:
            kwargs["jit_compile"] = bool(params.get("optimizer_jit_compile", False))
        return adamw_cls(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            weight_decay=weight_decay,
            **kwargs,
        )
    raise ValueError("Unsupported optimizer={!r}; expected 'adam' or 'adamw'.".format(optimizer_name))


def iterative_dimension_scale(params):
    mode = str(params.get("iterative_dimension_scaling", "none")).lower()
    valid_modes = {"none", "gradient_only", "balance_to_reference"}
    if mode not in valid_modes:
        raise ValueError(
            "iterative_dimension_scaling must be one of {}.".format(sorted(valid_modes))
        )
    if mode == "none":
        return mode, 1.0

    x_dim = int(params["x_dim"])
    reference_dim = int(params.get("iterative_reference_dim", 10))
    if x_dim < 1 or reference_dim < 1:
        raise ValueError("x_dim and iterative_reference_dim must be positive.")
    return mode, float(reference_dim) / float(x_dim)


class BGM(object):
    def __init__(self, params, timestamp=None, random_seed=None):
        super(BGM, self).__init__()
        self.params = params
        self.timestamp = timestamp
        if random_seed is not None:
            tf.keras.utils.set_random_seed(random_seed)
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
            tf.config.experimental.enable_op_determinism()
        if self.params['use_bnn']:
            self.g_net = BayesianVariationalNet(input_dim=params['z_dim'],output_dim = params['x_dim'], 
                                           model_name='g_net', nb_units=params['g_units'])
        else:
            self.g_net = BaseVariationalNet(input_dim=params['z_dim'],output_dim = params['x_dim'], 
                                           model_name='g_net', nb_units=params['g_units'])

        self.e_net = BaseFullyConnectedNet(input_dim=params['x_dim'],output_dim = params['z_dim'], 
                                        model_name='e_net', nb_units=params['e_units'])
            
        self.dz_net = Discriminator(input_dim=params['z_dim'],model_name='dz_net',
                                        nb_units=params['dz_units'])
        self.dx_net = Discriminator(input_dim=params['x_dim'],model_name='dx_net',
                                        nb_units=params['dx_units'])

        self.g_pre_optimizer = make_bgm_optimizer(params, params['lr'], beta_1=0.5, beta_2=0.9)
        self.d_pre_optimizer = make_bgm_optimizer(params, params['lr'], beta_1=0.5, beta_2=0.9)
        self.z_sampler = Gaussian_sampler(mean=np.zeros(params['z_dim']), sd=1.0)

        self.g_optimizer = make_bgm_optimizer(params, params['lr_theta'], beta_1=0.9, beta_2=0.99)
        self.posterior_optimizer = make_bgm_optimizer(params, params['lr_z'], beta_1=0.9, beta_2=0.99)
        
        self.initialize_nets()
        if self.timestamp is None:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            self.timestamp = now.strftime('%Y%m%d_%H%M%S')
        
        self.checkpoint_path = "{}/checkpoints/{}/{}".format(
            params['output_dir'], params['dataset'], self.timestamp)

        if self.params['save_model'] and not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        self.save_dir = "{}/results/{}/{}".format(
            params['output_dir'], params['dataset'], self.timestamp)

        if self.params['save_res'] and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)   

        self.ckpt = tf.train.Checkpoint(g_net = self.g_net,
                                    e_net = self.e_net,
                                    dz_net = self.dz_net,
                                    dx_net = self.dx_net,
                                    g_pre_optimizer = self.g_pre_optimizer,
                                    d_pre_optimizer = self.d_pre_optimizer,
                                    g_optimizer = self.g_optimizer,
                                    posterior_optimizer = self.posterior_optimizer)
        
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=100)                 

        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!') 

    def get_config(self):
        return {
                "params": self.params,
        }

    def initialize_nets(self, print_summary = False):
        self.g_net(np.zeros((1, self.params['z_dim'])))
        if print_summary:
            print(self.g_net.summary())

    def _decode_generator(self, data_z, training=True):
        outputs = self.g_net(data_z, training=training)
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
            raise ValueError("BGM generator must return a (mu, var) tuple.")
        mu_x, sigma_square_x = outputs
        return mu_x, sigma_square_x

    def _sample_decoder(self, mu_x, sigma_square_x):
        return self.g_net.reparameterize(mu_x, sigma_square_x)

    def _decoder_nll_from_parts(self, data_x, mu_x, sigma_square_x, obs_mask=None):
        ll_term = ((data_x - mu_x) ** 2) / (2 * sigma_square_x) + 0.5 * tf.math.log(sigma_square_x)
        if obs_mask is not None:
            ll_term = ll_term * tf.cast(obs_mask, ll_term.dtype)
        return tf.reduce_sum(ll_term, axis=1)

    def _decoder_nll(self, data_z, data_x, obs_mask=None, training=True):
        mu_x, sigma_square_x = self._decode_generator(data_z, training=training)
        return self._decoder_nll_from_parts(data_x, mu_x, sigma_square_x, obs_mask=obs_mask)

    def _variance_log_target_loss(self, sigma_square_x):
        weight = float(self.params.get('variance_log_target_weight', 0.0))
        if weight <= 0.0:
            return tf.constant(0.0, dtype=sigma_square_x.dtype)
        dtype = sigma_square_x.dtype
        eps = tf.cast(float(self.params.get('variance_log_eps', 1.0e-8)), dtype)
        target = tf.cast(
            float(self.params.get('variance_log_target', self.params.get('variance_target', 1.0e-2))),
            dtype,
        )
        safe_var = tf.maximum(sigma_square_x, eps)
        safe_target = tf.maximum(target, eps)
        return tf.reduce_mean(tf.square(tf.math.log(safe_var + eps) - tf.math.log(safe_target)))

    @tf.function
    def update_g_net(self, data_z, data_x):
        with tf.GradientTape() as gen_tape:
            mu_x, sigma_square_x = self._decode_generator(data_z)
            decoder_loss = self._decoder_nll_from_parts(data_x, mu_x, sigma_square_x)
            decoder_loss = tf.reduce_mean(decoder_loss)
            regularization_loss = tf.constant(0.0, dtype=decoder_loss.dtype)
            variance_log_target_weight = tf.cast(
                self.params.get('variance_log_target_weight', 0.0), decoder_loss.dtype
            )
            regularization_loss += variance_log_target_weight * self._variance_log_target_loss(sigma_square_x)

            variance_lower_bound = float(self.params.get('variance_lower_bound', 0.0))
            variance_lower_bound_weight = float(self.params.get('variance_lower_bound_weight', 0.0))
            if variance_lower_bound > 0.0 and variance_lower_bound_weight > 0.0:
                dtype = sigma_square_x.dtype
                lower_bound = tf.cast(variance_lower_bound, dtype)
                safe_sigma_square_x = tf.maximum(sigma_square_x, tf.cast(1.0e-12, dtype))
                log_gap = tf.nn.relu(tf.math.log(lower_bound) - tf.math.log(safe_sigma_square_x))
                variance_lower_bound_loss = tf.reduce_mean(tf.square(log_gap))
                regularization_loss += tf.cast(
                    variance_lower_bound_weight, decoder_loss.dtype
                ) * variance_lower_bound_loss
            
            if self.params['use_bnn']:
                loss_kl = sum(self.g_net.losses)
                regularization_loss += loss_kl * self.params['kl_weight']

            iterative_prior_mmd_weight = float(
                self.params.get('iterative_prior_mmd_weight', 0.0)
            )
            iterative_prior_moment_weight = float(
                self.params.get('iterative_prior_moment_weight', 0.0)
            )
            iterative_prior_covariance_weight = float(
                self.params.get('iterative_prior_covariance_weight', 0.0)
            )
            if (
                iterative_prior_mmd_weight > 0.0
                or iterative_prior_moment_weight > 0.0
                or iterative_prior_covariance_weight > 0.0
            ):
                prior_z = tf.random.normal(
                    shape=tf.stack([tf.shape(data_x)[0], tf.shape(data_z)[1]]),
                    dtype=data_x.dtype,
                )
                prior_mu_x, prior_sigma_square_x = self._decode_generator(prior_z)
                prior_x = self._sample_decoder(prior_mu_x, prior_sigma_square_x)
                real_mean = tf.reduce_mean(data_x, axis=0)
                prior_mean = tf.reduce_mean(prior_x, axis=0)
                real_var = tf.math.reduce_variance(data_x, axis=0)
                prior_var = tf.math.reduce_variance(prior_x, axis=0)
                if iterative_prior_moment_weight > 0.0:
                    prior_moment_loss = tf.reduce_mean(tf.square(real_mean - prior_mean))
                    prior_moment_loss += tf.reduce_mean(tf.square(real_var - prior_var))
                    regularization_loss += tf.cast(
                        iterative_prior_moment_weight,
                        decoder_loss.dtype,
                    ) * prior_moment_loss

                if iterative_prior_covariance_weight > 0.0:
                    denominator = tf.cast(tf.shape(data_x)[0] - 1, data_x.dtype)
                    real_centered = data_x - real_mean
                    prior_centered = prior_x - prior_mean
                    real_cov = tf.matmul(real_centered, real_centered, transpose_a=True) / denominator
                    prior_cov = tf.matmul(prior_centered, prior_centered, transpose_a=True) / denominator
                    prior_covariance_loss = tf.reduce_mean(tf.square(real_cov - prior_cov))
                    regularization_loss += tf.cast(
                        iterative_prior_covariance_weight,
                        decoder_loss.dtype,
                    ) * prior_covariance_loss

                if iterative_prior_mmd_weight > 0.0:
                    mmd_scales = tf.constant(
                        self.params.get('mmd_scales', [0.05, 0.1, 0.2, 0.5, 1.0]),
                        dtype=data_x.dtype,
                    )
                    real_pairwise = tf.expand_dims(data_x, axis=0) - tf.expand_dims(data_x, axis=1)
                    prior_pairwise = tf.expand_dims(prior_x, axis=0) - tf.expand_dims(prior_x, axis=1)
                    cross_pairwise = tf.expand_dims(data_x, axis=1) - tf.expand_dims(prior_x, axis=0)
                    prior_mmd_loss = tf.constant(0.0, dtype=data_x.dtype)
                    for scale in tf.unstack(mmd_scales):
                        bandwidth = tf.maximum(scale, tf.constant(1.0e-6, dtype=data_x.dtype))
                        kernel_scale = 2.0 * tf.square(bandwidth)
                        k_real = tf.exp(-tf.square(real_pairwise) / kernel_scale)
                        k_prior = tf.exp(-tf.square(prior_pairwise) / kernel_scale)
                        k_cross = tf.exp(-tf.square(cross_pairwise) / kernel_scale)
                        prior_mmd_loss += tf.reduce_mean(k_real + k_prior - 2.0 * k_cross)
                    prior_mmd_loss /= tf.cast(tf.size(mmd_scales), data_x.dtype)
                    regularization_loss += tf.cast(
                        iterative_prior_mmd_weight,
                        decoder_loss.dtype,
                    ) * prior_mmd_loss

            scaling_mode, scaling_value = iterative_dimension_scale(self.params)
            scale = tf.cast(scaling_value, decoder_loss.dtype)
            if scaling_mode == 'gradient_only':
                loss_x = scale * (decoder_loss + regularization_loss)
            elif scaling_mode == 'balance_to_reference':
                loss_x = scale * decoder_loss + regularization_loss
            else:
                loss_x = decoder_loss + regularization_loss

        g_gradients = gen_tape.gradient(loss_x, self.g_net.trainable_variables)
        
        self.g_optimizer.apply_gradients(zip(g_gradients, self.g_net.trainable_variables))
        return loss_x
        
    @tf.function
    def update_latent_variable_sgd(self, data_z, data_x):
        with tf.GradientTape() as tape:
            
            loss_px_z = self._decoder_nll(data_z, data_x)
            loss_px_z = tf.reduce_mean(loss_px_z)

            loss_prior_z =  tf.reduce_sum(data_z**2, axis=1)/2
            loss_prior_z = tf.reduce_mean(loss_prior_z)

            latent_regularization_loss = tf.constant(0.0, dtype=loss_px_z.dtype)
            iterative_latent_prior_mmd_weight = float(
                self.params.get('iterative_latent_prior_mmd_weight', 0.0)
            )
            iterative_latent_prior_moment_weight = float(
                self.params.get('iterative_latent_prior_moment_weight', 0.0)
            )
            iterative_latent_prior_covariance_weight = float(
                self.params.get('iterative_latent_prior_covariance_weight', 0.0)
            )
            if (
                iterative_latent_prior_mmd_weight > 0.0
                or iterative_latent_prior_moment_weight > 0.0
                or iterative_latent_prior_covariance_weight > 0.0
            ):
                prior_z = tf.random.normal(shape=tf.shape(data_z), dtype=data_z.dtype)
                z_mean = tf.reduce_mean(data_z, axis=0)
                z_var = tf.math.reduce_variance(data_z, axis=0)

                if iterative_latent_prior_moment_weight > 0.0:
                    latent_moment_loss = tf.reduce_mean(tf.square(z_mean))
                    latent_moment_loss += tf.reduce_mean(tf.square(z_var - 1.0))
                    latent_regularization_loss += tf.cast(
                        iterative_latent_prior_moment_weight,
                        data_z.dtype,
                    ) * latent_moment_loss

                if iterative_latent_prior_covariance_weight > 0.0:
                    batch_size = tf.cast(tf.maximum(tf.shape(data_z)[0] - 1, 1), data_z.dtype)
                    centered_z = data_z - z_mean
                    z_covariance = tf.matmul(centered_z, centered_z, transpose_a=True) / batch_size
                    identity = tf.eye(tf.shape(data_z)[1], dtype=data_z.dtype)
                    latent_covariance_loss = tf.reduce_mean(
                        tf.square(z_covariance - identity)
                    )
                    latent_regularization_loss += tf.cast(
                        iterative_latent_prior_covariance_weight,
                        data_z.dtype,
                    ) * latent_covariance_loss

                if iterative_latent_prior_mmd_weight > 0.0:
                    latent_mmd_scales = tf.constant(
                        self.params.get(
                            'iterative_latent_mmd_scales',
                            [0.1, 0.25, 0.5, 1.0, 2.0, 4.0],
                        ),
                        dtype=data_z.dtype,
                    )
                    z_pairwise_sq = tf.reduce_sum(
                        tf.square(tf.expand_dims(data_z, axis=0) - tf.expand_dims(data_z, axis=1)),
                        axis=-1,
                    )
                    prior_pairwise_sq = tf.reduce_sum(
                        tf.square(tf.expand_dims(prior_z, axis=0) - tf.expand_dims(prior_z, axis=1)),
                        axis=-1,
                    )
                    cross_pairwise_sq = tf.reduce_sum(
                        tf.square(tf.expand_dims(data_z, axis=0) - tf.expand_dims(prior_z, axis=1)),
                        axis=-1,
                    )
                    latent_mmd_loss = tf.constant(0.0, dtype=data_z.dtype)
                    for kernel_scale in tf.unstack(latent_mmd_scales):
                        latent_mmd_loss += tf.reduce_mean(
                            tf.exp(-z_pairwise_sq / kernel_scale)
                            + tf.exp(-prior_pairwise_sq / kernel_scale)
                            - 2.0 * tf.exp(-cross_pairwise_sq / kernel_scale)
                        )
                    latent_mmd_loss /= tf.cast(tf.size(latent_mmd_scales), data_z.dtype)
                    latent_regularization_loss += tf.cast(
                        iterative_latent_prior_mmd_weight,
                        data_z.dtype,
                    ) * latent_mmd_loss

            scaling_mode, scaling_value = iterative_dimension_scale(self.params)
            scale = tf.cast(scaling_value, loss_px_z.dtype)
            if scaling_mode == 'gradient_only':
                loss_postrior_z = scale * (loss_px_z + loss_prior_z + latent_regularization_loss)
            elif scaling_mode == 'balance_to_reference':
                loss_postrior_z = scale * loss_px_z + loss_prior_z + latent_regularization_loss
            else:
                loss_postrior_z = loss_px_z + loss_prior_z + latent_regularization_loss

        posterior_gradients = tape.gradient(loss_postrior_z, [data_z])
        self.posterior_optimizer.apply_gradients(zip(posterior_gradients, [data_z]))
        return loss_postrior_z
    
    @tf.function
    def train_disc_step(self, data_z, data_x):
        epsilon_z = tf.random.uniform([],minval=0., maxval=1.)
        epsilon_x = tf.random.uniform([],minval=0., maxval=1.)
        with tf.GradientTape(persistent=True) as disc_tape:
            with tf.GradientTape() as gpz_tape:
                data_z_ = self.e_net(data_x)
                data_z_hat = data_z*epsilon_z + data_z_*(1-epsilon_z)
                data_dz_hat = self.dz_net(data_z_hat)
            with tf.GradientTape() as gpx_tape:
                mu_x_, sigma_square_x_ = self._decode_generator(data_z)
                data_x_sample_ = self._sample_decoder(mu_x_, sigma_square_x_)
                data_x_ = mu_x_ if self.params.get('x_adv_use_mean', False) else data_x_sample_
                data_x_hat = data_x*epsilon_x + data_x_*(1-epsilon_x)
                data_dx_hat = self.dx_net(data_x_hat)
            
            data_dx_ = self.dx_net(data_x_)
            data_dz_ = self.dz_net(data_z_)
            
            data_dx = self.dx_net(data_x)
            data_dz = self.dz_net(data_z)
            
            dz_loss = (tf.reduce_mean((0.9*tf.ones_like(data_dz) - data_dz)**2) \
                +tf.reduce_mean((0.1*tf.ones_like(data_dz_) - data_dz_)**2))/2.0
            dx_loss = (tf.reduce_mean((0.9*tf.ones_like(data_dx) - data_dx)**2) \
                +tf.reduce_mean((0.1*tf.ones_like(data_dx_) - data_dx_)**2))/2.0
            
            grad_z = gpz_tape.gradient(data_dz_hat, data_z_hat)
            grad_norm_z = tf.sqrt(tf.reduce_sum(tf.square(grad_z), axis=1))
            gpz_loss = tf.reduce_mean(tf.square(grad_norm_z - 1.0))
            
            grad_x = gpx_tape.gradient(data_dx_hat, data_x_hat)
            grad_norm_x = tf.sqrt(tf.reduce_sum(tf.square(grad_x), axis=1))
            gpx_loss = tf.reduce_mean(tf.square(grad_norm_x - 1.0))
                
            d_loss = dx_loss + dz_loss + \
                    self.params['gamma']*(gpz_loss + gpx_loss)


        d_gradients = disc_tape.gradient(d_loss, self.dz_net.trainable_variables+self.dx_net.trainable_variables)
        
        self.d_pre_optimizer.apply_gradients(zip(d_gradients, self.dz_net.trainable_variables+self.dx_net.trainable_variables))
        
        return dz_loss, dx_loss, d_loss
    
    @tf.function
    def train_gen_step(self, data_z, data_x):
        with tf.GradientTape(persistent=True) as gen_tape:
            mu_x_, sigma_square_x_ = self._decode_generator(data_z)
            data_x_sample_ = self._sample_decoder(mu_x_, sigma_square_x_)
            data_x_ = mu_x_ if self.params.get('x_adv_use_mean', False) else data_x_sample_
            reg_loss = tf.reduce_mean(tf.square(sigma_square_x_))
            data_z_ = self.e_net(data_x)

            z_cycle_input = mu_x_ if self.params.get('z_cycle_use_mean', False) else data_x_sample_
            data_z__= self.e_net(z_cycle_input)
            mu_x__, sigma_square_x__ = self._decode_generator(data_z_)
            data_x__ = self._sample_decoder(mu_x__, sigma_square_x__)
            x_cycle_reconstruction = (
                mu_x__ if self.params.get('x_cycle_use_mean', False) else data_x__
            )
            
            data_dx_ = self.dx_net(data_x_)
            data_dz_ = self.dz_net(data_z_)
            
            l2_loss_x = tf.reduce_mean((data_x - x_cycle_reconstruction)**2)
            l2_loss_z = tf.reduce_mean((data_z - data_z__)**2)
            
            g_loss_adv = tf.reduce_mean((0.9*tf.ones_like(data_dx_)  - data_dx_)**2)
            e_loss_adv = tf.reduce_mean((0.9*tf.ones_like(data_dz_)  - data_dz_)**2)

            cycle_weight = tf.cast(self.params.get('cycle_weight', 10.0), g_loss_adv.dtype)
            x_cycle_weight = tf.cast(self.params.get('x_cycle_weight', cycle_weight), g_loss_adv.dtype)
            z_cycle_weight = tf.cast(self.params.get('z_cycle_weight', cycle_weight), g_loss_adv.dtype)
            variance_target_weight = tf.cast(
                self.params.get('variance_target_weight', 0.0), g_loss_adv.dtype
            )
            variance_target = tf.cast(
                self.params.get('variance_target', 0.0), g_loss_adv.dtype
            )
            variance_target_loss = tf.reduce_mean(tf.square(sigma_square_x_ - variance_target))
            variance_log_target_weight = tf.cast(
                self.params.get('variance_log_target_weight', 0.0), g_loss_adv.dtype
            )
            variance_log_target_loss = self._variance_log_target_loss(sigma_square_x_)
            g_adv_weight = tf.cast(self.params.get('g_adv_weight', 1.0), g_loss_adv.dtype)
            e_adv_weight = tf.cast(self.params.get('e_adv_weight', 1.0), g_loss_adv.dtype)
            marginal_moment_weight = tf.cast(
                self.params.get('marginal_moment_weight', 0.0), g_loss_adv.dtype
            )
            covariance_weight = tf.cast(
                self.params.get('covariance_weight', 0.0), g_loss_adv.dtype
            )
            marginal_mmd_weight = tf.cast(
                self.params.get('marginal_mmd_weight', 0.0), g_loss_adv.dtype
            )
            decorrelation_weight = tf.cast(
                self.params.get('decorrelation_weight', 0.0), g_loss_adv.dtype
            )

            real_mean = tf.reduce_mean(data_x, axis=0)
            gen_mean = tf.reduce_mean(data_x_, axis=0)
            real_var = tf.math.reduce_variance(data_x, axis=0)
            gen_var = tf.math.reduce_variance(data_x_, axis=0)
            marginal_moment_loss = tf.reduce_mean(tf.square(real_mean - gen_mean)) + tf.reduce_mean(
                tf.square(real_var - gen_var)
            )

            batch_size = tf.cast(tf.shape(data_x)[0] - 1, data_x.dtype)
            real_centered = data_x - real_mean
            gen_centered = data_x_ - gen_mean
            real_cov = tf.matmul(real_centered, real_centered, transpose_a=True) / batch_size
            gen_cov = tf.matmul(gen_centered, gen_centered, transpose_a=True) / batch_size
            covariance_loss = tf.reduce_mean(tf.square(real_cov - gen_cov))

            mmd_scales = tf.constant(
                self.params.get('mmd_scales', [0.05, 0.1, 0.2, 0.5, 1.0]),
                dtype=data_x.dtype,
            )
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
            decorrelation_target = str(self.params.get('decorrelation_target', 'identity')).lower()
            if decorrelation_target in {'train', 'batch', 'real', 'data'}:
                decorrelation_loss = correlation_alignment_loss(data_x_, data_x)
            elif decorrelation_target == 'identity':
                decorrelation_loss = offdiag_correlation_loss(data_x_)
            else:
                raise ValueError(
                    "decorrelation_target must be one of 'identity', 'train', 'batch', 'real', or 'data'."
                )
            g_e_loss = (
                g_adv_weight * g_loss_adv
                + e_adv_weight * e_loss_adv
                + x_cycle_weight * l2_loss_x
                + z_cycle_weight * l2_loss_z
                + self.params['alpha'] * reg_loss
                + variance_target_weight * variance_target_loss
                + variance_log_target_weight * variance_log_target_loss
                + marginal_moment_weight * marginal_moment_loss
                + covariance_weight * covariance_loss
                + marginal_mmd_weight * marginal_mmd_loss
                + decorrelation_weight * decorrelation_loss
            )

                
        g_e_gradients = gen_tape.gradient(g_e_loss, self.g_net.trainable_variables+self.e_net.trainable_variables)
        
        self.g_pre_optimizer.apply_gradients(zip(g_e_gradients, self.g_net.trainable_variables+self.e_net.trainable_variables))

        return g_loss_adv, e_loss_adv, l2_loss_z, l2_loss_x, reg_loss, g_e_loss
    

    def egm_init(self, data, egm_n_iter=10000, batch_size=32, egm_batches_per_eval=500, verbose=1):
        
        self.data_sampler = Base_sampler(x=data,y=data,v=data, batch_size=batch_size, normalize=False)
        print('EGM Initialization Starts ...')
        for batch_iter in range(egm_n_iter+1):
            for _ in range(self.params['g_d_freq']):
                batch_x,_,_ = self.data_sampler.next_batch()
                batch_z = self.z_sampler.get_batch(batch_size)
                dz_loss, dx_loss, d_loss = self.train_disc_step(batch_z, batch_x)

            batch_x,_,_ = self.data_sampler.next_batch()
            batch_z = self.z_sampler.get_batch(batch_size)
            g_loss_adv, e_loss_adv, l2_loss_z, l2_loss_x, sigma_square_loss, g_e_loss = self.train_gen_step(batch_z, batch_x)
            if batch_iter % egm_batches_per_eval == 0:
                
                loss_contents = (
                    'EGM Initialization Iter [%d] : g_loss_adv[%.4f], e_loss_adv [%.4f], l2_loss_z [%.4f], l2_loss_x [%.4f], '
                    'sd^2_loss[%.4f], g_e_loss [%.4f], dz_loss [%.4f], dx_loss[%.4f], d_loss [%.4f]'
                    % (batch_iter, g_loss_adv, e_loss_adv, l2_loss_z, l2_loss_x, sigma_square_loss, g_e_loss, dz_loss, dx_loss, d_loss)
                )
                if verbose:
                    print(loss_contents)
                data_z_ = self.e_net(data)
                data_x__, _ = self._decode_generator(data_z_, training=False)
                data_gen_1, sigma_square_x_1 = self.generate(nb_samples=5000)
                data_gen_12, sigma_square_x_12 = self.generate(nb_samples=5000,use_x_sd=False)
                if self.params['save_res']:
                    np.savez('%s/init_data_gen_at_%d.npz'%(self.save_dir, batch_iter),
                            gen1=data_gen_1, gen12=data_gen_12,
                            z=data_z_, x_rec=data_x__, var1=sigma_square_x_1, var12=sigma_square_x_12
                            )
                if self.params['save_model']:
                    base_path = self.checkpoint_path + f"/weights_at_egm_init_{batch_iter}"
                    self.e_net.save_weights(f"{base_path}_encoder.weights.h5")
                    self.g_net.save_weights(f"{base_path}_generator.weights.h5")
                    print('Saving checkpoint for egm_init at {}'.format(base_path))


        print('EGM Initialization Ends.')

    def fit(self, data,
            batch_size=32, epochs=100, epochs_per_eval=5,
            use_egm_init=True, egm_n_iter=20000, egm_batches_per_eval=500, verbose=1):
        if self.params['save_res']:
            f_params = open('{}/params.txt'.format(self.save_dir),'w')
            f_params.write(str(self.params))
            f_params.close()
        
        if use_egm_init:
            self.egm_init(data, egm_n_iter=egm_n_iter, egm_batches_per_eval=egm_batches_per_eval, batch_size=batch_size, verbose=verbose)
            print('Initialize latent variables Z with e(V)...')
            data_z_init = self.e_net(data)
        else:
            print('Random initialization of latent variables Z...')
            data_z_init = np.random.normal(0, 1, size = (len(data), self.params['z_dim'])).astype('float32')

        self.data_z = tf.Variable(data_z_init, name="Latent Variable",trainable=True)

        self.history_loss = []
        print('Iterative Updating Starts ...')
        for epoch in range(epochs+1):
            sample_idx = np.random.choice(len(data), len(data), replace=False)
            
            with tqdm(total=len(data) // batch_size, desc=f"Epoch {epoch}/{epochs}", unit="batch") as batch_bar:
                for i in range(0,len(data) - batch_size + 1,batch_size):
                    batch_idx = sample_idx[i:i+batch_size]
                    batch_z = tf.Variable(tf.gather(self.data_z, batch_idx, axis = 0), name='batch_z', trainable=True)
                    batch_x = data[batch_idx,:]
                    loss_x = self.update_g_net(batch_z, batch_x)

                    loss_postrior_z = self.update_latent_variable_sgd(batch_z, batch_x)

                    self.data_z.scatter_nd_update(
                        indices=tf.expand_dims(batch_idx, axis=1),
                        updates=batch_z                             
                    )
                    
                    loss_contents = (
                        'loss_x: [%.4f], loss_postrior_z: [%.4f]'
                        % (loss_x, loss_postrior_z)
                    )
                    batch_bar.set_postfix_str(loss_contents)
                    batch_bar.update(1)
            
            if epoch % epochs_per_eval == 0:
                if self.params['save_model']:
                    base_path = self.checkpoint_path + f"/weights_at_{epoch}"
                    self.g_net.save_weights(f"{base_path}_generator.weights.h5")
                    print('Saving checkpoint for epoch {} at {}'.format(epoch, base_path))
                        
                data_gen_1, sigma_square_x_1 = self.generate(nb_samples=5000)
                data_gen_12, sigma_square_x_12 = self.generate(nb_samples=5000,use_x_sd=False)
                if self.params['save_res']:
                    np.savez('%s/data_gen_at_%d.npz'%(self.save_dir, epoch),
                            gen1=data_gen_1, gen12=data_gen_12,
                            z=self.data_z.numpy(), var1=sigma_square_x_1, var12=sigma_square_x_12
                            )

    @tf.function
    def generate(self, nb_samples=1000, use_x_sd=True):
        data_z = tf.random.normal(shape=(nb_samples, self.params['z_dim']), mean=0.0, stddev=1.0)

        mu_x, sigma_square_x = self._decode_generator(data_z, training=False)

        if use_x_sd:
            data_x_gen = self._sample_decoder(mu_x, sigma_square_x)
        else:
            data_x_gen = mu_x
        return data_x_gen, sigma_square_x

    @tf.function
    def predict_on_posteriors(self, data_posterior_z):
        n_mcmc = tf.shape(data_posterior_z)[0]
        n_samples = tf.shape(data_posterior_z)[1]

        data_posterior_z_flat = tf.reshape(data_posterior_z, [-1, self.params['z_dim']])
        mu_x_flat, sigma_square_x_flat = self._decode_generator(data_posterior_z_flat, training=False)

        data_x_pred_flat = self._sample_decoder(mu_x_flat, sigma_square_x_flat)
        data_x_pred = tf.reshape(data_x_pred_flat, [n_mcmc, n_samples, self.params['x_dim']])

        return data_x_pred

    def predict(self, data, alpha=0.05, return_samples=False, bs=100, n_mcmc=5000, burn_in=5000, step_size=0.01, num_leapfrog_steps=10, seed=42):
        assert 0 < alpha < 1, "The significance level 'alpha' must be greater than 0 and less than 1."

        if not isinstance(data, tf.Tensor):
            data_tf = tf.convert_to_tensor(data, dtype=tf.float32)
        else:
            data_tf = tf.cast(data, tf.float32)

        n_data_samples = data_tf.shape[0]

        is_nan_tf = tf.math.is_nan(data_tf)
        is_obs_tf = tf.logical_not(is_nan_tf)

        data_clean_tf = tf.where(is_nan_tf,
                                 tf.zeros_like(data_tf),
                                 data_tf)

        is_obs_np = is_obs_tf.numpy()
        ind_x1_list = [
            np.where(row)[0].tolist()
            for row in is_obs_np
        ]
        
        data_posterior_z = self.tfp_mcmc_sampler(
            data=data_clean_tf,
            ind_x1=ind_x1_list,
            n_mcmc=n_mcmc,
            burn_in=burn_in,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            seed=seed
        )
        data_x_pred_all = []
        
        for i in range(0, n_data_samples, bs):
            batch_posterior_z = data_posterior_z[:, i:i + bs, :]
            data_x_batch_pred = self.predict_on_posteriors(batch_posterior_z)
            data_x_batch_pred = data_x_batch_pred.numpy()
            data_x_pred_all.append(data_x_batch_pred)

        data_x_pred_all = np.concatenate(data_x_pred_all, axis=1)
        
        data_np = data_tf.numpy()
        miss_mask_full = np.isnan(data_np).astype(np.float32)
        obs_mask_full = 1.0 - miss_mask_full
        data_obs_np = np.nan_to_num(data_np, nan=0.0)

        miss_mask_flat = miss_mask_full.astype(bool)
        same_pattern = np.all(miss_mask_flat == miss_mask_flat[0])

        if same_pattern:
            miss_idx = np.where(miss_mask_flat[0])[0]
            if miss_idx.size == 0:
                pred_interval = np.zeros((n_data_samples, 0, 2), dtype=np.float32)
            else:
                dim_samples = data_x_pred_all[:, :, miss_idx]
                lower = np.quantile(dim_samples, alpha / 2.0, axis=0)
                upper = np.quantile(dim_samples, 1.0 - alpha / 2.0, axis=0)
                pred_interval = np.stack([lower, upper], axis=-1)
        else:
            pred_interval = []
            for i in range(n_data_samples):
                miss_idx_i = np.where(miss_mask_flat[i])[0]
                if miss_idx_i.size == 0:
                    pred_interval.append(np.zeros((0, 2), dtype=np.float32))
                    continue
                dim_samples_i = data_x_pred_all[:, i, miss_idx_i]
                lower_i = np.quantile(dim_samples_i, alpha / 2.0, axis=0)
                upper_i = np.quantile(dim_samples_i, 1.0 - alpha / 2.0, axis=0)
                intervals_i = np.stack([lower_i, upper_i], axis=-1)
                pred_interval.append(intervals_i)

        if return_samples:
            return data_x_pred_all, pred_interval
        else:
            data_imputed = np.mean(data_x_pred_all, axis=0)
            data_imputed = miss_mask_full * data_imputed + obs_mask_full * data_obs_np
            return data_imputed, pred_interval

    @tf.function
    def get_log_posterior(self, data_z, data_x, ind_x1=None, obs_mask=None):
        mu_x, sigma_square_x = self._decode_generator(data_z, training=False)

        if ind_x1 is None:
            loss_px_z = self._decoder_nll_from_parts(data_x, mu_x, sigma_square_x)
        else:
            data_x_cond = tf.gather(data_x, ind_x1, batch_dims=1)
            mu_x_cond = tf.gather(mu_x, ind_x1, batch_dims=1)
            sigma_square_x_cond = tf.gather(sigma_square_x, ind_x1, batch_dims=1)
            loss_px_z = self._decoder_nll_from_parts(
                data_x_cond,
                mu_x_cond,
                sigma_square_x_cond,
                obs_mask=obs_mask,
            )

        loss_prior_z = tf.reduce_sum(data_z**2, axis=1) / 2

        log_posterior = -(loss_prior_z + loss_px_z)
        return log_posterior


    def tfp_mcmc_sampler(self, data, ind_x1=None, n_mcmc=3000, burn_in=5000, 
                        step_size=0.01, num_leapfrog_steps=10, seed=42):
        if not isinstance(data, tf.Tensor):
            data = tf.convert_to_tensor(data, dtype=tf.float32)
        
        n_samples = data.shape[0]
        z_dim = self.params['z_dim']

        ind_x1_tensor = None
        obs_mask = None

        if ind_x1 is not None:
            if isinstance(ind_x1, (list, tuple)) and len(ind_x1) > 0 and isinstance(ind_x1[0], (list, tuple)):
                assert len(ind_x1) == n_samples, \
                    f"len(ind_x1)={len(ind_x1)} != n_samples={n_samples}"

                max_len = max(len(row) for row in ind_x1) if n_samples > 0 else 0
                assert max_len > 0, f"No observed features"

                ind_mat = np.zeros((n_samples, max_len), dtype=np.int32)
                mask_mat = np.zeros((n_samples, max_len), dtype=np.float32)

                for i, row in enumerate(ind_x1):
                    L = len(row)
                    if L > 0:
                        ind_mat[i, :L] = np.array(row, dtype=np.int32)
                        mask_mat[i, :L] = 1.0

                ind_x1_tensor = tf.constant(ind_mat, dtype=tf.int32)
                obs_mask = tf.constant(mask_mat, dtype=tf.float32)

            else:
                ind_x1_tensor = tf.convert_to_tensor(ind_x1, dtype=tf.int32)

                if ind_x1_tensor.shape.rank == 1:
                    K = tf.shape(ind_x1_tensor)[0]
                    ind_x1_tensor = tf.broadcast_to(ind_x1_tensor[tf.newaxis, :],
                                                    [n_samples, K])
                elif ind_x1_tensor.shape.rank != 2:
                    raise ValueError("ind_x1 must be rank 1 or 2 if tensor-like.")

                obs_mask = tf.ones_like(ind_x1_tensor, dtype=tf.float32)
        
        initial_state = tf.random.normal(
            shape=(n_samples, z_dim), 
            seed=seed, 
            dtype=tf.float32
        )
        
        def target_log_prob_fn(z):
            return self.get_log_posterior(z, data, ind_x1_tensor, obs_mask)
        
        hmc_kernel = tfm.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps
        )
        
        adaptive_kernel = tfm.SimpleStepSizeAdaptation(
            inner_kernel=hmc_kernel,
            num_adaptation_steps=int(burn_in * 0.8),
            target_accept_prob=0.75
        )
        
        @tf.function
        def run_mcmc():
            samples, kernel_results = tfm.sample_chain(
                num_results=n_mcmc,
                num_burnin_steps=burn_in,
                current_state=initial_state,
                kernel=adaptive_kernel,
                trace_fn=lambda _, pkr: pkr.inner_results.is_accepted
            )
            return samples, kernel_results
        
        samples, is_accepted = run_mcmc()
        
        acceptance_rate = tf.reduce_mean(tf.cast(is_accepted, tf.float32))
        print(f"TFP MCMC Acceptance Rate: {acceptance_rate:.4f}")
        
        return samples
