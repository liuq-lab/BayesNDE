import datetime
import os

import dateutil.tz
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from bayesgm.datasets import Base_sampler, Gaussian_sampler

from ..networks import (
    Discriminator,
    MNISTDiscriminator,
    MNISTEncoderConv,
    MNISTGenerator,
)
from .base import BGM

class MNISTBGM(BGM):
    def __init__(self, params, timestamp=None, random_seed=None):
        self.params = params
        self.timestamp = timestamp
        if random_seed is not None:
            tf.keras.utils.set_random_seed(random_seed)
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
            tf.config.experimental.enable_op_determinism()

        self.g_net = MNISTGenerator(z_dim=params['z_dim'], filters=32,
                                    use_bnn=params['use_bnn'], name='g_net')
        self.e_net = MNISTEncoderConv(z_dim=params['z_dim'], filters=32, name='e_net')
        self.dz_net = Discriminator(input_dim=params['z_dim'], model_name='dz_net',
                                    nb_units=params['dz_units'])
        self.dx_net = MNISTDiscriminator(filters=64, name='dx_net')

        self.g_pre_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.5, beta_2=0.9)
        self.d_pre_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.5, beta_2=0.9)
        self.z_sampler = Gaussian_sampler(mean=np.zeros(params['z_dim']), sd=1.0)

        self.g_optimizer = tf.keras.optimizers.Adam(params['lr_theta'], beta_1=0.9, beta_2=0.99)
        self.posterior_optimizer = tf.keras.optimizers.Adam(params['lr_z'], beta_1=0.9, beta_2=0.99)
        
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

    @tf.function
    def update_g_net(self, data_z, data_x):
        with tf.GradientTape() as gen_tape:
            mu_x, sigma_square_x = self.g_net(data_z)
            x_logits = self.g_net.reparameterize(mu_x, sigma_square_x)

            x_logits = tf.clip_by_value(x_logits, -10, 10)
            log_px_z = tf.reduce_sum(
                data_x * x_logits - tf.nn.softplus(x_logits), 
                axis=[1, 2, 3]
            )
            loss_x = -tf.reduce_mean(log_px_z)
            
            if self.params['use_bnn']:
                loss_kl = sum(self.g_net.losses)
                loss_x += loss_kl * self.params['kl_weight']

        g_gradients = gen_tape.gradient(loss_x, self.g_net.trainable_variables)
        
        self.g_optimizer.apply_gradients(zip(g_gradients, self.g_net.trainable_variables))
        return loss_x
        
    @tf.function
    def update_latent_variable_sgd(self, data_z, data_x):
        with tf.GradientTape() as tape:
            
            mu_x, sigma_square_x = self.g_net(data_z)
            x_logits = self.g_net.reparameterize(mu_x, sigma_square_x)
            
            x_logits = tf.clip_by_value(x_logits, -10, 10)
            log_px_z = tf.reduce_sum(
                data_x * x_logits - tf.nn.softplus(x_logits), 
                axis=[1, 2, 3]
            )
            loss_px_z = -tf.reduce_mean(log_px_z)

            loss_prior_z =  tf.reduce_sum(data_z**2, axis=1)/2
            loss_prior_z = tf.reduce_mean(loss_prior_z)

            loss_postrior_z = loss_px_z + loss_prior_z

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
                mu_x_, sigma_square_x_ = self.g_net(data_z)
                x_logits_ = self.g_net.reparameterize(mu_x_, sigma_square_x_)
                data_x_ = tf.nn.sigmoid(x_logits_)
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
            grad_norm_x = tf.sqrt(tf.reduce_sum(tf.square(grad_x), axis=[1, 2, 3]))
            gpx_loss = tf.reduce_mean(tf.square(grad_norm_x - 1.0))
                
            d_loss = dx_loss + dz_loss + \
                    self.params['gamma']*(gpz_loss + gpx_loss)

        d_gradients = disc_tape.gradient(d_loss, self.dz_net.trainable_variables+self.dx_net.trainable_variables)
        
        self.d_pre_optimizer.apply_gradients(zip(d_gradients, self.dz_net.trainable_variables+self.dx_net.trainable_variables))
        
        return dz_loss, dx_loss, d_loss
    
    @tf.function
    def train_gen_step(self, data_z, data_x):
        with tf.GradientTape(persistent=True) as gen_tape:
            mu_x_, sigma_square_x_ = self.g_net(data_z)
            x_logits_ = self.g_net.reparameterize(mu_x_, sigma_square_x_)
            data_x_ = tf.nn.sigmoid(x_logits_)
            reg_loss = tf.reduce_mean(tf.square(sigma_square_x_))
            data_z_ = self.e_net(data_x)

            data_z__= self.e_net(data_x_)
            mu_x__, sigma_square_x__ = self.g_net(data_z_)
            x_logits__ = self.g_net.reparameterize(mu_x__, sigma_square_x__)
            data_x__ = tf.nn.sigmoid(x_logits__)
            
            data_dx_ = self.dx_net(data_x_)
            data_dz_ = self.dz_net(data_z_)
            
            l2_loss_x = tf.reduce_mean((data_x - data_x__)**2)
            l2_loss_z = tf.reduce_mean((data_z - data_z__)**2)
            
            g_loss_adv = tf.reduce_mean((0.9*tf.ones_like(data_dx_)  - data_dx_)**2)
            e_loss_adv = tf.reduce_mean((0.9*tf.ones_like(data_dz_)  - data_dz_)**2)

            g_e_loss = g_loss_adv + e_loss_adv + 10 * (l2_loss_x + l2_loss_z) + self.params['alpha'] * reg_loss
                
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
                mu_x__, sigma_square_x__ = self.g_net(data_z_)
                x_logits__ = self.g_net.reparameterize(mu_x__, sigma_square_x__)
                data_x__ = tf.nn.sigmoid(x_logits__)
                data_gen = self.generate(nb_samples=5000)
                np.savez('%s/init_data_gen_at_%d.npz'%(self.save_dir, batch_iter),
                        data_gen=data_gen, z=data_z_, x_rec=data_x__)
                if self.params['save_model']:
                    base_path = self.checkpoint_path + f"/weights_at_egm_init_{batch_iter}"
                    self.e_net.save_weights(f"{base_path}_encoder.weights.h5")
                    self.g_net.save_weights(f"{base_path}_generator.weights.h5")
                    print('Saving checkpoint for egm_init at {}'.format(base_path))


        print('EGM Initialization Ends.')

    def fit(self, data,
            batch_size=32, epochs=100, epochs_per_eval=5,
            use_egm_init=True, egm_n_iter=10000, egm_batches_per_eval=500, verbose=1):
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
                        
                data_gen = self.generate(nb_samples=5000)
                if self.params['save_res']:
                    np.savez('%s/data_gen_at_%d.npz'%(self.save_dir, epoch),
                            gen=data_gen,
                            z=self.data_z.numpy()
                            )

    @tf.function
    def generate(self, nb_samples=1000):
        data_z = tf.random.normal(shape=(nb_samples, self.params['z_dim']), mean=0.0, stddev=1.0)

        mu_x, sigma_square_x = self.g_net(data_z, training=False)
        x_logits = self.g_net.reparameterize(mu_x, sigma_square_x)
        data_x_pred = tf.nn.sigmoid(x_logits)

        return data_x_pred

    @tf.function
    def predict_on_posteriors(self, data_posterior_z):
        n_mcmc = tf.shape(data_posterior_z)[0]
        n_samples = tf.shape(data_posterior_z)[1]

        data_posterior_z_flat = tf.reshape(data_posterior_z, [-1, self.params['z_dim']])
        mu_x_flat, sigma_square_x_flat = self.g_net(data_posterior_z_flat)
        x_logits_flat = self.g_net.reparameterize(mu_x_flat, sigma_square_x_flat)

        data_x_pred_flat = tf.nn.sigmoid(x_logits_flat)
        data_x_pred = tf.reshape(data_x_pred_flat, [n_mcmc, n_samples, 28, 28, 1])

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

        is_obs_flat_tf = tf.reshape(is_obs_tf, [n_data_samples, -1])
        is_obs_flat_np = is_obs_flat_tf.numpy()

        ind_x1_list = [
            np.where(row)[0].tolist()
            for row in is_obs_flat_np
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

        n_mcmc_samples = data_x_pred_all.shape[0]
        flat_pred = data_x_pred_all.reshape(n_mcmc_samples,
                                            n_data_samples,
                                            -1)

        miss_mask_flat = miss_mask_full.reshape(n_data_samples, -1).astype(bool)

        same_pattern = np.all(miss_mask_flat == miss_mask_flat[0])

        if same_pattern:
            miss_idx = np.where(miss_mask_flat[0])[0]
            if miss_idx.size == 0:
                pred_interval = np.zeros((n_data_samples, 0, 2), dtype=np.float32)
            else:
                pix_samples = flat_pred[:, :, miss_idx]
                lower = np.quantile(pix_samples, alpha / 2.0, axis=0)
                upper = np.quantile(pix_samples, 1.0 - alpha / 2.0, axis=0)
                pred_interval = np.stack([lower, upper], axis=-1)
        else:
            pred_interval = []
            for i in range(n_data_samples):
                miss_idx_i = np.where(miss_mask_flat[i])[0]
                if miss_idx_i.size == 0:
                    pred_interval.append(np.zeros((0, 2), dtype=np.float32))
                    continue
                pix_samples_i = flat_pred[:, i, miss_idx_i]
                lower_i = np.quantile(pix_samples_i, alpha / 2.0, axis=0)
                upper_i = np.quantile(pix_samples_i, 1.0 - alpha / 2.0, axis=0)
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
        mu_x, sigma_square_x = self.g_net(data_z)
        x_logits = self.g_net.reparameterize(mu_x, sigma_square_x)
        x_logits = tf.clip_by_value(x_logits, -10, 10)

        batch_size = tf.shape(data_x)[0]
        data_x_flat = tf.reshape(data_x, [batch_size, -1])
        x_logits_flat = tf.reshape(x_logits, [batch_size, -1])

        if ind_x1 is None:
            ll_term = data_x_flat * x_logits_flat - tf.nn.softplus(x_logits_flat)
            log_px_z = tf.reduce_sum(ll_term, axis=1)
        else:
            data_x_cond   = tf.gather(data_x_flat, ind_x1, batch_dims=1)
            x_logits_cond = tf.gather(x_logits_flat, ind_x1, batch_dims=1)

            ll_term = data_x_cond * x_logits_cond - tf.nn.softplus(x_logits_cond)

            if obs_mask is not None:
                ll_term = ll_term * obs_mask

            log_px_z = tf.reduce_sum(ll_term, axis=1)
        
        log_prior_z = -0.5 * tf.reduce_sum(data_z**2, axis=1)
        return log_prior_z + log_px_z
