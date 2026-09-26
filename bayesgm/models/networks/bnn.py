import tensorflow as tf
import tensorflow_probability as tfp

class BayesianFullyConnectedNet(tf.keras.Model):
    def __init__(self, input_dim, output_dim, model_name, nb_units=[256, 256, 256]):
        super(BayesianFullyConnectedNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_name = model_name
        self.nb_units = nb_units
        self.all_layers = []
        
        self.norm_layer = tf.keras.layers.BatchNormalization()
        for i in range(len(nb_units) + 1):
            units = self.output_dim if i == len(nb_units) else self.nb_units[i]
            bayesian_layer = tfp.layers.DenseFlipout(
                units=units,
                activation=None
            )
            self.all_layers.append(bayesian_layer)
            
    def call(self, inputs, training=True):
        x = self.norm_layer(inputs)
        for i, bayesian_layer in enumerate(self.all_layers[:-1]):
            with tf.name_scope("%s_layer_%d" % (self.model_name, i+1)):
                x = bayesian_layer(x)
                x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        
        bayesian_layer = self.all_layers[-1]
        with tf.name_scope("%s_layer_output" % self.model_name):
            output = bayesian_layer(x)
        return output
    
class BayesianVariationalNet(tf.keras.Model):
    def __init__(self, input_dim, output_dim, model_name, nb_units=[256, 256, 256]):
        super(BayesianVariationalNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_name = model_name
        self.nb_units = nb_units
        self.all_layers = []
        
        self.norm_layer = tf.keras.layers.BatchNormalization()

        kernel_prior_fn = lambda dtype, shape, name, trainable, add_variable_fn: tfp.distributions.Independent(
                    tfp.distributions.Normal(loc=tf.zeros(shape, dtype=dtype), scale=0.1),
                    reinterpreted_batch_ndims=len(shape)
                )

        for i in range(len(nb_units)):
            bayesian_layer = tfp.layers.DenseFlipout(
                units=self.nb_units[i],
                activation=None,
                kernel_prior_fn=kernel_prior_fn,
                bias_prior_fn=kernel_prior_fn
            )
            self.all_layers.append(bayesian_layer)
        self.mean_layer = tfp.layers.DenseFlipout(
                units=self.output_dim,
                activation=None,
                kernel_prior_fn=kernel_prior_fn,
                bias_prior_fn=kernel_prior_fn
            )
        self.var_layer = tfp.layers.DenseFlipout(
                units=self.output_dim,
                activation=None,
                kernel_prior_fn=kernel_prior_fn,
                bias_prior_fn=kernel_prior_fn
            )
            
    def call(self, inputs, eps=1e-6, training=True):
        x = self.norm_layer(inputs, training=training)
        for i, bayesian_layer in enumerate(self.all_layers):
            with tf.name_scope("%s_layer_%d" % (self.model_name, i + 1)):
                x = bayesian_layer(x, training=training)
                x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
                
        with tf.name_scope("%s_layer_output" % self.model_name):
            mean = self.mean_layer(x, training=training)
            var = self.var_layer(x, training=training)
            var = tf.nn.softplus(var) + eps
        return mean, var
    
    def reparameterize(self, mean, var):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.sqrt(var) + mean
