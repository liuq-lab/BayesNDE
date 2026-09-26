import tensorflow as tf
import tensorflow_probability as tfp

class BaseFullyConnectedNet(tf.keras.Model):
    def __init__(self, input_dim, output_dim, model_name, nb_units=[256, 256, 256], batchnorm=False):  
        super(BaseFullyConnectedNet, self).__init__()
        self.input_layer = tf.keras.layers.Input((input_dim,))
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_name = model_name
        self.nb_units = nb_units
        self.batchnorm = batchnorm
        self.all_layers = []
        """ Builds the FC stacks. """
        for i in range(len(nb_units) + 1):
            units = self.output_dim if i == len(nb_units) else self.nb_units[i]
            fc_layer = tf.keras.layers.Dense(
                units = units,
                activation = None,
                kernel_regularizer = tf.keras.regularizers.L2(1e-4),
                bias_regularizer = tf.keras.regularizers.L2(1e-4)
            )   
            norm_layer = tf.keras.layers.BatchNormalization()
            self.all_layers.append([fc_layer, norm_layer])
        
        self.out = self.call(self.input_layer)

    def call(self, inputs, training=True):
        for i, layers in enumerate(self.all_layers[:-1]):
            fc_layer, norm_layer = layers
            with tf.name_scope("%s_layer_%d" % (self.model_name, i+1)):
                x = fc_layer(inputs) if i==0 else fc_layer(x)
                if self.batchnorm:
                    x = norm_layer(x)
                x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        fc_layer, norm_layer = self.all_layers[-1]
        with tf.name_scope("%s_layer_ouput" % self.model_name):
            output = fc_layer(x)
        return output

class BaseVariationalNet(tf.keras.Model):
    def __init__(self, input_dim, output_dim, model_name, nb_units=[256, 256, 256]):
        super(BaseVariationalNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_name = model_name
        self.nb_units = nb_units
        self.all_layers = []
        
        self.norm_layer = tf.keras.layers.BatchNormalization()
        
        for i in range(len(nb_units)):
            dense_layer = tf.keras.layers.Dense(
                units=self.nb_units[i],
                activation=None
            )
            self.all_layers.append(dense_layer)
            
        self.mean_layer = tf.keras.layers.Dense(
                units=self.output_dim,
                activation=None
            )
            
        self.var_layer = tf.keras.layers.Dense(
                units=self.output_dim,
                activation=None
            )
            
    def call(self, inputs, eps=1e-6, training=True):
        x = self.norm_layer(inputs, training=training)
        for i, bayesian_layer in enumerate(self.all_layers):
            with tf.name_scope("%s_layer_%d" % (self.model_name, i+1)):
                x = bayesian_layer(x)
                x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        
        with tf.name_scope("%s_layer_output" % self.model_name):
            mean = self.mean_layer(x)
            var = self.var_layer(x)
            var = tf.nn.softplus(var) + eps
        return mean, var
    
    def reparameterize(self, mean, var):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.sqrt(var) + mean


class Discriminator(tf.keras.Model):
    def __init__(self, input_dim, model_name, nb_units=[256, 256], batchnorm=True):  
        super(Discriminator, self).__init__()
        self.input_layer = tf.keras.layers.Input((input_dim,))
        self.input_dim = input_dim
        self.model_name = model_name
        self.nb_units = nb_units
        self.batchnorm = batchnorm
        self.all_layers = []
        """Builds the FC stacks."""
        for i in range(len(self.nb_units)+1):
            units = 1 if i == len(self.nb_units) else self.nb_units[i]
            fc_layer = tf.keras.layers.Dense(
                units = units,
                activation = None
            )
            norm_layer = tf.keras.layers.BatchNormalization()

            self.all_layers.append([fc_layer, norm_layer])
        self.out = self.call(self.input_layer)

    def call(self, inputs, training=True):
            
        for i, layers in enumerate(self.all_layers[:-1]):
            fc_layer, norm_layer = layers
            with tf.name_scope("%s_layer_%d" % (self.model_name,i+1)):
                x = fc_layer(inputs) if i==0 else fc_layer(x)
                if self.batchnorm:
                    x = norm_layer(x)
                x = tf.keras.activations.tanh(x)
        fc_layer, norm_layer = self.all_layers[-1]
        with tf.name_scope("%s_layer_ouput" % self.model_name):
            output = fc_layer(x)
        return output
