import tensorflow as tf
import case_vae.vae.layers as layers
import numpy as np
import sys

class printeverybatch(tf.keras.Model):
    def train_step(self, data):
        x, y = data
        #tf.print('new batch:')
        #tf.print('X:')
        #tf.print(x,summarize=-1)
        #tf.print('Y:')
        #tf.print(y,summarize=-1)
        return super().train_step(data)

# ******************************************** #
#           quantile regression loss           #
# ******************************************** #

def quantile_loss(quantile):
    @tf.function
    def loss(target, pred):
        err = target - pred

        return tf.where(err>=0, quantile*err, (quantile-1)*err)
    return loss

def vector_quantiles_loss(quantiles):
    @tf.function
    def loss(target, preds):
        err = -1*(preds - target)

        return tf.where(err>=0, np.array(quantiles)*err, (np.array(quantiles)-1)*err)
    return loss

def vector_expectiles_loss(quantiles):
    @tf.function
    def loss(target, preds):
        err = (preds - target)*(preds - target)

        return tf.where(target>preds, np.array(quantiles)*err, (1-np.array(quantiles))*err)
    return loss

def lambda_quantile_loss(quantile):
    @tf.function
    def loss(target, pred):

        err1 = tf.squeeze(target) - pred[...,0]
        err2 = tf.squeeze(target) - pred[...,1]
        w = 0.5

        err = tf.squeeze(target) - (w*pred[...,0]-(1-w)*pred[...,1])

        return tf.where(err>=0, quantile*err, (quantile-1)*err)
    return loss

# ******************************************** #
#           quantile regression models         #
# ******************************************** #


class QuantileRegression():

    def __init__(self, quantile, n_layers=5, n_nodes=20, x_mu_std=(0.,1.), optimizer='adam', initializer='he_uniform', activation='elu'):
        self.quantile = quantile
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.x_mu_std = x_mu_std
        self.optimizer = optimizer
        self.initializer = initializer
        self.activation = activation

    def build(self):
        inputs = tf.keras.Input(shape=(1,))
        x = layers.StdNormalization(*self.x_mu_std)(inputs)
        for _ in range(self.n_layers):
            x = tf.keras.layers.Dense(self.n_nodes, kernel_initializer=self.initializer, activation=self.activation)(x)
        outputs = tf.keras.layers.Dense(1, kernel_initializer=self.initializer)(x)
        #model = printeverybatch(inputs, outputs)
        model = tf.keras.Model(inputs, outputs)
        model.compile(loss=quantile_loss(self.quantile), optimizer=self.optimizer) # Adam(lr=1e-3) TODO: add learning rate
        model.summary()
        return model



class LambdaQuantileRegression():

    def __init__(self, quantile, n_layers=5, n_nodes=20, x_mu_std=(0.,1.), optimizer='adam', initializer='he_uniform', activation='elu', poldeg=5):
        self.quantile = quantile
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.x_mu_std = x_mu_std
        self.optimizer = optimizer
        self.initializer = initializer
        self.activation = activation
        self.poldeg = poldeg

    def build(self):

        inputs = tf.keras.Input(shape=(1,))
        normx = layers.StdNormalization(*self.x_mu_std)(inputs)

        activation = 'linear'

        x_cubed = tf.keras.layers.Lambda(lambda x:x**3)(normx)
        x_squared = tf.keras.layers.Lambda(lambda x:x**2)(normx)
        hidden = tf.keras.layers.Concatenate()([x_cubed,x_squared,normx])
        outputs = tf.keras.layers.Dense(1, activation = activation)(hidden)
        
        model = tf.keras.Model(inputs, outputs)
        model.compile(loss=quantile_loss(self.quantile), optimizer=self.optimizer)
        model.summary()
        return model



class LambdaBernsteinQuantileRegression():

    def __init__(self, quantile, n_layers=5, n_nodes=20, x_mu_std=(0.,1.), optimizer='adam', initializer='he_uniform', activation='elu', poldeg=5):
        self.quantile = quantile
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.x_mu_std = x_mu_std
        self.optimizer = optimizer
        self.initializer = initializer
        self.activation = activation
        self.poldeg = poldeg

    def build(self):

        activation = 'linear'

        inputs = tf.keras.Input(shape=(1,))
        normx = layers.StdNormalization(*self.x_mu_std)(inputs)

        b03 = tf.keras.layers.Lambda(lambda x: 1 - 3*x + 3*x**2 - x**3)(normx)
        b13 = tf.keras.layers.Lambda(lambda x: 3*x - 6*x**2 + 3*x**3)(normx)
        b23 = tf.keras.layers.Lambda(lambda x: 3*x**2 - 3*x**3)(normx)
        b33 = tf.keras.layers.Lambda(lambda x: x**3)(normx)

        hidden = tf.keras.layers.Concatenate()([b03, b13, b23, b33])
        outputs = tf.keras.layers.Dense(1, activation = activation)(hidden)

        model = tf.keras.Model(inputs, outputs)
        model.compile(loss=quantile_loss(self.quantile), optimizer=self.optimizer)

        model.summary()
        return model
        


class VectorQuantileRegression():

    def __init__(self, quantiles, n_layers=5, n_nodes=20, x_mu_std=(0.,1.), optimizer='adam', initializer='he_uniform', activation='elu'):
        self.quantiles = quantiles
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.x_mu_std = x_mu_std
        self.optimizer = optimizer
        self.initializer = initializer
        self.activation = activation

    def build(self):
        inputs = tf.keras.Input(shape=(1,))
        x = layers.StdNormalization(*self.x_mu_std)(inputs)
        for _ in range(self.n_layers):
            x = tf.keras.layers.Dense(self.n_nodes, kernel_initializer=self.initializer, activation=self.activation)(x)
        outputs = tf.keras.layers.Dense(len(self.quantiles), kernel_initializer=self.initializer)(x)
        #model = printeverybatch(inputs, outputs)
        model = tf.keras.Model(inputs, outputs)
        model.compile(loss=vector_quantiles_loss(self.quantiles), optimizer=self.optimizer) # Adam(lr=1e-3) TODO: add learning rate
        model.summary()
        return model


class VectorExpectileRegression():

    def __init__(self, quantiles, n_layers=5, n_nodes=20, x_mu_std=(0.,1.), optimizer='adam', initializer='he_uniform', activation='elu'):
        self.quantiles = quantiles
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.x_mu_std = x_mu_std
        self.optimizer = optimizer
        self.initializer = initializer
        self.activation = activation

    def build(self):
        inputs = tf.keras.Input(shape=(1,))
        x = layers.StdNormalization(*self.x_mu_std)(inputs)
        for _ in range(self.n_layers):
            x = tf.keras.layers.Dense(self.n_nodes, kernel_initializer=self.initializer, activation=self.activation)(x)
        outputs = tf.keras.layers.Dense(len(self.quantiles), kernel_initializer=self.initializer)(x)
        #model = printeverybatch(inputs, outputs)
        model = tf.keras.Model(inputs, outputs)
        model.compile(loss=vector_expectiles_loss(self.quantiles), optimizer=self.optimizer) # Adam(lr=1e-3) TODO: add learning rate
        model.summary()
        return model




