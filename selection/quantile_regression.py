import tensorflow as tf
import vande.vae.layers as layers
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
        #tf.print("\n targett:", target, output_stream=sys.stdout)
        #tf.print("\n targett shape:", target.shape, output_stream=sys.stdout)
        #tf.print("\n pred:", pred, output_stream=sys.stdout)
        #tf.print("\n pred shape:", pred.shape, output_stream=sys.stdout)
        #tf.print("\n err:", err, output_stream=sys.stdout)
        #tf.print("\n pred[0]:", pred[0], output_stream=sys.stdout)

        #tf.print("\n target:", target, output_stream=sys.stdout)
        #tf.print("\n pred:", pred, output_stream=sys.stdout)

        return tf.where(err>=0, quantile*err, (quantile-1)*err)
    return loss

def vector_quantiles_loss(quantiles):
    @tf.function
    def loss(target, preds):
        err = -1*(preds - target)

        #tf.print("\n PPPPPP target:", target, output_stream=sys.stdout)
        #tf.print("\n PPPPPP preds:", preds, output_stream=sys.stdout)

        return tf.where(err>=0, np.array(quantiles)*err, (np.array(quantiles)-1)*err)
    return loss


def lambda_quantile_loss(quantile):
    @tf.function
    def loss(target, pred):
        #poly = pred[:,0] + pred[:,1]*pow(pred[:,-1],1) + pred[:,2]*pow(pred[:,-1],2) + pred[:,3]*pow(pred[:,-1],3) + pred[:,4]*pow(pred[:,-1],4)


        #tf.print("\n target:", tf.squeeze(target), output_stream=sys.stdout)
        #tf.print("\n target shape:", tf.squeeze(target).shape, output_stream=sys.stdout)

        err1 = tf.squeeze(target) - pred[...,0]
        err2 = tf.squeeze(target) - pred[...,1]
        w = 0.5

        err = tf.squeeze(target) - (w*pred[...,0]-(1-w)*pred[...,1])
        #err = w*err1 + (1-w)*err2
        
        #tf.print("\n pred:", pred, output_stream=sys.stdout)
        #tf.print("\n pred shape:", pred.shape, output_stream=sys.stdout)
        #tf.print("\n err1:", err1, output_stream=sys.stdout)
        #tf.print("\n err1 shape:", err1.shape, output_stream=sys.stdout)
        #tf.print("\n pred[...,0]:", pred[...,0], output_stream=sys.stdout)
        #tf.print("\n pred[...,0].shape:", pred[...,0].shape, output_stream=sys.stdout)
        
        #tf.print("\n pred:", poly, output_stream=sys.stdout)

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


class Lambda1QuantileRegression():

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

        # Below: works!!

        def poly(x):
            val = x[:,0]
            for i in range(1,5):
                val += x[:,i] * (x[:,-1]**i) 
            return val

        inputs = tf.keras.Input(shape=(1,))
        normx = layers.StdNormalization(*self.x_mu_std)(inputs)
        x = tf.keras.layers.Dense(self.n_nodes, kernel_initializer=self.initializer, activation=self.activation)(normx)
        for _ in range(self.n_layers-1):
            x = tf.keras.layers.Dense(self.n_nodes, kernel_initializer=self.initializer, activation=self.activation)(x)

        # last layer before lambda layer has to have degree(pol) nodes
        x = tf.keras.layers.Dense(self.poldeg, kernel_initializer=self.initializer, activation=self.activation)(x)

        # now final lambda layer
        x = tf.keras.layers.concatenate([x, normx])

        outputs = tf.keras.layers.Lambda(poly,output_shape=(1,))(x)

        model = tf.keras.Model(inputs, outputs)
        model.compile(loss=quantile_loss(self.quantile), optimizer=self.optimizer) # Adam(lr=1e-3) TODO: add learning rate
        model.summary()
        return model

        
        




class Lambda2QuantileRegression():

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

        def poly(x):
            val = x[:,0]
            for i in range(1,5):
                val += x[:,i] * (x[:,-1]**i) 
            return val

        inputs = tf.keras.Input(shape=(1,))
        normx = layers.StdNormalization(*self.x_mu_std)(inputs)
        x = tf.keras.layers.Dense(self.n_nodes, kernel_initializer=self.initializer, activation=self.activation)(normx)
        for _ in range(self.n_layers-1):
            x = tf.keras.layers.Dense(self.n_nodes, kernel_initializer=self.initializer, activation=self.activation)(x)

        output_1 = tf.keras.layers.Dense(1, kernel_initializer=self.initializer)(x)

        #model = printeverybatch(inputs, outputs)

        x_quintupled = tf.keras.layers.Lambda(lambda x:x**5)(normx)
        x_quadrupled = tf.keras.layers.Lambda(lambda x:x**4)(normx)
        x_cubed = tf.keras.layers.Lambda(lambda x:x**3)(normx)
        x_squared = tf.keras.layers.Lambda(lambda x:x**2)(normx)
        hidden = tf.keras.layers.Concatenate()([x_quintupled,x_quadrupled,x_cubed,x_squared,normx])
        output_2 = tf.keras.layers.Dense(1, activation = 'linear')(hidden)
        
        outputs = tf.keras.layers.Concatenate()([output_1,output_2])

        model = tf.keras.Model(inputs, outputs)
        model.compile(loss=lambda_quantile_loss(self.quantile), optimizer=self.optimizer) # Adam(lr=1e-3) TODO: add learning rate
        model.summary()
        return model



class Lambda3QuantileRegression():

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

        x_quintupled = tf.keras.layers.Lambda(lambda x:x**5)(normx)
        x_quadrupled = tf.keras.layers.Lambda(lambda x:x**4)(normx)
        x_cubed = tf.keras.layers.Lambda(lambda x:x**3)(normx)
        x_squared = tf.keras.layers.Lambda(lambda x:x**2)(normx)
        hidden = tf.keras.layers.Concatenate()([x_quintupled,x_quadrupled,x_cubed,x_squared,normx])
        outputs = tf.keras.layers.Dense(1, activation = activation)(hidden)
        
        model = tf.keras.Model(inputs, outputs)
        model.compile(loss=quantile_loss(self.quantile), optimizer=self.optimizer) # Adam(lr=1e-3) TODO: add learning rate
        model.summary()
        return model



class Lambda4QuantileRegression():

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
        model.compile(loss=quantile_loss(self.quantile), optimizer=self.optimizer) # Adam(lr=1e-3) TODO: add learning rate
        model.summary()
        return model

        


class Lambda5QuantileRegression():

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

        x_squared = tf.keras.layers.Lambda(lambda x:x**2)(normx)
        hidden = tf.keras.layers.Concatenate()([x_squared,normx])
        outputs = tf.keras.layers.Dense(1, activation = activation)(hidden)
        
        model = tf.keras.Model(inputs, outputs)
        model.compile(loss=quantile_loss(self.quantile), optimizer=self.optimizer) # Adam(lr=1e-3) TODO: add learning rate
        model.summary()
        return model



class Lambda6QuantileRegression():

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

        x_quadrupled = tf.keras.layers.Lambda(lambda x:x**4)(normx)
        x_cubed = tf.keras.layers.Lambda(lambda x:x**3)(normx)
        x_squared = tf.keras.layers.Lambda(lambda x:x**2)(normx)
        hidden = tf.keras.layers.Concatenate()([x_quadrupled,x_cubed,x_squared,normx])
        outputs = tf.keras.layers.Dense(1, activation = activation)(hidden)
        
        model = tf.keras.Model(inputs, outputs)
        model.compile(loss=quantile_loss(self.quantile), optimizer=self.optimizer) # Adam(lr=1e-3) TODO: add learning rate
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





