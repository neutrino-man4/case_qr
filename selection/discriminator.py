import numpy as np
import time
from abc import ABCMeta, abstractmethod
import tensorflow as tf
import sklearn.ensemble as scikit
import case_qr_fixed.selection.quantile_regression as qr
import case_paths.jet_sample as js
import case_vae.training as train
import case_vae.vae.layers as layers
from tensorflow.keras.callbacks import LambdaCallback, Callback

class GetWeights(Callback):
    # Keras callback which collects values of weights and biases at each epoch
    def __init__(self):
        super(GetWeights, self).__init__()
        self.weight_dict = {}

    def on_epoch_end(self, epoch, logs=None):
        # this function runs at the end of each epoch

        # loop over each layer and get weights and biases
        for layer_i in range(len(self.model.layers)):
            print(layer_i)
            print(self.model.layers)
            w = self.model.layers[layer_i].get_weights()[0]
            b = self.model.layers[layer_i].get_weights()[1]
            print('Layer %s has weights of shape %s and biases of shape %s' %(
                layer_i, np.shape(w), np.shape(b)))

            # save all weights and biases inside a dictionary
            if epoch == 0:
                # create array to hold weights and biases
                self.weight_dict['w_'+str(layer_i+1)] = w
                self.weight_dict['b_'+str(layer_i+1)] = b
            else:
                # append new weights to previously-created weights array
                self.weight_dict['w_'+str(layer_i+1)] = np.dstack(
                    (self.weight_dict['w_'+str(layer_i+1)], w))
                # append new weights to previously-created weights array
                self.weight_dict['b_'+str(layer_i+1)] = np.dstack(
                    (self.weight_dict['b_'+str(layer_i+1)], b))

class Discriminator(metaclass=ABCMeta):

    def __init__(self, quantile, loss_strategy):
        self.loss_strategy = loss_strategy
        self.quantile = quantile
        self.mjj_key = 'mJJ'

    @abstractmethod
    def fit(self, jet_sample):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass 

    @abstractmethod
    def predict(self, data):
        '''predict cut for each example in data'''
        pass

    @abstractmethod
    def select(self, jet_sample):
        pass

    def __repr__(self):
        return '{}% qnt, {} strategy'.format(str(self.quantile*100), self.loss_strategy.title_str)


class FlatCutDiscriminator(Discriminator):

    def fit(self, jet_sample):
        loss = self.loss_strategy(jet_sample)
        self.cut = np.percentile( loss, (1.-self.quantile)*100 )
        
    def predict(self, jet_sample):
        return np.asarray([self.cut]*len(jet_sample))

    def select(self, jet_sample):
        loss = self.loss_strategy(jet_sample)
        return loss > self.cut

    def __repr__(self):
        return 'Flat Cut: ' + Discriminator.__repr__(self)


class QRDiscriminator(Discriminator):

    def __init__(self, quantile, loss_strategy, batch_sz=128, epochs=100, learning_rate=0.001, optimizer=tf.keras.optimizers.Adam, **model_params):
        Discriminator.__init__(self, quantile, loss_strategy)
        self.batch_sz = batch_sz
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer() #optimizer(learning_rate)
        self.model_params = model_params

    @tf.function
    def training_step(self, x_batch, y_batch):
        # Open a GradientTape to record the operations run in forward pass
        # import ipdb; ipdb.set_trace()
        print("XXXXXXXXXXXXXXXXXXXX")
        with tf.GradientTape() as tape:
            predictions = self.model(x_batch, training=True)
            tf.print("\n y_batch:", y_batch, output_stream=sys.stdout)
            tf.print("\n y_batch shape:", y_batch.shape, output_stream=sys.stdout)
            loss_value = tf.math.reduce_mean(qr.quantile_loss(y_batch, predictions, self.quantile))

        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss_value

    def training_epoch(self, train_dataset):
        # Iterate over the batches of the dataset.
        train_loss = 0.
        for step, (x_batch, y_batch) in enumerate(train_dataset):
            loss_value = self.training_step(x_batch, y_batch)
            train_loss += loss_value
            if step % 10000 == 0:
                print("Step {}: lr {:.3e}, loss {:.4f}".format(step, self.optimizer.learning_rate.numpy(), loss_value))
        return train_loss / (step + 1)

    def valid_epoch(self, valid_dataset):
        # Iterate over the batches of the dataset.
        valid_loss = 0.
        for step, (x_batch, y_batch) in enumerate(valid_dataset):
            predictions = self.model(x_batch, training=False)
            valid_loss += tf.math.reduce_mean(qr.quantile_loss(y_batch, predictions, self.quantile))
        return valid_loss / (step + 1)


    def make_training_datasets(self, train_sample, valid_sample):
        x_train = train_sample[self.mjj_key]
        y_train = self.loss_strategy(train_sample)
        x_valid = valid_sample[self.mjj_key]
        y_valid = self.loss_strategy(valid_sample)
        return (x_train, y_train), (x_valid, y_valid)

    def fit(self, train_sample, valid_sample):
        # process the input
        (x_train, y_train), (x_valid, y_valid) = self.make_training_datasets(train_sample, valid_sample)
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(self.batch_sz)#.shuffle(self.batch_sz*10)
        valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(self.batch_sz)

        # build the regressor
        self.regressor = qr.QuantileRegressionV2(**self.model_params)
        # self.model = self.regressor.make_model(x_mean_std=(np.mean(x_train), np.std(x_train)), y_mean_std=(np.mean(y_train), np.std(y_train)))
        self.model = self.regressor.make_model(x_min_max=(np.min(x_train), np.max(x_train)), y_min_max=(np.min(y_train), np.max(y_train)))
        
        # build loss arrays and callbacks
        losses_train = []
        losses_valid = []
        train_stop = train.Stopper(optimizer=self.optimizer)

        # run training
        for epoch in range(self.epochs):
            start_time = time.time()
            losses_train.append(self.training_epoch(train_dataset))
            losses_valid.append(self.valid_epoch(valid_dataset))
            # print epoch results
            print('### [Epoch {} - {:.2f} sec]: train loss {:.3f}, val loss {:.3f} (mean / batch) ###'.format(epoch, time.time()-start_time, losses_train[-1], losses_valid[-1]))
            if train_stop.check_stop_training(losses_valid):
                print('!!! stopping training !!!')
                break

        return losses_train, losses_valid


    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path, custom_objects={'StdNormalization': layers.StdNormalization, 'StdUnnormalization': layers.StdUnnormalization}, compile=False)
        print('loaded model ', self.model)

    def predict(self, data):
        if isinstance(data, js.JetSample):
            data = data[self.mjj_key]
        #print("Horo")
        #print(data)
        return self.model(data, training=False)

    def select(self, jet_sample):
        print("Huru")
        #print(jet_sample)
        loss_cut = self.predict(jet_sample)
        print(loss_cut)
        print(loss_cut.shape)
        if loss_cut.shape == self.loss_strategy(jet_sample).shape:
            return self.loss_strategy(jet_sample) > loss_cut
        else:
            return self.loss_strategy(jet_sample) > loss_cut[1::2]

    def __repr__(self):
        return 'QR Cut: ' + Discriminator.__repr__(self)


class LQRDiscriminator(Discriminator):

    def __init__(self, quantile, loss_strategy, batch_sz=128, epochs=100, learning_rate=0.001, optimizer=tf.keras.optimizers.Adam, **model_params):
        Discriminator.__init__(self, quantile, loss_strategy)
        self.batch_sz = batch_sz
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer() #optimizer(learning_rate)
        self.model_params = model_params

    @tf.function
    def training_step(self, x_batch, y_batch):
        # Open a GradientTape to record the operations run in forward pass
        # import ipdb; ipdb.set_trace()
        with tf.GradientTape() as tape:
            predictions = self.model(x_batch, training=True)
            loss_value = tf.math.reduce_mean(qr.lambda_quantile_loss(y_batch, predictions, self.quantile))

        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss_value

    def training_epoch(self, train_dataset):
        # Iterate over the batches of the dataset.
        train_loss = 0.
        for step, (x_batch, y_batch) in enumerate(train_dataset):
            loss_value = self.training_step(x_batch, y_batch)
            train_loss += loss_value
            if step % 10000 == 0:
                print("Step {}: lr {:.3e}, loss {:.4f}".format(step, self.optimizer.learning_rate.numpy(), loss_value))
        return train_loss / (step + 1)

    def valid_epoch(self, valid_dataset):
        # Iterate over the batches of the dataset.
        valid_loss = 0.
        for step, (x_batch, y_batch) in enumerate(valid_dataset):
            predictions = self.model(x_batch, training=False)
            valid_loss += tf.math.reduce_mean(qr.quantile_loss(y_batch, predictions, self.quantile))
        return valid_loss / (step + 1)


    def make_training_datasets(self, train_sample, valid_sample):
        x_train = train_sample[self.mjj_key]
        y_train = self.loss_strategy(train_sample)
        x_valid = valid_sample[self.mjj_key]
        y_valid = self.loss_strategy(valid_sample)
        return (x_train, y_train), (x_valid, y_valid)

    def fit(self, train_sample, valid_sample):
        # process the input
        (x_train, y_train), (x_valid, y_valid) = self.make_training_datasets(train_sample, valid_sample)
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(self.batch_sz)#.shuffle(self.batch_sz*10)
        valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(self.batch_sz)

        # build the regressor
        self.regressor = qr.QuantileRegressionV2(**self.model_params)
        # self.model = self.regressor.make_model(x_mean_std=(np.mean(x_train), np.std(x_train)), y_mean_std=(np.mean(y_train), np.std(y_train)))
        self.model = self.regressor.make_model(x_min_max=(np.min(x_train), np.max(x_train)), y_min_max=(np.min(y_train), np.max(y_train)))
        
        # build loss arrays and callbacks
        losses_train = []
        losses_valid = []
        train_stop = train.Stopper(optimizer=self.optimizer)

        # run training
        for epoch in range(self.epochs):
            start_time = time.time()
            losses_train.append(self.training_epoch(train_dataset))
            losses_valid.append(self.valid_epoch(valid_dataset))
            # print epoch results
            print('### [Epoch {} - {:.2f} sec]: train loss {:.3f}, val loss {:.3f} (mean / batch) ###'.format(epoch, time.time()-start_time, losses_train[-1], losses_valid[-1]))
            if train_stop.check_stop_training(losses_valid):
                print('!!! stopping training !!!')
                break

        return losses_train, losses_valid


    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path, custom_objects={'StdNormalization': layers.StdNormalization, 'StdUnnormalization': layers.StdUnnormalization}, compile=False)
        print('loaded model ', self.model)

    def predict(self, data):
        if isinstance(data, js.JetSample):
            data = data[self.mjj_key]
        #print("Horo")
        #print(data)
        return self.model(data, training=False)

    def select(self, jet_sample):
        print("Huru")
        #print(jet_sample)
        loss_cut = self.predict(jet_sample)
        print(loss_cut)
        print(loss_cut.shape)
        if loss_cut.shape == self.loss_strategy(jet_sample).shape:
            return self.loss_strategy(jet_sample) > loss_cut
        else:
            return self.loss_strategy(jet_sample) > loss_cut[1::2]

    def __repr__(self):
        return 'QR Cut: ' + Discriminator.__repr__(self)


class QRDiscriminator_KerasAPI(QRDiscriminator):
    """docstring for QRDiscriminator_KerasAPI"""
    def __init__(self, **kwargs):
        super(QRDiscriminator_KerasAPI, self).__init__(**kwargs)

    def fit(self, train_sample, valid_sample):
        # prepare training set
        (x_train, y_train), (x_valid, y_valid) = self.make_training_datasets(train_sample, valid_sample)

        self.model = qr.QuantileRegression(quantile=self.quantile, x_mu_std=(np.mean(x_train), np.std(x_train)), **self.model_params).build()
        self.history = self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_sz, verbose=2, validation_data=(x_valid, y_valid), shuffle=True, \
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1), tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3, verbose=1)])
       
        return self.history.history['loss'], self.history.history['val_loss']

    def predict(self, data):
        if isinstance(data, js.JetSample):
            data = data[self.mjj_key]
        xx = data #self.scale_input(data)
        #print("HERE")
        #print(xx)

        predicted = self.model.predict(xx).flatten() 
        # return self.unscale_output(predicted)
        return predicted




class LambdaQRDiscriminator_KerasAPI(QRDiscriminator):
    """docstring for QRDiscriminator_KerasAPI"""
    def __init__(self, **kwargs):
        super(L4QRDiscriminator_KerasAPI, self).__init__(**kwargs)

    def fit(self, train_sample, valid_sample):
        # prepare training set
        (x_train, y_train), (x_valid, y_valid) = self.make_training_datasets(train_sample, valid_sample)

        self.model = qr.LambdaQuantileRegression(quantile=self.quantile, x_mu_std=(np.mean(x_train), np.std(x_train)), **self.model_params).build()
        print(x_train)
        print(y_train)
        #gw = GetWeights()
        self.history = self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_sz, verbose=2, validation_data=(x_valid, y_valid), shuffle=True, \
                                      callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1), tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3, verbose=1)])

        #for epoch,weights in weights_dict.items():
        #    print("Weights for 2nd Layer of epoch #",epoch+1)
        #    print(weights)
        #    print("Bias for 2nd Layer of epoch #",epoch+1)
        #    print(weights)

        return self.history.history['loss'], self.history.history['val_loss']

    def predict(self, data):
        if isinstance(data, js.JetSample):
            data = data[self.mjj_key]
        xx = data #self.scale_input(data)
        #print("HERE")
        #print(xx)

        predicted = self.model.predict(xx).flatten() 
        # return self.unscale_output(predicted)
        return predicted
        



class LBSQRDiscriminator_KerasAPI(QRDiscriminator):
    """docstring for QRDiscriminator_KerasAPI"""
    def __init__(self, **kwargs):
        super(LBSQRDiscriminator_KerasAPI, self).__init__(**kwargs)

    def fit(self, train_sample, valid_sample):
        # prepare training set
        (x_train, y_train), (x_valid, y_valid) = self.make_training_datasets(train_sample, valid_sample)

        self.model = qr.LambdaBernsteinQuantileRegression(quantile=self.quantile, x_mu_std=(np.mean(x_train), np.std(x_train)), **self.model_params).build()
        print(x_train)
        print(y_train)
        #gw = GetWeights()
        self.history = self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_sz, verbose=2, validation_data=(x_valid, y_valid), shuffle=True, \
                                      callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1), tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3, verbose=1)])

        #for epoch,weights in weights_dict.items():
        #    print("Weights for 2nd Layer of epoch #",epoch+1)
        #    print(weights)
        #    print("Bias for 2nd Layer of epoch #",epoch+1)
        #    print(weights)

        return self.history.history['loss'], self.history.history['val_loss']

    def predict(self, data):
        if isinstance(data, js.JetSample):
            data = data[self.mjj_key]
        xx = data #self.scale_input(data)
        #print("HERE")
        #print(xx)

        predicted = self.model.predict(xx).flatten() 
        # return self.unscale_output(predicted)
        return predicted
        
        

class GBRDiscriminator(Discriminator):

    def fit(self, jet_sample):
        self.model = scikit.GradientBoostingRegressor(loss='quantile', alpha=1-self.quantile, learning_rate=.01, max_depth=2, verbose=2)



###############################
########## Vector QR ##########
###############################

class VDiscriminator(metaclass=ABCMeta):

    def __init__(self, quantiles, loss_strategy):
        self.loss_strategy = loss_strategy
        self.quantiles = quantiles
        self.mjj_key = 'mJJ'

    @abstractmethod
    def fit(self, jet_sample):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass 

    @abstractmethod
    def predict(self, data):
        '''predict cut for each example in data'''
        pass

    @abstractmethod
    def select(self, jet_sample):
        pass

    def __repr__(self):
        return f'{self.quantiles*100} qnts, {self.loss_strategy.title_str} strategy'


class VQRDiscriminator(VDiscriminator):

    def __init__(self, quantiles, loss_strategy, batch_sz=128, epochs=100, learning_rate=0.001, optimizer=tf.keras.optimizers.Adam, **model_params):
        VDiscriminator.__init__(self, quantiles, loss_strategy)
        self.batch_sz = batch_sz
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer() #optimizer(learning_rate)
        self.model_params = model_params

    @tf.function
    def training_step(self, x_batch, y_batch):
        # Open a GradientTape to record the operations run in forward pass
        # import ipdb; ipdb.set_trace()
        print("Z-Z-Z-Z-Z-Z-Z-Z-Z")
        with tf.GradientTape() as tape:
            predictions = self.model(x_batch, training=True)
            tf.print("\n y_batch:", y_batch, output_stream=sys.stdout)
            tf.print("\n y_batch shape:", y_batch.shape, output_stream=sys.stdout)
            loss_value = tf.math.reduce_mean(qr.vquantile_loss(y_batch, predictions, self.quantiles))

        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss_value

    def training_epoch(self, train_dataset):
        # Iterate over the batches of the dataset.
        train_loss = 0.
        for step, (x_batch, y_batch) in enumerate(train_dataset):
            loss_value = self.training_step(x_batch, y_batch)
            train_loss += loss_value
            if step % 10000 == 0:
                print("Step {}: lr {:.3e}, loss {:.4f}".format(step, self.optimizer.learning_rate.numpy(), loss_value))
        return train_loss / (step + 1)

    def valid_epoch(self, valid_dataset):
        # Iterate over the batches of the dataset.
        valid_loss = 0.
        for step, (x_batch, y_batch) in enumerate(valid_dataset):
            predictions = self.model(x_batch, training=False)
            valid_loss += tf.math.reduce_mean(qr.quantile_loss(y_batch, predictions, self.quantile))
        return valid_loss / (step + 1)


    def make_training_datasets(self, train_sample, valid_sample):
        x_train = train_sample[self.mjj_key]
        y_train = self.loss_strategy(train_sample)
        x_valid = valid_sample[self.mjj_key]
        y_valid = self.loss_strategy(valid_sample)
        return (x_train, y_train), (x_valid, y_valid)

    def fit(self, train_sample, valid_sample):
        # process the input
        (x_train, y_train), (x_valid, y_valid) = self.make_training_datasets(train_sample, valid_sample)
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(self.batch_sz)#.shuffle(self.batch_sz*10)
        valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(self.batch_sz)

        # build the regressor
        self.regressor = qr.QuantileRegressionV2(**self.model_params)
        # self.model = self.regressor.make_model(x_mean_std=(np.mean(x_train), np.std(x_train)), y_mean_std=(np.mean(y_train), np.std(y_train)))
        self.model = self.regressor.make_model(x_min_max=(np.min(x_train), np.max(x_train)), y_min_max=(np.min(y_train), np.max(y_train)))
        
        # build loss arrays and callbacks
        losses_train = []
        losses_valid = []
        train_stop = train.Stopper(optimizer=self.optimizer)

        # run training
        for epoch in range(self.epochs):
            start_time = time.time()
            losses_train.append(self.training_epoch(train_dataset))
            losses_valid.append(self.valid_epoch(valid_dataset))
            # print epoch results
            print('### [Epoch {} - {:.2f} sec]: train loss {:.3f}, val loss {:.3f} (mean / batch) ###'.format(epoch, time.time()-start_time, losses_train[-1], losses_valid[-1]))
            if train_stop.check_stop_training(losses_valid):
                print('!!! stopping training !!!')
                break

        return losses_train, losses_valid


    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path, custom_objects={'StdNormalization': layers.StdNormalization, 'StdUnnormalization': layers.StdUnnormalization}, compile=False)
        print('loaded model ', self.model)

    def predict(self, data):
        if isinstance(data, js.JetSample):
            data = data[self.mjj_key]
        #print("Horo")
        #print(data)
        return self.model(data, training=False)

    def select(self, jet_sample, whichquant):
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("LSDJASDJDAJSDJDASJDJASJDASD")
        print("Huru")
        print(len(jet_sample))
        loss_cut = self.predict(jet_sample)

        split_loss_cut = np.array(np.split(np.array(loss_cut),len(jet_sample)),dtype=object)   

        print(split_loss_cut)
        print(split_loss_cut.shape)
        if split_loss_cut[:,whichquant].shape == self.loss_strategy(jet_sample).shape:
            print("!!!!!!!!!!!!!!!!!")
            return self.loss_strategy(jet_sample) > split_loss_cut[:,whichquant]
        else:
            return self.loss_strategy(jet_sample) > loss_cut[1::2]

    def __repr__(self):
        return 'QR Cut: ' + VDiscriminator.__repr__(self)

class VQRv1Discriminator_KerasAPI(VQRDiscriminator):
    """docstring for QRDiscriminator_KerasAPI"""
    def __init__(self, **kwargs):
        super(VQRv1Discriminator_KerasAPI, self).__init__(**kwargs)

    def fit(self, train_sample, valid_sample):
        # prepare training set
        (x_train, y_train), (x_valid, y_valid) = self.make_training_datasets(train_sample, valid_sample)

        self.model = qr.VectorQuantileRegression(quantiles=self.quantiles, x_mu_std=(np.mean(x_train), np.std(x_train)), **self.model_params).build()
        self.history = self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_sz, verbose=2, validation_data=(x_valid, y_valid), shuffle=True, \
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1), tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3, verbose=1)])
       
        return self.history.history['loss'], self.history.history['val_loss']

    def predict(self, data,flatten=True):
        if isinstance(data, js.JetSample):
            data = data[self.mjj_key]
        xx = data #self.scale_input(data)

        predicted = self.model.predict(xx) 
        if flatten: 
        # return self.unscale_output(predicted)
            return predicted.flatten()
        return predicted


class VERv1Discriminator_KerasAPI(VQRDiscriminator):
    """docstring for QRDiscriminator_KerasAPI"""
    def __init__(self, **kwargs):
        super(VERv1Discriminator_KerasAPI, self).__init__(**kwargs)

    def fit(self, train_sample, valid_sample):
        # prepare training set
        (x_train, y_train), (x_valid, y_valid) = self.make_training_datasets(train_sample, valid_sample)

        self.model = qr.VectorExpectileRegression(quantiles=self.quantiles, x_mu_std=(np.mean(x_train), np.std(x_train)), **self.model_params).build()
        self.history = self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_sz, verbose=2, validation_data=(x_valid, y_valid), shuffle=True, \
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1), tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3, verbose=1)])
       
        return self.history.history['loss'], self.history.history['val_loss']

    def predict(self, data,flatten=True):
        if isinstance(data, js.JetSample):
            data = data[self.mjj_key]
        xx = data #self.scale_input(data)

        predicted = self.model.predict(xx) 
        if flatten: 
        # return self.unscale_output(predicted)
            return predicted.flatten()
        return predicted
    