from keras.models import Sequential, load_model
from keras.callbacks import History, EarlyStopping, Callback
from keras import layers
from keras.engine.input_layer import Input
from keras import backend as K
from keras.models import Model
import matplotlib.pyplot as plt
from keras import objectives
import numpy as np
import os
from model._globals import Config
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# config
config = Config('config.yaml')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress tensorflow CPU speedup warnings


def get_gru_vae_model(X_train, logger, _id, train=False):
    '''Train gru vae model according to specifications in config.yaml or load pre-trained model.

    Args:
        X_train (np array):numpy array of training inputs with dimensions [timesteps,input_dim,]
        logger (object):logging object
        train (bool):If False,will attempt to load existing model from repo

    Returns:
        model (object):Trained Keras gru vae model
    '''
    cbs = [History(),
           EarlyStopping(monitor='vae_loss', patience=config.patience, min_delta=config.min_delta, verbose=0)]
    # build encoder model
    inputs = Input(shape=(X_train.shape[1], 1,))
    x = layers.Conv1D(config.filters[0], config.kernel_size, strides=config.strides, activation=config.activation,
                      )(inputs)
    x = layers.MaxPool1D(config.max_pooling)(x)
    x = layers.Conv1D(config.filters[1], config.kernel_size, strides=config.strides, activation=config.activation,
                      )(x)
    x = layers.MaxPool1D(config.max_pooling)(x)
    # shape info need to build decoder model
    shape = K.int_shape(x)
    x = layers.GRU(config.layers[0], activation=config.activation, return_sequences=True)(x)
    x = layers.Dropout(config.dropout)(x)

    x = layers.GRU(config.layers[0], activation=config.activation, return_sequences=False)(x)
    x = layers.Dropout(config.dropout)(x)
    # generate latent vector
    z_mean = layers.Dense(config.layers[1])(x)
    z_log_var = layers.Dense(config.layers[1])(x)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], config.layers[1]),
                                  mean=0., stddev=1.)
        return z_mean + K.exp(z_log_var) * epsilon

    z = layers.Lambda(sampling)([z_mean, z_log_var])

    # build decoder model
    z_decoded = layers.RepeatVector(shape[1])(z)
    z_decoded = layers.GRU(shape[2], activation=config.activation, return_sequences=True)(z_decoded)
    z_decoded = layers.GRU(shape[2], activation=config.activation, return_sequences=True)(z_decoded)
    z_decoded = layers.UpSampling1D(config.max_pooling)(z_decoded)

    def Conv1DTranspose(input_tensor, filters, kernel_size, strides, activation):
        x = layers.Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
        x = layers.Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1),
                                   activation=activation)(x)
        x = layers.Lambda(lambda x: K.squeeze(x, axis=2))(x)
        return x

    z_decoded = Conv1DTranspose(z_decoded, filters=config.filters[1], kernel_size=config.kernel_size,
                                activation=config.activation,
                                strides=config.strides)

    z_decoded = layers.Flatten()(z_decoded)
    z_decoded = layers.Dense(X_train.shape[1], activation=config.activation)(z_decoded)

    def vae_loss(inputs, z_decoded):
        z_decoded= K.expand_dims(z_decoded, -1)
        xent_loss = objectives.mse(inputs, z_decoded)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
        loss = xent_loss + kl_loss
        return loss

    vae = Model(inputs, z_decoded)

    if not train and os.path.exists(os.path.join('result/%s/models' % _id, "gru_vae_model.h5")):
        logger.info("Loading pre-trained model")
        return load_model(os.path.join('result/%s/models' % _id, "gru_vae_model.h5"),
                          custom_objects={'vae_loss': vae_loss, 'config': config})

    elif (not train and not os.path.exists(os.path.join('result/%s/models' % _id, "gru_vae_model.h5"))) or train:

        if not train:
            logger.info("Training new model from scratch.")

        vae.summary()
        vae.compile(optimizer=config.optimizer, loss=vae_loss)
        history = vae.fit(X_train, X_train, batch_size=config.lstm_batch_size, epochs=config.epochs,
                          callbacks=cbs, verbose=True)
        plt.plot(history.history['loss'])
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.savefig(os.path.join('result/%s/models' % _id, "gru_vae_model_loss.png"), dpi=300)
        vae.save(os.path.join('result/%s/models' % _id, "gru_vae_model.h5"))
        return vae


def _get_gru_vae_model(X_train, logger, _id, train=False):
    '''Train gru vae model according to specifications in config.yaml or load pre-trained model.

    Args:
        X_train (np array):numpy array of training inputs with dimensions [timesteps,input_dim,]
        logger (object):logging object
        train (bool):If False,will attempt to load existing model from repo

    Returns:
        model (object):Trained Keras gru vae model
    '''

    if not train and os.path.exists(os.path.join('data/%s/models' % _id, "model.h5")):
        logger.info("Loading pre-trained model")
        return load_model(os.path.join('data/%s/models' % _id, "model.h5"))

    elif (not train and not os.path.exists(os.path.join('data/%s/models' % _id, "model.h5"))) or train:

        if not train:
            logger.info("Training new model from scratch.")

        cbs = [History(),
               EarlyStopping(monitor='val_loss', patience=config.patience, min_delta=config.min_delta, verbose=0)]
        # build encoder model
        inputs = Input(shape=(X_train.shape[1], 1,))
        x = layers.Conv1D(config.filters[0], config.kernel_size, strides=config.strides, activation=config.activation,
                          )(inputs)
        x = layers.MaxPool1D(config.max_pooling)(x)
        x = layers.Conv1D(config.filters[1], config.kernel_size, strides=config.strides, activation=config.activation,
                          )(x)
        x = layers.MaxPool1D(config.max_pooling)(x)
        # shape info need to build decoder model
        shape = K.int_shape(x)
        x = layers.GRU(config.layers[0], activation=config.activation, return_sequences=True)(x)
        x = layers.Dropout(config.dropout)(x)

        x = layers.GRU(config.layers[0], activation=config.activation, return_sequences=False)(x)
        x = layers.Dropout(config.dropout)(x)
        # generate latent vector
        z_mean = layers.Dense(config.layers[1])(x)
        z_log_var = layers.Dense(config.layers[1])(x)

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], config.layers[1]),
                                      mean=0., stddev=1.)
            return z_mean + K.exp(z_log_var) * epsilon

        z = layers.Lambda(sampling)([z_mean, z_log_var])

        # build decoder model
        z_decoded = layers.RepeatVector(shape[1])(z)
        z_decoded = layers.GRU(shape[2], activation=config.activation, return_sequences=True)(z_decoded)
        z_decoded = layers.GRU(shape[2], activation=config.activation, return_sequences=True)(z_decoded)
        z_decoded = layers.UpSampling1D(config.max_pooling)(z_decoded)

        z_decoded = layers.Conv1D(config.filters[1], config.kernel_size, strides=config.strides,
                                  activation=config.activation,
                                  )(z_decoded)

        def Conv1DTranspose(input_tensor, filters, kernel_size, strides, activation):
            x = layers.Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
            x = layers.Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1),
                                       activation=activation)(x)
            x = layers.Lambda(lambda x: K.squeeze(x, axis=2))(x)
            return x

        z_decoded = Conv1DTranspose(z_decoded, filters=config.filters[1], kernel_size=config.kernel_size,
                                    activation=config.activation,
                                    strides=config.strides)

        z_decoded = layers.Flatten()(z_decoded)
        z_decoded = layers.Dense(X_train.shape[1], activation=config.activation)(z_decoded)

        class CustomVariationalLayer(layers.Layer):
            def vae_loss(self, x, z_decoded):
                x = K.flatten(x)
                z_decoded = K.flatten(z_decoded)
                xent_loss = objectives.binary_crossentropy(x, z_decoded)
                kl_loss = -.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
                return K.mean(xent_loss + kl_loss)

            def call(self, inputs, **kwargs):
                x = inputs[0]
                z_decoded = inputs[1]
                loss = self.vae_loss(x, z_decoded)
                self.add_loss(loss, inputs=inputs)
                return x

        y = CustomVariationalLayer()([inputs, z_decoded])

        vae = Model(inputs, y)
        vae.compile(optimizer=config.optimizer, loss=None)
        vae.summary()
        history = vae.fit(x=X_train, y=None, batch_size=config.lstm_batch_size, epochs=config.epochs, verbose=True)
        plt.plot(history.history['loss'])
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.savefig(os.path.join('result/%s/models' % _id, "model_loss.png"), dpi=300)
        vae.save(os.path.join('result/%s/models' % _id, "model.h5"))
        return vae
