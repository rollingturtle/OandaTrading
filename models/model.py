import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU

def set_seeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def cw(df):
    c0, c1 = np.bincount(df["dir"])
    w0 = (1 / c0) * (len(df)) / 2
    w1 = (1 / c1) * (len(df)) / 2
    print("cw: using class 0 weight {}, class 1 weight {}".format(w0,w1))
    return {0: w0, 1: w1}

optimizer = Adam(lr=0.0001)

#Todo: make the list of available models in a pythonic way... this sucks

############################# model definitions #################################


def ffn(xtrain,
        rate=0.2,
        hu_list=(128, 32, 16)):
    xtrain_numpy = xtrain.to_numpy()
    i = Input(shape=xtrain_numpy[0].shape)
    x = i
    for d in hu_list:
        x= Dropout(rate, seed=100)(x)
        x = Dense(d)(x)
        x = LeakyReLU(alpha=0.05)(x)
    x = Dense(1, activation="sigmoid")(x)

    model = Model(i, x)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['acc'])
    model.summary()
    return model

def dnn1(hu_list=(128, 32, 16),
         dropout=False,
         rate=0.2,
         regularize=True,
         reg=l2(0.0005),
         optimizer=optimizer,
         input_dim=None):
    '''
    Implements simple feedforward fully connected layer
    :param hu_list: tuple of dimensions, one for each layer
    :param dropout: boolean
    :param rate: dropout rate
    :param regularize: whether to add further regularization
    :param reg: the kind or regularizer to add
    :param optimizer: optimizer to use
    :param input_dim: dimension of inputs
    :return: keras model
    '''

    if not regularize:
        reg = None
    model = Sequential()
    model.add(Dense(hu_list[0], input_dim=input_dim, activity_regularizer=reg)) #, activation="relu"))
    if dropout:
        model.add(Dropout(rate, seed=100))
    model.add(LeakyReLU(alpha=0.05))
    for layer_dim in hu_list[1:]:
        model.add(Dense(layer_dim,  activity_regularizer=reg)) #, activation="relu"))
        if dropout:
            model.add(Dropout(rate, seed=100))
        model.add(LeakyReLU(alpha=0.05))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['acc'])
    model.summary()
    return model


def LSTM_dnn(dropout=False,
         rate=0.2,
         regularize=True,
         reg=l2(0.0005),  #l1(0.0005)
         inputs=None):

    #inputs = np.array(inputs)
    #inputs = inputs.reshape(inputs.shape[1], inputs.shape[2])
    inputs = keras.layers.Input(shape=(inputs.shape[1],inputs.shape[2])) #, inputs.shape[2]))
    lstm_out = keras.layers.LSTM(32)(inputs)
    outputs = keras.layers.Dense(1, activation="sigmoid")(lstm_out)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['acc'])
    model.summary()


    # # create and fit the LSTM network
    # model = Sequential()
    # model.add(LSTM(4, input_shape=(1, look_back)))
    # model.add(Dense(1))
    # model.compile(loss='mean_squared_error', optimizer='adam')
    # model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    return model

def sequence_of_VAEs():
    pass

def recurrent_VAE():
    pass


available_models = {"dnn1": dnn1, "LSTM_dnn": LSTM_dnn, "ffn": ffn }
