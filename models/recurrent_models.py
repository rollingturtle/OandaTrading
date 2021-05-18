import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l1, l2
from keras.optimizers import Adam
from keras.layers import LeakyReLU

import random
import numpy as np


def set_seeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

optimizer = keras.optimizers.Adam(learning_rate=0.0001)


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
