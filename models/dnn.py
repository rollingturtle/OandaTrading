import random
import numpy as np

import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l1, l2
from keras.optimizers import Adam
from keras.layers import LeakyReLU



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


def dnn1(hu_list=(128, 32, 16),
         dropout=False,
         rate=0.2,
         regularize=True,
         reg=l2(0.0005),  #l1(0.0005)
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
        model.add(Dense(layer_dim,  activity_regularizer=reg, activation="relu"))
        if dropout:
            model.add(Dropout(rate, seed=100))
        #model.add(LeakyReLU(alpha=0.05))

    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['acc'])
    model.summary()
    return model


def create_better_model(hu_list=[64, 32, 16],
                        dropout=False,
                        rate=0.2,
                        regularize=False,
                        reg=l1(0.0005),
                        optimizer=optimizer,
                        input_dim=None):

    if not regularize:
        reg = None

    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(input_dim,)))
    # model.add(tf.keras.layers.Dense(32, activation='relu'))
    # model = Sequential()

    model.add(tf.keras.layers.Dense(hu_list[0], activity_regularizer=reg, activation="relu"))
    if dropout:
        model.add(tf.keras.layers.Dropout(rate, seed=100))

    for layer_units in hu_list:
        model.add(tf.keras.layers.Dense(layer_units, activation="relu", activity_regularizer=reg))
        if dropout:
            model.add(tf.keras.layers.Dropout(rate, seed=100))

    # Final binary output
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['acc'])
    return model
