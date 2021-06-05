import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU, ReLU

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




'''
input_img = keras.Input(shape=(28, 28, 1))

x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

'''

def ffn(xtrain,
        rate=0.1,
        hu_list=(128, 32, 16)):
    xtrain_numpy = xtrain.to_numpy()
    i = Input(shape=xtrain_numpy[0].shape)
    x = i
    for d in hu_list:
        x= Dropout(rate, seed=100)(x)
        x = Dense(d)(x)
        x = ReLU()(x) #LeakyReLU(alpha=0.05)
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

def LSTM_dnn(dropout=0.0, inputs=None):
    inputs = keras.layers.Input(shape=(inputs.shape[1],inputs.shape[2]))
    lstm_out = keras.layers.LSTM(64, dropout=dropout)(inputs) # Todo: make dimension of LSTM a param
    outputs = keras.layers.Dense(1, activation="sigmoid")(lstm_out)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['acc'])
    model.summary()
    return model

def LSTM_dnn_all_states(dropout=0.0, inputs=None):
    inputs = keras.layers.Input(shape=(inputs.shape[1],inputs.shape[2]))
    lstm_out = keras.layers.LSTM(32,dropout=dropout,
                                 return_sequences=True)(inputs) # Todo: make dimension of LSTM a param
    lstm_out_flatten = keras.layers.Flatten()(lstm_out)
    #lstm_out_flatten = keras.layers.AveragePooling1D(pool_size=inputs.shape[1])(lstm_out)

    x = Dropout(dropout, seed=100)(lstm_out_flatten)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['acc'])
    model.summary()
    return model


def sequence_of_VAEs():
    pass

def recurrent_VAE_cell(inputs, previous_notes):
    inputs = keras.layers.Input(shape=(inputs.shape[1]))
    previous_notes = keras.layers.Input(shape=(previous_notes.shape[1],
                                               previous_notes.shape[2]))



    pass

##### Update this as models are added
available_models = {"dnn1": dnn1,
                    "LSTM_dnn": LSTM_dnn,
                    "LSTM_dnn_all_states": LSTM_dnn_all_states,
                    "ffn": ffn }

'''
https://keras.io/api/layers/recurrent_layers/rnn/

class MinimalRNNCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = backend.dot(inputs, self.kernel)
        output = h + backend.dot(prev_output, self.recurrent_kernel)
        return output, [output]

# Let's use this cell in a RNN layer:

cell = MinimalRNNCell(32)
x = keras.Input((None, 5))
layer = RNN(cell)
y = layer(x)

# Here's how to use the cell to build a stacked RNN:

cells = [MinimalRNNCell(32), MinimalRNNCell(64)]
x = keras.Input((None, 5))
layer = RNN(cells)
y = layer(x)


autoencoder
input = layers.Input(shape=(28, 28, 1))

# Encoder
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)

# Decoder
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

# Autoencoder
autoencoder = Model(input, x)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.summary()

'''