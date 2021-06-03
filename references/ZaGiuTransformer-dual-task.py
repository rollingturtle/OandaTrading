#!/usr/bin/env python
# coding: utf-8

# ### Testing Transformer for Stock **predictions**

# notes about next steps to take:
# 0) Test the following: smoothing solo dei dati di training, e fare il training con due task, uno di regressione, e uno di classificazione, in parallelo. Validare e Fare il Test su dati reali renderebbe l’intero sistema utilizzabile in quanto ad ogni nuovo minuto ricomputi lo smoothing, lo dai in pasto al modello il quale pero’ e’ stato trained per fare regression rispetto al valore reale del next minute, e per “impararlo meglio” anche a classificare l’evento secondo una label interessante o meno che creiamo noi nei dati, come hai fatto tu fino ad ora. Sono un po’ fuso ora…  a presto!
# 
# 1) make the aveage option just an option with a flag and not with doubling the model
# 
# 2) make the transformer implementation a full one mimicking what done in NLP Udemy course. This would open the possibility to play more with seq to seq, and not only with regression or classification based on the representations learned by the transformer encoder
# 
# 4) develop tools to detect the change of regime, from stationary to non stationary. The idea is that non stationarity express itself through transitories, to reach new states of stationarity which can be described by pdfs that we can learn. It would be important to recognize when this transitories begin to raise the "alarm", schedule more frequent trainings, one more recent data. And how to detect the end of a transitory and the return to a stationary regime? need a buffer period of training and check on test predictions. When accuracy reaches accetable level on both short and long term predictions , we can assume stationairy has come back and decrease the learning schedule and trust long term predictions again.

# In[1]:



import numpy as np
import pandas as pd
import os, datetime
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
print('Tensorflow version: {}'.format(tf.__version__))

import matplotlib.pyplot as plt
plt.style.use('seaborn')

import warnings
warnings.filterwarnings('ignore')


# In[2]:


from google.colab import drive
drive.mount('/drive')


# In[3]:


# check compute backend
tf.test.gpu_device_name()


# # HyperParameters

# In[4]:


# review if these hyperparams are appropriate 
batch_size = 32
seq_len = 128
epochs_num = 4 #35, 2 for testing

# attention related parameters
d_k = 256 # keys dims
d_v = 256 # value dims
n_heads = 12 # multihead attentions has mutliple heads
ff_dim = 256  # after attention we have 2 ffn in a sequence. This should be their internal size


# # Loading BTC and ETH data and adding smooothed columns

# In[5]:


data_path = "/drive/MyDrive/ML_DATA/ZaGiu/btc_eth_merged.csv"


# In[6]:


df = pd.read_csv(data_path, delimiter=',', 
                 usecols=['time', 'datetime', 'open', 'high', 'low', 'close'])#, 'volumeto'])

# making a copy of it for later comparison
import copy
dforig = copy.deepcopy(df)

# # Replace 0 to avoid dividing by 0 later on
# df['volumeto'].replace(to_replace=0, method='ffill', inplace=True) 
df.sort_values('time', inplace=True)

# Apply moving average with a window of moving_average_width minutes to all columns
moving_average_width = 100
df[['open_sm', 'high_sm', 'low_sm', 'close_sm']] = df[['open', 'high', 'low', 'close']].rolling(moving_average_width).mean()
# from documentation:
#   By default, the result is set to the right edge of the window. This can be changed to the center of the window by setting center=True. 

# Drop all rows with NaN values
df.dropna(how='any', axis=0, inplace=True) 
df.head()


# # Selecting data to work on: BTC only for now (to start with)

# In[7]:


df


# In[8]:


btc_data = df[['open', 'close', 'high', 'low', 'open_sm', 'high_sm', 
               'low_sm', 'close_sm','time', 'datetime']]
btc_data


# In[9]:


# Selecting a subset to keep computation short.
# After all, including a very long history may not help. It might be even harmful if we assume that the data generating process 
# is non stationary
# in the non statioary scenario however, we might take the position that the statistical changes are slow 
# wrt a limited time interval (more than 1 month is a lot but it 
# it is what I am taking here). In case shorten the history is not helping to deal with non stationarity, 
# conditioning absolute values like abs value of open could be of help
# here in the first instance we consider a limited short time in the past to make predictions. 
# The hope is that in that period the data generative process is stationary.
# so we can learn useful representations of historical data that can be useful to make predictions.
# Conersely, if the data process is not stationary, we may have little hope to build useful representations
# without conditioning on time or on a proxy of it, like absolute price for example
# last 50000 minutes from the dataset = 50000/1440 = 34.7 days
back_in_time = 50000 # minutes
btc_data = btc_data[-back_in_time:]
dforig = dforig[-back_in_time:]



# In[10]:


btc_data


# # # Visualizing BTC Data

# In[11]:


plt.figure(figsize = (18,9))
k_time = 5000 # time step for x axis
plt.plot(range(btc_data.shape[0]),(btc_data['low']+btc_data['high'])/2.0)
plt.xticks(range(0,btc_data.shape[0],k_time),btc_data['datetime'].loc[::k_time],rotation=45)
plt.xlabel('datetime',fontsize=18)
plt.ylabel('Mid Price(low+high)/2',fontsize=18)


# In[12]:


fig = plt.figure(figsize=(15,10))
st = fig.suptitle("BTC close Price", fontsize=20)
st.set_y(0.92)

k_time = 100000
ax1 = fig.add_subplot(211)
ax1.plot(df['close'], label='BTC close Price')
ax1.set_xticks(range(0, df.shape[0], k_time))
ax1.set_xticktargets(df['datetime'].loc[::k_time])
ax1.set_ylabel('close Price', fontsize=18)
ax1.legend(loc="upper left", fontsize=12)


# In[13]:


fig = plt.figure(figsize=(18,10))
st = fig.suptitle("BTC close Price Comparison - Real and Smoothed", fontsize=20)
st.set_y(0.92)

ax1 = fig.add_subplot(311)
ax1.plot(btc_data['close'], label='BTC close Price smoothed')
ax1.plot(dforig['close'][-back_in_time + moving_average_width:], label='BTC close Price original')
# ax1.set_xticks(range(0, df.shape[0], 1464))
# ax1.set_xticktargets(df['datetime'].loc[::1464])
ax1.set_ylabel('close Price', fontsize=18)
ax1.legend(loc="upper left", fontsize=12)

ax1 = fig.add_subplot(312)
ax1.plot(dforig['close'][-back_in_time + moving_average_width:], label='BTC close Price original')
# ax1.set_xticks(range(0, df.shape[0], 1464))
# ax1.set_xticktargets(df['datetime'].loc[::1464])
ax1.set_ylabel('close Price', fontsize=18)
ax1.legend(loc="upper left", fontsize=12)

ax1 = fig.add_subplot(313)
ax1.plot(btc_data['close_sm'], label='BTC close Price smoothed')
# ax1.set_xticks(range(0, df.shape[0], 1464))
# ax1.set_xticktargets(df['datetime'].loc[::1464])
ax1.set_ylabel('close Price', fontsize=18)
ax1.legend(loc="upper left", fontsize=12)


# # Classification Task Definitions, addition of target/targets columns for regression and calssifications

# In[14]:


print(btc_data['close'])
btc_data['close'].shift()


# In[15]:


# Generating targets for 3 binary classification tasks
# Need to generate a label for each instant
# targets assignment policy:
# if mid_(t+1) - mid_t > 5 ==>  (BUY)
# if mid_(t+1) - mid_t < -5 ==>  (SELL)
# otherwise ==> (DoNothing)
buy_threshold = 5 # positive variation of price from perious minute
sell_threshold = -5 # negative version of it

#params for high gain and long_term prediction
buy_threshold_long_interval = 40 #positive delta in dollars of interest in the long interval
sell_threshold_long_interval = -40
long_interval = 40 # duration in minutes of the long interval 

# .shift pushes AHEAD the sequence of 1 step, or n steps with .shift(n)
# I want to identify the events associated to a value increase, decrease, or not big change 
# associated to the current event
# This does not represent a strategy, just tasks to enforce more knowledge in the representations learned
# over which we do regression
close_deltas = btc_data['close'] - btc_data['close'].shift()
btc_data['buy_flag'] = close_deltas
btc_data['sell_flag'] = close_deltas
# these 2 operations above will generate the first line with NaNs

# removing NaNs
btc_data.dropna(subset=['buy_flag', 'sell_flag'], inplace=True) # drop rows where nan are present

# making them truth values (targets)
btc_data['buy_flag'] = btc_data['buy_flag']  > buy_threshold
btc_data['sell_flag'] = btc_data['sell_flag'] < sell_threshold
# 3rd redundant task
# do nothing might be a redundant task: check if forcing it adds any value...
btc_data['donothing_flag'] = (close_deltas > sell_threshold) & (close_deltas < buy_threshold)

'''
# example of how to setup longterm prediction task, out of the representation built so far
# without going autoregressive, as we should probably consider, including transofmer
# decoder... here only representations built by encoder will be used for that
# hoping that mixing per minute data and smoothed data on interval similar to 40 mind
# will provide enough info for the task

close_delta_long_term =  btc_data['close'] - btc_data['close'].shift(long_interval)
btc_data['buy_flag_long_term'] = close_delta_long_term
btc_data['sell_flag_long_term'] = close_delta_long_term

# removing NaN
btc_data.dropna(subset=['buy_buy_flag_long_termflag', 'sell_flag_long_term'], inplace=True) # drop rows where nan are present

# making them truth values (targets)
btc_data['buy_flag_long_term'] = btc_data['buy_flag_long_term']  > buy_threshold_long_interval
btc_data['sell_flag_long_term'] = btc_data['sell_flag_long_term'] < sell_threshold_long_interval
# 3rd redundant task
# do nothing might be a redundant task: check if forcing it adds any value...
btc_data['donothing_flag'] = (close_deltas > sell_threshold_long_interval) & (close_deltas < buy_threshold_long_interval)


'''

# trick to make targets numericals
btc_data = btc_data * 1
btc_data


# # Calculate normalized percentage change for all columns

# In[16]:


'''Calculate percentage change'''

btc_data['open'] = btc_data['open'].pct_change() # Create arithmetic returns column
btc_data['high'] = btc_data['high'].pct_change() # Create arithmetic returns column
btc_data['low'] = btc_data['low'].pct_change() # Create arithmetic returns column
btc_data['close'] = btc_data['close'].pct_change() # Create arithmetic returns column
btc_data['open_sm'] = btc_data['open_sm'].pct_change() # Create arithmetic returns column
btc_data['high_sm'] = btc_data['high_sm'].pct_change() # Create arithmetic returns column
btc_data['low_sm'] = btc_data['low_sm'].pct_change() # Create arithmetic returns column
btc_data['close_sm'] = btc_data['close_sm'].pct_change() # Create arithmetic returns column

btc_data.dropna(how='any', axis=0, inplace=True) # Drop all rows with NaN values


# In[17]:


btc_data


# # Normalize Price Change Percentage columns

# In[18]:


###############################################################################
'''Normalize price change columns'''

min_return = min(btc_data[['open', 'high', 'low', 'close']].min(axis=0)) #, 'open_sm', 'high_sm', 'low_sm', 'close_sm']].min(axis=0))
max_return = max(btc_data[['open', 'high', 'low', 'close']].max(axis=0)) #, 'open_sm', 'high_sm', 'low_sm', 'close_sm']].max(axis=0))
denominator = max_return - min_return
off_set = min_return

print("raw data...")
print(min_return)
print(max_return )                                                 
print("offset and denominator")
print(off_set, denominator)

# Min-max normalize price columns (0-1 range)
btc_data['open'] = (btc_data['open'] - off_set) / denominator
btc_data['high'] = (btc_data['high'] - off_set) / denominator
btc_data['low'] = (btc_data['low'] - off_set) / denominator
btc_data['close'] = (btc_data['close'] - off_set) / denominator


'''Normalize Smoothed Price change Columnss''' 
min_return = min(btc_data[['open_sm', 'high_sm', 'low_sm', 'close_sm']].min(axis=0))
max_return = max(btc_data[['open_sm', 'high_sm', 'low_sm', 'close_sm']].max(axis=0))                                                                  
denominator = max_return - min_return
off_set = min_return

print("Smoothed only")
print(min_return)
print(max_return)


print("offset and denominator")
print(off_set, denominator)
if 1:
    btc_data['open_sm'] = (btc_data['open_sm'] - off_set) / denominator
    btc_data['high_sm'] = (btc_data['high_sm'] - off_set) / denominator
    btc_data['low_sm'] = (btc_data['low_sm'] - off_set) / denominator
    btc_data['close_sm'] = (btc_data['close_sm'] - off_set) / denominator


# In[19]:


btc_data


# In[20]:


plt.figure(figsize = (18,9))
with plt.rc_context({'axes.edgecolor':'orange', 'xtick.color':'red', 'ytick.color':'green', 'figure.facecolor':'white'}):
  # Temporary rc parameters in effect
  k_time = 5000
  plt.plot(range(btc_data.shape[0]), btc_data['open'])
  plt.plot(range(btc_data.shape[0]), btc_data['open_sm'])


  plt.xticks(range(0,btc_data.shape[0],k_time),btc_data['datetime'].loc[::k_time],rotation=45)
  plt.xlabel('time',fontsize=18)
  plt.ylabel('Open change % real and smoothed',fontsize=18)
  #plt.legend(loc="upper left", fontsize=12)


# # Splitting Data into Train, Validation, Test sets

# In[21]:


'''Create training, validation and test split'''

times = sorted(btc_data.index.values)
last_10pct = sorted(btc_data.index.values)[-int(0.1*len(times))] # Last 10% of series
last_20pct = sorted(btc_data.index.values)[-int(0.2*len(times))] # Last 20% of series

btc_data_train = btc_data[(btc_data.index < last_20pct)]  # Training data are 80% of total data
btc_data_val = btc_data[(btc_data.index >= last_20pct) & (btc_data.index < last_10pct)]
btc_data_test = btc_data[(btc_data.index >= last_10pct)]


# Convert pandas columns into arrays
train_data = btc_data_train.values
val_data = btc_data_val.values
test_data = btc_data_test.values
print('Training data shape: {}'.format(train_data.shape))
print('Validation data shape: {}'.format(val_data.shape))
print('Test data shape: {}'.format(test_data.shape))


# # Plot daily data separation

# In[22]:


fig = plt.figure(figsize=(15,12))
st = fig.suptitle("Data Separation", fontsize=20)
st.set_y(0.95)

###############################################################################

ax1 = fig.add_subplot(211)
ax1.plot(np.arange(train_data.shape[0]), btc_data_train['close'], label='Training data')

ax1.plot(np.arange(train_data.shape[0], 
                   train_data.shape[0]+val_data.shape[0]), btc_data_val['close'], label='Validation data')

ax1.plot(np.arange(train_data.shape[0]+val_data.shape[0], 
                   train_data.shape[0]+val_data.shape[0]+test_data.shape[0]), btc_data_test['close'], label='Test data')
ax1.set_xlabel('Date')
ax1.set_ylabel('Normalized Closing Returns')
ax1.set_title("close Price", fontsize=18)
ax1.legend(loc="best", fontsize=12)


# # Create chunks (input sequences!) of training, validation and test data

# In[23]:


# open,	close,	high,	low,	open_sm,	close_sm,	high_sm,	low_sm,	mid,	buy_flag,	sell_flag,	donothing_flag,	datetime
# Training data
X_train, y_train = [], []
for i in range(seq_len, len(train_data)):
  X_train.append(train_data[i-seq_len:i, 0:8]) # Chunks of training data with a length of 128 df-rows
  a = np.array([train_data[:,7][i]]).astype('float32')
  b = np.array(train_data[:, 10:13][i-1]).astype('float32')
  #print(a,b, a.shape, b.shape)
  c = np.concatenate((a,b))
  #print(a,b,c)
  y_train.append(c) #Value of 7th column (close Price) of df-row 128+1 and tasks label
X_train, y_train = np.array(X_train).astype('float32'), np.array(y_train)


###############################################################################

# Validation data
X_val, y_val = [], []
for i in range(seq_len, len(val_data)):
    X_val.append(val_data[i-seq_len:i, 0:8])
    a = np.array([val_data[:,7][i]]).astype('float32')
    b = np.array(val_data[:, 10:13][i-1]).astype('float32')
    #print(a,b, a.shape, b.shape)
    c = np.concatenate((a,b))
    #print(a,b,c)
    y_val.append(c) #Value of 7th column (close Price) of df-row 128+1 and tasks label
X_val, y_val = np.array(X_val).astype('float32'), np.array(y_val).astype('float32')

###############################################################################

# Test data
X_test, y_test = [], []
for i in range(seq_len, len(test_data)):
    X_test.append(test_data[i-seq_len:i, 0:8])
    a = np.array([test_data[:,7][i]]).astype('float32')
    b = np.array(test_data[:, 10:13][i-1]).astype('float32')
    #print(a,b, a.shape, b.shape)
    c = np.concatenate((a,b))
    #print(a,b,c)
    y_test.append(c) #Value of 7th column (close Price) of df-row 128+1 and tasks label 
X_test, y_test = np.array(X_test).astype('float32'), np.array(y_test).astype('float32')

print('Training set shape', X_train.shape, y_train.shape)
print('Validation set shape', X_val.shape, y_val.shape)
print('Testing set shape' ,X_test.shape, y_test.shape)


# In[24]:


y_train


# # Time to Vector

# In[25]:


class Time2Vector(Layer):
  def __init__(self, seq_len, **kwargs):
    super(Time2Vector, self).__init__()
    self.seq_len = seq_len

  def build(self, input_shape):
    '''Initialize weights and biases with shape (batch, seq_len)'''
    self.weights_linear = self.add_weight(name='weight_linear',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)
    
    self.bias_linear = self.add_weight(name='bias_linear',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)
    
    self.weights_periodic = self.add_weight(name='weight_periodic',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)

    self.bias_periodic = self.add_weight(name='bias_periodic',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)

  def call(self, x):
    '''Calculate linear and periodic time features'''
    x = tf.math.reduce_mean(x[:,:,:4], axis=-1) 
    time_linear = self.weights_linear * x + self.bias_linear # Linear time feature
    time_linear = tf.expand_dims(time_linear, axis=-1) # Add dimension (batch, seq_len, 1)
    
    time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
    time_periodic = tf.expand_dims(time_periodic, axis=-1) # Add dimension (batch, seq_len, 1)
    return tf.concat([time_linear, time_periodic], axis=-1) # shape = (batch, seq_len, 2)
   
  def get_config(self): # Needed for saving and loading model with custom layer
    config = super().get_config().copy()
    config.update({'seq_len': self.seq_len})
    return config


# # PositionalEncoding (not used for now)

# In[26]:


# class PositionalEncoding(layers.Layer):

#     def __init__(self):
#         super(PositionalEncoding, self).__init__()
    
#     def get_angles(self, pos, i, d_model):
#         angles = 1 / np.power(10000., (2*(i//2)) / np.float32(d_model))
#         return pos * angles

#     def call(self, inputs):
#         seq_length = inputs.shape.as_list()[-2]
#         d_model = inputs.shape.as_list()[-1]
#         angles = self.get_angles(np.arange(seq_length)[:, np.newaxis],
#                                  np.arange(d_model)[np.newaxis, :],
#                                  d_model)
#         angles[:, 0::2] = np.sin(angles[:, 0::2])
#         angles[:, 1::2] = np.cos(angles[:, 1::2])
#         pos_encoding = angles[np.newaxis, ...]
#         return inputs + tf.cast(pos_encoding, tf.float32)


# # Transformer

# In[27]:


class SingleAttention(Layer):
  def __init__(self, d_k, d_v):
    super(SingleAttention, self).__init__()
    self.d_k = d_k
    self.d_v = d_v

  def build(self, input_shape):
    self.query = Dense(self.d_k, 
                       input_shape=input_shape, 
                       kernel_initializer='glorot_uniform', 
                       bias_initializer='glorot_uniform')
    
    self.key = Dense(self.d_k, 
                     input_shape=input_shape, 
                     kernel_initializer='glorot_uniform', 
                     bias_initializer='glorot_uniform')
    
    self.value = Dense(self.d_v, 
                       input_shape=input_shape, 
                       kernel_initializer='glorot_uniform', 
                       bias_initializer='glorot_uniform')

  def call(self, inputs): # inputs = (in_seq, in_seq, in_seq)
    q = self.query(inputs[0])
    k = self.key(inputs[1])

    attn_weights = tf.matmul(q, k, transpose_b=True)
    attn_weights = tf.map_fn(lambda x: x/np.sqrt(self.d_k), attn_weights)
    attn_weights = tf.nn.softmax(attn_weights, axis=-1)
    
    v = self.value(inputs[2])
    attn_out = tf.matmul(attn_weights, v)
    return attn_out    

#############################################################################

class MultiAttention(Layer):
  def __init__(self, d_k, d_v, n_heads):
    super(MultiAttention, self).__init__()
    self.d_k = d_k
    self.d_v = d_v
    self.n_heads = n_heads
    self.attn_heads = list()

  def build(self, input_shape):
    for n in range(self.n_heads):
      self.attn_heads.append(SingleAttention(self.d_k, self.d_v))  
    
    # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1]=7 
    self.linear = Dense(input_shape[0][-1], 
                        input_shape=input_shape, 
                        kernel_initializer='glorot_uniform', 
                        bias_initializer='glorot_uniform')

  def call(self, inputs):
    attn = [self.attn_heads[i](inputs) for i in range(self.n_heads)]
    concat_attn = tf.concat(attn, axis=-1)
    multi_linear = self.linear(concat_attn)
    return multi_linear   

#############################################################################

class TransformerEncoder(Layer):
  def __init__(self, d_k, d_v, n_heads, ff_dim, dropout=0.1, **kwargs):
    super(TransformerEncoder, self).__init__()
    self.d_k = d_k
    self.d_v = d_v
    self.n_heads = n_heads
    self.ff_dim = ff_dim
    self.attn_heads = list()
    self.dropout_rate = dropout

  def build(self, input_shape):
    self.attn_multi = MultiAttention(self.d_k, self.d_v, self.n_heads)
    self.attn_dropout = Dropout(self.dropout_rate)
    self.attn_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)

    self.ff_conv1D_1 = Conv1D(filters=self.ff_dim, kernel_size=1, activation='relu')
    # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1] = 7 
    self.ff_conv1D_2 = Conv1D(filters=input_shape[0][-1], kernel_size=1) 
    self.ff_dropout = Dropout(self.dropout_rate)
    self.ff_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)    
  
  def call(self, inputs): # inputs = (in_seq, in_seq, in_seq)
    attn_layer = self.attn_multi(inputs)
    attn_layer = self.attn_dropout(attn_layer)
    attn_layer = self.attn_normalize(inputs[0] + attn_layer)

    ff_layer = self.ff_conv1D_1(attn_layer)
    ff_layer = self.ff_conv1D_2(ff_layer)
    ff_layer = self.ff_dropout(ff_layer)
    ff_layer = self.ff_normalize(inputs[0] + ff_layer)
    return ff_layer 

  def get_config(self): # Needed for saving and loading model with custom layer
    config = super().get_config().copy()
    config.update({'d_k': self.d_k,
                   'd_v': self.d_v,
                   'n_heads': self.n_heads,
                   'ff_dim': self.ff_dim,
                   'attn_heads': self.attn_heads,
                   'dropout_rate': self.dropout_rate})
    return config


# # Model

# In[28]:


def create_model():
  '''Initialize time and transformer layers'''
  time_embedding = Time2Vector(seq_len)
  attn_layer1 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
  attn_layer2 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
  attn_layer3 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)

  '''Construct model'''
  #Training set shape (39860, 128, 8) (39860, 4)
  #Validation set shape (4870, 128, 8) (4870, 4)
  #Testing set shape (4870, 128, 8) (4870, 4)
  #TODO: make this 8 a parameter
  in_seq = Input(shape=(seq_len, 8)) 
  x = time_embedding(in_seq)
  x = Concatenate(axis=-1)([in_seq, x])
  x = attn_layer1((x, x, x))
  x = attn_layer2((x, x, x))
  x = attn_layer3((x, x, x))
  
  # todo: review the following, I do not like this avg pooling....
  x = GlobalAveragePooling1D(data_format='channels_first')(x)  

  xt = Dropout(0.1)(x)

  #regression tasl
  x = Dense(32, activation='relu')(xt)
  x = Dropout(0.1)(x)
  out = Dense(1, activation='linear', name="out")(x)

  #Classification on buy/ no buy
  xb = Dense(32, activation='relu')(xt) 
  xb = Dropout(0.1)(xb)
  outb = Dense(1, activation='sigmoid', name="outb")(xb)# use_bias=False???
  #Classification on sell/no sell
  xs = Dense(32, activation='relu')(xt)
  xs = Dropout(0.1)(xs)
  outs = Dense(1, activation='sigmoid', name="outs")(xs)# use_bias=False???
  #Classification on donothing/ do not do nothing
  xdn = Dense(32, activation='relu')(xt)
  xdn = Dropout(0.1)(xdn)
  outdn = Dense(1, activation='sigmoid', name="outdn")(xdn)

  model = Model(inputs=in_seq, outputs=[out, outb, outs, outdn])

  # define two dictionaries: one that specifies the loss method for
  # each output of the network along with a second dictionary that
  # specifies the weight per loss
  losses = {
    "out": "mse",
    "outb": "binary_crossentropy",
    "outs": "binary_crossentropy",
    "outdn": "binary_crossentropy",
  }
  lossWeights = {"out": 1., "outb": 1.0, "outs": 1.0, "outdn": 1.}
  # TODO: link loss and loss weigths dicts to model outputs and expected outputs!

  model.compile(loss=losses, loss_weights=lossWeights,
                optimizer='adam', metrics=['mae', 'mape'])
  
  return model


#TODO: how to visualize validation precision and recall during training?
model = create_model()
model.summary()

model_file = "/drive/MyDrive/ML_DATA/ZaGiu/Transformer+TimeEmbedding.hdf5"
callback = tf.keras.callbacks.ModelCheckpoint(model_file, 
                                              monitor='val_loss', 
                                              save_best_only=True, verbose=1)
# try to load from previous runs to make it faster hopefully
# # The model weights (that are considered the best) are loaded into the model.
# model.load_weights(checkpoint_filepath)

reuse_model = True
if not reuse_model:
    history = model.fit(X_train, [y_train[:,0],y_train[:,1],
                                  y_train[:,2],y_train[:,3]], 
                        batch_size=batch_size, 
                        epochs=20,  #epochs_num, 
                        callbacks=[callback],
                        validation_data=(X_val, [y_val[:,0],y_val[:,1],
                                                y_val[:,2],y_val[:,3]]))

model = tf.keras.models.load_model(model_file,
                                   custom_objects={'Time2Vector': Time2Vector, 
                                                   'SingleAttention': SingleAttention,
                                                   'MultiAttention': MultiAttention,
                                                   'TransformerEncoder': TransformerEncoder})




# In[29]:


###############################################################################
'''Calculate predictions and metrics'''

print("Calcuate predictions on the different datasets...")
#Calculate predication for training, validation and test data
#train_pred = model.predict(X_train)
#val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

print("Calcuate selected evaluation metrics..")
#Print evaluation metrics for all datasets
#train_eval = model.evaluate(X_train, [y_train[:,0],y_train[:,1],
#                              y_train[:,2],y_train[:,3]], verbose=0)
#val_eval = model.evaluate(X_val, [y_val[:,0],y_val[:,1],
#                              y_val[:,2],y_val[:,3]], verbose=0)
test_eval = model.evaluate(X_test, [y_test[:,0],y_test[:,1],
                                    y_test[:,2],y_test[:,3]], verbose=0)
print(' ')
print('Evaluation metrics')
#print('Training Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(train_eval[0], train_eval[1], train_eval[2]))
#print('Validation Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(val_eval[0], val_eval[1], val_eval[2]))
print('Test Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(test_eval[0], test_eval[1], test_eval[2]))


# # Displaying results

# In[30]:


###############################################################################
'''Display results'''
#loss: 1.8826 - out_loss: 4.0764e-05 - outb_loss: 0.6742 - outs_loss: 0.6738 - outdn_loss: 0.5345
#               out_mae: 0.0035 - out_mape: 68010.8828 
#               outb_mae: 0.4806 - outb_mape: 245931344.0000 
#               outs_mae: 0.4794 - outs_mape: 244003552.0000 
#               outdn_mae: 0.3543 - outdn_mape: 194948912.0000

fig = plt.figure(figsize=(15,20))
st = fig.suptitle("Transformer + TimeEmbedding Model", fontsize=22)
st.set_y(0.92)

#Plot training data results
ax11 = fig.add_subplot(311)
ax11.plot(train_data[:, 7], label='BTC mid Returns')
ax11.plot(np.arange(seq_len, train_pred[0].shape[0]+seq_len), train_pred[0][:,0], linewidth=3, 
          label='Predicted BTC mid Returns')
ax11.set_title("Training Data", fontsize=18)
ax11.set_xlabel('Date')
ax11.set_ylabel('BTC mid Returns')
ax11.legend(loc="best", fontsize=12)

#Plot validation data results
ax21 = fig.add_subplot(312)
ax21.plot(val_data[:, 7], label='BTC Closing Returns')
ax21.plot(np.arange(seq_len, val_pred[0].shape[0]+seq_len), val_pred[0][:,0], linewidth=3, 
          label='Predicted BTC mid Returns')
ax21.set_title("Validation Data", fontsize=18)
ax21.set_xlabel('Date')
ax21.set_ylabel('BTC Closing Returns')
ax21.legend(loc="best", fontsize=12)

#Plot test data results
ax31 = fig.add_subplot(313)
ax31.plot(test_data[:, 7], label='BTC Closmiding Returns')
ax31.plot(np.arange(seq_len, test_pred[0].shape[0]+seq_len), test_pred[0][:,0], linewidth=3, 
          label='Predicted BTC mid Returns')
ax31.set_title("Test Data", fontsize=18)
ax31.set_xlabel('Date')
ax31.set_ylabel('BTC mid Returns')
ax31.legend(loc="best", fontsize=12)


# # Precision and Recall of the binary classification tasks

# In[31]:


print("Shape of predictions:")
print([test_pred[l].shape for l in range(0,4)])
print("Shapes of targets:")
print([y_test[:,l].shape for l in range(0,4)])


# In[50]:


from sklearn.metrics import confusion_matrix
kind_of_task = dict()
kind_of_task[0] = "BUY"
kind_of_task[1] = "SELL"
kind_of_task[2] = "DONOTHING"

'''
Precision: ability of a classification model to return only relevant instances
Recall: ability of a classification model to identify all relevant instances
'''

thresholds = np.array(range(25,95,5))/100
lx = thresholds.shape[0]
precision = np.zeros((3,lx))
recall = np.zeros((3,lx))
F1 = np.zeros((3,lx))

for ll in range(3):
    l = ll + 1
    print("\n\n\n***Confusion Matrix for case **{}** ".format(kind_of_task[ll]))
    col = 0
    for t in thresholds:
        yp = np.zeros(len(y_test[:,l]))

        print("USING threshold to set predictions to 1 at {}".format(t))
        yp[(np.squeeze(test_pred[l])) > t] = 1# pred beyond the threshold, set to 1

        c = confusion_matrix(y_test[:,l], yp)
        #print("Confusion matrix for this binary task is:")
        #print(c)
        #print(c.sum())

        #print('total 0:',(y_test[:,l]==0).sum())
        #print(c/sum(c))

        precision_s = c[1,1]/sum(c[:,1]) 
        recall_s = c[1,1]/sum(c[1,:])
        F1_s = 2 * (precision_s * recall_s)/(precision_s + recall_s)

        precision[ll, col] = precision_s
        recall[ll, col] = recall_s
        F1[ll, col] = F1_s

        print("Precision: {}\nRecall: {} \nF1: {}".format(precision_s, recall_s,                                                          F1_s))
        col+=1


fig = plt.figure(figsize=(18,20))
with plt.rc_context({'axes.edgecolor':'orange', 'xtick.color':'red', 'ytick.color':'green', 'figure.facecolor':'white'}):
    st = fig.suptitle("Recall and Precision as function of 1-threshold", fontsize=20)
    st.set_y(0.92)

    ax1 = fig.add_subplot(311)
    ax1.plot(thresholds, precision[0,:], label='BUY',  marker=11)
    ax1.plot(thresholds, precision[1,:], label='SELL',  marker=11)
    ax1.plot(thresholds, precision[2,:], label='DONOTHING',  marker=11)
    ax1.set_ylabel('Precision', fontsize=18)
    ax1.set_xlabel('1-Threshold')
    ax1.legend(loc="upper left", fontsize=12)

    ax2 = fig.add_subplot(312)
    ax2.plot(thresholds, recall[0,:], label='BUY',  marker=11)
    ax2.plot(thresholds, recall[1,:], label='SELL',  marker=11)
    ax2.plot(thresholds, recall[2,:], label='DONOTHING',  marker=11)
    ax2.set_ylabel('Recall', fontsize=18)
    ax2.set_xlabel('1-Threshold')
    ax2.legend(loc="upper left", fontsize=12)

    ax3 = fig.add_subplot(313)
    ax3.plot(thresholds, F1[0,:], label='BUY',  marker=11)
    ax3.plot(thresholds, F1[1,:], label='SELL',  marker=11)
    ax3.plot(thresholds, F1[2,:], label='DONOTHING',  marker=11)
    ax3.set_ylabel('F1', fontsize=18)
    ax3.set_xlabel('1-Threshold')
    ax3.legend(loc="upper left", fontsize=12)


# # Model architecture overview

# In[ ]:


tf.keras.utils.plot_model(
    model,
    to_file="BTC_Transformer+TimeEmbedding.png",
    show_shapes=True,
    show_layer_names=True,
    expand_nested=True,
    dpi=96,)


# # Model metrics

# In[ ]:


'''Display model metrics'''

fig = plt.figure(figsize=(15,20))
st = fig.suptitle("Transformer + TimeEmbedding Model Metrics", fontsize=22)
st.set_y(0.92)

#Plot model loss
ax1 = fig.add_subplot(311)
ax1.plot(history.history['loss'], label='Training loss (MSE)')
ax1.plot(history.history['val_loss'], label='Validation loss (MSE)')
ax1.set_title("Model loss", fontsize=18)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (MSE)')
ax1.legend(loc="best", fontsize=12)

#Plot MAE
ax2 = fig.add_subplot(312)
ax2.plot(history.history['mae'], label='Training MAE')
ax2.plot(history.history['val_mae'], label='Validation MAE')
ax2.set_title("Model metric - Mean average error (MAE)", fontsize=18)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Mean average error (MAE)')
ax2.legend(loc="best", fontsize=12)

#Plot MAPE
ax3 = fig.add_subplot(313)
ax3.plot(history.history['mape'], label='Training MAPE')
ax3.plot(history.history['val_mape'], label='Validation MAPE')
ax3.set_title("Model metric - Mean average percentage error (MAPE)", fontsize=18)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Mean average percentage error (MAPE)')
ax3.legend(loc="best", fontsize=12)

