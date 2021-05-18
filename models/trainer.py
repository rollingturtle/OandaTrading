import tensorflow as tf
from tensorflow import keras
import pandas as pd
import logging
import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append('../')

import configs.config as cfg
from models.dnn import *
from models.recurrent_models import *
import configs.EUR_PLN_2 as cfginst
import common.utils as u

# Todo: make the trainer a class that takes in instrument configuration file and kind of model to train

def find_string(list_of_strings, s):
    index = -1
    for i,ss in enumerate(list_of_strings):
        if s in ss:
            index = i
            return list_of_strings[index]
    return -1

set_seeds(100)

# set instrument to work with
instrument = cfginst.instrument

# get or generate datafiles files and folders, if do not exist
namefiles_dict = {}
namefiles_dict = u.creates_filenames_dict(cfginst.instrument, namefiles_dict, cfg)

# Todo: make that if I want to work on an instrument, trainer calls getprepare to get the data if not present
#loading data
assert os.path.exists(namefiles_dict["base_data_folder_name"]), "Base data folder DO NOT exists!"
train_data = pd.read_csv(namefiles_dict["train_filename"], index_col=None, header=0)
test_data = pd.read_csv(namefiles_dict["test_filename"], index_col=None, header=0)
# valid not used for now, using keras support but that uses
# std and mean computed on the train+valid data
train_labels = pd.read_csv(namefiles_dict["train_labl_filename"],index_col=None, header=0)
test_labels = pd.read_csv(namefiles_dict["test_labl_filename"], index_col=None, header=0)

# Todo: make this step unified and linked to instrument specific configuration
# exctract relevant columns for training (features!)
all_cols = train_data.columns
lagged_cols = []
for col in all_cols:
    if 'lag' in col:
        lagged_cols.append(col)

# reoder features
lagged_cols_reordered = []
for lag in range(1,cfginst.lags+1):
    for feat in cfginst.features:
        r = find_string(all_cols, feat + "_lag_" + str(lag))
        if r != -1:
            lagged_cols_reordered.append(r)


print("reodered features are:", lagged_cols_reordered)

print("trainer.py: Lagged columns which are the input to the model:")
print(lagged_cols)
print("trainer.py: how many Lagged columns:")
print(len(lagged_cols))
print(train_data.head())

assert (not train_data[lagged_cols].isnull().values.any()), "NANs in Training Data"
assert (not train_labels["dir"].isnull().values.any()), "NANs in LABELS"

logging.info("Creating the NN model...")
logging.info("Training the NN model...")
DNN=False
LSTM=True


if DNN:
    model = dnn1(dropout = True,
                 rate=0.1,
                 input_dim = len(lagged_cols))
    model.fit(x = train_data[lagged_cols],
              y = train_labels["dir"],
              epochs = 53,
              verbose = True,
              validation_split = 0.1,
              shuffle = True,
              batch_size=64,
              class_weight = cw(train_labels))

else: #LSTM:
    numpy_train = train_data[lagged_cols_reordered].\
        to_numpy().reshape(-1, len(cfginst.features), cfginst.lags)
    print("numpy_train.shape ",numpy_train.shape)
    model = LSTM_dnn(dropout = True,
                 rate=0.1,
                 inputs = numpy_train)

    model.fit(x = numpy_train, #train_data[lagged_cols],
              y = train_labels["dir"].to_numpy(),
              epochs = 53,
              verbose = True,
              validation_split = 0.1,
              shuffle = True,
              batch_size=64,
              class_weight = cw(train_labels))

print("\n")
print("main: Evaluating the model on in-sample data (training data)")
model.evaluate(train_data[lagged_cols], train_labels["dir"], verbose=True) # evaluate the fit on the train set

# testing predictions
print("main: testing predictions for later trading applications")
pred = model.predict(train_data[lagged_cols], verbose=True)
print(pred)

print("main: valuating the model on out-of-sample data (test data)")
model.evaluate(test_data[lagged_cols], test_labels["dir"], verbose=True)

# Todo: save the model under folder for specific configuration (See conf_name under EUR_USD_1.py)
model_folder = namefiles_dict["model_folder"]
if not os.path.exists(model_folder):
    logging.info("trainer: specific model folder does not exist: creating it...")
    os.mkdir(model_folder)
    # Todo add choice to break out?

model.save(model_folder + "/DNN_model.h5")
print("main:Trained model save to " + model_folder +"/DNN_model.h5")