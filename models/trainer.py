import pandas as pd
import logging
import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append('../')

import configs.config as cfg
from models.dnn import *
import configs.EUR_PLN_2 as cfginst
import common.utils as u


set_seeds(100)

# set instrument to work with
instrument = cfginst.instrument

# get or generate datafiles files and folders, if do not exist
namefiles_dict = {}
namefiles_dict = u.creates_filenames_dict(cfginst.instrument, namefiles_dict, cfg)

#loading data
assert os.path.exists(namefiles_dict["base_data_folder_name"]), "Base data folder DO NOT exists!"
train_filename = namefiles_dict["train_filename"]
valid_filename = namefiles_dict["valid_filename"]
test_filename = namefiles_dict["test_filename"]
train_labl_filename = namefiles_dict["train_labl_filename"]
valid_labl_filename = namefiles_dict["valid_labl_filename"]
test_labl_filename = namefiles_dict["test_labl_filename"]

train_data = pd.read_csv(train_filename, index_col=None, header=0)
test_data = pd.read_csv(test_filename, index_col=None, header=0)
# valid not used for now, using keras support but that uses
# std and mean computed on the train+valid data
train_labels = pd.read_csv(train_labl_filename,index_col=None, header=0)
test_labels = pd.read_csv(test_labl_filename, index_col=None, header=0)

# exctract relevant columns for training (features!)
all_cols = train_data.columns
lagged_cols = []
for col in all_cols:
    if 'lag' in col:
        lagged_cols.append(col)

print("Lagged columns which are the input to the model")
print(lagged_cols)
print("how many Lagged columns:")
print(len(lagged_cols))
print(train_data.head())


assert (not train_data[lagged_cols].isnull().values.any()), "NANs in Training Data"
assert (not train_labels["dir"].isnull().values.any()), "NANs in LABELS"

logging.info("Creating the NN model...")
model = dnn1(dropout = True,
             rate=0.1,
             input_dim = len(lagged_cols))

logging.info("Training the NN model...")
model.fit(x = train_data[lagged_cols],
          y = train_labels["dir"],
          epochs = 53,
          verbose = True,
          validation_split = 0.1,
          shuffle = True,
          class_weight = cw(train_labels))

print("\n")
print("main: Evaluating the model on in-sample data (training data)")
model.evaluate(train_data[lagged_cols], train_labels["dir"], verbose=True) # evaluate the fit on the train set

# testing predictions
print("main: testing predictions for later trading applications")
pred = model.predict(train_data[lagged_cols], verbose=True)
# print("cols are: ",lagged_cols)
# print("they are {}".format(len(lagged_cols)))
# print(train_data[lagged_cols])
print(pred)

print("main: valuating the model on out-of-sample data (test data)")
model.evaluate(test_data[lagged_cols], test_labels["dir"], verbose=True)

# Todo: save the model under folder for specific configuration (See conf_name under EUR_USD_1.py)

model_folder = cfg.proj_path + "/TrainedModels/" + instrument
if not os.path.exists(model_folder):
    logging.info("trainer: specific model folder does not exist: creating it...")
    os.mkdir(model_folder)
    # Todo add choice to break out?

model.save(model_folder + "/DNN_model.h5")
print("main:Trained model save to " + model_folder +"/DNN_model.h5")