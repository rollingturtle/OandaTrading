import pandas as pd
import numpy as np
import tpqoa
from datetime import datetime, timedelta
import time
import logging
import os
import os.path
import pickle

import sys
sys.path.append('../')
import configs.config as cfg

from models.dnn import *
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import configs.EUR_USD_1 as eu

set_seeds(100)

instrument = eu.instrument

#loading data
base_data_folder_name = cfg.data_path + instrument + "/"
train_folder = base_data_folder_name + "Train/"
valid_folder = base_data_folder_name + "Valid/"
test_folder = base_data_folder_name + "Test/"

assert os.path.exists(base_data_folder_name), "Base data folder DO NOT exists!"

train_filename = train_folder + "train.csv"
valid_filename = valid_folder + "valid.csv"
test_filename = test_folder + "test.csv"
train_labl_filename = train_folder + "trainlabels.csv"
valid_labl_filename = valid_folder + "validlabels.csv"
test_labl_filename = test_folder + "testlabels.csv"

#, engine='openpyxl') #, parse_dates = ["time"], index_col = "time")
train_data = pd.read_csv(train_filename, index_col=None, header=0)
test_data = pd.read_csv(test_filename, index_col=None, header=0)
#valid not used for now, using keras support but that uses std and mean computed on the train+valid data
train_labels = pd.read_csv(train_labl_filename,index_col=None, header=0)
test_labels = pd.read_csv(test_labl_filename, index_col=None, header=0)


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

print("TEST if there ar NANs in data or labels")
print("NANs in DATA: ", train_data[lagged_cols].isnull().values.any(),
      "\nNANs in LABELS: ", train_labels["dir"].isnull().values.any())

logging.info("Creating the NN model...")
model = create_model(dropout = True, input_dim = len(lagged_cols))
#model = create_better_model(dropout = False, input_dim = len(cols))

logging.info("Training the NN model...")
model.fit(x = train_data[lagged_cols],
          y = train_labels["dir"],
          epochs = 80,
          verbose = True,
          validation_split = 0.2,
          shuffle = False,
          class_weight = cw(train_labels))

print("\n")
print("main: Evaluating the model on in-sample data (training data)")
model.evaluate(train_data[lagged_cols], train_labels["dir"]) # evaluate the fit on the train set

# testing predictions
print("main: testing predictions for later trading applications")
pred = model.predict(train_data[lagged_cols])
# print("cols are: ",lagged_cols)
# print("they are {}".format(len(lagged_cols)))
# print(train_data[lagged_cols])
# print(pred)

print("main: valuating the model on out-of-sample data (test data)")
model.evaluate(test_data[lagged_cols], test_labels["dir"])

# Todo: save the model once trained to reuse it without training
model.save(cfg.proj_path + "/TrainedModels/" + instrument +"/DNN_model.h5")
logging.info("Trained model save to " + cfg.proj_path + "/TrainedModels/" + instrument +"/DNN_model.h5")