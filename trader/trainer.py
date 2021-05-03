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

set_seeds(100)

instrument = "EUR_USD"
#loading data
base_data_folder_name = cfg.data_path + str(instrument) + "/"
train_folder = base_data_folder_name + "Train/"
valid_folder = base_data_folder_name + "Valid/"
test_folder = base_data_folder_name + "Test/"

assert os.path.exists(base_data_folder_name), "Base data folder DO NOT exists!"

train_filename = train_folder + "train.xlsx"
valid_filename = valid_folder + "valid.xlsx"
test_filename = test_folder + "test.xlsx"
train_labl_filename = train_folder + "trainlabels.xlsx"
valid_labl_filename = valid_folder + "validlabels.xlsx"
test_labl_filename = test_folder + "testlabels.xlsx"

train_data = pd.read_excel(train_filename, index_col=None,
                           header=0, engine='openpyxl') #, parse_dates = ["time"], index_col = "time")
test_data = pd.read_excel(test_filename, index_col=None,
                          header=0, engine='openpyxl') #, parse_dates = ["time"], index_col = "time")
train_labels = pd.read_excel(train_labl_filename,index_col=None,
                             header=0, engine='openpyxl') #, parse_dates = ["time"], index_col = "time")
test_labels = pd.read_excel(test_labl_filename, index_col=None,
                            header=0, engine='openpyxl') #, parse_dates = ["time"], index_col = "time")


all_cols = train_data.columns
cols = []
for col in all_cols:
    if 'lag' in col and "boll" not in col:
        cols.append(col)

print(train_data[cols].isnull().values.any() , train_labels["dir"].isnull().values.any())

logging.info("Creating the NN model...")
model = create_model(dropout = True, input_dim = len(cols))
#model = create_better_model(dropout = False, input_dim = len(cols))
logging.info("Training the NN model...")
model.fit(x = train_data[cols],
          y = train_labels["dir"],
          epochs = 80,
          verbose = True,
          validation_split = 0.2,
          shuffle = False,
          class_weight = cw(train_labels))

print("\n")
time.sleep(2)
logging.info("Evaluating the model on in-sample data (training data)")
model.evaluate(train_data[cols], train_labels["dir"]) # evaluate the fit on the train set
# pred = model.predict(train_data[cols])

logging.info("\nEvaluating the model on out-of-sample data (test data)")
model.evaluate(test_data[cols], test_labels["dir"])

# Todo: save the model once trained to reuse it without training
model.save(cfg.proj_path + "/TrainedModels/" + "DNN_model.h5")
