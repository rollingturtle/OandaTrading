import pandas as pd
import numpy as np
import tpqoa
from datetime import datetime, timedelta
import time
import logging
import os
import os.path

import sys
sys.path.append('../')
import configs.config as cfg

from models.dnn import *
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

instrument = "EUR_USD"
#loading data
base_data_folder_name = cfg.data_path + str(instrument) + "/"
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

train_data = pd.read_csv(train_filename) #, parse_dates = ["time"], index_col = "time")
train_labels = pd.read_csv(train_labl_filename) #, parse_dates = ["time"], index_col = "time")


set_seeds(100)

all_cols = train_data.columns
cols = []
#features = ["dir", "sma", "boll", "min", "max", "mom", "vol"]
for col in all_cols:
    if 'lag' in col:
        cols.append(col)

model = create_model(dropout = True, input_dim = len(cols)) # hl = 3, hu = 50,
model.fit(x = train_data[cols], y = train_labels["dir"], epochs = 50, verbose = True,
          validation_split = 0.2, shuffle = False, class_weight = cw(train_labels))
