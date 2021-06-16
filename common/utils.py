import pandas as pd
import numpy as np
# display
import matplotlib
if __name__ == "__main__":
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')

import tpqoa
from datetime import datetime, timedelta
import time
import logging
import os
import os.path
import pickle
import itertools

import sys
sys.path.append('../')


def make_features(df, sma_int, window, ref_price = None,
                  fwsma1 = 3, fwsma2 = 5, mom_win = 3, epsilon=10e-8,
                  live_trading=False):
    ''' Creates features  and targets, using ref_price and spread data as input
     sma_int and window are used to compute sma feature and bollinger related features
        '''

    # Todo: should i differentiate between features for learining and inference time?
    # training: i use t to create targets and lags to generate input data
    # inference testing: as above
    # inference live trading: t is THE MOST recent input
    #
    # log returns: fundamental quantity
    df["returns"] = np.log(df[ref_price] / df[ref_price].shift())

    df["sma_diff"] = df[ref_price].rolling(window).mean() - df[ref_price].rolling(sma_int).mean()

    df["boll1"] = (df[ref_price] - df[ref_price].rolling(window).mean()) #/ df[ref_price].rolling(window).std()
    df["boll_std"] = df[ref_price].rolling(window).std() # this can go to 0!!!

    df["min"] = df[ref_price].rolling(window).min() / df[ref_price] - 1
    df["max"] = df[ref_price].rolling(window).max() / df[ref_price] - 1
    df["mom"] = df["returns"].rolling(mom_win).mean()
    df["vol"] = df["returns"].rolling(window).std()
    df.dropna(inplace=True)

    df["boll"] = df["boll1"] /(epsilon +  df["boll_std"])
    df.drop(columns=["boll_std","boll1"], inplace=True)

    half_spread = 0.5 * np.array(df["spread"])
    # make targets. NO shifts here, as we will generate n lags (t-1, .., t-n) for each instant t
    # Then, legs will be the inputs to the model and the targets at current time t will be the labels/sought values
    # target: identify market direction
    df["dir"] = np.where(df["returns"] > 0, 1, 0)

    if not live_trading: # during trading I do not want to delete the most recent item
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=fwsma1)
        df["fw_sma1"] = df[ref_price].rolling(window=indexer).mean() # , min_periods=1
        df["fw_sma1"].shift(-1) # want lag1 to be the first element of the fw sma to predict

        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=fwsma2)
        df["fw_sma2"] = df[ref_price].rolling(window=indexer).mean() # , min_periods=1
        df["fw_sma2"].shift(-1)
        df.dropna(inplace=True)

        df["dir_sma1"] = np.where(df["fw_sma1"] > df[ref_price], 1, 0)
        df["dir_sma2"] = np.where(df["fw_sma2"] > df[ref_price], 1, 0)
        df.drop(columns=["fw_sma1", "fw_sma2"], inplace=True)
        # df["profit_over_spread"] = np.where(df["returns"] > np.log(1 + half_spread), 1, 0)  # profit over spread
        # df["loss_over_spread"] = np.where(df["returns"] < np.log(1 - half_spread), 1, 0)  # loss under spread
        # Todo: consider a label based on whether or not the rolling mean of the log returns
        # in the next steps is positive or negative

    return df

def make_lagged_features(df, features, lags=5):
        ''' Add lagged features to data. Default lag is 5'''
        #cols = []
        for f in features:
            for lag in range(1, lags + 1):
                col = "{}_lag_{}".format(f, lag)
                df[col] = df[f].shift(lag)
                #cols.append(col)

        df.dropna(inplace=True)
        return df

def creates_filenames_dict(instrument, namefiles_dict, cfg):
    '''

    :param instrument: oanda instrument for which data folders will be created
    :param namefiles_dict: dictionary in which we write filenames and folders for data storing
    :param cfg: config object holding main general parameters
    :return: dictionary of filenames and folder for data storage
    '''
    #folders
    namefiles_dict["base_data_folder_name"]  = cfg.data_path + str(instrument) + "/"
    namefiles_dict["model_folder"] = cfg.proj_path + "/TrainedModels/" + str(instrument) + "/"
    namefiles_dict["train_folder"]  = namefiles_dict["base_data_folder_name"]  + "Train/"
    namefiles_dict["valid_folder"]  = namefiles_dict["base_data_folder_name"]  + "Valid/"
    namefiles_dict["test_folder"] = namefiles_dict["base_data_folder_name"]  + "Test/"

    #files
    namefiles_dict["raw_data_file_name"] = namefiles_dict["base_data_folder_name"] + "raw_data.csv"
    namefiles_dict["raw_data_featured_resampled_file_name"] = namefiles_dict["base_data_folder_name"] + \
                                                              "raw_data_featured_resampled.csv"
    namefiles_dict["train_filename"] = namefiles_dict["train_folder"] + "train.csv"
    namefiles_dict["valid_filename"] = namefiles_dict["valid_folder"] + "valid.csv"
    namefiles_dict["test_filename"] = namefiles_dict["test_folder"] + "test.csv"
    namefiles_dict["train_labl_filename"] = namefiles_dict["train_folder"] + "traintargets.csv"
    namefiles_dict["valid_labl_filename"] = namefiles_dict["valid_folder"] + "validtargets.csv"
    namefiles_dict["test_labl_filename"] = namefiles_dict["test_folder"] + "testtargets.csv"
    namefiles_dict["params"] = namefiles_dict["train_folder"] + "params.pkl"
    return namefiles_dict

def find_string(list_of_strings, s):
    for i, ss in enumerate(list_of_strings):
        if s in ss:
            index = i
            return list_of_strings[index]
    return -1

def get_split_points(df, split_pcs):
    train_split = int(len(df) * split_pcs[0])
    val_split = int(len(df) * (split_pcs[0] + split_pcs[1]))
    return train_split, val_split


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
  else:
      print('Confusion matrix, without normalization')

  print(cm)

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()
  return