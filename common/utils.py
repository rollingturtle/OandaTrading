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


def make_features(df, sma_int, window, ref_price = None, fwsma1 = 3, fwsma2 = 5, mom_win = 3, epsilon=10e-8):
    ''' Creates features  and targets, using ref_price as input and half_spread
    (estimation of half cost for each position.
     sma_int and window are used to compute sma feature and bollinger related features

     Add some features out of the instrument raw data.
        Any technical indicator can be added as feature
        pip = 0.0001 fourth price decimal
        spread = 1.5
        spread in currency units = 1.5 * 0.0001

        e.g. ASK price > BID price.
        ASK - BID = Spread
        Spread can be expressed in PIPS
        PIP = (0.0001)^-1 * (ASK - BID) in the reference currency (the one we use to pay what we buy)
        e.g.:for the amount of 1$, means you pay 1.00015$ = 1 *(1 + 0,00015)

        Here we use estimate proportional transaction cost as 0.007%

         lret(t) = log( price(t) / price(t-1))

        Assume positions like:
        price(t) > price(t-1) *(1 + 0,00007)  BUY (go long) ==> lret(t) > lg(1 + 0,00007)  go long
        price(t) < price(t-1) * (1 - 0,,0007) SELL (go short)==> lret(t) < lg(1 - 0,00007) go short
        '''

    # Todo: now df contains o,l,h, volume and spread data along with reference price
    # Todo: add then new features AND above all, new tasks like:
    # Todo: predict the sma direction computed like [now, now+1, ... now+K]
    #
    # log returns: fundamental quantity
    df["returns"] = np.log(df[ref_price] / df[ref_price].shift())

    df["sma_diff"] = df[ref_price].rolling(window).mean() - df[ref_price].rolling(sma_int).mean()

    df["boll1"] = (df[ref_price] - df[ref_price].rolling(window).mean()) #/ df[ref_price].rolling(window).std()
    df["boll_std"] = df[ref_price].rolling(window).std() # this can go to 0!!!

    df["min"] = df[ref_price].rolling(window).min() / df[ref_price] - 1
    df["max"] = df[ref_price].rolling(window).max() / df[ref_price] - 1
    df["mom"] = df["returns"].rolling(mom_win).mean()  # todo make this 3 a param
    df["vol"] = df["returns"].rolling(window).std()
    df.dropna(inplace=True)

    df["boll"] = df["boll1"] /(epsilon +  df["boll_std"])
    df.drop(columns=["boll_std","boll1"], inplace=True)


    half_spread = 0.5 * np.array(df["spread"])
    # make targets
    # target: identify market direction
    df["dir"] = np.where(df["returns"] > 0, 1, 0)

    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=fwsma1)
    df["fw_sma1"] = df[ref_price].rolling(window=indexer).mean() # , min_periods=1
    df["fw_sma1"].shift()

    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=fwsma2) # Todo: add param for this
    df["fw_sma2"] = df[ref_price].rolling(window=indexer).mean() # , min_periods=1
    df["fw_sma2"].shift()
    df.dropna(inplace=True)

    df["dir_sma1"] = np.where(df["fw_sma1"] > df[ref_price], 1, 0)
    df["dir_sma2"] = np.where(df["fw_sma2"] > df[ref_price], 1, 0)
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