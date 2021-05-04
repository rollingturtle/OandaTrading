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


def make_features(df, sma_int, window, hspread_ptc, ref_price):

    # log returns: fundamental quantity
    df["returns"] = np.log(df[ref_price] / df[ref_price].shift())

    # same labels to identify direction and amount of change
    df["dir"] = np.where(df["returns"] > 0, 1, 0)  # market direction
    df["profit_over_spread"] = np.where(df["returns"] > np.log(1 + hspread_ptc), 1, 0)  # profit over spread
    df["loss_over_spread"] = np.where(df["returns"] < np.log(1 - hspread_ptc), 1, 0)  # loss under spread

    # features: watch out for some of them to diverge (boll)
    df["sma"] = df[ref_price].rolling(window).mean() - df[ref_price].rolling(sma_int).mean()

    df["boll"] = (df[ref_price] - df[ref_price].rolling(window).mean()) #/ df[ref_price].rolling(window).std()
    df["boll_std"] = df[ref_price].rolling(window).std()
    df["min"] = df[ref_price].rolling(window).min() / df[ref_price] - 1
    df["max"] = df[ref_price].rolling(window).max() / df[ref_price] - 1
    df["mom"] = df["returns"].rolling(3).mean()  # todo make this a param
    df["vol"] = df["returns"].rolling(window).std()
    df.dropna(inplace=True)

    df["boll"] = df["boll"] / df["boll_std"]
    df.drop(columns=["boll_std"], inplace=True)
    #print("utils.make_features: df[boll]: ", df["boll"])

    return df


def make_lagged_features(df, features, lags=5):
        ''' Add lagged features to data. Default lag is 5'''
        cols = []
        for f in features:
            for lag in range(1, lags + 1):
                col = "{}_lag_{}".format(f, lag)
                df[col] = df[f].shift(lag)
                cols.append(col)

        df.dropna(inplace=True)
        return df