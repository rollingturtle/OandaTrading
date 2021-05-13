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
    ''' Creates features  and labels, using ref_price as input and hspread_ptc
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

    # log returns: fundamental quantity
    df["returns"] = np.log(df[ref_price] / df[ref_price].shift())

    # same labels to identify direction and amount of change
    df["dir"] = np.where(df["returns"] > 0, 1, 0)  # market direction
    df["profit_over_spread"] = np.where(df["returns"] > np.log(1 + hspread_ptc), 1, 0)  # profit over spread
    df["loss_over_spread"] = np.where(df["returns"] < np.log(1 - hspread_ptc), 1, 0)  # loss under spread

    # features: watch out for some of them to diverge (boll)
    df["sma"] = df[ref_price].rolling(window).mean() - df[ref_price].rolling(sma_int).mean()

    df["boll1"] = (df[ref_price] - df[ref_price].rolling(window).mean()) #/ df[ref_price].rolling(window).std()
    df["boll_std"] = df[ref_price].rolling(window).std() # todo: this can go to 0!!!
    df["min"] = df[ref_price].rolling(window).min() / df[ref_price] - 1
    df["max"] = df[ref_price].rolling(window).max() / df[ref_price] - 1
    df["mom"] = df["returns"].rolling(3).mean()  # todo make this a param
    df["vol"] = df["returns"].rolling(window).std()
    df.dropna(inplace=True)

    df["boll"] = df["boll1"] * df["boll_std"]  # TODO: replace for test /with *... see the meaning of rescaling with std
    #df.drop(columns=["boll_std","boll1"], inplace=True)
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