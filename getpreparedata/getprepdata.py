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


class OandaDataCollector():
    '''
    Class that downloads data from Oanda for a specific instrument, creates feature, and lagged features.
    It can save the data into a set of folders defined in the config module
    '''
    def __init__(self, instrument,
                 conf_file
                 ):

        self.raw_data = None
        self.raw_data_resampled = None
        self.last_bar = None
        self.api_oanda = tpqoa.tpqoa(conf_file)
        self.instrument = instrument

        self.base_data_folder_name = cfg.data_path + str(self.instrument) + "/"
        self.train_folder = self.base_data_folder_name + "Train/"
        self.valid_folder = self.base_data_folder_name + "Valid/"
        self.test_folder = self.base_data_folder_name + "Test/"

        if os.path.exists(self.base_data_folder_name):
            logging.info("Base folder exists: overwriting files!")
            # Todo add choice to break out
        else:
            logging.info("Non existent Base folder: creating it...")
            os.mkdir(self.base_data_folder_name)
            os.mkdir(self.train_folder)
            os.mkdir(self.valid_folder)
            os.mkdir(self.test_folder)

        self.train_filename = self.train_folder + "train.xlsx"
        self.valid_filename = self.valid_folder + "valid.xlsx"
        self.test_filename = self.test_folder + "test.xlsx"

        self.train_labl_filename = self.train_folder + "trainlabels.xlsx"
        self.valid_labl_filename = self.valid_folder + "validlabels.xlsx"
        self.test_labl_filename = self.test_folder + "testlabels.xlsx"
        return

    def get_most_recent(self, granul="S5", days = 2):
        '''
        Get the most recent data for the instrument for which the object was created
        :param granul: base frequency for raw data being downloaded
        :param days: number of past days (from today) we are downloading data for
        '''

        # set start and end date for historical data retrieval
        now = datetime.utcnow()
        now = now - timedelta(microseconds = now.microsecond) # Oanda does not deal with microseconds
        past = now - timedelta(days = days)

        # get historical data up to now
        logging.info("get_most_recent: calling tpqoa get_history....")
        self.raw_data = self.api_oanda.get_history(
            instrument = self.instrument,
            start = past, end = now,
            granularity = granul,
            price = "M",
            localize = False).dropna() #c.dropna().to_frame()
        return

    def make_features(self, window = 10, sma_int=5, hspread_ptc=0.007):
        '''
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
        '''
        df = self.raw_data
        ref_price = 'c'  #close price as reference
        # Todo: do this check better
        logging.info("Resampled DF")
        assert len(df[ref_price]) > sma_int, \
            "the dataframe lenght is not greater than the Simple Moving Average interval"
        assert len(df[ref_price]) > window, \
            "the dataframe lenght is not greater than the Bollinger window"

        # log returns: fundamental quantity
        df["returns"] = np.log(df[ref_price] / df[ref_price].shift())

        # same labels to identify direction and amount of change
        df["dir"] = np.where(df["returns"] > 0, 1, 0)  # market direction
        # Todo 1 - review these: likely to be wrong! returns are logret, ptc is estimated proportional cost per transaction
        df["profit_over_spread"] = np.where(df["returns"] > np.log(1 + hspread_ptc), 1, 0)  # profit over spread
        df["loss_over_spread"] = np.where(df["returns"] < np.log(1 - hspread_ptc), 1, 0)  # loss under spread
        df["sma"] = df[ref_price].rolling(window).mean() - df[ref_price].rolling(sma_int).mean()
        df["boll"] = (df[ref_price] - df[ref_price].rolling(window).mean()) / df[ref_price].rolling(window).std()
        df["min"] = df[ref_price].rolling(window).min() / df[ref_price] - 1
        df["max"] = df[ref_price].rolling(window).max() / df[ref_price] - 1
        df["mom"] = df["returns"].rolling(3).mean()  #todo make this a param
        df["vol"] = df["returns"].rolling(window).std()

        df.dropna(inplace=True)
        return

    def resample_data(self,  brl="1min"):
        '''
        Resample data already obtained to the frequency defined by brl
        :param brl: default 1min
        '''
        bar_length = pd.to_timedelta(brl)
        # resampling data at the desired bar length,
        # holding the last value of the bar period (.last())
        # label right is to set the index to the end of the bar time
        # dropna is here to remove weekends.
        # iloc to remove the last bar/row typically incomplete
        self.raw_data_resampled = self.raw_data.resample(bar_length,
                            label = "right").last().dropna().iloc[:-1]
        return

    def make_lagged_features(self, lags=5):
        ''' Add lagged features to data. Default lag is 5'''
        cols = []
        features = ["dir", "sma", "boll", "min", "max", "mom", "vol"]
        for f in features:
            for lag in range(1, lags + 1):
                col = "{}_lag_{}".format(f, lag)
                self.raw_data[col] = self.raw_data[f].shift(lag)
                cols.append(col)
        self.raw_data.dropna(inplace=True)
        return

    def make_3_datasets(self, split_pcs=(0.7, 0.15, 0.15)):
        '''
        Generate 3 datasets for ML training/evaluation/test and save to files.
        Default percentages are (0.7, 0.15, 0.15)
        '''

        if self.raw_data_resampled is not None: # it was populated in precedence
            df = self.raw_data_resampled
        else:
            df = self.raw_data

        assert sum(split_pcs) == 1, "make_3_datasets: split points are not dividing the unity"

        train_split = int(len(df) * split_pcs[0])
        val_split = int(len(df) *(split_pcs[0] + split_pcs[1]))

        self.train_ds = df.iloc[:train_split].copy()
        self.validation_ds = df.iloc[train_split:val_split].copy()
        self.test_ds = df.iloc[val_split:].copy()
        return

    def standardize(self):
        '''
        standardize the 3 datasets, using mean and std from train dataset
        to be called after make_3_datasets
        '''
        mu, std = self.train_ds.mean(), self.train_ds.std()

        # standardize all of them using training data derived statistics
        self.train_ds_std = (self.train_ds - mu) / std
        self.validation_ds_std = (self.validation_ds - mu) / std
        self.test_ds_std = (self.test_ds - mu) / std

        params = {"mu": mu, "std": std}
        pickle.dump(params, open(self.train_folder  + "params.pkl", "wb"))
        logging.info('standardize: saving params to file {}'.
                     format(self.train_folder  + "params.pkl"))

        return

    def save_to_file(self):
        '''Save the previously formed dataset to disk'''

        logging.info('Saving data input files to {}'.format(self.base_data_folder_name))

        self.train_ds_std.to_excel(self.train_filename, index = False, header=True)
        logging.info("Save train_ds to {}".format(self.train_filename))
        self.validation_ds_std.to_excel(self.valid_filename, index = False, header=True)
        logging.info("Save valid_ds to {}".format(self.valid_filename))
        self.test_ds_std.to_excel(self.test_filename, index = False, header=True)
        logging.info("Save test_ds to {}".format(self.test_filename))

        logging.info('Saving data label files to {}'.format(self.base_data_folder_name))

        labels = ["dir", "profit_over_spread", "loss_over_spread"]
        self.train_ds[labels].to_excel(self.train_labl_filename, index = False, header=True)
        self.validation_ds[labels].to_excel(self.valid_labl_filename, index = False, header=True)
        self.test_ds[labels].to_excel(self.test_labl_filename, index = False, header=True)
        return


if __name__ == '__main__':
    # main executes functional test

    # parameters for data collection
    instrument = "EUR_USD"
    brl = "1min" # bar lenght for resampling

    odc = OandaDataCollector(instrument, cfg.conf_file)
    logging.info('OandaDataCollector object created for instrument {}'.format(instrument))

    # actual data collection of most recent data
    logging.info('OandaDataCollector data collection starts...')
    odc.get_most_recent(days = 30)
    odc.make_features()
    odc.make_lagged_features()
    odc.resample_data(brl = brl)
    odc.make_3_datasets()
    odc.standardize()
    odc.save_to_file()

    print("All row data downloaded from Oanda for instrument {}".format(instrument))
    print(odc.raw_data.info(),  end="\n  ******** \n")
    print("Re-sampled data for bar length {} from Oanda for instrument {}".format(brl, instrument))
    print(odc.raw_data_resampled.info(),  end="\n  ******** \n")



