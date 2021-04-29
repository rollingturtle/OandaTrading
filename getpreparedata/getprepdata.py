import pandas as pd
import numpy as np
import tpqoa
from datetime import datetime, timedelta
import time
import logging


import sys
sys.path.append('../')
import configs.config as cfg


class OandaDataCollector():
    def __init__(self,
                 conf_file
                 ):

        self.raw_data = None
        self.raw_data_resampled = None
        self.last_bar = None
        self.api_oanda = tpqoa.tpqoa(conf_file)


    def get_most_recent(self, instrument, granul="S5", days = 2):
        #time.sleep(2) # TODO: to avoid issue to get up to now, we need to set a delay? What it is for??
        # set start and end date for historical data retrieval
        now = datetime.utcnow()
        now = now - timedelta(microseconds = now.microsecond) # Oanda does not deal with microseconds
        past = now - timedelta(days = days)
        # get historical data up to now
        self.raw_data = self.api_oanda.get_history(
            instrument = instrument,
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
        df["profit_over_spread"] = np.where(df["returns"] > hspread_ptc, 1, 0)  # profit over spread
        df["loss_over_spread"] = np.where(df["returns"] < 1 - hspread_ptc, 1, 0)  # loss under spread

        df["sma"] = df[ref_price].rolling(window).mean() - df[ref_price].rolling(sma_int).mean()
        df["boll"] = (df[ref_price] - df[ref_price].rolling(window).mean()) / df[ref_price].rolling(window).std()
        df["min"] = df[ref_price].rolling(window).min() / df[ref_price] - 1
        df["max"] = df[ref_price].rolling(window).max() / df[ref_price] - 1
        df["mom"] = df["returns"].rolling(3).mean()  #todo make this a param
        df["vol"] = df["returns"].rolling(window).std()

        df.dropna(inplace=True)
        return

    def resample_data(self,  brl="1min"):
        bar_length = pd.to_timedelta(brl)

        # resampling data at the desired bar length, holding the last value of the bar period (.last())
        # label right is to set the index to the end of the bar time
        self.raw_data_resampled = self.raw_data.resample(bar_length,
                            label = "right").last().dropna().iloc[:-1]
        # dropna is here to remove weekends. iloc to remove the last bar/row typically incomplete
        return


    def make_lagged_features(self, lags=5):
        ''' Add lagged features to data'''
        cols = []
        features = ["dir", "sma", "boll", "min", "max", "mom", "vol"]
        for f in features:
            for lag in range(1, lags + 1):
                col = "{}_lag_{}".format(f, lag)
                self.raw_data[col] = self.raw_data[f].shift(lag)
                cols.append(col)
        self.raw_data.dropna(inplace=True)
        return


    def make_3_datasets(self, split_pcs=(0.7, 0.15, 0.15), save_to_file=True):
        ''' Generate 3 datasets for ML training/evaluation/test and save to files'''

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

        if save_to_file:
            # TODO: save to DATA folder
            print(self.train_ds.info())

        pass
    # Todo 2: method which splits data in 2 or 3 parts and save it into prepared folder under instrument name


if __name__ == '__main__':
    # main executes functional test

    odc = OandaDataCollector(cfg.conf_file)
    logging.info('OandaDataCollector object created')
    # parameters for data collection
    instrument = "EUR_USD"
    brl = "1min" # bar lenght for resampling
    # actual data collection of most recent data
    logging.info('OandaDataCollector data collection starts...')
    odc.get_most_recent(instrument)
    odc.make_features()
    odc.make_lagged_features()
    odc.resample_data(brl = brl)
    odc.make_3_datasets()
    print("All row data downloaded from Oanda for instrument {}".format(instrument))
    print(odc.raw_data.info(),  end="\n  ******** \n")
    print("Re-sampled data for bar length {} from Oanda for instrument {}".format(brl, instrument))
    print(odc.raw_data_resampled.info(),  end="\n  ******** \n")



