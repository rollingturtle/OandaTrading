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
        now = now - timedelta(microseconds = now.microsecond)
        past = now - timedelta(days = days)
        # get historical data up to now
        self.raw_data = self.api_oanda.get_history(
            instrument = instrument,
            start = past, end = now,
            granularity = granul,
            price = "M",
            localize = False).dropna() #c.dropna().to_frame()
        return


    def make_features(self, window = 10, sma_int=5):
        df = self.raw_data
        symbol = 'c'
        # Todo: do this check better
        logging.info("Resampled DF")
        assert len(df[symbol]) > sma_int, \
            "the dataframe lenght is not greater than the Simple Moving Average interval"
        assert len(df[symbol]) > window, \
            "the dataframe lenght is not greater than the Bollinger window"
        df["returns"] = np.log(df[symbol] / df[symbol].shift())
        df["dir_1step"] = np.where(df["returns"] > 0, 1, 0)
        df["sma"] = df[symbol].rolling(window).mean() - df[symbol].rolling(sma_int).mean()
        df["boll"] = (df[symbol] - df[symbol].rolling(window).mean()) / df[symbol].rolling(window).std()
        df["min"] = df[symbol].rolling(window).min() / df[symbol] - 1
        df["max"] = df[symbol].rolling(window).max() / df[symbol] - 1
        df["mom"] = df["returns"].rolling(3).mean()
        df["vol"] = df["returns"].rolling(window).std()
        df.dropna(inplace=True)
        return

    def resample_data(self,  brl):
        bar_length = pd.to_timedelta(brl)
        #self.raw_data_bfrs.rename(columns = {"c":instrument},
        #                          inplace = True) # "c" stands for close price

        # resampling data at the desired bar lenght, holding the last value of the bar period (.last())
        self.raw_data_resampled = self.raw_data.resample(bar_length,
                            label = "right").last().dropna().iloc[:-1]
        return




if __name__ == '__main__':
    #logging.basicConfig(level=logging.INFO)

    # configuration file for oanda access
    conf_file = cfg.config_path + "oanda.cfg"

    odc = OandaDataCollector(conf_file)
    logging.info('OandaDataCollector object created')

    # parameters for data collection
    instrument = "EUR_USD"
    brl = "1min" # bar lenght for resampling

    # actual data collection of most recent data
    logging.info('OandaDataCollector data collection starts...')
    odc.get_most_recent(instrument)
    odc.make_features()
    odc.resample_data(brl = brl)


    print("All row data downloaded from Oanda for instrument {}".format(instrument))
    print(odc.raw_data.info(),  end="\n  ******** \n")

    print("Re-sampled data for bar length {} from Oanda for instrument {}".format(brl, instrument))
    print(odc.raw_data_resampled.info(),  end="\n  ******** \n")



