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
        self.raw_data_bfrs = None
        self.last_bar = None
        self.api_oanda = tpqoa.tpqoa(conf_file)


    def get_most_recent(self, instrument, brl, days = 2, window = 10, sma_int=5):

        time.sleep(2) # TODO: to avoid issue to get up to now, we need to set a delay? What it is for??
        bar_length = pd.to_timedelta(brl)

        # set start and end date for historical data retrieval
        now = datetime.utcnow()
        now = now - timedelta(microseconds = now.microsecond)
        past = now - timedelta(days = days)

        # get historical data up to now
        self.raw_data_bfrs = self.api_oanda.get_history(instrument = instrument,
                                                        start = past, end = now,
                                                        granularity = "S5",
                                                        price = "M",
                                                        localize = False).dropna() #c.dropna().to_frame()

        #self.raw_data_bfrs.rename(columns = {"c":instrument},
        #                          inplace = True) # "c" stands for close price

        # resampling data at the desired bar lenght, holding the last value of the bar period (.last())
        df = self.raw_data_bfrs.resample(bar_length,
                                         label = "right").last().dropna().iloc[:-1] # label = "right"

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


        # hold copy of the re-sampled data
        self.raw_data = df.copy()

        # TODO: review if this is for checking whether there is more data to collect?
        #self.last_bar = self.raw_data.index[-1]
        #if pd.to_datetime(datetime.utcnow()).tz_localize("UTC") - self.last_bar < bar_length:




if __name__ == '__main__':
    #logging.basicConfig(level=logging.INFO)

    # configuration file for oanda access
    conf_file = cfg.config_path + "oanda.cfg"

    odc = OandaDataCollector(conf_file)
    logging.info('OandaDataCollector object created')

    # parameters for data collection
    instrument = "EUR_USD"
    brl = "5min" # bar lenght for resampling

    # actual data collection of most recent data
    logging.info('OandaDataCollector data collection starts...')
    odc.get_most_recent(instrument, brl = brl)


    print("All row data downloaded from Oanda for instrument {}".format(instrument))
    print(odc.raw_data_bfrs.info(),  end="\n  ******** \n")

    print("Re-sampled data for bar length {} from Oanda for instrument {}".format(brl, instrument))
    print(odc.raw_data.info(),  end="\n  ******** \n")



