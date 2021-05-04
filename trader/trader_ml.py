import pandas as pd
import numpy as np
import tpqoa
import pickle
from datetime import datetime, timedelta
from models.dnn import set_seeds
import keras
import logging

import sys
sys.path.append('../')
import configs.config as cfg
from common import utils as u
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# change this import pointing to the wanted/needed configuration for the main to work
import configs.EUR_USD_1 as cfginst

set_seeds(100)


class DNNTrader(tpqoa.tpqoa):
    def __init__(self, conf_file,
                 instrument,
                 bar_length,
                 window,
                 lags,
                 units,
                 model,
                 mu,
                 std,
                 hspread_ptc,
                 sma_int,
                 features,
                 h_prob_th,
                 l_prob_th):

        super().__init__(conf_file)

        self.position = 0
        self.instrument = instrument
        self.window = window
        self.bar_length = bar_length
        self.lags = lags
        self.units = units
        self.model = model
        self.mu = mu
        self.std = std
        self.tick_data = pd.DataFrame()
        self.hist_data = None
        self.min_length = None
        self.raw_data = None
        self.data = None
        self.profits = []
        self.hspread_ptc = hspread_ptc
        self.sma_int = sma_int
        self.features = features
        self.h_prob_th = h_prob_th
        self.l_prob_th = l_prob_th

    def get_most_recent(self, days=5, granul="S5"):
        print("SEQUENCE: get_most_recent")
        now = datetime.utcnow()
        now = now - timedelta(microseconds=now.microsecond)
        past = now - timedelta(days=days)

        # # Todo review why price M is not supported here!!
        # df = self.get_history(instrument=self.instrument, start=past, end=now,
        #                       granularity=granul, price="B").c.dropna().to_frame() #price="M" raise ValueError("price must be either 'B' or 'A'.")
        # df.rename(columns={"c": self.instrument}, inplace=True)
        # # Todo: why issues here?
        # #df.index = df.index.tz_localize("UTC") #TypeError: Already tz-aware, use tz_convert to convert.

        logging.info("get_most_recent: getting recent history to prepend to tick data so as to compute features...")
        df = self.get_history(
            instrument=self.instrument,
            start=past,
            end=now,
            granularity=granul,
            price = "M",
            localize=False).c.dropna().to_frame()
        df.rename(columns={"c": self.instrument}, inplace=True)
        df.info()

        logging.info("get_most_recent: resampling recent history to chosen bar length in line" +
                     " with the training data used to train the model")
        df = df.resample(self.bar_length, label="right").last().dropna().iloc[:-1]
        self.hist_data = df.copy()
        self.min_length = len(self.hist_data) + 1
        df.info()
        return

    def resample_and_join(self):
        print("SEQUENCE: resample_and_join")
        self.raw_data = self.hist_data.append(
                            self.tick_data.resample(self.bar_length,
                                    label="right").last().ffill().iloc[:-1])
        #print("self.raw_data", self.raw_data.head())
        return

    def prepare_data(self):
        print("SEQUENCE: prepare_data")

        #print("self.raw_data.head():\n  ",self.raw_data.head())
        # create features
        df = self.raw_data.reset_index(drop=True, inplace=False)
        #print("df = self.raw_data.copy(): \n", df.info)
        #self.raw_data \
        df = u.make_features(df,
                            self.sma_int,
                            self.window,
                            self.hspread_ptc,
                            ref_price = self.instrument )

        #self.data = u.make_lagged_features(self.raw_data, self.features, self.lags)
        #print("u.make_features(df,..:\n",df.info)
        df = u.make_lagged_features(df, self.features, self.lags)
        #print("u.make_lagged_features(df,: \n",df.info)
        self.data = df.copy()
        #print(self.data.info)

        #print("prepare_data: ", self.data.columns)
        return

    def predict(self):
        print("SEQUENCE: predict")

        df = self.data.copy()
        df_s = (df - self.mu) / self.std
        #print("predict: ", self.features)

        all_cols = self.data.columns
        lagged_cols = []
        for col in all_cols:
            if 'lag' in col:
                lagged_cols.append(col)
        #print("predict: ", lagged_cols)
        #print(self.data.head())

        df["proba"] = self.model.predict(df_s[lagged_cols])
        self.data = df.copy()

    def on_success(self, time, bid, ask):
        print("SEQUENCE: on_success")
        #print("time, bid, ask", time, bid, ask)

        print(self.ticks, end=" ")

        # store and resample tick data and join with historical data
        df = pd.DataFrame({self.instrument: (ask + bid) / 2},
                          index=[pd.to_datetime(time)])
        self.tick_data = self.tick_data.append(df)
        #print("self.tick_data.head()\n", self.tick_data.head())
        self.resample_and_join()

        # only if new bar has been added:
        if len(self.raw_data) > self.min_length - 1:
            self.min_length += 1

            self.prepare_data()
            self.predict()
            print("on_success: predicted probabilty for next bar is ", self.data["proba"].iloc[-1])
            # orders and trades

            if self.position == 0:
                if self.data["proba"].iloc[-1] > self.h_prob_th: #. 0.53:
                    order = self.create_order(self.instrument, self.units, suppress=True, ret=True)
                    self.report_trade(order, "GOING LONG")
                    self.position = 1
                elif self.data["proba"].iloc[-1] < self.l_prob_th: # 0.47:
                    order = self.create_order(self.instrument, -self.units, suppress=True, ret=True)
                    self.report_trade(order, "GOING SHORT")
                    self.position = -1

            elif self.position == -1:
                if self.data["proba"].iloc[-1] > self.h_prob_th: #0.53:
                    order = self.create_order(self.instrument, self.units * 2, suppress=True, ret=True)
                    self.report_trade(order, "GOING LONG")
                    self.position = 1

            elif self.position == 1:
                if self.data["proba"].iloc[-1] < self.l_prob_th: #0.47:
                    order = self.create_order(self.instrument, -self.units * 2, suppress=True, ret=True)
                    self.report_trade(order, "GOING SHORT")
                    self.position = -1
        return

    def report_trade(self, order, going):
        print("SEQUENCE: report_trade")

        print(order)
        time = order["time"]
        units = order["units"]
        # Todo: review this workaround to make it run as price and pl keys are missing
        print("order.keys(): ", order.keys())
        if "price" in order.keys():
            price = order["price"]
            pl = float(order["pl"])
            self.profits.append(pl)
            cumpl = sum(self.profits)
            print("\n" + 100 * "-")
            print("{} | {}".format(time, going))
            print("{} | units = {} | price = {} | P&L = {} | Cum P&L = {}".format(time, units, price, pl, cumpl))
            print(100 * "-" + "\n")


if __name__ == "__main__":
    import configs.EUR_USD_1 as eu
    base_data_folder_name = cfg.data_path + str(cfginst.instrument) + "/"
    train_folder = base_data_folder_name + "Train/"
    valid_folder = base_data_folder_name + "Valid/"
    test_folder = base_data_folder_name + "Test/"
    params = pickle.load(open(train_folder + "params.pkl", "rb"))
    mu = params["mu"]
    std = params["std"]

    model = keras.models.load_model(cfg.trained_models_path +
                                    cfginst.instrument + "/DNN_model.h5")

    trader = DNNTrader(cfg.conf_file,
                       instrument=cfginst.instrument,
                       bar_length=cfginst.brl,
                       window=cfginst.window,
                       lags=cfginst.lags,
                       units=100000,
                       model=model,
                       mu=mu, std=std,
                       hspread_ptc=cfginst.hspread_ptc,
                       sma_int=cfginst.sma_int,
                       features=cfginst.features,
                       h_prob_th=eu.higher_go_long,
                       l_prob_th=eu.lower_go_short)

    trader.get_most_recent(days=cfginst.days_inference, granul=cfginst.granul)  # get historical data
    logging.info("main: most recent historical data obtained and resampled, now starting streaming data and trading...")
    trader.stream_data(cfginst.instrument, stop=cfginst.stop_trading)  # streaming & trading here!!!!

    if trader.position != 0:
        print("Closing position as we are ending trading!")
        close_order = trader.create_order(instrument=cfginst.instrument,
                                          units=-trader.position * trader.units,
                                          suppress=True, ret=True)  # close Final Position
        trader.report_trade(close_order, "GOING NEUTRAL")  # report Final Trade
