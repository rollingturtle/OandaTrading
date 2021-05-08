import pandas as pd
import tpqoa
import pickle
from datetime import datetime, timedelta
from models.dnn import set_seeds
import keras
import logging
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
import configs.config as cfg
from common import utils as u
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from  strategies.strategies import Strategy_1

from functools import partial

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

        self.strategy = Strategy_1(instrument=self.instrument,
                                   order_fun= self.create_order, #partial(self.create_order,
                                                #      self.instrument, suppress=True, ret=True),
                                    report_fun= self.report_trade,
                                    live_or_test="live")

    def fwtest(self, data, labels):
        pass

    def backtest(self, data, labels):
        '''
        test the (model,strategy) pair on the passed, and compare the buy&hold with
        the chosen strategy. Ideally estimation of trading costs should be included
        '''
        test_outs = pd.DataFrame()
        self.data = data.copy()
        test_outs["pred"] = self.predict(TESTING=True)

        # calculate Strategy Returns: I need access to returns here!!!
        # but likely the returns I have access here are normalized..
        # Todo: review how to grant access to returns here, and see if access to normalized is doable
        test_outs["strategy"] = test_outs["pred"] * data["returns"]

        # determine when a trade takes place
        test_outs["trades"] = test_outs["pred"].diff().fillna(0).abs()

        # subtract transaction costs from return when trade takes place
        test_outs.strategy = test_outs.strategy - test_outs.trades * self.hspread_ptc

        # calculate cumulative returns for strategy & buy and hold
        test_outs["creturns"] = data["returns"].cumsum().apply(np.exp)
        test_outs["cstrategy"] = test_outs['strategy'].cumsum().apply(np.exp)
        results = test_outs

        # absolute performance of the strategy
        perf = results["cstrategy"].iloc[-1]
        # out-/underperformance of strategy
        outperf = perf - results["creturns"].iloc[-1]

        # plot results
        print("plotting cumulative results of buy&hold and strategys")
        title = "{} | TC = {}".format(self.instrument, self.hspread_ptc)

        plt.figure()
        results[["cstrategy"]].plot(title=title, figsize=(12, 8)) #,  "creturns"
        plt.show

        plt.figure()

        plt.plot([i for i in range(10)])
        plt.show()

        return round(perf, 6), round(outperf, 6)
        # load passed data and visualize it
        # from passed data, resample if necessary,
        # create features and lagged features using other class methods possibly (preparedata)
        # if backtest, in-sample/out-sample is done on training/test data, as it probably should be, lagged features
        # are already available!
        # initial position is 0
        # for t over data:
        #   make prediction on t using lagged features from data
        #   use prediction to feed strategy
        #  if any change of position, update cum P&L, considering trading costs
        # visualize plot buy&hold vs strategy (w/o trading costs)
        # no the position for each time step, as predicted, does not depend on the previous position
        # what is actually dependendent on the previous position is whether or not the current position
        # must correspond to an action to be taken (going long, going short)
        # for some strageies, the position is either -1 or 1, being 0 only at the beginning or end of
        # trading session
        pass

    def get_most_recent(self, days=5, granul="S5"):
        '''Get most recent data that will be prepended to the stream data, to have enough data to
        compute all the necessary features and be able to trade from the first bar'''
        print("SEQUENCE: get_most_recent")
        now = datetime.utcnow()
        now = now - timedelta(microseconds=now.microsecond)
        past = now - timedelta(days=days)

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

        print("\nhistory dataframe information:\n")
        df.info()

        logging.info("get_most_recent: resampling recent history to chosen bar length in line" +
                     " with the training data used to train the model")
        df = df.resample(self.bar_length, label="right").last().dropna().iloc[:-1]
        self.hist_data = df.copy()
        self.min_length = len(self.hist_data) + 1

        print("\nresampled history dataframe information:\n")
        df.info()
        return

    def resample_and_join(self):
        print("\nSEQUENCE: resample_and_join")
        self.raw_data = self.hist_data.append(
                            self.tick_data.resample(self.bar_length,
                                    label="right").last().ffill().iloc[:-1])
        #print("self.raw_data", self.raw_data.head())
        return

    def prepare_data(self):
        print("SEQUENCE: prepare_data")

        # create features
        df = self.raw_data.reset_index(drop=True, inplace=False)
        df = u.make_features(df,
                            self.sma_int,
                            self.window,
                            self.hspread_ptc,
                            ref_price = self.instrument )

        df = u.make_lagged_features(df, self.features, self.lags)
        self.data = df.copy()
        return

    def predict(self, TESTING=False):
        print("SEQUENCE: predict")

        df = self.data.copy()
        if not TESTING:
            # if we are trading live (not TESTING) we need to normalize the data using
            # training dataset statistics
            df = (df - self.mu) / self.std

        # get feature columns
        all_cols = self.data.columns
        lagged_cols = []
        for col in all_cols:
            if 'lag' in col:
                lagged_cols.append(col)

        df["proba"] = self.model.predict(df[lagged_cols])

        self.data = df.copy()

        if TESTING:
            return df["proba"]
        return


    def on_success(self, time, bid, ask):
        print("SEQUENCE: on_success")

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

            # here we apply a strategy based on the probabilities. Many strategies are possible
            # Todo: externalize the strategy and make this class support historical data, for backtesting purposes

            # from functool import partial
            self.position = self.strategy.act(
                       position=self.position,
                       prob_up=self.data["proba"].iloc[-1],
                       thr_up=self.h_prob_th,
                       thr_low=self.l_prob_th,
                       units = self.units)

            #   better idea is to instantiate a strategy object, init with order_fun, report_fun
            # mode live_or_test, and leave other params for the act call
            # if self.position == 0:
            #     if self.data["proba"].iloc[-1] > self.h_prob_th: #. 0.53:
            #         order = self.create_order(self.instrument, self.units, suppress=True, ret=True)
            #         self.report_trade(order, "GOING LONG")
            #         self.position = 1
            #     elif self.data["proba"].iloc[-1] < self.l_prob_th: # 0.47:
            #         order = self.create_order(self.instrument, -self.units, suppress=True, ret=True)
            #         self.report_trade(order, "GOING SHORT")
            #         self.position = -1
            #
            # elif self.position == -1:
            #     if self.data["proba"].iloc[-1] > self.h_prob_th: #0.53:
            #         order = self.create_order(self.instrument, self.units * 2, suppress=True, ret=True)
            #         self.report_trade(order, "GOING LONG")
            #         self.position = 1
            #
            # elif self.position == 1:
            #     if self.data["proba"].iloc[-1] < self.l_prob_th: #0.47:
            #         order = self.create_order(self.instrument, -self.units * 2, suppress=True, ret=True)
            #         self.report_trade(order, "GOING SHORT")
            #         self.position = -1
        return

    def report_trade(self, order, going):
        print("SEQUENCE: report_trade")

        print(order)
        time = order["time"]
        units = order["units"]
        # Todo: review this workaround to make it run as price and pl keys are missing sometimes
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

    # change this import pointing to the wanted/needed configuration for the main to work
    import configs.EUR_USD_1 as cfginst

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
                       units=cfginst.units,
                       model=model,
                       mu=mu, std=std,
                       hspread_ptc=cfginst.hspread_ptc,
                       sma_int=cfginst.sma_int,
                       features=cfginst.features,
                       h_prob_th=cfginst.higher_go_long,
                       l_prob_th=cfginst.lower_go_short)

    TRADING = 0
    BCKTESTING, FWTESTING = (1,0) if not TRADING else (0,0)

    if TRADING:
        trader.get_most_recent(days=cfginst.days_inference, granul=cfginst.granul)  # get historical data
        logging.info("main: most recent historical data obtained and resampled" +
                     "now starting streaming data and trading...")
        trader.stream_data(cfginst.instrument, stop=cfginst.stop_trading)  # streaming & trading here!!!!

        if trader.position != 0:
            print("Closing position as we are ending trading!")
            close_order = trader.create_order(instrument=cfginst.instrument,
                                              units=-trader.position * trader.units,
                                              suppress=True, ret=True)  # close Final Position
            trader.report_trade(close_order, "GOING NEUTRAL")  # report Final Trade
    else: # TESTING
        import configs.EUR_USD_1 as eu

        instrument = eu.instrument
        # loading data
        base_data_folder_name = cfg.data_path + instrument + "/"
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

        train_data = pd.read_csv(train_filename, index_col=None, header=0)
        test_data = pd.read_csv(test_filename, index_col=None, header=0)
        # valid not used for now, using keras support but that uses
        # std and mean computed on the train+valid data
        train_labels = pd.read_csv(train_labl_filename, index_col=None, header=0)
        test_labels = pd.read_csv(test_labl_filename, index_col=None, header=0)

        #trader.prepare_data() ### necessary? maybe not if I take data prepared by getpreparedata.py
        if BCKTESTING:
            trader.backtest(train_data, train_labels)

        else: # fwtesting
            trader.fwtest(test_data, test_labels)
