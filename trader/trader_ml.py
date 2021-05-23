import pandas as pd
import tpqoa
import pickle
from datetime import datetime, timedelta
from models.model import set_seeds
from tensorflow import keras
import logging
import numpy as np
import matplotlib
if __name__ == "__main__":
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')

import sys
sys.path.append('../')
import configs.config as cfg
from common import utils as u
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from  strategies.strategies import Strategy_1

# set_seeds(100)


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

    def test(self, data, labels):
        '''
        test the (model,strategy) pair on the passed, and compare the buy&hold with
        the chosen strategy. Ideally estimation of trading costs should be included
        '''
        #overwrite strategy to set test mode TODO: change this with testing before changing
        self.strategy = Strategy_1(instrument=self.instrument,
                                   order_fun=self.create_order,
                                   report_fun=self.report_trade,
                                   live_or_test="test")
        test_outs = pd.DataFrame()
        self.data = data.copy()
        test_outs["probs"] = self.predict(TESTING=True)
        #test_outs["position"] = test_outs["pred"]
        # Todo: implement strategy here!!
        test_outs["position"] = self.strategy.act(
            prob_up=test_outs["probs"],
            thr_up=self.h_prob_th,
            thr_low=self.l_prob_th)
        # calculate Strategy Returns: I need access to returns here!!!
        # but likely the returns I have access here are normalized..
        # Todo: review how to grant access to returns here, and see if access to normalized is doable
        test_outs["strategy_gross"] = test_outs["position"] * \
                                      (data["returns"] * self.std["returns"]
                                                         + self.mu["returns"]) #denormalizing
        # determine when a trade takes place
        test_outs["trades"] = test_outs["position"].diff().fillna(0).abs()
        # subtract transaction costs from return when trade takes place
        test_outs['strategy'] = test_outs["strategy_gross"] - test_outs["trades"] * self.hspread_ptc
        # calculate cumulative returns for strategy & buy and hold
        test_outs["creturns"] = (data["returns"] * self.std["returns"]
                                 + self.mu["returns"]).cumsum().apply(np.exp)
        test_outs["cstrategy"] = test_outs['strategy'].cumsum().apply(np.exp)
        test_outs["cstrategy_gross"] = test_outs['strategy_gross'].cumsum().apply(np.exp)
        results = test_outs
        # absolute performance of the strategy
        perf = results["cstrategy"].iloc[-1]
        # out-/underperformance of strategy
        outperf = perf - results["creturns"].iloc[-1]
        print("outperf is ", outperf)
        # plot results
        # todo: review why figures are not shown as expected
        print("plotting cumulative results of buy&hold and strategy")
        title = "{} | Transaction Cost = {}".format(self.instrument, self.hspread_ptc)
        results[["cstrategy",  "creturns", "cstrategy_gross"]].\
            plot(title=title, figsize=(12, 8))
        plt.show()
        plt.figure(figsize=(12, 8))
        plt.title("positions")
        #plt.plot(test_outs["trades"])
        plt.plot(test_outs["position"])
        plt.xlabel("time")
        plt.ylabel("positions")
        plt.show()
        # reset strategy to live
        #overwrite strategy to set test mode
        self.strategy = Strategy_1(instrument=self.instrument,
                                   order_fun=self.create_order,
                                   report_fun=self.report_trade,
                                   live_or_test="live")
        return round(perf, 6), round(outperf, 6)

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
        self.raw_data = self.hist_data.append(
                            self.tick_data.resample(self.bar_length,
                                    label="right").last().ffill().iloc[:-1])
        return

    def prepare_data(self):
        print("\nSEQUENCE: prepare_data")
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
        print("\nSEQUENCE: predict")
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
        else:
            return

    def on_success(self, time, bid, ask):
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
        return

    def report_trade(self, order, going):
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
        return


if __name__ == "__main__":

    ####  IMPORTANT ####
    ####  change this import pointing to the
    ####  wanted/needed configuration
    import configs.EUR_PLN_2 as cfginst

    instrument = cfginst.instrument

    # get or generate datafiles files and folders, if do not exist
    namefiles_dict = {}
    namefiles_dict = u.creates_filenames_dict(instrument, namefiles_dict, cfg)
    #load params for data standardization
    params = pickle.load(open(namefiles_dict["params"], "rb"))
    mu = params["mu"]
    std = params["std"]
    # load trained model
    model = keras.models.load_model(cfg.trained_models_path +
                                    instrument + "/DNN_model.h5")
    # create trader object using instrument configuration details

    print("Layers of model being used are: ")
    print(model.layers)
    trader = DNNTrader(cfg.conf_file,
                       instrument=instrument,
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

    # either live trading or testing (back or fw testing)
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
        # loading data
        assert os.path.exists(namefiles_dict["base_data_folder_name"]), "Base data folder DO NOT exists!"
        train_data = pd.read_csv(namefiles_dict["train_filename"],
                                 index_col="time", parse_dates=True, header=0)
        test_data = pd.read_csv(namefiles_dict["test_filename"],
                                index_col="time", parse_dates=True, header=0)
        # valid not used for now, using keras support but that uses
        # std and mean computed on the train+valid data
        train_labels = pd.read_csv(namefiles_dict["train_labl_filename"],
                                   index_col="time", parse_dates=True, header=0)
        test_labels = pd.read_csv(namefiles_dict["test_labl_filename"],
                                  index_col="time", parse_dates=True, header=0)

        #trader.prepare_data() ### necessary? maybe not if I take data prepared by getpreparedata.py
        if BCKTESTING:
            trader.test(train_data, train_labels)

        else: # fwtesting
            trader.test(test_data, test_labels)
