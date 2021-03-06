# system
import logging
import os
import os.path
import sys
import os
sys.path.append('../')
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# data
import pandas as pd
import tpqoa
from datetime import datetime, timedelta
import numpy as np
import pickle

# display
import matplotlib
if __name__ == "__main__":
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# keras
from models.model import set_seeds
from tensorflow import keras

# project modules
import configs.config as cfg
import models.model as m
import common.utils as u
from  strategies.strategies import Strategy_1

import threading

################################ OaNDa (ond) base class #########################################
class ond: # Todo: should be abstract class?
    '''
    Base class for classes directly communicating to Oanda service
    '''
    def __init__(self,conf_file=None,  # oanda user conf file: todo: should I set it here?? NO
                 instrument_file=None): # instrument conf file

        self.instrument_file = instrument_file
        self.instrument = instrument_file.instrument
        self.targets = instrument_file.targets  # "dir", "profit_over_spread", "loss_over_spread"]
        self.features = instrument_file.features
        self.lags = instrument_file.lags

        self.raw_data = pd.DataFrame()

        self.namefiles_dict = {}
        self.namefiles_dict = u.creates_filenames_dict(
            self.instrument_file.instrument,
            self.namefiles_dict, cfg)

        # tpqoa object to download historical data
        self.api_oanda = tpqoa.tpqoa(conf_file)

        return

    def _get_history(self, instrument = None,
                                    start = None,
                                    end = None,
                                    granularity = None,
                                    price = None,
                                    localize = False):
        '''
        helper function used in multithreaded data collection
        '''
        print("Thread started with price ", price)
        df =  self.api_oanda.get_history(
                                    instrument,
                                    start,
                                    end,
                                    granularity,
                                    price,
                                    localize).dropna() #.to_frame() #).c.dropna()...
        if price == "A":
            self.ask_df = df.copy()
            print("Got ASK data ", self.ask_df.columns)
        elif price =="B":
            self.bid_df = df.copy()
            print("Got BID data ", self.bid_df.columns)
        else:
            print("price not recognized!")

        i = self.barrier.wait() #sync with other tasks
        if i == 0:
            # Only one thread needs to print this
            print("passed the barrier")
        return

    def get_most_recent(self, granul="S5", days = 1):
        '''
        Get the most recent data for the instrument for which the object was created
        :param granul: base frequency for raw data being downloaded
        :param days: number of past days (from today) we are downloading data for
        '''

        # set start and end date for historical data retrieval
        now = datetime.utcnow()
        now = now - timedelta(microseconds = now.microsecond)
        past = now - timedelta(days = days)

        # get historical data up to now
        logging.info("get_most_recent: getting historical data from Oanda... ")
        self.barrier = threading.Barrier(2,timeout=20)
        ask_thread = threading.Thread(target=self._get_history,
                                  args=(self.instrument, past, now, granul, "A", False))
        bid_thread = threading.Thread(target=self._get_history,
                                  args=(self.instrument, past, now, granul,"B", False))
        ask_thread.start()
        bid_thread.start()
        ask_thread.join()
        bid_thread.join()

        # combine price information to derive the spread
        for p in "ohlc":
            self.raw_data['sprd_{}'.format(p)] = self.ask_df['{}'.format(p)] - \
                                                 self.bid_df['{}'.format(p)]
            self.raw_data['ask_{}'.format(p)] = self.ask_df['{}'.format(p)]
            self.raw_data['bid_{}'.format(p)] = self.bid_df['{}'.format(p)]
        self.raw_data['volume'] = (self.ask_df["volume"] +  self.bid_df["volume"])/2

        # averaging o,h,c,l values between ask and bid
        for p in "ohlc":
            self.raw_data[p] = (self.ask_df[p] + self.bid_df[p])/2

        # averaging spread across o,h,l,c
        col = self.raw_data.loc[:, ['sprd_{}'.format(p) for p in "ohlc"]]
        self.raw_data["spread"] = col.mean(axis=1)

        # dropping all temporary columns
        self.raw_data.drop(columns=[ p for p in self.raw_data.columns if "_" in p ], inplace=True)

        # renaming c as instrument as it will be used as reference price
        self.raw_data.rename(columns={"c": self.instrument}, inplace=True)

        print("get_most_recent: self.raw_data.info() ", self.raw_data.info())

        # check for any missing data: TODO: is this useful here??
        assert (not self.raw_data.isnull().values.any()), \
            "get_most_recent: 1-NANs in self.raw_data_featured" #
        return

################################ Get DaTa (gdt) class #########################################
class gdt(ond):
    '''
    Class that inherits from class ond, dedicated to get historical data, create features and
    create the 3 datasets.
    Main methods are:
        self.get_most_recent(granul, days)
        self.make_features()
        self.make_lagged_features(lags)
        self.resample_data(barlength)
        self.make_3_datasets()
        self.standardize()
        self.save_to_file()
    '''

    def __init__(self,
                 conf_file, # General conf Oanda conf file
                 instrument_file):  # instrument conf file

        print("gdt: using conf file and instrument file:  \n{} \n{} ", conf_file, instrument_file)
        super(gdt, self).__init__(conf_file, instrument_file)

        # raw tick data, and resampled version which has precise time interval
        self.raw_data = pd.DataFrame()
        self.raw_data_featured = pd.DataFrame()
        self.raw_data_featured_resampled = pd.DataFrame()

        # 3 datasets as generated by the division of raw_data_resampled, but not standardized
        # these datasets contain however labelling information that should be used, due to the fact
        # that standardization affects their value as well
        self.train_ds = pd.DataFrame()
        self.validation_ds = pd.DataFrame()
        self.test_ds = pd.DataFrame()

        # 3 datasets obtained from the 3 above via standardization.
        # Labelling values for market direction are not to be taken from here
        self.train_ds_std = pd.DataFrame()
        self.validation_ds_std = pd.DataFrame()
        self.test_ds_std = pd.DataFrame()

        # dict to host mu and std for all features, computed on training data
        self.params = {}

        # dataframes to hold ask and bid historical prices
        self.ask_df = pd.DataFrame()
        self.bid_df = pd.DataFrame()

        # tpqoa object to download historical data
        #self.api_oanda = tpqoa.tpqoa(conf_file)

        gdt.check_create_path(self.namefiles_dict)
        return

    @classmethod
    def check_create_path(cls, namefiles_dict):
        if os.path.exists(namefiles_dict["base_data_folder_name"]):
            logging.info("__init__: Base folder exists: you may be overwriting existing files!")
            # Todo add choice to break out?
        else:
            logging.info("__init__: Non existent Base folder: creating it...")
            os.mkdir(namefiles_dict["base_data_folder_name"])
            os.mkdir(namefiles_dict["train_folder"])
            os.mkdir(namefiles_dict["valid_folder"])
            os.mkdir(namefiles_dict["test_folder"])
        return

    @classmethod
    def _load_data_from_file(cls, namefiles_dict, raw_only=True):
        '''load raw data and process data from file'''

        # raw data
        raw_data = pd.read_csv(
            namefiles_dict["raw_data_file_name"], index_col="time", parse_dates=True, header=0)
        raw_data_featured_resampled = pd.read_csv(
            namefiles_dict["raw_data_featured_resampled_file_name"],
                             index_col="time", parse_dates=True, header=0)

        if raw_only:
            return

        # loading 3 datasets, standardized, which contains also the columns for targets
        train_ds_std = \
            pd.read_csv(namefiles_dict["train_filename"], index_col="time", parse_dates=True, header=0)
        validation_ds_std = \
            pd.read_csv(namefiles_dict["valid_filename"], index_col="time", parse_dates=True, header=0)
        test_ds_std = \
            pd.read_csv(namefiles_dict["test_filename"], index_col="time", parse_dates=True, header=0)

        # normalization params
        params = pickle.load(open(namefiles_dict["train_folder"]  + "params.pkl", "rb"))

        return raw_data, raw_data_featured_resampled, train_ds_std, validation_ds_std, \
                    test_ds_std, params

    def load_data_from_file(self, raw_only=False):
        self.raw_data, \
        self.raw_data_featured_resampled,\
        self.train_ds_std, \
        self.validation_ds_std,\
        self.test_ds_std,\
        self.params = gdt._load_data_from_file(self.namefiles_dict, raw_only=False)
        return

    def report(self):
        '''provides insights on data memorized in the odc object'''
        # Todo: check this, as it might make temporary copies of the data... useless
        datalist = {"raw_data"                   : self.raw_data,
                         "raw_data_featured_resampled": self.raw_data_featured_resampled,
                         "train_ds_std"               : self.train_ds_std,
                         "validation_ds_std"          : self.validation_ds_std,
                         "test_ds_std"                : self.test_ds_std}

        for k, df in datalist.items():
            if df is not None:
                print("\nreport: Display info for dataframe {}".format(k))
                #df.info()
                df.describe()
                print("\n")
            else:
                print("\nreport: Dataframe {} is None".format(k))
        print("\nreport: displaying information about training set statistics")
        print("report: params is {}".format(self.params))
        return

    def make_features(self, window = 10, sma_int=5):
        '''
        creates features using utils.make_features
        '''
        df = self.raw_data.copy()
        ref_price = self.instrument

        # Todo: do this check better
        assert (len(df[ref_price]) > sma_int), \
            "make_features: the dataframe length is not greater than the Simple Moving Average interval"
        assert (len(df[ref_price]) > window), \
            "make_features: the dataframe length is not greater than the Bollinger window"

        # Creating features from ref_price
        self.raw_data_featured = u.make_features(df, sma_int, window, ref_price=ref_price )

        logging.info("make_features: created new features and added to self.raw_data_featured")
        #print("make_features: self.raw_data_featured.columns ", self.raw_data_featured.columns)

        assert (not self.raw_data_featured.isnull().values.any()), \
            "make_features: 1-NANs in self.raw_data_featured"
        return

    def make_lagged_features(self, lags=5):
        '''
        Add lagged features to data.
        Creates features using utils.make_lagged_features
        '''
        assert (not self.raw_data_featured.isnull().values.any()), \
            "make_lagged_features: 1-NANs in self.raw_data_featured"

        # Creating lagged features, not based only on reference price but on the list self.features
        self.raw_data_featured = \
            u.make_lagged_features(self.raw_data_featured, self.features,lags=lags)

        assert (not self.raw_data_featured.isnull().values.any()), \
            "make_lagged_features: 2-NANs in self.raw_data_featured"

        logging.info("make_lagged_features: created new features and added to self.raw_data_featured")
        return

    def resample_data(self,  brl="1min"):
        '''
        Resample data already obtained to the frequency defined by brl
        :param brl: default 1min
        '''

        bar_length = pd.to_timedelta(brl)
        assert (not self.raw_data_featured.isnull().values.any()), \
            "resample_data: 1-NANs in self.raw_data_featured"

        # resampling data at the desired bar length,
        # label = "right" - is to set the index to the end of the bar time
        # .last() - holding the last value of the bar period
        # dropna - to remove weekends.
        # iloc[:-1] -  to remove the last bar/row typically incomplete

        # Actual resampling of data
        self.raw_data_featured_resampled = self.raw_data_featured.resample(bar_length,
                            label = "right").last().dropna().iloc[:-1]

        assert (not self.raw_data_featured.isnull().values.any()), \
            "resample_data: 2-NANs in self.raw_data_featured"
        assert (not self.raw_data_featured_resampled.isnull().values.any()), \
            "resample_data: 2-NANs in self.raw_data_featured_resampled"
        logging.info("resample_data: resampled the just created new features, into self.raw_data_featured_resampled")
        return


    def make_3_datasets(self, split_pcs=(0.7, 0.10, 0.20)):
        '''
        Generate 3 datasets for ML training/evaluation/test and save to files.
        '''
        if self.raw_data_featured_resampled is not None:  # it was populated in precedence
            df = self.raw_data_featured_resampled
        else:
            logging.info("make_3_datasets: ERROR: self.raw_data_featured_resampled was empty!")
            exit()
        assert (sum(split_pcs) == 1), "make_3_datasets: split points are not dividing the unity"
        assert (not self.raw_data_featured_resampled.isnull().values.any()), \
            "make_3_datasets: NANs in self.raw_data_featured_resampled"

        # Making 3 input datasets
        ## Select relevant columns for input data
        input_cols = [col for col in df.columns if "lag_" in col]
        # train_split = int(len(df) * split_pcs[0])
        # val_split = int(len(df) * (split_pcs[0] + split_pcs[1]))

        train_split, val_split = u.get_split_points(df, split_pcs)

        self.train_ds = df[input_cols].iloc[:train_split].copy()
        self.validation_ds = df[input_cols].iloc[train_split:val_split].copy()
        self.test_ds = df[input_cols].iloc[val_split:].copy()

        # Making 3 target datasets
        ## Select relevant columns for target data
        target_cols = [col for col in df.columns if ("dir" in col and "lag_" not in col)]
        self.train_targets = df[target_cols].iloc[:train_split].copy()
        self.validation_targets = df[target_cols].iloc[train_split:val_split].copy()
        self.test_targets = df[target_cols].iloc[val_split:].copy()
        #print("make_3_datasets: self.train_ds.info() ", self.train_ds.info())
        return

    def standardize(self):
        '''
        standardize the 3 datasets, using mean and std from train dataset
        to be called after make_3_datasets
        '''
        logging.info("standardize: subtracting mean and dividing by standard deviation, for each feature!")

        # Standardization of all columns in the 3 datasets using mean, std, from train data
        mu, std = self.train_ds.mean(), self.train_ds.std()
        self.params = {"mu": mu, "std": std}
        self.train_ds_std = (self.train_ds - mu) / std
        self.validation_ds_std = (self.validation_ds - mu) / std
        self.test_ds_std = (self.test_ds - mu) / std

        #print("standardize: self.train_ds_std.info() ", self.train_ds_std.info())
        assert (not self.train_ds_std.isnull().values.any()), "standardize: NANs in Training Data"
        return

    def save_to_file(self):
        '''Save the previously formed datasets to disk'''

        # Saving each generated data file to disk
        self.raw_data.to_csv(self.namefiles_dict["raw_data_file_name"],
                             index = True, header=True)
        self.raw_data_featured_resampled.to_csv(
            self.namefiles_dict["raw_data_featured_resampled_file_name"],
                             index = True, header=True)

        self.train_ds_std.to_csv(self.namefiles_dict["train_filename"],
                             index = True, header=True)
        self.validation_ds_std.to_csv(self.namefiles_dict["valid_filename"],
                             index = True, header=True)
        self.test_ds_std.to_csv(self.namefiles_dict["test_filename"],
                             index = True, header=True)

        self.train_targets.to_csv(self.namefiles_dict["train_labl_filename"],
                             index = True, header=True)
        self.validation_targets.to_csv(self.namefiles_dict["valid_labl_filename"],
                             index = True, header=True)
        self.test_targets.to_csv(self.namefiles_dict["test_labl_filename"],
                             index = True, header=True)

        pickle.dump(self.params, open(self.namefiles_dict["train_folder"]  + "params.pkl", "wb"))

        logging.info("save_to_file: Saved raw data and resampled raw data to {} and to {}".format(
                self.namefiles_dict["raw_data_file_name"],
                  self.namefiles_dict["raw_data_featured_resampled_file_name"]))
        logging.info('saved_to_file: saving params to file {}'.format(
            self.namefiles_dict["train_folder"]  + "params.pkl"))
        return


################################ TRaiNer (trn) class #########################################
class trn(ond):
    '''
    Class that inherits from class, dedicated to traind ML models on data obtained by gdt objects
    create the 3 datasets.
    Main methods are:
        TBD...
    '''

    def __init__(self,
                 conf_file=None, # General conf Oanda conf file
                 instrument_file=None,  # instrument conf file
                 model_id=None):

        super(trn, self).__init__(instrument_file=instrument_file, conf_file=conf_file)

        assert model_id in m.available_models, \
            "DL_Trainer: init: model_id not among known models"
        self.model_id = model_id

        self.model = None
        self.train_data = None
        self.test_data = None
        self.validation_data = None
        self.validation_targets = None
        self.train_targets = None
        self.test_targets = None
        self.lagged_cols = []
        self.lagged_cols_reordered = []
        return

    def load_train_data(self):
        assert os.path.exists(self.namefiles_dict["base_data_folder_name"]), \
            "Base data folder DO NOT exists!"
        self.train_data = pd.read_csv(self.namefiles_dict["train_filename"],
                                   index_col="time", parse_dates=True, header=0)
        self.test_data = pd.read_csv(self.namefiles_dict["test_filename"],
                                   index_col="time", parse_dates=True, header=0)
        self.validation_data = pd.read_csv(self.namefiles_dict["valid_filename"],
                                   index_col="time", parse_dates=True, header=0)

        self.train_targets = pd.read_csv(self.namefiles_dict["train_labl_filename"],
                                   index_col="time", parse_dates=True, header=0)
        self.test_targets = pd.read_csv(self.namefiles_dict["test_labl_filename"],
                                   index_col="time", parse_dates=True, header=0)
        self.validation_targets = pd.read_csv(self.namefiles_dict["valid_labl_filename"],
                                   index_col="time", parse_dates=True, header=0)

        # Todo: make this step unified and linked to instrument specific configuration
        # exctract relevant columns for training (features!)
        all_cols = self.train_data.columns
        for col in all_cols:
            if 'lag' in col:
                self.lagged_cols.append(col)

        # reorder features
        for lag in range(1, self.lags + 1):
            for feat in self.features:
                r = u.find_string(all_cols, feat + "_lag_" + str(lag))
                if r != -1:
                    self.lagged_cols_reordered.append(r)

        print("load_train_data: reordered features are:", self.lagged_cols_reordered)
        # print("load_train_data: Lagged columns which are the input to the model:")
        # print(self.lagged_cols)
        print("load_train_data: how many Lagged columns:")
        print(len(self.lagged_cols))
        # print("self.train_data.head()\n",self.train_data.head())
        assert (not self.train_data[self.lagged_cols].isnull().values.any()), \
            "NANs in Training Data"
        assert (not self.train_targets["dir"].isnull().values.any()), \
            "NANs in targets"
        return # train and test data loaded. Validation carved out by Keras from training data

    def set_model(self, dropout=None):
        if self.model_id == "dnn1": # Todo: do this using the dictionary of models in model.py
            self.model = m.dnn1(dropout = True, # dropout TODO: harmonize use of dropout here
                              rate=0.1,
                              input_dim = len(self.lagged_cols))
        elif self.model_id == "LSTM_dnn":  #Todo: review if makes sense to inherit for each model...
            self.model = m.LSTM_dnn(dropout = dropout,
                                    inputs = np.zeros((1, self.instrument_file.lags,
                                                       len(self.instrument_file.features))))
        elif self.model_id == "LSTM_dnn_all_states":  #Todo: review if makes sense to inherit for each model...
            self.model = m.LSTM_dnn_all_states(dropout = dropout,
                                    inputs = np.zeros((1, self.instrument_file.lags,
                                                       len(self.instrument_file.features))))
        elif self.model_id == "LSTM_dnn_all_states_mout":  #Todo: review if makes sense to inherit for each model...
            self.model = m.LSTM_dnn_all_states_mout(dropout = dropout,
                                    inputs = np.zeros((1, self.instrument_file.lags,
                                                       len(self.instrument_file.features))))
        elif self.model_id == "ffn":
            self.model = m.ffn(self.train_data[self.lagged_cols],
                              rate=0.1)

        # visualize some details of the model NOT YET TRAINED
        # get some visualization before learning (on weights)
        # Todo: do this better to generate more informative insight
        # plt.hist(self.model.layers[2].get_weights()[0])
        # plt.show()
        # plt.figure()
        #
        # plt.hist(self.model.layers[2].get_weights()[1])
        # plt.show()
        # plt.figure()
        print("trn.set_model: model set to ", self.model_id)
        return

    def train_model(self, epochs=30):  # Todo: explode this using gradient_tape
        '''
        It trains the specified model on the training data already obtained
        '''
        print("train_model: using model ", self.model_id)
        if self.model_id == "ffn" or self.model_id == "ffn": #todo : do this better

            r = self.model.fit(x=self.train_data[self.lagged_cols],
                          y=self.train_targets["dir"],
                          epochs=epochs,
                          verbose=True,
                          validation_data=(self.validation_data[self.lagged_cols],
                                           self.validation_targets["dir"]),
                          shuffle=True,
                          batch_size=64,
                          class_weight=m.cw(self.train_targets))

        elif self.model_id == "LSTM_dnn" or \
                self.model_id ==  "LSTM_dnn_all_states":
            # inputs: A 3D tensor with shape [batch, timesteps, feature].

            numpy_train = trn._get_3d_tensor(
                self.train_data[self.lagged_cols_reordered].to_numpy(),
                    self.instrument_file)
            numpy_val = trn._get_3d_tensor(
                self.validation_data[self.lagged_cols_reordered].to_numpy(),
                    self.instrument_file)

            print("numpy_train.shape ",numpy_train.shape)

            r = self.model.fit(x = numpy_train,
                      y = self.train_targets["dir"].to_numpy(),
                      epochs = epochs,
                      verbose = True,
                      validation_data=(numpy_val,
                                       self.validation_targets["dir"].to_numpy()),
                      shuffle = True,
                      batch_size=64,  # Todo make BS a param
                      class_weight = m.cw(self.train_targets))

        elif self.model_id == "LSTM_dnn_all_states_mout":
            # inputs: A 3D tensor with shape [batch, timesteps, feature].

            numpy_train = trn._get_3d_tensor(
                self.train_data[self.lagged_cols_reordered].to_numpy(),
                    self.instrument_file)
            numpy_val = trn._get_3d_tensor(
                self.validation_data[self.lagged_cols_reordered].to_numpy(),
                    self.instrument_file)

            print("numpy_train.shape ",numpy_train.shape)
            numpy_train_targets = self.train_targets[["dir", "dir_sma1", "dir_sma2"]].to_numpy()
            numpy_val_targets = self.validation_targets[["dir", "dir_sma1", "dir_sma2"]].to_numpy()

            r = self.model.fit(x = numpy_train,
                      y = [numpy_train_targets[:,0],numpy_train_targets[:,1],numpy_train_targets[:,2]],
                      epochs = epochs,
                      verbose = True,
                      validation_data=(numpy_val,
                            [numpy_val_targets[:, 0], numpy_val_targets[:, 1], numpy_val_targets[:, 2]]),
                      shuffle = True,
                      batch_size=64) #,  # Todo make BS a param
                      #class_weight = m.cw_multi(self.train_targets))

        # get some visualization of the effect of learning (on weights, loss, acc)
        # plt.hist(self.model.layers[2].get_weights()[0])
        # plt.show()
        # plt.figure()
        print("PLOTTING Train/Validation loss and accuracy")
        plt.plot(r.history['loss'], label="loss")
        plt.plot(r.history['val_loss'], label="val_loss")
        plt.legend()
        plt.show()


        acc_str = "acc" if self.model_id != "LSTM_dnn_all_states_mout" else "out_acc"
        plt.figure()
        plt.plot(r.history[acc_str], label=acc_str)
        plt.plot(r.history['val_' + acc_str], label="val_" + acc_str)
        plt.legend()
        plt.show()
        plt.figure()
        return

    @classmethod
    def _get_3d_tensor(cls, twodim_np_tensor, cfginst):
        return twodim_np_tensor.reshape(-1,
                                        cfginst.lags,
                                        len(cfginst.features))

    def save_model(self):
        model_folder = self.namefiles_dict["model_folder"]
        # Todo: save the model under folder for specific configuration (See conf_name under EUR_USD_1.py)
        if not os.path.exists(model_folder):
            logging.info("trainer: specific model folder does not exist: creating it...")
            os.mkdir(model_folder)
            # Todo add choice to break out?

        model_file_name = model_folder + "/" + str(self.model_id) + ".h5"
        self.model.save(model_file_name)

        print(model_file_name)
        return

    def evaluate_model(self):
        print("\n")
        if self.model_id == "ffn" or self.model_id == "ffn":  # todo : do this better

            print("main: Evaluating the model on in-sample data (training data)")
            self.model.evaluate(self.train_data[self.lagged_cols], self.train_targets["dir"], verbose=True)
            print("main: valuating the model on out-of-sample data (test data)")
            self.model.evaluate(self.test_data[self.lagged_cols], self.test_targets["dir"], verbose=True)

            # Todo: why evaluate does not show the accuracy?
        elif self.model_id == "LSTM_dnn" or \
                self.model_id ==  "LSTM_dnn_all_states":

            print("main: Evaluating the model on in-sample data (training data)")
            numpy_eval = self._get_3d_tensor(self.train_data[self.lagged_cols_reordered]. \
                to_numpy(), cfginst)
            self.model.evaluate(numpy_eval, self.train_targets["dir"].to_numpy(), verbose=True)

            print("main: valuating the model on out-of-sample data (test data)")
            numpy_test = self._get_3d_tensor(self.test_data[self.lagged_cols_reordered]. \
                to_numpy(), cfginst)
            self.model.evaluate(numpy_test, self.test_targets["dir"].to_numpy(), verbose=True)

        elif self.model_id == "LSTM_dnn_all_states_mout":
            # inputs: A 3D tensor with shape [batch, timesteps, feature].

            numpy_train_targets = self.train_targets[["dir", "dir_sma1", "dir_sma2"]].to_numpy()
            numpy_val_targets = self.validation_targets[["dir", "dir_sma1", "dir_sma2"]].to_numpy()
            numpy_test_targets = self.test_targets["dir", "dir_sma1", "dir_sma2"].to_numpy()

            print("main: Evaluating the model on in-sample data (training data)")
            numpy_eval = self._get_3d_tensor(self.train_data[self.lagged_cols_reordered]. \
                                             to_numpy(), cfginst)
            self.model.evaluate(numpy_eval,
                                [numpy_train_targets[:,0],numpy_train_targets[:,1],numpy_train_targets[:,2]],
                                verbose=True)

            print("main: valuating the model on out-of-sample data (test data)")
            numpy_test = self._get_3d_tensor(self.test_data[self.lagged_cols_reordered]. \
                                             to_numpy(), cfginst)
            self.model.evaluate(numpy_test,
                                [numpy_test_targets[:, 0], numpy_test_targets[:, 1], numpy_test_targets[:, 2]],
                                verbose=True)


        #keras.metrics.confusion_matrix(self.test_targets["dir"], y_pred)

        return

    def make_predictions(self):
        print("make_predictions: just testing predictions for later trading applications")
        if self.model_id == "ffn" or self.model_id == "ffn":  # todo : do this better
            pred = self.model.predict(self.train_data[self.lagged_cols], verbose=True)
        elif self.model_id == "LSTM_dnn" or \
                self.model_id ==  "LSTM_dnn_all_states":
            numpy_train = self._get_3d_tensor(self.train_data[self.lagged_cols_reordered]. \
                to_numpy(), cfginst)
            pred = self.model.predict(numpy_train, verbose=True)
        print(pred)

        print("make_predictions: confusion matrix on train data for market direction next time step")
        if self.model_id == "LSTM_dnn_all_states_mout":
            pred = pred[:,0]
        print(keras.metrics.confusion_matrix(self.test_targets["dir"], pred))
        return


################################ TRaDer (trd) class #########################################
class trd(ond, tpqoa.tpqoa):
    '''
    Trading class to get tick data and trade or to back/forward test using the data already obtained
    '''

    def __init__(self,
                 conf_file=None,
                 instrument_file=None,
                 model_id=None,
                 mu=None,
                 std=None):

        tpqoa.tpqoa.__init__(self, conf_file=conf_file)
        ond.__init__(self, instrument_file=instrument_file, conf_file=conf_file)

        self.position = 0
        self.window = instrument_file.window
        self.bar_length = instrument_file.brl
        self.units = instrument_file.units
        self.model_id = model_id
        self.model = None
        self.mu = mu
        self.std = std
        self.tick_data = pd.DataFrame()
        self.hist_data = pd.DataFrame()
        self.min_length = None
        self.raw_data = pd.DataFrame()
        self.data = pd.DataFrame()
        self.profits = []

        self.barrier = None

        # dataframes to hold ask and bid historical prices
        self.ask_df = pd.DataFrame()
        self.bid_df = pd.DataFrame()
        #self.half_spread = instrument_file.half_spread
        self.sma_int = instrument_file.sma_int
        self.features = instrument_file.features
        self.h_prob_th = instrument_file.higher_go_long
        self.l_prob_th = instrument_file.lower_go_short
        self.strategy = Strategy_1(instrument=self.instrument,
                                   order_fun= self.create_order,
                                   report_fun= self.report_trade)
        self.set_model()
        return


    def set_model(self): # Todo: make this a class method, and also it is shared with trn... make it better

        self.model = keras.models.load_model(cfg.trained_models_path +
                                        self.instrument + "/" + self.model_id + ".h5")
        print("Layers of model being used are: ")
        print(self.model.layers)
        return


    def test(self, data, FW=True):
        '''
        test the (model,strategy) pair on data, and compare the buy&hold with
        the chosen strategy. Ideally estimation of trading costs should be included

        :param data: data used to obtain data for test model performance
        :param FW: kind of testing - BackWard or ForWard
        :return: absolute and delta value of cumulative pnl
        '''

        test_outs = pd.DataFrame()
        self.data = data.copy()
        (train_split, val_split) = u.get_split_points(self.data, cfginst.split_pcs)

        if FW:
            self.data = self.data.iloc[train_split:] #val_split:] using validation data as well to test..  review!
            test_data = self.data
            test_type = "FW_TESTING"
        else:
            self.data = self.data.iloc[:train_split]
            test_data = self.data
            test_type = "BW_TESTING"

        # Predictions
        test_outs["probs"] = self.predict(TESTING=True)

        # Strategy in action
        test_outs["position"] = self.strategy.act(
            prob_up=test_outs["probs"],
            thr_up=self.h_prob_th,
            thr_low=self.l_prob_th,
            live_or_test="test")

        # calculate Strategy Returns
        test_outs["strategy_gross"] = test_outs["position"] * (test_data["returns"])

        # determine when a trade takes place
        test_outs["trades"] = test_outs["position"].diff().fillna(0).abs()

        # subtract transaction costs from return when trade takes place
        # returns are log returns, so I need to take the exp,
        # test_outs['strategy'] = (test_outs["strategy_gross"].apply(np.exp) -
        #                         test_outs["trades"] * test_data["spread"]/2).apply(np.log)

        test_outs['trading_costs'] = test_outs["trades"] * test_data["spread"]/2


        # calculate cumulative returns for strategy & buy and hold
        test_outs["creturns"] = (data["returns"]).cumsum().apply(np.exp)
        # test_outs["cstrategy"] = test_outs['strategy'].cumsum().apply(np.exp)
        test_outs["cstrategy_gross"] = test_outs['strategy_gross'].cumsum().apply(np.exp)
        test_outs["cstrategy"] = test_outs["cstrategy_gross"] - test_outs['trading_costs'].cumsum()

        results = test_outs

        # absolute performance of the strategy
        perf = results["cstrategy"].iloc[-1]

        # out-/underperformance of strategy
        outperf = perf - results["creturns"].iloc[-1]
        print("outperf is ", outperf)

        # plot results
        print("plotting cumulative results of buy&hold and strategy")

        title = "{} {} | Avg Transaction Cost = {}".format(
            self.instrument, test_type, test_data["spread"].mean())
        results[["cstrategy",  "creturns", "cstrategy_gross"]].\
            plot(title=title, figsize=(12, 8))
        plt.show()

        plt.title("{} {} | positions".format(self.instrument, test_type))
        plt.plot(test_outs["position"])
        plt.xlabel("time")
        plt.ylabel("positions")
        plt.show()

        return round(perf, 6), round(outperf, 6)


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
            print("{} | units = {} | price = {} | P&L = {} | Cum P&L = {}".\
                  format(time, units, price, pl, cumpl))
            print(100 * "-" + "\n")
        return

    def resample_hist_data(self):
        logging.info("resample_hist_data: resampling recent history to chosen bar length in line" +
                     " with the training data used to train the model")
        self.raw_data = self.raw_data.resample(self.bar_length, label="right").\
                            last().dropna().iloc[:-1]

        self.hist_data = self.raw_data.copy()
        self.min_length = len(self.hist_data) + 1

        print("\nresampled history dataframe information:\n")
        self.raw_data.info()
        return

    def resample_and_join(self):
        self.raw_data = \
            self.hist_data.append( # todo: why hist_data is not updated?
                self.tick_data.resample(self.bar_length,
                                        label="right")\
                                        .last()\
                                        .ffill()\
                                        .iloc[:-1])

        return

    def prepare_data(self):
        logging.info("\nprepare_data: creating features")

        # Todo: make sure the params for features match the feature params used to generate training!!
        # create base features
        df = self.raw_data.reset_index(drop=True, inplace=False)# Todo: why is this here???
        df = u.make_features(df,
                            self.sma_int,
                            self.window,
                            ref_price = self.instrument,
                            live_trading=True )

        # create lagged features: this is where our tick data becomes "lag_1" in the input to the model...
        # Todo: make sure of it!
        df = u.make_lagged_features(df, self.features, self.lags)
        self.data = df.copy()

        return

    def predict(self, TESTING=False):
        print("\nSEQUENCE: predict")
        df = self.data.copy()
        if not TESTING:
            # if we are trading live (not TESTING) we need to normalize the data using
            # training dataset statistics
            df = (df - self.mu) / self.std # Todo: review application of this!
        # get feature columns
        all_cols = self.data.columns
        lagged_cols = []
        lagged_cols_reordered = []


        if self.model_id == "ffn" or self.model_id == "ffn":
            for col in all_cols:
                if 'lag' in col:
                    lagged_cols.append(col)
            df["proba"] = self.model.predict(df[lagged_cols])

        elif self.model_id == "LSTM_dnn" \
                or self.model_id == "LSTM_dnn_all_states"\
                or self.model_id == "LSTM_dnn_all_states_mout":
            # reoder features
            for lag in range(1, self.lags + 1):
                for feat in self.features:
                    r = u.find_string(all_cols, feat + "_lag_" + str(lag))
                    if r != -1:
                        lagged_cols_reordered.append(r)

            numpy_train = self._get_3d_tensor(df[lagged_cols_reordered]. \
                to_numpy())
            all_probs = self.model.predict(numpy_train, verbose=True) # all probs is a list!!
            df["proba"] = all_probs if len(all_probs) <= 1 else all_probs[0]
            #todo: review - taking only out in case of LSTM_dnn_all_states_mout

        self.data = df.copy()
        if TESTING:
            # Todo Add here generation of confusion matrix:
            #keras.metrics.confusion_matrix(y, y_hat)
            # y_hat obtained from np.argmax of predicted probability out or just the class id (0/1 for binclass)?
            return df["proba"]
        else:
            return

    def _get_3d_tensor(self, twodim_np_tensor):
            return twodim_np_tensor.reshape(-1,
                                            self.instrument_file.lags,
                                            len(self.instrument_file.features))

    def on_success(self, time, bid, ask):
        print("\nReceived ticks: ", self.ticks, end=" ")

        # store and resample tick data and join with historical data
        # ask_df = pd.DataFrame({self.instrument: ask },
        #                   index=[pd.to_datetime(time)])
        # bid_df = pd.DataFrame({self.instrument: bid},
        #                       index=[pd.to_datetime(time)])
        ref_price = pd.DataFrame({self.instrument: (ask + bid)/2  },
                          index=[pd.to_datetime(time)])
        spread = pd.DataFrame({"spread": (ask - bid)},
                              index=[pd.to_datetime(time)])

        # from bid and ask price I need to recreate the data useful to extend the  used for training
        # so I need to emulate what I do in gdt .make_features and .make_lagged_features
        # but this is done in trd.prepare_data
        # here I need to prepare the tick_data for it to joined the past live data
        # which will then be passed to make_feature, which feed make_lagged_features
        # to obtain suitable input data for the model to make predictions
        # so I need to create average instrument price (named "instrument") and spread
        # (named "spread") columns in a dataframe

        self.tick_data = self.tick_data.append(
                            pd.concat([ref_price, spread], axis=1, join='inner'))
        # visualize the data added to the tick_data
        print("ONSUCCESS: self.tick_data.tail()\n", self.tick_data.tail())

        # resample and join data
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
                       units = self.units,
                       live_or_test="live")
        return


################################################## ctrl ###############################################
class ctrl:
    '''
    Class for controlling data acquisition, model training, testing and trading of
    ML based trading strategies
    '''

    def __init__(self,
                 conf_file, # oanda user conf file
                 instrument_files):  # list of instrument conf files - for now only 1 element supported!

        self.instrument_files = instrument_files
        self.instruments = [t.instrument for t in self.instrument_files]
        self.targets = [t.targets for t in self.instrument_files]  # "dir", "profit_over_spread", "loss_over_spread"]
        self.features = [t.features for t in self.instrument_files]
        self.lags = [t.lags for t in self.instrument_files]

        self.namefiles_dicts = []
        for f in self.instrument_files:
            namefiles_dict = {}
            namefiles_dict = u.creates_filenames_dict(
                f.instrument,
                namefiles_dict, cfg)
            self.namefiles_dicts.append(namefiles_dict)

        # Create the first of the fundamental objects to operate:
        # getting data.
        # train model, test and trade, will be created later, when clarified
        # if we have all the necessary parts (data, mu, std at least)

        # many objects to get data? or just one that deals internally with multithread.
        # for now make many
        self.gdt_list = []
        for f in instrument_files:
            b = gdt(f, conf_file)
            self.gdt_list.append(b)

        return

        # # many objects to train the model on a specific data and targets
        # self.model_id = model_id
        # self.trn_list = []
        # for f in instrument_files:
        #     b = trn(f, conf_file, self.model_id)
        #     self.trn_list.append(b)
        #
        # # only one trader is needed, as it will take a model trained on the joint data
        # # and it will get tick data from a set of threads, one for each instrument, and t
        # # the model can have multiple outputs predicting the market trend for each instrument
        # # for now trader supports only one instrument however so let's pass the first in the list
        #
        # self.trader = trd(self.instrument_files[0], conf_file,
        # return



if __name__ == '__main__':
    '''
    __main__ execute functional test, or it can be used to download data for later use
    '''

    ####  IMPORTANT ####
    ####  change this import pointing to the
    ####  wanted/needed configuration
    import configs.EUR_PLN_2 as cfginst

    ######################## data  ########################
    odc = gdt(instrument_file=cfginst, conf_file=cfg.conf_file)

    print('OandaDataCollector object created for instrument {}'.format(cfginst.instrument))
    NEW_DATA = False
    REPORT_only = True
    if not REPORT_only:
        if NEW_DATA:
            # actual data collection of most recent data
            print('OandaDataCollector data collection starts...')
            odc.get_most_recent(granul=cfginst.granul, days = cfginst.days)
            print("All row data downloaded from Oanda for instrument {}".format(cfginst.instrument))
            print(odc.raw_data.info(), end="\n  ******** \n")
        else:
            # reloading raw data already dowloaded in precedence and using this to
            # create features
            # useful when rebuilding new features and datastes but no need to download data from Oanda
            print('OandaDataCollector is loading from disk only RAW data: DATASETS are being re-built...')
            odc.load_data_from_file(raw_only=True)

        odc.make_features()
        odc.make_lagged_features(lags=cfginst.lags)
        odc.resample_data(brl = cfginst.brl)

        odc.make_3_datasets()
        odc.standardize()
        odc.save_to_file()

    else: # Just report on data available
        print('OandaDataCollector all data (RAW and DATASETS) is loading from disk...')
        odc.load_data_from_file()
        odc.report()


    ######################## train  ########################
    model_id = "LSTM_dnn_all_states_mout" # "LSTM_dnn" #"ffn" #""dnn1" # #

    TRAIN = True
    EPOCHS = 50
    DROPOUT = 0.1
    if TRAIN:
        model_trainer = trn(instrument_file=cfginst,
                            conf_file=cfg.conf_file,
                            model_id=model_id)

        logging.info("Loading data and creating the NN model...")
        model_trainer.load_train_data()
        model_trainer.set_model(dropout=DROPOUT)

        logging.info("Training the NN model...")
        model_trainer.train_model(epochs=EPOCHS)

        logging.info("Saving the NN model...")
        model_trainer.save_model()

        logging.info("Evaluating the NN model...")
        model_trainer.evaluate_model()

        logging.info("Predictions with the NN model...")
        model_trainer.make_predictions()

        # Todo: would it better to make a parent class for training and inherit from that to create ad-hoc trainers
        # per each model in models?? seems more what I want...

    ######################## trade ########################
    # Todo: review, should not be necessary to pass the instrument here...
    instrument = cfginst.instrument
    # get or generate datafiles files and folders, if do not exist
    namefiles_dict = {}
    namefiles_dict = u.creates_filenames_dict(instrument, namefiles_dict, cfg)

    # load params for data standardization
    params = pickle.load(open(namefiles_dict["params"], "rb"))
    mu = params["mu"]
    std = params["std"]

    # create trader object using instrument configuration details
    trader = trd(conf_file=cfg.conf_file,
                 instrument_file=cfginst,
                 model_id=model_id,
                 mu=mu,
                 std=std)

    # either live trading or testing (back or fw testing)
    TRADING = 0
    BCKTESTING, FWTESTING = (1, 0) if not TRADING else (0, 0)  # todo: do it better!!


    if TRADING:
        # get historical data to compute features that requires past data
        trader.get_most_recent(days=cfginst.days_inference, granul=cfginst.granul)

        # resample historical data
        trader.resample_hist_data()
        logging.info("main: most recent historical data obtained and resampled" +
                     "now starting streaming data and trading...")

        # streaming & trading here!!!!
        trader.stream_data(cfginst.instrument, stop=cfginst.stop_trading)

        if trader.position != 0:
            print("Closing position as we are ending trading!")
            close_order = trader.create_order(instrument=cfginst.instrument,
                                              units=-trader.position * trader.units,
                                              suppress=True, ret=True)  # close Final Position
            trader.report_trade(close_order, "GOING NEUTRAL")  # report Final Trade
    else:  # TESTING
        # loading data
        assert os.path.exists(namefiles_dict["base_data_folder_name"]),\
            "Base data folder DO NOT exists!"

        train_data = pd.read_csv(namefiles_dict["train_filename"],
                                 index_col="time", parse_dates=True, header=0)
        test_data = pd.read_csv(namefiles_dict["test_filename"],
                                index_col="time", parse_dates=True, header=0)

        train_targets = pd.read_csv(namefiles_dict["train_labl_filename"],
                                    index_col="time", parse_dates=True, header=0)
        test_targets = pd.read_csv(namefiles_dict["test_labl_filename"],
                                   index_col="time", parse_dates=True, header=0)

        data_with_returns = pd.read_csv(namefiles_dict["raw_data_featured_resampled_file_name"],
                                    index_col="time", parse_dates=True, header=0)

        if BCKTESTING:
            trader.test(data_with_returns, FW=False)
        else:  # fwtesting
            trader.test(data_with_returns) #Todo: must pass here data with returns, relative to test!