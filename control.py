import pandas as pd
import tpqoa
from datetime import datetime, timedelta
import logging
import os
import os.path
import pickle

import sys
sys.path.append('../')
import configs.config as cfg
from common import utils as u

import fun_classes as fncl


if __name__ == '__main__':
    '''
    __main__ execute functional test, or it can be used to download data for later use
    '''

    ####  IMPORTANT ####
    ####  change this import pointing to the
    ####  wanted/needed configuration
    import configs.EUR_PLN_2 as cfginst

    odc = fncl.gdt(conf_file=cfg.conf_file, instrument_file=cfginst)

    #GET DATA

    print('OandaDataCollector object created for instrument {}'.format(cfginst.instrument))
    NEW_DATA = True
    if NEW_DATA:
        # actual data collection of most recent data
        print('OandaDataCollector data collection starts...')
        odc.get_most_recent(granul=cfginst.granul, days = cfginst.days)
        odc.make_features()
        odc.make_lagged_features(lags=cfginst.lags)
        odc.resample_data(brl = cfginst.brl)
        odc.make_3_datasets()
        odc.standardize()
        odc.save_to_file()

        print("All row data downloaded from Oanda for instrument {}".format(cfginst.instrument))
        print(odc.raw_data.info(),  end="\n  ******** \n")
        print("Re-sampled data for bar length {} from Oanda for instrument {}".format(
                                                                cfginst.brl, cfginst.instrument))
        print(odc.raw_data_featured_resampled.info(),  end="\n  ******** \n")
    else:
        print('OandaDataCollector data is loading from disk...')
        odc.load_data_from_file()
        odc.report()
        # odc.make_features()
        # odc.make_lagged_features()
        # odc.report()

    # TRAIN

    # set instrument to work with
    instrument = cfginst.instrument

    # # get or generate datafiles files and folders, if do not exist
    # namefiles_dict = {}
    # namefiles_dict = u.creates_filenames_dict(cfginst.instrument, namefiles_dict, cfg)

    # Todo: do this selection better and not via string. This should reference via dict to the model
    model_id = "LSTM_dnn_all_states"# "LSTM_dnn" #"ffn" #""dnn1"
    model_trainer = fncl.trn(conf_file=cfg.conf_file,
                             instrument_file=cfginst,
                             model_id=model_id)

    logging.info("Loading data and creating the NN model...")
    model_trainer.load_train_data()
    model_trainer.set_model()

    logging.info("Training the NN model...")
    model_trainer.train_model(epochs=50)

    logging.info("Evaluating the NN model...")
    model_trainer.evaluate_model()

    logging.info("Predictions with the NN model...")
    model_trainer.make_predictions()

    logging.info("Saving the NN model...")
    model_trainer.save_model()

    # Todo: would it better to make a parent class for training and inherit from that to create ad-hoc trainers
    # per each model in models?? seems more what I want...

    # TRADE
    namefiles_dict = {}
    namefiles_dict = u.creates_filenames_dict(
        cfginst.instrument,
        namefiles_dict, cfg)

#load params for data standardization
    params = pickle.load(open(namefiles_dict["params"], "rb"))
    mu = params["mu"]
    std = params["std"]
    # load trained model
    model_id = "LSTM_dnn" #"ffn" #DNN_model

    # create trader object using instrument configuration details
    trader = DNNTrader(cfg.conf_file,
                       instrument_file=cfginst,
                       model_id=model_id,
                       mu=mu, std=std)

    instrument = cfginst.instrument

    # get or generate datafiles files and folders, if do not exist
    namefiles_dict = {}
    namefiles_dict = u.creates_filenames_dict(instrument, namefiles_dict, cfg)

    # either live trading or testing (back or fw testing)
    TRADING = 1
    BCKTESTING, FWTESTING = (1,0) if not TRADING else (0,0) #todo: do it better!!

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

