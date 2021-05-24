from abc import ABCMeta, abstractmethod
import pandas as pd
import logging
import sys
import os
import matplotlib
if __name__ == "__main__":
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')

os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append('../')

import configs.config as cfg
import models.model as m
import common.utils as u


################################ Trainer Class

# class Trainer(metaclass=ABCMeta):
#     @abstractmethod
#     def act(self,
#             position=0,
#             prob_up=0.5,
#             thr_up=.53,
#             thr_low=.47,
#             units=1000):
#         pass


class DL_Trainer():
    '''
    Implements a class that trains a DL model on data relative to a specific instrument
    '''
    def __init__(self,
                 instrument_file,
                 model_id):

        self.instrument_file = instrument_file
        self.instrument = instrument_file.instrument
        self.features = instrument_file.features
        self.lags = instrument_file.lags

        self.namefiles_dict = {}
        self.namefiles_dict = u.creates_filenames_dict(
                                  self.instrument_file.instrument,
                                    self.namefiles_dict, cfg)

        assert model_id in m.available_models, \
            "DL_Trainer: init: model_id not among known models"
        self.model_id = model_id

        self.model = None
        self.train_data = None
        self.test_data = None
        self.validation_data = None
        self.validation_labels = None
        self.train_labels = None
        self.test_labels = None
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

        self.train_labels = pd.read_csv(self.namefiles_dict["train_labl_filename"],
                                   index_col="time", parse_dates=True, header=0)
        self.test_labels = pd.read_csv(self.namefiles_dict["test_labl_filename"],
                                   index_col="time", parse_dates=True, header=0)
        self.validation_labels = pd.read_csv(self.namefiles_dict["valid_labl_filename"],
                                   index_col="time", parse_dates=True, header=0)

        # Todo: make this step unified and linked to instrument specific configuration
        # exctract relevant columns for training (features!)
        all_cols = self.train_data.columns
        for col in all_cols:
            if 'lag' in col:
                self.lagged_cols.append(col)
        # reoder features
        for lag in range(1, self.lags + 1):
            for feat in self.features:
                r = self._find_string(all_cols, feat + "_lag_" + str(lag))
                if r != -1:
                    self.lagged_cols_reordered.append(r)
        print("load_train_data: reordered features are:", self.lagged_cols_reordered)
        print("load_train_data: Lagged columns which are the input to the model:")
        print(self.lagged_cols)
        print("load_train_data: how many Lagged columns:")
        print(len(self.lagged_cols))
        print("self.train_data.head()\n",self.train_data.head())
        assert (not self.train_data[self.lagged_cols].isnull().values.any()), \
            "NANs in Training Data"
        assert (not self.train_labels["dir"].isnull().values.any()), \
            "NANs in LABELS"
        return # train and test data loaded. Validation carved out by Keras from training data

    def set_model(self):
        if self.model_id == "dnn1":
            self.model = m.dnn1(dropout = True,
                              rate=0.1,
                              input_dim = len(self.lagged_cols))
        elif self.model_id == "LSTM_dnn":  #Todo: review if makes sense to inherit for each model...
            self.model = m.LSTM_dnn(dropout = True,
                                    rate=0.1,
                                    inputs = numpy_train)
        elif self.model_id == "ffn":
            self.model = m.ffn(self.train_data[self.lagged_cols],
                              rate=0.1)

        # visualize some details of the model NOT YET TRAINED
        # get some visualization before learning (on weights)
        # Todo: do this better to generate more informative insight
        plt.hist(self.model.layers[2].get_weights()[0])
        plt.show()
        plt.figure()

        plt.hist(self.model.layers[2].get_weights()[1])
        plt.show()
        plt.figure()
        return

    def train_model(self, epochs=30):  # Todo: explode this using gradient_tape
        r = self.model.fit(x=self.train_data[self.lagged_cols],
                      y=self.train_labels["dir"],
                      epochs=epochs,
                      verbose=True,
                      validation_data=(self.validation_data[self.lagged_cols], self.validation_labels["dir"]),
                      shuffle=True,
                      batch_size=64,
                      class_weight=m.cw(self.train_labels))
        # get some visualization of the effect of learning (on weights, loss, acc)
        plt.hist(self.model.layers[2].get_weights()[0])
        plt.show()
        plt.figure()
        plt.plot(r.history['loss'], label="loss")
        plt.plot(r.history['val_loss'], label="val_loss")
        plt.legend()
        plt.show()
        plt.figure()
        plt.plot(r.history['acc'], label="acc")
        plt.plot(r.history['val_acc'], label="val_acc")
        plt.legend()
        plt.show()
        plt.figure()
        return

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
        print("main: Evaluating the model on in-sample data (training data)")
        self.model.evaluate(self.train_data[self.lagged_cols], self.train_labels["dir"], verbose=True)
        print("main: valuating the model on out-of-sample data (test data)")
        self.model.evaluate(self.test_data[self.lagged_cols], self.test_labels["dir"], verbose=True)
        # Todo: why evaluate does not show the accuracy?
        return

    def make_predictions(self):
        print("main: just testing predictions for later trading applications")
        pred = self.model.predict(self.train_data[self.lagged_cols], verbose=True)
        print(pred)
        return

    def _find_string(self, list_of_strings, s):
        for i, ss in enumerate(list_of_strings):
            if s in ss:
                index = i
                return list_of_strings[index]
        return -1


def find_string(list_of_strings, s):
    for i,ss in enumerate(list_of_strings):
        if s in ss:
            index = i
            return list_of_strings[index]
    return -1




if __name__ == "__main__":
    '''
    __main__ executes functional test
    '''
    # Todo: turn this into a real functional test. And unittests?

    m.set_seeds(100)

    ####  IMPORTANT ####
    ####  change this import pointing to the
    ####  wanted/needed configuration
    import configs.EUR_PLN_1 as cfginst

    # set instrument to work with
    instrument = cfginst.instrument

    # get or generate datafiles files and folders, if do not exist
    namefiles_dict = {}
    namefiles_dict = u.creates_filenames_dict(cfginst.instrument, namefiles_dict, cfg)

    # Todo: do this selection better and not via string. This should reference via dict to the model
    model_id = "ffn" #""dnn1"
    model_trainer = DL_Trainer(cfginst, model_id)

    logging.info("Loading data and creating the NN model...")
    model_trainer.load_train_data()
    model_trainer.set_model()

    logging.info("Training the NN model...")
    model_trainer.train_model(epochs=70)

    logging.info("Evaluating the NN model...")
    model_trainer.evaluate_model()

    logging.info("Predictions with the NN model...")
    model_trainer.make_predictions()

    logging.info("Saving the NN model...")
    model_trainer.save_model()

    # Todo: would it better to make a parent class for training and inherit from that to create ad-hoc trainers
    # per each model in models?? seems more what I want...
    #
    # if DNN:
    #    ....
    #
    # else: #LSTM:
    #     numpy_train = train_data[lagged_cols_reordered].\
    #         to_numpy().reshape(-1, len(cfginst.features), cfginst.lags)
    #     print("numpy_train.shape ",numpy_train.shape)
    #     model = LSTM_dnn(dropout = True,
    #                  rate=0.1,
    #                  inputs = numpy_train)
    #
    #     model.fit(x = numpy_train, #train_data[lagged_cols],
    #               y = train_labels["dir"].to_numpy(),
    #               epochs = 20,
    #               verbose = True,
    #               validation_split = 0.1,
    #               shuffle = True,
    #               batch_size=64,
    #               class_weight = cw(train_labels))
    #
    #     print("\n")
    #     print("main: Evaluating the model on in-sample data (training data)")
    #     model.evaluate(numpy_train, train_labels["dir"].to_numpy(), verbose=True)
    #
    #     # testing predictions
    #     print("main: testing predictions for later trading applications")
    #     pred = model.predict(numpy_train, verbose=True)
    #     print(pred)
    #
    #     print("main: valuating the model on out-of-sample data (test data)")
    #     model.evaluate(numpy_train, test_labels["dir"].to_numpy(), verbose=True)
    #
    # model_folder = namefiles_dict["model_folder"]
    # # Todo: save the model under folder for specific configuration (See conf_name under EUR_USD_1.py)
    # if not os.path.exists(model_folder):
    #     logging.info("trainer: specific model folder does not exist: creating it...")
    #     os.mkdir(model_folder)
    #
    #
    # if DNN:
    #     # Todo add choice to break out?
    #     model.save(model_folder + "/DNN_model.h5")
    #     print("main:Trained model save to " + model_folder +"/DNN_model.h5")
    # else: #LSTM:
    #     # Todo add choice to break out?
    #     model.save(model_folder + "/LSTM_model.h5")
    #     print("main:Trained model save to " + model_folder + "/LSTM_model.h5")
    #
    #
