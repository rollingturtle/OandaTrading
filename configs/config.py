import os
import logging

# configuration parameters that are not related to specific instrument being considered

proj_path = "/Users/ANDREA/PycharmProjects/OandaTrading/"
config_path = proj_path + "configs/"
data_path = proj_path + "Data/"
trained_models_path = proj_path + "TrainedModels/"

# configuration file for oanda access
conf_file = config_path + "oanda.cfg"

# general logging level
logging.basicConfig(level=logging.INFO)
