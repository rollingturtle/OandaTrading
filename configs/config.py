# configuration parameters that are not related to specific instrument
import logging
import os

# Todo: make it the import parametric? so know what case is what...?
# local use
proj_path = os.path.abspath("./") + "/"

# for colab:
#proj_path = '/content/drive/MyDrive/drive_on_mac/OandaTrading/'
config_path = proj_path + "configs/"
data_path = proj_path + "Data/"
trained_models_path = proj_path + "TrainedModels/"

# configuration file for oanda access
conf_file = config_path + "oanda.cfg"

# general logging level
logging.basicConfig(level=logging.INFO)
