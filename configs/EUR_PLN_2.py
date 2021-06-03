# configuration file for a specific case:
# case: instrument, features, various numeric parameters used to create features
# for each case, we store numeric params (not derived by data)
# this data must be shared between the training phase and the trading time
# for the same kind of features must be computed at trading time
# for the ML model to work
import configs as cfg
import os
import logging

instrument = "EUR_PLN"
conf_name = instrument + "_2" # Change this according to the config file name!
brl = "1min"  # bar lenght for resampling
window = 10
sma_int = 5
half_spread = 0.00007
targets = ["dir", "dir_sma1", "dir_sma2"]
features = ["dir", "dir_sma1", "dir_sma2", "sma_diff",
            "boll", "min", "max", "mom", "vol", "volume",
            "o", "h", "l", "spread", instrument]


granul="S5"
days = 2
days_inference = 3
lags = 12
split_pcs = (0.7, 0.10, 0.20)
stop_trading = 500

#trading thresholds for probability
lower_go_short = 0.2#0.4
higher_go_long = 0.8#0.6

# bulk trade
units = 1000
