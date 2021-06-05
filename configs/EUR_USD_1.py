# configuration file for a specific case:
# case: instrument, features, various numeric parameters used to create features
# for each case, we store numeric params (not derived by data)
# this data must be shared between the training phase and the trading time
# for the same kind of features must be computed at trading time
# for the ML model to work

instrument = "EUR_USD"
conf_name = instrument + "_1"
brl = "1min"  # bar lenght for resampling
window = 10
sma_int = 5
half_spread = 0.00007

targets = ["dir", "dir_sma1", "dir_sma2"]
# Todo:  "volume" could be included in the training but tick data does not contain it. Oanda
# provides it every 5 second, Aleksander says. How could i get it? historical data every 5 secs?
# Todo "dir_sma1", "dir_sma2" have been removed from the list of features, as they may
# cause look-into-the-future bias: is that confirmed?
features = ["dir", "sma_diff",
            "boll", "min", "max", "mom", "vol",
            "spread", instrument] #  # "volume", "o", "h", "l",


granul="S5"
days = 30
days_inference = 3
lags = 8
split_pcs = (0.7, 0.20, 0.10)
stop_trading = 500

#trading thresholds for probability
lower_go_short = 0.4#0.4
higher_go_long = 0.6#0.6

# bulk trade
units = 1000

