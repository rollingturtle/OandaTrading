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
targets = ["dir", "profit_over_spread", "loss_over_spread"]
features = ["dir", "sma", "boll", "min", "max", "mom", "vol",
                 "profit_over_spread", "loss_over_spread"]

granul="S5"
days = 20
days_inference = 3
lags = 5
split_pcs = (0.7, 0.15, 0.15)
stop_trading = 300

#trading thresholds for probability
lower_go_short = 0.3#0.4
higher_go_long = 0.7#0.6

# bulk trade
units = 1000
