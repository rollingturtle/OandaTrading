# numeric params (not derived by data) that must be common among the
# data collection phase and the actual inference/trading moment

instrument = "EUR_USD"
conf_name = instrument + "_1"
brl = "1min"  # bar lenght for resampling
window = 10
sma_int = 5
hspread_ptc = 0.00007
labels = ["dir", "profit_over_spread", "loss_over_spread"]
features = ["dir", "sma", "boll", "min", "max", "mom", "vol",
                 "profit_over_spread", "loss_over_spread"]

granul="S5"
days = 20#20
days_inference = 3
lags = 5
split_pcs = (0.7, 0.15, 0.15)
stop_trading = 300

#trading thresholds for probability
lower_go_short = 0.4999 #0.47
higher_go_long = 0.53

