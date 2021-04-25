
import pandas as pd
import numpy as np
import fxcmpy

api = fxcmpy.fxcmpy(config_file = r"C:\Users\hagma\AlgoTrading\Part4_Materials\FXCM\FXCM.cfg") # please insert full file path

tick_data = None
resamp = None
bar_length = "5s"
window = 1
ticks = 0
position = 0
units = 100
instrument = "EUR/USD"
min_length = window + 1


col = ["tradeId", "amountK", "currency", "grossPL", "isBuy"]


def con_trader(data, dataframe):
    
    global tick_data, resamp, ticks, position, min_length # global variables
    
    ticks += 1
    print(ticks, end = " ")
    
    # store and resample tick data
    tick_data = dataframe.iloc[:, :2]
    resamp = tick_data.resample(bar_length, label = "right").last().ffill().iloc[:-1]
    resamp["Mid"] = (resamp.Ask + resamp.Bid)/2
    
    # prepare data & define strategy
    resamp["returns"] = np.log(resamp.Mid / resamp.Mid.shift(1))
    resamp["position"] = -np.sign(resamp.returns.rolling(window).mean())
 
    # executing trades
    if len(resamp) > min_length - 1: # if a new bar is added: go through if/elif
        min_length += 1
        if resamp["position"].iloc[-1] == 1: # if signal is long
            if position == 0:
                order = api.create_market_buy_order(instrument, units) # buy 1 * units if position is neutral ("1 trade")
                print(2*"\n" + "{} | GO LONG | unreal P&L: {}".format(str(order.get_time()),
                                                                      api.get_open_positions().grossPL.sum()) + "\n")
            elif position == -1:
                order = api.create_market_buy_order(instrument, 2 * units) # buy 2 * units if position is short ("2 trades") 
                print(2*"\n" + "{} | GO LONG | unreal P&L: {}".format(str(order.get_time()),
                                                                      api.get_open_positions().grossPL.sum()) + "\n")
            position = 1
        elif resamp["position"].iloc[-1] == -1: # if signal is short
            if position == 0:
                order = api.create_market_sell_order(instrument, units) # sell 1 * units if position is neutral ("1 trade")
                print(2*"\n" + "{} | GO SHORT | unreal P&L: {}".format(str(order.get_time()), 
                                                                       api.get_open_positions().grossPL.sum()) + "\n")
            elif position == 1:
                order = api.create_market_sell_order(instrument, 2 * units) # sell 2 * units if position is long ("2 trades")
                print(2*"\n" + "{} | GO SHORT | unreal P&L: {}".format(str(order.get_time()),
                                                                       api.get_open_positions().grossPL.sum()) + "\n")
            position = -1    
            
    # define trading stop
    if ticks > 50:
        api.close_all_for_symbol(instrument)
        api.unsubscribe_market_data("EUR/USD")
        print(2*"\n" + "{} | GO NEUTRAL".format(str(tick_data.index[-1])) + "\n")
        print(api.get_closed_positions_summary()[col])

api.subscribe_market_data("EUR/USD", (con_trader, ))
