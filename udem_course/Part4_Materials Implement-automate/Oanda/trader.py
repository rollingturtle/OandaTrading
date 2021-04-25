
import pandas as pd
import numpy as np
import tpqoa

class ConTrader(tpqoa.tpqoa):
    def __init__(self, conf_file, instrument, bar_length, window, units):
        super().__init__(conf_file)
        self.position = 0
        self.instrument = instrument
        self.window = window
        self.bar_length = bar_length
        self.units = units
        self.tick_data = pd.DataFrame()
        self.min_length = self.window + 1
        self.profits = [] # store p&l for all trades
    
    def resample_data(self):
        self.data = self.tick_data.resample(self.bar_length, label = "right").last().ffill().iloc[:-1]
    
    def prepare_data(self):
        self.data["returns"] = np.log(self.data.mid / self.data.mid.shift(1))
        self.data["position"] = -np.sign(self.data.returns.rolling(self.window).mean())
    
    def on_success(self, time, bid, ask):
        print(self.ticks, end = " ")
        
        # store and resample tick data
        df = pd.DataFrame({"bid":bid, "ask":ask, "mid":(ask + bid)/2}, 
                          index = [pd.to_datetime(time)])
        self.tick_data = self.tick_data.append(df)
        self.resample_data()
        
        # prepare data & define strategy
        self.prepare_data()
        
        # executing trades
        if len(self.data) > self.min_length - 1:
            self.min_length += 1
            if self.data["position"].iloc[-1] == 1:
                if self.position == 0:
                    order = self.create_order(self.instrument, self.units, suppress = True, ret = True)
                    self.report_trade(order, "GOING LONG")
                elif self.position == -1:
                    order = self.create_order(self.instrument, self.units * 2, suppress = True, ret = True) 
                    self.report_trade(order, "GOING LONG")
                self.position = 1
            elif self.data["position"].iloc[-1] == -1: 
                if self.position == 0:
                    order = self.create_order(self.instrument, -self.units, suppress = True, ret = True)
                    self.report_trade(order, "GOING SHORT")
                elif self.position == 1:
                    order = self.create_order(self.instrument, -self.units * 2, suppress = True, ret = True)
                    self.report_trade(order, "GOING SHORT")
                self.position = -1
    
    def report_trade(self, order, going):
        time = order["time"]
        units = order["units"]
        price = order["price"]
        pl = float(order["pl"])
        self.profits.append(pl)
        cumpl = sum(self.profits)
        print("\n" + 100* "-")
        print("{} | {}".format(time, going))
        print("{} | units = {} | price = {} | P&L = {} | Cum P&L = {}".format(time, units, price, pl, cumpl))
        print(100 * "-" + "\n")



if __name__ == "__main__":
    trader = ConTrader(r"C:\Users\hagma\AlgoTrading\Part4_Materials\Oanda\oanda.cfg", "EUR_USD", "5s", 1, 100000)
    trader.stream_data(trader.instrument, stop = 50) # trading (50 ticks)
    close_order = trader.create_order(trader.instrument, units = -trader.position * trader.units, 
                                      suppress = True, ret = True) # close Final Position
    trader.report_trade(close_order, "GOING NEUTRAL") # report Final Trade


