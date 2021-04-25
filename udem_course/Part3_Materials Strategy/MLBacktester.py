
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
plt.style.use("seaborn")

class MLBacktester():
    ''' Class for the vectorized backtesting of Machine Learning-based trading strategies (Classification).

    Attributes
    ==========
    symbol: str
        ticker symbol with which to work with
    start: str
        start date for data retrieval
    end: str
        end date for data retrieval
    tc: float
        proportional transaction costs per trade

    Methods
    =======
    get_data:
        retrieves and prepares the raw data
    select_data:
        selects a sub-set of the data
    prepare_features:
        prepares the features for the model fitting/prediction 
    fit_model:
        fitting the ML model
    test_strategy:
        runs the backtest for the ML-based strategy
    plot_results:
        plots the performance of the strategy compared to buy and hold
    '''

    def __init__(self, symbol, start, end, tc):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.tc = tc
        self.model = LogisticRegression(C = 1e6, max_iter = 100000, multi_class = "ovr")
        self.results = None
        self.get_data()
    
    def __repr__(self):
        rep = "MLBacktester(symbol = {}, start = {}, end = {}, tc = {})"
        return rep.format(self.symbol, self.start, self.end, self.tc)
                             
    def get_data(self):
        ''' Retrieves and prepares the data.
        '''
        raw = pd.read_csv("five_minute_pairs.csv", parse_dates = ["time"], index_col = "time")
        raw = raw[self.symbol].to_frame().dropna()
        raw = raw.loc[self.start:self.end]
        raw.rename(columns={self.symbol: "price"}, inplace=True)
        raw["returns"] = np.log(raw / raw.shift(1))
        self.data = raw.dropna()
        return raw
                             
    def select_data(self, start, end):
        ''' Selects sub-sets of the financial data (training set/test set).
        '''
        data = self.data.loc[start:end].copy()
        return data
    
    def prepare_features(self, start, end):
        ''' Prepares the feature columns for the fitting/prediction steps.
        '''
        self.data_subset = self.select_data(start, end)
        self.feature_columns = []
        for lag in range(1, self.lags + 1):
            col = "lag{}".format(lag)
            self.data_subset[col] = self.data_subset["returns"].shift(lag)
            self.feature_columns.append(col)
        self.data_subset.dropna(inplace=True)
        
    def fit_model(self, start, end):
        ''' Fitting the ML Model.
        '''
        self.prepare_features(start, end)
        self.model.fit(self.data_subset[self.feature_columns], np.sign(self.data_subset["returns"]))
        
    def test_strategy(self, start_train, end_train, start_test, end_test, lags=5):
        ''' Backtests the trading strategy.
        '''
        self.lags = lags
        
        # fit the model on the training set
        self.fit_model(start_train, end_train)
        
        # prepare the test set
        self.prepare_features(start_test, end_test)
        
        # make predictions on the test set
        prediction = self.model.predict(self.data_subset[self.feature_columns])
        self.data_subset["pred"] = prediction
        
        # calculate Strategy Returns
        self.data_subset["strategy"] = self.data_subset["pred"] * self.data_subset["returns"]
        
        # determine when a trade takes place
        self.data_subset["trades"] = self.data_subset["pred"].diff().fillna(0).abs()
        
        # subtract transaction costs from return when trade takes place
        self.data_subset.strategy = self.data_subset.strategy - self.data_subset.trades * self.tc
        
        # calculate cumulative returns for strategy & buy and hold
        self.data_subset["creturns"] = self.data_subset["returns"].cumsum().apply(np.exp)
        self.data_subset["cstrategy"] = self.data_subset['strategy'].cumsum().apply(np.exp)
        self.results = self.data_subset
        
        # absolute performance of the strategy
        perf = self.results["cstrategy"].iloc[-1]
        # out-/underperformance of strategy
        outperf = perf - self.results["creturns"].iloc[-1]
        
        return round(perf, 6), round(outperf, 6)
        
    def plot_results(self):
        ''' Plots the cumulative performance of the trading strategy
        compared to buy and hold.
        '''
        if self.results is None:
            print("No results to plot yet. Run a strategy.")
        else: 
            title = "{} | TC = {}".format(self.symbol, self.tc)
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))
