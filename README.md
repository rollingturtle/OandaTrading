# OandaTrading
Tools to operate with Oanda using Machine Learning. 
It requires a personal `oanda.cfg` file under `configs` folder which is used to authenticate to the 
Oanda service https://trade.oanda.com/

Currently, tools can be divided in 3 categories: 

1. getting and preparing the data: download historical data, make features, make lagged features, prepare datasets
1. train a model: take a model defined under models, train a model, save the trained model under trained models
1. trade with ML: live trading and back/fw testing. Download a small chunk of most recent historical data, get stream of live data, concatenate them, make features, lagged features, run ML, get probabilities, apply selected strategy and issue orders accordingly
Ideally there should be a set of strategies, a set of models, a set of possible features-set, a set of instruments to trade.

To create the conda environment: `conda create --name myenv --file env-details.txt`
