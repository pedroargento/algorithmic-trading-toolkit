import stock_selection as Selection
import covariance_models as Covariance
import factor_models as Factors 
import strategy as strategy

import data_processing as data

from sklearn.pipeline import Pipeline


"""
Create a sklearn pipeline with steps to get from percentage returns to selected stocks and portfolio weights.
This example pipeline includes:
        Step 1: Compute covariance matrix with a model from covariance.py. Accepts any covariance function that get stocks returns as parameters and returns 
        a square matrix

        Step 2: Factor Model: Uses PCA to select independent factors from the covariance matrix. PCA from factor_models.py custom implementation so 
        it can use a provided covariance matrix instead of using the sample covariance like Skelearn.Decomposition implementation.
        
        Setp 3: Select stocks based on factor clustering, choosing the best number of clusters based on silhouette metric. Equal weights portfolio with stocks that are 
        the closest of its cluster center.
        """ 
strategy_pipe = Pipeline([('cov', Covariance.ShrinkageCovariance(model = Covariance.ConstantCovarianceModel)), 
                 ('factors', Factors.PCAFactors(number_of_factors=3)),                         
                 ('stocks', Selection.ClusteringSelection(min_number_of_stocks=2))])           


"""
Instanciate a Strategy from strategy.py passing:
 1) the strategy pipeline, 
 2) stop loss and stop profit function: any function that receives a dataframe of owned shares and shares prices accross time and returns
  a bool variable (passed as 0 value, not yet implemented)
 3) Type of market: list of markets that the strategy is meant to be applied to ('010' = spot market. See ./data/SeriesHistoricas_Layout.pdf for codes).
 4) stock universe: list of stocks that can be traded by the strategy.
"""
strat  = strategy.Strategy(strategy_pipe, 0, 0, type_of_market= ['010'], stock_universe= ['AAPL34', 'EALT4', 'PETR4', 'PETR3'])

"""
Fit the strategy using data between days passed as parameters. Also receives a data streamer, that is any function that returns series of stock prices.
files downloaded from http://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/historico/mercado-a-vista/cotacoes-historicas/
"""
strat.fit('2019-01-01', '2019-12-01', data.bovespa_file_process)

"""
Simulate the strategy with $1000 initial investment between days passed as parameter. Rebalances the strategy (restores target weights) every 'rebalence period' 
days (rebalance_period = 0 means never rebalance). Returns a pandas.DataFrame with shares owned across time and evolution of portfolio value.
"""
results = strat.simulate(1000, '2020-01-01', '2020-12-01', data.bovespa_file_process, rebalance_period=0)
results.plot()
print(results)