import pandas as pd
import data_processing as data
import stock_selection as selection
import factor_models as factors
import numpy as np

import covariance_models as cov

class Strategy():

    def __init__(self, strategy_pipeline, stop_loss, stop_profit, type_of_market = None, stock_universe = None):
        self.stop_loss = stop_loss
        self.stop_profit = stop_profit
        self.type_of_market = type_of_market
        self.stock_universe = stock_universe

        self.strategy_pipeline = strategy_pipeline

    
    def fit(self, start_date, end_date, data_stream, **kwargs):
        data = data_stream(start_date, end_date, type_of_market = self.type_of_market, stocks = self.stock_universe, **kwargs)
        prices = pd.pivot_table(data, values = 'premed', index = 'date', columns = 'cod_neg')
        returns = prices.dropna(axis=1).pct_change().dropna()

        self.strategy = self.strategy_pipeline.fit_transform(returns)

    def simulate(self, cash_investment, start_date, end_date, data_stream, rebalance_period = 0, **kwargs):
        data = data_stream(start_date, end_date, stocks = self.strategy.index, **kwargs)
        prices = pd.pivot_table(data, values = 'premed', index = 'date', columns = 'cod_neg')
        shares = pd.DataFrame(np.zeros(prices.shape), index = prices.index, columns = prices.columns)

        starting_shares = self._rebalance(cash_investment, prices.iloc[0])
        shares.iloc[0] = starting_shares

        portfolio_value = [(starting_shares*prices.iloc[0]).sum()]
        for idx, _ in enumerate(prices.index):
            if idx > 0:
                current_shares = shares.iloc[idx-1]
                current_portfolio_value = (current_shares*prices.iloc[idx]).sum()
                portfolio_value.append(current_portfolio_value)

                if rebalance_period != 0 and idx%rebalance_period == 0:
                    new_shares = self._rebalance(current_portfolio_value, prices.iloc[idx])
                    shares.iloc[idx] = new_shares
                else:
                    shares.iloc[idx] = shares.iloc[idx - 1]


        shares['portfolio_value'] = portfolio_value
        return shares

    def _rebalance(self, portfolio_value, prices):
        return (portfolio_value*self.strategy.weight).T/prices
