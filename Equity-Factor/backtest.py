#backtest.py

'''
"yfinance|yf" lets us download stock data
"pandas|pd" lets us work with tables of data
'''
import pandas as pd
import numpy as np


'''Forms top minus bottom [*]decile portfolio (monthly rebalanced)'''
def form_long_short_portfolio(factor_data, returns_data):

    '''ME = month end, convert daily data to monthly - "reasample("ME").last()" takes last day of each month as monthly factor value'''
    monthly_factor = factor_data.resample("ME").last()
    monthly_returns = returns_data.resample("ME").sum()

    portfolio_returns = []
    dispersion_series = []
    weights_history = []


    '''go through each month except last'''
    for i in range(len(monthly_factor.index) - 1):

        date = monthly_factor.index[i]

        '''get factor values at this month'''
        factor_values = monthly_factor.loc[date]

        '''get next month's returns'''
        next_month_returns = monthly_returns.iloc[i + 1]

        '''
        - You only rank valid stocks
        - Returns match the filtered stocks
        '''
        factor_values = factor_values.dropna()
        next_month_returns = next_month_returns.loc[factor_values.index]

        '''rank stocks into 10 buckets - split sinto 10 equal groups'''
        n_buckets = min(10, len(factor_values))
        deciles = pd.qcut(factor_values.rank(), n_buckets, labels=False, duplicates="drop")
        '''If you use fewer than 10 stocks (like custom list),
        pd.qcut(..., 10) can throw: "ValueError:"'''

        weights = pd.Series(0, index=factor_values.index)

        # Make sure weights are float, not int
        weights = weights.astype(float)

        # Long decile 9
        long_mask = (deciles == 9)
        if long_mask.sum() > 0:
            weights[long_mask] = 1 / long_mask.sum()
        else:
            weights[long_mask] = 0.0

        # Short decile 0
        short_mask = (deciles == 0)
        if short_mask.sum() > 0:
            weights[short_mask] = -1 / short_mask.sum()
        else:
            weights[short_mask] = 0.0


        weights_history.append(weights)


        '''top or best 10%'''
        long = next_month_returns[deciles == 9].mean()

        '''bottom or worst 10%'''
        short = next_month_returns[deciles == 0].mean()

        dispersion = next_month_returns.std()
        dispersion_series.append(dispersion)


        '''long minus short return - profit from best stocks minus loss from worst stocks'''
        portfolio_returns.append(long - short)

    '''return portfolio returns as a pandas Series with dates as index'''
    returns_series = pd.Series(portfolio_returns, index=monthly_factor.index[:-1])
    dispersion_series = pd.Series(dispersion_series, index=monthly_factor.index[:-1])

    weights_df = pd.DataFrame(weights_history, index=monthly_factor.index[:-1])

    return returns_series, dispersion_series, weights_df




def sharpe_ratio(returns):
    '''annualised Sharpe ratio'''
    return np.sqrt(12) * returns.mean() / returns.std()

def market_regime_filter(returns, market_returns):
    '''Bull if market above 200-day MA'''

    market_price = (1 + market_returns).cumprod()
    ma_200 = market_price.rolling(200).mean()
    '''smooth price over 200 days to identify long-term trend'''

    regime = market_price > ma_200
    '''
    - price > MA = Bull, 
    - price < MA = Bear
    '''

    return regime

def max_drawdown(returns):
    '''calculate maximum drawdown'''
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

'''how much position changed'''
def calculate_turnover(weights):
    return weights.diff().abs().sum(axis=1)
    '''
    - "abs()" ignores direction
    - sum() is total trading needed
    '''

'''High turnover reduces implementability due to transaction costs.'''

def concentration(weights):
    return (weights ** 2).sum(axis=1)
    '''
    - (weights ** 2) penalises large positions, 
    - sum() gives concentration risk
    - higher concentration = more risk
    '''


'''[*] A "decile portfolio" is a method in finance and investment analysis that 
ranks a universe of assets (such as stocks or mutual funds) based on a specific 
metric—like performance, P/E ratio, or risk—and divides them into ten equal 
groups (deciles), each representing 10% of the total, to analyze, compare, 
and rank their performance.'''