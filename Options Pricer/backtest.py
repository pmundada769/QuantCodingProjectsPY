#backtest.py

'''
"pandas|pd" lets us work with tables of data
"numpy|np" lets us do maths on arrays of numbers
'''
import pandas as pd
import numpy as np


'''Forms top minus bottom [*]decile portfolio (monthly rebalanced)'''
def form_long_short_portfolio(factor_data, returns_data):

    '''ME = month end, convert daily data to monthly - "resample("ME").last()" takes last day of each month as monthly factor value'''
    monthly_factor  = factor_data.resample("ME").last()
    monthly_returns = returns_data.resample("ME").sum()

    portfolio_returns  = []
    dispersion_series  = []
    weights_history    = []

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
        factor_values      = factor_values.dropna()
        next_month_returns = next_month_returns.loc[factor_values.index]

        '''rank stocks into 10 buckets - split into 10 equal groups'''
        n_buckets = min(10, len(factor_values))
        deciles   = pd.qcut(factor_values.rank(), n_buckets, labels=False, duplicates="drop")
        '''If you use fewer than 10 stocks (like custom list),
        pd.qcut(..., 10) can throw: "ValueError:"'''

        weights = pd.Series(0, index=factor_values.index)

        '''make sure weights are float, not int'''
        weights = weights.astype(float)

        '''long decile 9 - best momentum stocks'''
        long_mask = (deciles == n_buckets - 1)
        if long_mask.sum() > 0:
            weights[long_mask] = 1 / long_mask.sum()
        else:
            weights[long_mask] = 0.0

        '''short decile 0 - worst momentum stocks'''
        short_mask = (deciles == 0)
        if short_mask.sum() > 0:
            weights[short_mask] = -1 / short_mask.sum()
        else:
            weights[short_mask] = 0.0

        weights_history.append(weights)

        '''top or best 10%'''
        long = next_month_returns[long_mask].mean()

        '''bottom or worst 10%'''
        short = next_month_returns[short_mask].mean()

        '''cross-sectional spread of returns this month'''
        dispersion = next_month_returns.std()
        dispersion_series.append(dispersion)

        '''long minus short return - profit from best stocks minus loss from worst stocks'''
        portfolio_returns.append(long - short)

    '''return portfolio returns as a pandas Series with dates as index'''
    returns_series    = pd.Series(portfolio_returns, index=monthly_factor.index[:-1])
    dispersion_series = pd.Series(dispersion_series, index=monthly_factor.index[:-1])
    weights_df        = pd.DataFrame(weights_history, index=monthly_factor.index[:-1])

    return returns_series, dispersion_series, weights_df


def sharpe_ratio(returns):
    '''annualised Sharpe ratio - sqrt(12) scales monthly to yearly'''
    return np.sqrt(12) * returns.mean() / returns.std()

def sortino_ratio(returns):
    '''
    - like Sharpe but only penalises downside volatility, not upside
    - downside_std uses only negative months in the denominator
    - more realistic for asymmetric strategies like momentum
    '''
    downside_returns = returns[returns < 0]
    downside_std     = downside_returns.std()

    '''avoid division by zero if no losing months'''
    if downside_std == 0:
        return np.nan

    '''annualise with sqrt(12) same as Sharpe'''
    return np.sqrt(12) * returns.mean() / downside_std

def calmar_ratio(returns):
    '''
    - annualised return divided by the absolute max drawdown
    - measures how much return you earn per unit of worst-case loss
    - higher = better recovery from drawdowns
    '''
    ann_return = returns.mean() * 12
    dd         = max_drawdown(returns)

    '''avoid division by zero if no drawdown'''
    if dd == 0:
        return np.nan

    '''divide by absolute value so positive = good'''
    return ann_return / abs(dd)

def market_regime_filter(returns, market_returns):
    '''Bull if market above 200-day MA'''

    market_price = (1 + market_returns).cumprod()
    ma_200       = market_price.rolling(200).mean()
    '''smooth price over 200 days to identify long-term trend'''

    regime = market_price > ma_200
    '''
    - price > MA = Bull
    - price < MA = Bear
    '''

    return regime

def max_drawdown(returns):
    '''calculate maximum drawdown - largest peak-to-trough decline'''
    cumulative = (1 + returns).cumprod()
    peak       = cumulative.cummax()
    drawdown   = (cumulative - peak) / peak
    return drawdown.min()

def rolling_sharpe(returns, window=12):
    '''
    - Sharpe ratio computed over a rolling window of months
    - window=12 means trailing 12-month Sharpe at each point in time
    - shows whether strategy quality is stable or deteriorating
    '''
    rolling_mean = returns.rolling(window).mean()
    rolling_std  = returns.rolling(window).std()

    '''annualise same way as static Sharpe'''
    return np.sqrt(12) * rolling_mean / rolling_std

def hit_rate(returns):
    '''
    - fraction of months where the strategy made money
    - 0.5 = random, >0.6 = consistent, <0.4 = worrying
    '''
    return (returns > 0).mean()

def average_win_loss(returns):
    '''
    - average gain in winning months vs average loss in losing months
    - ratio > 1 means wins are bigger than losses even if hit rate < 0.5
    '''
    wins   = returns[returns > 0].mean()
    losses = returns[returns < 0].mean()

    '''return both so caller can print or compare'''
    return wins, losses

'''how much the portfolio changed month-to-month (trading activity)'''
def calculate_turnover(weights):
    return weights.diff().abs().sum(axis=1)
    '''
    - "abs()" ignores direction
    - sum() is total trading needed
    - high turnover → higher transaction costs
    '''

def concentration(weights):
    return (weights ** 2).sum(axis=1)
    '''
    - (weights ** 2) penalises large positions
    - sum() gives concentration risk
    - higher concentration = more risk from single names
    '''

def transaction_cost_drag(returns, turnover, cost_per_trade=0.001):
    '''
    - estimates how much transaction costs reduce strategy returns
    - cost_per_trade = 0.001 means 10bps round-trip (realistic for liquid equities)
    - drag = turnover * cost_per_trade subtracted from gross returns
    '''
    cost   = turnover * cost_per_trade
    net    = returns.values - cost.reindex(returns.index).fillna(0).values

    '''return net returns as a Series with same index as gross returns'''
    return pd.Series(net, index=returns.index)


'''[*] A "decile portfolio" is a method in finance and investment analysis that 
ranks a universe of assets (such as stocks or mutual funds) based on a specific 
metric—like performance, P/E ratio, or risk—and divides them into ten equal 
groups (deciles), each representing 10% of the total, to analyze, compare, 
and rank their performance.'''