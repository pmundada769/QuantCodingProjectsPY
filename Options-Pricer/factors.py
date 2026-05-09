#factors.py

'''"pandas|pd" allows users to create tables of equity factor data'''
import pandas as pd

'''12month momentum excluding last month ("12 - 1" momentum)'''
def momentum_factor(price_data):

    '''12month return (252 or 21*12) - ∆price 252 trading days (1Y)'''
    long_term = price_data.pct_change(252)

    '''1month return (21) - ∆price 21 trading days (1M)'''
    short_term = price_data.pct_change(21)

    '''subtract recent month to avoid short-term reversal noise - ∆price 252;21'''
    momentum = long_term - short_term

    '''return momentum to whoever asks'''
    return momentum

'''3month rolling volatility'''
def volatility_factor(returns):

    '''calculate rolling 63 trading day standard deviation - 63 = 21*3 (3M or Q-quarter)'''
    volatility = returns.rolling(63).std()

    '''return volatility to whoever asks'''
    return volatility

'''value factor using price-to-book proxy (inverse of 2Y price change)'''
def value_factor(price_data):

    '''
    - True value needs balance sheet data (P/B ratio from fundamentals)
    - We proxy value as the inverse of 2-year price appreciation
    - Stocks that have risen least over 2Y are "cheap" by this measure
    - 504 trading days ≈ 2 years
    '''
    two_year_return = price_data.pct_change(504)

    '''invert: low past return = high value score'''
    value = -two_year_return

    '''return value scores to whoever asks'''
    return value

'''mean reversion factor - short-term reversal (1 month)'''
def reversal_factor(returns):

    '''
    - stocks that fell hardest last month tend to bounce (and vice versa)
    - 21 trading days = 1 month
    - this is the opposite signal to momentum, useful for blending
    '''
    one_month_return = returns.rolling(21).sum()

    '''invert: worst recent performer = best reversal candidate'''
    reversal = -one_month_return

    '''return reversal scores to whoever asks'''
    return reversal

'''quality factor using earnings stability proxy (rolling return consistency)'''
def quality_factor(returns):

    '''
    - true quality uses ROE, debt-to-equity, earnings stability from fundamentals
    - we proxy quality as rolling Sharpe: mean(returns) / std(returns) over 126 days
    - stable growers get a high score; erratic stocks get a low score
    - 126 trading days ≈ 6 months (21*6)
    '''
    rolling_mean = returns.rolling(126).mean()
    rolling_std  = returns.rolling(126).std()

    '''divide mean by std - higher = more consistent, smoother returns'''
    quality = rolling_mean / (rolling_std + 1e-8)
    '''add tiny constant (1e-8) to denominator to avoid dividing by zero'''

    '''return quality scores to whoever asks'''
    return quality

'''composite factor - equal-weight blend of momentum, value, quality'''
def composite_factor(price_data, returns):

    '''
    - blending uncorrelated factors reduces drawdowns and smooths returns
    - each factor is z-scored first so they live on the same scale
    - then averaged: composite = (z_mom + z_val + z_qual) / 3
    '''

    mom  = momentum_factor(price_data)
    val  = value_factor(price_data)
    qual = quality_factor(returns)

    '''z-score each factor cross-sectionally (across stocks, not time)'''
    def cross_sectional_zscore(df):
        '''subtract row mean, divide by row std - each date normalised to zero mean, unit variance'''
        return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-8, axis=0)

    z_mom  = cross_sectional_zscore(mom)
    z_val  = cross_sectional_zscore(val)
    z_qual = cross_sectional_zscore(qual)

    '''average the three z-scored signals'''
    composite = (z_mom + z_val + z_qual) / 3

    '''return blended composite signal to whoever asks'''
    return composite