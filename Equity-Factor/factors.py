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
