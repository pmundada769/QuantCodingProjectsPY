#data.py

'''
"yfinance|yf" lets us download stock data
"pandas|pd" lets us work with tables of data
'''
import yfinance as yf
'''[*] yf.pdr_override()'''

import pandas as pd

'''downloads adjusted close prices and returns daily returns; "stock name", "start date"'''
def get_price_data(tickers, start_date):

    '''
    - download historical price data for multiple stocks using yf.download()
    - "auto_adjust=True" avoids broken adjusted-close logic
    - "threads=False" prevents SQLite cache collisions when downloading multiple stocks
    '''
    data = yf.download(tickers, start=start_date, auto_adjust=True, threads=False)["Close"]
    '''[**]'''

    '''calculate daily percentage returns/price changes'''
    returns = data.pct_change()

    '''return both price data and returns to whoever asks'''
    return data, returns

'''downloads benchmark index data for regime and performance comparison'''
def get_benchmark_data(ticker, start_date):

    '''
    - same logic as get_price_data but for a single benchmark ticker
    - typically called with "SPY" (S&P 500 ETF) or "^GSPC" (S&P 500 index)
    - returns a Series not a DataFrame since there is only one ticker
    '''
    data = yf.download(ticker, start=start_date, auto_adjust=True, threads=False)["Close"]

    '''squeeze DataFrame down to a Series if only one column returned'''
    if isinstance(data, pd.DataFrame):
        data = data.squeeze()

    '''calculate daily percentage returns'''
    returns = data.pct_change()

    '''return price and returns to whoever asks'''
    return data, returns

'''filters tickers to remove those with too many missing trading days'''
def clean_price_data(price_data, max_missing_pct=0.05):

    '''
    - stocks with >5% missing data are unreliable for factor ranking
    - max_missing_pct=0.05 means drop any column with more than 5% NaN rows
    - keeps the universe clean without over-filtering
    '''
    missing_pct = price_data.isna().mean()

    '''keep only tickers below the missing threshold'''
    clean_tickers = missing_pct[missing_pct <= max_missing_pct].index.tolist()

    '''return filtered price data to whoever asks'''
    return price_data[clean_tickers]

'''
[*] yf.pdr_override() got removed by yfinance
[**] Yahoo does not return "Adj Close" anymore. It returns "Close" only.
'''