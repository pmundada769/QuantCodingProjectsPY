#data.py

'''
"yfinance|yf" lets us download stock data
pdr_override() hijacks pandas data reader to use and fetch yf instead 
of default data source whilst maintaining DataFrame format

"pandas|pd" lets us work with tables of data
'''
import yfinance as yf
'''[*] yf.pdr_override()'''
  
import pandas as pd   

'''downloads adjusted close prices and returns daily returns; "stock name", "start date"'''
def get_price_data(tickers, start_date): 
   
    '''
    - download historical price data for multiple stocks using yf.download(),
    - "auto_adjust=True" avoids broken adjusted-close logic,
    - "threads=False" prevents SQLite cache collisions when downloading multiple stocks
    '''
    data = yf.download(tickers, start=start_date, auto_adjust=True, threads=False)["Close"] #["Adj Close"]
    '''[**]'''

    '''calculate daily percentage returns/price changes'''
    returns = data.pct_change()

    '''return both price data and returns to whoever asks'''
    return data, returns

'''
[*] yf.pdr_override() got removed by yfinance
[**] Yahoo does not return "Adj Close" anymore. It returns "Close" only.
'''