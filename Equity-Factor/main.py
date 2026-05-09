
'''
- "get_price_data" gets stock data, 
- "momentum_factor" and "volatility_factor" compute factors, 
- "form_long_short_portfolio" backtests the strategy, 
- "sharpe_ratio" and "max_drawdown" compute performance metrics
'''
from data import get_price_data
from factors import momentum_factor, volatility_factor
from backtest import calculate_turnover
from backtest import form_long_short_portfolio, market_regime_filter, sharpe_ratio, max_drawdown
import yfinance as yf
import requests
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt

'''[1]: define universe of stock tickers to analyse'''
CUSTOM_TICKERS = ["AAPL", "MSFT", "NVDA", "META"]

def get_sp500():
    import os
    file_path = "sp500_cache.csv"
    if os.path.exists(file_path):
        table = pd.read_csv(file_path)
    else:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        table = pd.read_html(StringIO(response.text))[0]
        table.to_csv(file_path, index=False)
    return table["Symbol"].tolist()

def filter_by_market_cap(tickers, min_cap, max_cap):
    selected = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            cap = info.get("marketCap", None)
            if cap and min_cap <= cap <= max_cap:
                selected.append(t)
        except:
            pass
    return selected

USE_SP500 = False  # Set to True to use S&P 500 universe, False for custom list
if USE_SP500:
    tickers = get_sp500()
    tickers = filter_by_market_cap(tickers, 1e7, 1e13) 
else:
    tickers = ["BEEM", "QBTS", "PLTR", "VKTX", "FOUR", "TOST", "MSTR", "PSIX", "RIOT", "AZTR"]  # Custom list of tickers

'''[2]: get data'''
prices, returns = get_price_data(tickers, "2015-01-01")

'''[3]: compute factor'''
momentum = momentum_factor(prices)
volatility = volatility_factor(returns)

'''[4]: backtest'''
strategy_returns, dispersion, weights = form_long_short_portfolio(momentum, returns)
vol_strategy_returns, vol_dispersion, vol_weights = form_long_short_portfolio(volatility, returns)

spy_prices, spy_returns = get_price_data(["SPY"], "2015-01-01")
if isinstance(spy_returns, pd.Series):
    spy_returns = spy_returns.to_frame("SPY")

regime = market_regime_filter(returns.mean(axis=1), spy_returns.squeeze())

# --- Monthly regime for analysis ---
monthly_regime = regime.resample("ME").last()
monthly_regime = monthly_regime.reindex(strategy_returns.index, method='ffill').fillna(True).astype(bool)

# --- Safe Monthly Strategy Returns ---
strategy_returns_monthly = strategy_returns.resample("ME").last().dropna()

print("Strategy returns (first 12 rows):")
print(strategy_returns.head(12))

print("Monthly strategy returns after resample:")
print(strategy_returns_monthly.head(12))

# --- Cumulative returns for strategy (plot first, do not block) ---
strategy_cumulative = (1 + strategy_returns_monthly).cumprod()
strategy_cumulative.index = strategy_cumulative.index.to_period('M').to_timestamp()
strategy_cumulative.plot(title="Momentum Long-Short Strategy", grid=True)
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.show(block=False)  # <-- non-blocking

# --- Regime breakdown for bull/bear ---
bull_returns = strategy_returns_monthly[monthly_regime]
bear_returns = strategy_returns_monthly[~monthly_regime]
print("Bull Sharpe:", sharpe_ratio(bull_returns))
print("Bear Sharpe:", sharpe_ratio(bear_returns))

'''[5]: performance metrics'''
print("Sharpe:", sharpe_ratio(strategy_returns))
print("Max Drawdown:", max_drawdown(strategy_returns))
print("Volatility Factor Sharpe:", sharpe_ratio(vol_strategy_returns))
print("Volatility Factor Max DD:", max_drawdown(vol_strategy_returns))
print("Average Cross-Sectional Dispersion:", dispersion.mean())
turnover = calculate_turnover(weights)
print("Average Monthly Turnover:", turnover.mean())

# --- Dynamic Top/Bottom Ranking Table with Next-Month Returns ---
TOP_N = 5
monthly_factor = momentum.resample("ME").last().dropna(how="all")
monthly_returns = returns.resample("ME").sum()
next_month_returns = monthly_returns.shift(-1)

ranking_table = []
for date in monthly_factor.index[:-1]:
    factor_values = monthly_factor.loc[date].dropna()
    if len(factor_values) == 0:
        continue
    next_returns = next_month_returns.loc[date].reindex(factor_values.index).fillna(0)
    top_stocks = factor_values.nlargest(TOP_N).index.tolist()
    bottom_stocks = factor_values.nsmallest(TOP_N).index.tolist()
    ranking_table.append({
        "Date": date.strftime("%Y-%m-%d"),
        "Top Stocks": top_stocks,
        "Top Returns": next_returns[top_stocks].round(4).tolist(),
        "Bottom Stocks": bottom_stocks,
        "Bottom Returns": next_returns[bottom_stocks].round(4).tolist()
    })

ranking_df = pd.DataFrame(ranking_table).set_index("Date")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
print("\nTop/Bottom Ranking Table (last 12 months):")
print(ranking_df.tail(12))

# --- Individual stock cumulative returns ---
cumulative_stocks = (1 + returns.fillna(0)).cumprod()
cumulative_stocks.plot(title="Individual Stock Cumulative Returns", grid=True)
plt.show(block=False)

# --- Compare multiple stocks ---
selected = [t for t in ["TOST", "FOUR", "BEEM"] if t in cumulative_stocks.columns]
if selected:
    cumulative_stocks[selected].plot(title=f"Cumulative Returns: {', '.join(selected)}", grid=True)
    plt.show(block=False)

# --- Focus on one stock ---
focus_stock = "VKTX"
if focus_stock in cumulative_stocks.columns:
    plt.figure()
    cumulative_stocks[focus_stock].plot(title=f"Cumulative Returns: {focus_stock}", grid=True)
    plt.show(block=False)

# --- Final blocking call to show all remaining figures ---
plt.show()

# --- After plotting strategy cumulative returns ---
strategy_cumulative.plot(title="Momentum Long-Short Strategy", grid=True)
plt.show()

# --- Section 2: Top/Bottom Ranking Table based on Momentum ---
TOP_N = 5
monthly_factor = momentum.resample("ME").last().dropna(how="all")
monthly_returns = returns.resample("ME").sum()
next_month_returns = monthly_returns.shift(-1)

ranking_table = []

for date in monthly_factor.index[:-1]:
    factor_values = monthly_factor.loc[date].dropna()
    if len(factor_values) == 0:
        continue
    next_returns = next_month_returns.loc[date].reindex(factor_values.index).fillna(0)
    top_stocks = factor_values.nlargest(TOP_N).index.tolist()
    bottom_stocks = factor_values.nsmallest(TOP_N).index.tolist()
    ranking_table.append({
        "Date": date.strftime("%Y-%m-%d"),
        "Top Stocks": top_stocks,
        "Top Expected Returns": next_returns[top_stocks].round(4).tolist(),
        "Bottom Stocks": bottom_stocks,
        "Bottom Expected Returns": next_returns[bottom_stocks].round(4).tolist()
    })

ranking_df = pd.DataFrame(ranking_table).set_index("Date")
print("\nTop/Bottom Ranking Table (last 12 months):")
print(ranking_df.tail(12))

# === Forecast Picks based on most recent momentum ===
TOP_N = 1
latest_momentum = monthly_factor.iloc[-1].dropna()
top_forecast = latest_momentum.nlargest(TOP_N).index.tolist()
bottom_forecast = latest_momentum.nsmallest(TOP_N).index.tolist()

print(f"\nForecast for next month based on latest momentum")
print("\nForecast Top Momentum Stocks:", top_forecast)
print("\nForecast Bottom Momentum Stocks:", bottom_forecast)
