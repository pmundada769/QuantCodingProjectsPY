#main.py

'''
- "get_price_data" gets stock data
- "momentum_factor", "volatility_factor", "value_factor", "quality_factor", "composite_factor" compute factors
- "form_long_short_portfolio" backtests each strategy
- "sharpe_ratio", "sortino_ratio", "calmar_ratio", "max_drawdown" compute performance metrics
- "hit_rate", "average_win_loss", "rolling_sharpe" give deeper strategy diagnostics
- "transaction_cost_drag" shows net-of-cost performance
'''
from data import get_price_data, get_benchmark_data, clean_price_data
from factors import momentum_factor, volatility_factor, value_factor, quality_factor, composite_factor
from backtest import (
    form_long_short_portfolio,
    market_regime_filter,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    max_drawdown,
    rolling_sharpe,
    hit_rate,
    average_win_loss,
    calculate_turnover,
    concentration,
    transaction_cost_drag,
)
import pandas as pd
import requests
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
        url     = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        table = pd.read_html(StringIO(response.text))[0]
        table.to_csv(file_path, index=False)
    return table["Symbol"].tolist()

USE_SP500 = False  # Set to True to use S&P 500 universe, False for custom list
if USE_SP500:
    tickers = get_sp500()
else:
    tickers = ["BEEM", "QBTS", "PLTR", "VKTX", "FOUR", "TOST", "MSTR", "PSIX", "RIOT", "AZTR"]

'''[2]: get and clean data'''
prices, returns = get_price_data(tickers, "2015-01-01")

'''remove stocks with too many gaps before computing any factors'''
prices  = clean_price_data(prices)
returns = prices.pct_change()

'''[3]: compute all factors'''
momentum  = momentum_factor(prices)
volatility = volatility_factor(returns)
value      = value_factor(prices)
quality    = quality_factor(returns)
composite  = composite_factor(prices, returns)

'''[4]: backtest all factors'''
strategy_returns,   dispersion,     weights     = form_long_short_portfolio(momentum,   returns)
vol_returns,        vol_dispersion, vol_weights = form_long_short_portfolio(volatility,  returns)
value_returns,      _,              val_weights = form_long_short_portfolio(value,       returns)
quality_returns,    _,              qual_weights = form_long_short_portfolio(quality,    returns)
composite_returns,  _,              comp_weights = form_long_short_portfolio(composite, returns)

'''[5]: get benchmark (SPY) for regime filter'''
spy_prices, spy_returns = get_benchmark_data("SPY", "2015-01-01")

regime         = market_regime_filter(returns.mean(axis=1), spy_returns)
monthly_regime = regime.resample("ME").last()
monthly_regime = monthly_regime.reindex(strategy_returns.index, method="ffill").fillna(True).astype(bool)

'''[6]: monthly strategy returns'''
strategy_returns_monthly = strategy_returns.resample("ME").last().dropna()

print("Strategy returns (first 12 rows):")
print(strategy_returns.head(12))

print("\nMonthly strategy returns after resample:")
print(strategy_returns_monthly.tail(12))

'''[7]: transaction-cost-adjusted returns'''
turnover      = calculate_turnover(weights)
net_returns   = transaction_cost_drag(strategy_returns_monthly, turnover)

'''[8]: performance metrics - momentum strategy'''
bull_returns  = strategy_returns_monthly[monthly_regime]
bear_returns  = strategy_returns_monthly[~monthly_regime]
wins, losses  = average_win_loss(strategy_returns_monthly)

print("\n--- MOMENTUM STRATEGY METRICS ---")
print("Bull Sharpe:               ", sharpe_ratio(bull_returns))
print("Bear Sharpe:               ", sharpe_ratio(bear_returns))
print("Sharpe (gross):            ", sharpe_ratio(strategy_returns_monthly))
print("Sharpe (net of costs):     ", sharpe_ratio(net_returns))
print("Sortino Ratio:             ", sortino_ratio(strategy_returns_monthly))
print("Calmar Ratio:              ", calmar_ratio(strategy_returns_monthly))
print("Max Drawdown:              ", max_drawdown(strategy_returns_monthly))
print("Hit Rate:                  ", hit_rate(strategy_returns_monthly))
print("Avg Win Month:             ", wins)
print("Avg Loss Month:            ", losses)
print("Avg Cross-Sectional Disp:  ", dispersion.mean())
print("Avg Monthly Turnover:      ", turnover.mean())

'''[9]: compare all factors side by side'''
print("\n--- FACTOR COMPARISON ---")
factor_map = {
    "Momentum":  strategy_returns_monthly,
    "Volatility": vol_returns.resample("ME").last().dropna(),
    "Value":      value_returns.resample("ME").last().dropna(),
    "Quality":    quality_returns.resample("ME").last().dropna(),
    "Composite":  composite_returns.resample("ME").last().dropna(),
}

for name, ret in factor_map.items():
    print(f"{name:<12}  Sharpe: {sharpe_ratio(ret):.3f}   MaxDD: {max_drawdown(ret):.3f}   HitRate: {hit_rate(ret):.2f}")

'''[10]: rolling sharpe - shows whether strategy is improving or degrading over time'''
roll_sharpe = rolling_sharpe(strategy_returns_monthly, window=12)

'''[11]: cumulative returns plot - momentum strategy'''
strategy_cumulative = (1 + strategy_returns_monthly).cumprod()
strategy_cumulative.index = strategy_cumulative.index.to_period("M").to_timestamp()
strategy_cumulative.plot(title="Momentum Long-Short Strategy", grid=True)
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.show(block=False)

'''rolling 12-month Sharpe - visual check for alpha decay'''
roll_sharpe.index = roll_sharpe.index.to_period("M").to_timestamp()
roll_sharpe.plot(title="Rolling 12-Month Sharpe (Momentum)", grid=True)
plt.axhline(0, color="red", linestyle="--", linewidth=1)
plt.ylabel("Sharpe Ratio")
plt.show(block=False)

'''[12]: compare all factor cumulative returns on one chart'''
fig, ax = plt.subplots(figsize=(12, 6))
for name, ret in factor_map.items():
    ret_monthly = ret.resample("ME").last().dropna() if not hasattr(ret, "resample") else ret
    cum = (1 + ret_monthly).cumprod()
    cum.index = cum.index.to_period("M").to_timestamp()
    ax.plot(cum, label=name)
ax.set_title("All Factor Cumulative Returns")
ax.set_ylabel("Cumulative Returns")
ax.legend()
ax.grid(True)
plt.show(block=False)

'''[13]: individual stock cumulative returns'''
cumulative_stocks = (1 + returns.fillna(0)).cumprod()
cumulative_stocks.plot(title="Individual Stock Cumulative Returns", grid=True)
plt.show(block=False)

'''compare a few selected stocks'''
selected = [t for t in ["TOST", "FOUR", "BEEM"] if t in cumulative_stocks.columns]
if selected:
    cumulative_stocks[selected].plot(title=f"Cumulative Returns: {', '.join(selected)}", grid=True)
    plt.show(block=False)

'''focus on one stock'''
focus_stock = "VKTX"
if focus_stock in cumulative_stocks.columns:
    plt.figure()
    cumulative_stocks[focus_stock].plot(title=f"Cumulative Returns: {focus_stock}", grid=True)
    plt.show(block=False)

'''[14]: Top/Bottom ranking table with next-month returns'''
TOP_N          = 5
monthly_factor = momentum.resample("ME").last().dropna(how="all")
monthly_returns = returns.resample("ME").sum()
next_month_returns = monthly_returns.shift(-1)

ranking_table = []
for date in monthly_factor.index[:-1]:
    factor_values = monthly_factor.loc[date].dropna()
    if len(factor_values) == 0:
        continue
    next_returns  = next_month_returns.loc[date].reindex(factor_values.index).fillna(0)
    top_stocks    = factor_values.nlargest(TOP_N).index.tolist()
    bottom_stocks = factor_values.nsmallest(TOP_N).index.tolist()
    ranking_table.append({
        "Date":             date.strftime("%Y-%m-%d"),
        "Top Stocks":       top_stocks,
        "Top Returns":      next_returns[top_stocks].round(4).tolist(),
        "Bottom Stocks":    bottom_stocks,
        "Bottom Returns":   next_returns[bottom_stocks].round(4).tolist(),
    })

ranking_df = pd.DataFrame(ranking_table).set_index("Date")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
print("\nTop/Bottom Ranking Table (last 12 months):")
print(ranking_df.tail(12))

'''[15]: forecast picks based on most recent momentum'''
TOP_N          = 1
latest_momentum = monthly_factor.iloc[-1].dropna()
top_forecast    = latest_momentum.nlargest(TOP_N).index.tolist()
bottom_forecast = latest_momentum.nsmallest(TOP_N).index.tolist()

print(f"\nForecast for next month based on latest momentum")
print("\nForecast Top Momentum Stocks:   ", top_forecast)
print("\nForecast Bottom Momentum Stocks:", bottom_forecast)

'''[16]: final blocking show - keeps all figures open'''
plt.show()