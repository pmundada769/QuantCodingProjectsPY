# Equity Factor Model

A quantitative equity research tool that ranks stocks by factor signals — momentum, volatility, value, quality — and backtests long-short portfolios to measure whether those signals actually predict future returns.

Built in Python. No finance background required beyond what is explained below.

---

## How to run it

```bash
# Step 1: Open VSCode terminal (Ctrl + `)

# Step 2: Navigate to your project and activate your virtual environment
cd Desktop/quant_projects
source venv/bin/activate          # Mac / Linux
# venv\Scripts\activate           # Windows

# Step 3: Navigate into the project folder
cd equity_factor

# Step 4: Run
python main.py

# If you get an import error, try:
python -m main
```

You will see performance metrics printed in the terminal, and several matplotlib charts will open.

---

## What it produces

**In the terminal:**
- Momentum strategy metrics (Sharpe, Sortino, Calmar, Max Drawdown, Hit Rate, Win/Loss)
- Factor comparison table across all five factors
- Top/bottom ranking table for the last 12 months
- Forecast: which stocks have the strongest/weakest momentum signal right now

**As charts:**
- Cumulative returns of the momentum long-short strategy
- Rolling 12-month Sharpe (shows if the edge is fading)
- All five factors overlaid on one chart
- Individual stock cumulative returns
- Focused single-stock chart

---

## Files

| File | What it does |
|---|---|
| `main.py` | Runs everything — the entry point |
| `data.py` | Downloads and cleans price data from Yahoo Finance |
| `factors.py` | Computes factor signals (momentum, value, quality, etc.) |
| `backtest.py` | Backtests portfolios and computes performance metrics |
| `sp500_cache.csv` | Cached list of S&P 500 tickers so Wikipedia is not scraped every run |

---

## Key parameters you can change

All in `main.py`:

| Parameter | Where | What changing it does |
|---|---|---|
| `tickers = [...]` | Line 53 | Change the stocks you analyse |
| `USE_SP500 = False` | Line 49 | Set to `True` to run on the full S&P 500 |
| `"2015-01-01"` | Line 56 | Change the start date of the backtest |
| `TOP_N = 5` | Line 178 | How many top/bottom stocks appear in the ranking table |
| `TOP_N = 1` | Line 206 | How many stocks to forecast for next month |
| `focus_stock = "VKTX"` | Line 171 | Which single stock to plot on its own chart |
| `selected = [...]` | Line 165 | Which stocks to compare on one chart |
| `window=12` | Line 129 | Rolling Sharpe window in months |

In `factors.py`:

| Parameter | Where | What changing it does |
|---|---|---|
| `252` | `momentum_factor` | Long-term lookback (252 trading days = 1 year) |
| `21` | `momentum_factor` | Short-term exclusion (21 trading days = 1 month) |
| `63` | `volatility_factor` | Volatility window (63 = 3 months) |
| `504` | `value_factor` | Value lookback (504 = 2 years) |
| `126` | `quality_factor` | Quality window (126 = 6 months) |

In `backtest.py`:

| Parameter | Where | What changing it does |
|---|---|---|
| `cost_per_trade=0.001` | `transaction_cost_drag` | Cost assumption (0.001 = 10 basis points per trade) |

---

## How to add a new stock

Find the ticker list in `main.py` and add any Yahoo Finance ticker:
```python
tickers = ["BEEM", "QBTS", "PLTR", "AAPL", "TSLA"]
```
The ticker must exist on Yahoo Finance and have at least 1 year of price history (3+ years recommended).

---

## How to switch to the S&P 500 universe

```python
USE_SP500 = True   # was False
```
The first run scrapes Wikipedia. After that it reads from `sp500_cache.csv`. Running 500 tickers takes several minutes.

---

## How to evaluate results

| Metric | Good | Concerning |
|---|---|---|
| Sharpe Ratio | > 1.0 | < 0.5 |
| Sortino Ratio | > 1.2 | < 0.6 |
| Calmar Ratio | > 0.5 | < 0.2 |
| Max Drawdown | > -30% | < -60% |
| Hit Rate | > 55% | < 45% |
| Monthly Turnover | < 0.5 | > 0.8 |
| Rolling Sharpe | Stable or rising | Declining toward zero |

The rolling Sharpe chart is the most important diagnostic. A trend toward zero means the signal's edge is fading.

---

## Dependencies

```bash
pip install yfinance pandas numpy matplotlib requests
```

---

*Pranshu Mundada*