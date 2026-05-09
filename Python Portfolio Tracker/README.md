# Python Portfolio Tracker

Reads a CSV of your holdings, fetches live prices via yfinance, and shows
P&L, sector exposure, cumulative returns, drawdown, and performance metrics vs SPY.

---

## How to run

```bash
cd Desktop/CODING/"Quant Projects"/"Python Portfolio Tracker"
source venv/bin/activate
pip install -r requirements.txt

# edit holdings.csv with your real positions first
streamlit run app.py
```

---

## Files

| File | What it does |
|---|---|
| `tracker.py` | Loads CSV, fetches live prices, computes P&L, sector breakdown, metrics |
| `charts.py` | Cumulative return, P&L bar, sector pie, drawdown, histogram, movers |
| `app.py` | Streamlit dashboard with file upload |
| `holdings.csv` | Sample portfolio — replace with your real holdings |
| `requirements.txt` | Dependencies |

---

## Holdings CSV format

```
ticker,shares,avg_cost,sector
AAPL,50,145.20,Technology
MSFT,30,280.50,Technology
JPM,40,155.30,Financials
```

- `ticker` — Yahoo Finance symbol
- `shares` — number of shares you own
- `avg_cost` — your average purchase price per share
- `sector` — used for the sector breakdown chart (can be anything)

Upload your own CSV in the sidebar, or replace `holdings.csv` directly.

---

## Dashboard tabs

- **Cumulative Return** — portfolio value growth vs SPY benchmark
- **P&L** — dollar P&L and % return per holding, best/worst performers
- **Sectors** — pie chart of allocation + sector return table
- **Drawdown** — peak-to-trough decline over the history period
- **Distribution** — histogram of daily returns
- **Holdings Table** — full table with live prices, values, P&L coloured green/red

---

## Performance metrics

| Metric | What it means |
|---|---|
| Total Return | Cumulative % gain over the selected period |
| Ann. Return | Annualised return (compound) |
| Ann. Volatility | Annualised standard deviation of daily returns |
| Sharpe Ratio | Return per unit of risk (annualised) |
| Sortino Ratio | Like Sharpe but only penalises downside moves |
| Max Drawdown | Worst peak-to-trough loss over the period |
| Hit Rate | % of days the portfolio was up |
| Beta vs SPY | Market sensitivity (if benchmark enabled) |
| Alpha vs SPY | Daily alpha annualised (if benchmark enabled) |

---

## Data

- **Prices:** Live from Yahoo Finance, refreshed every 5 minutes via Streamlit cache
- **Your cost basis:** From your CSV — never changes until you update the file

---

*Pranshu Mundada*