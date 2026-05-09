# Professional Momentum Backtest (vectorbt)

High-performance momentum backtest using vectorbt — a vectorised backtesting library built on NumPy. Includes transaction costs, slippage, multiple position sizing methods, rolling Sharpe, underwater curve, monthly return calendar, and a parameter sweep across lookbacks and cost levels.

---

## How to run

```bash
cd Desktop/CODING/"Quant Projects"/"VBT Backtest"
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

vectorbt is optional — the backtest falls back to a manual numpy/pandas implementation if it is not installed. All results are identical.

---

## What vectorbt adds

Traditional backtests loop over dates one by one. vectorbt does everything in vectorised array operations simultaneously, making it ~100x faster. This enables:

- Running 100 parameter combinations in the time a loop-based backtest runs 1
- Real-time parameter sweeps in the dashboard
- Professional-grade portfolio analytics (sharpe, sortino, calmar, drawdown) in one line

---

## Dashboard tabs

- **Equity Curve** — cumulative net-of-costs return + rolling 252-day Sharpe
- **Drawdown** — underwater curve showing every drawdown period
- **Distribution** — histogram of daily returns
- **Monthly Calendar** — green/red return heatmap by year and month
- **Parameter Sweep** — 16-combination grid: 4 lookbacks × 4 cost levels, Sharpe heatmap

---

## What to say in an interview

I implemented a professional momentum backtest using vectorbt's vectorised portfolio engine. The strategy uses 12-minus-1 cross-sectional momentum with monthly rebalancing, equal-weight or inverse-vol position sizing, and explicit transaction cost + slippage modelling. The parameter sweep runs 16 combinations across lookback windows and cost scenarios, with the Sharpe heatmap showing which combinations are robust vs which depend on low costs. The rolling 252-day Sharpe shows where the strategy's edge was strong and where it decayed.

---

## Dependencies

```bash
pip install numpy pandas yfinance plotly streamlit
pip install vectorbt   # optional — manual fallback available
```

---

*Pranshu Mundada*