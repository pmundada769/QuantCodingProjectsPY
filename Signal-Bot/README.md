# Unified Trading Signal Bot — Capstone

A complete systematic trading system. Four signal models feed into a weighted ensemble, a vol-targeting risk layer scales positions, a drawdown stop zeroes out during crashes, and an execution layer generates paper trade orders via the Alpaca API.

---

## How to run

```bash
cd Desktop/CODING/"Quant Projects"/"Signal Bot"
source venv/bin/activate
pip install -r requirements.txt

streamlit run app.py

python cli.py --tickers SPY QQQ TLT GLD EEM --target-vol 15
python cli.py --tickers SPY QQQ TLT GLD EEM VNQ DBC HYG --start 2010-01-01
```

For Alpaca paper trading: free account at `alpaca.markets` → Settings → API Keys → Paper Trading.

---

## Architecture

```
╔══════════════════════════════════════════════════════════════╗
║                      SIGNAL LAYER                            ║
║  TSMOM (30%)    │  Cross-Sect Momentum (25%)                 ║
║  Vol Regime (20%) │  SMA Trend (15%)  │  Sentiment (10%)    ║
╚═══════════════════════════╤══════════════════════════════════╝
                            │ weighted average → composite [-1,+1]
                            ▼
╔══════════════════════════════════════════════════════════════╗
║                      RISK LAYER                              ║
║  Vol targeting:  position = composite × (target_vol / σ)    ║
║  Drawdown stop:  position = 0 if portfolio DD > threshold   ║
║  Leverage cap:   clip to ±2×                                 ║
╚═══════════════════════════╤══════════════════════════════════╝
                            │
                            ▼
╔══════════════════════════════════════════════════════════════╗
║                   EXECUTION LAYER                            ║
║  Alpaca paper trading API — market orders, day fill          ║
║  Dry run mode: shows orders without submitting               ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Signal descriptions

| Signal | Default Weight | Logic |
|---|---|---|
| TSMOM | 30% | Sign of 12-month trailing return. +1 = long, -1 = short |
| Cross-Sect Momentum | 25% | Rank within universe, rescaled to [-1, +1] |
| Vol Regime | 20% | Reduce when market vol is elevated vs historical average |
| SMA Trend | 15% | 50-day MA vs 200-day MA crossover |
| Sentiment | 10% | RSS headline sentiment z-score (optional) |

All weights are adjustable via sliders. Auto-normalised to sum to 1.

---

## Risk layer details

**Volatility targeting**
`position = signal × (target_vol / realised_60d_vol)`
Keeps portfolio volatility approximately constant regardless of market regime.
If an asset is calm (low realised vol), you take a bigger position. If it's volatile, smaller.

**Drawdown stop**
When the portfolio falls more than the threshold (default 20%) from its peak, all positions are set to zero. They reactivate on recovery.
This prevents the catastrophic losses that kill momentum strategies during reversals.

**Signal agreement**
When all four signals point the same direction, agreement = 100%. This is a confidence indicator. High agreement + large position = highest conviction.

---

## Dashboard tabs

- **Equity Curve** — cumulative backtest return + rolling 252-day Sharpe
- **Signal History** — daily composite signal bar chart per asset
- **Position Heatmap** — monthly long/short positions across all assets
- **Drawdown** — underwater curve with DD stop level marked
- **Weight Sensitivity** — removes each signal one at a time, shows Δ Sharpe
- **Execute Orders** — proposed order table + Alpaca submission

---

## What to say in an interview

I built a unified systematic trading system that aggregates four signals — time-series momentum, cross-sectional momentum, a volatility regime filter, and a moving average trend indicator — into a weighted ensemble. Each signal outputs a score between -1 and +1. The ensemble is a weighted average, normalised to sum to 1. I then apply volatility targeting — scaling each position by target_vol / realised_vol — which keeps portfolio volatility approximately constant. A drawdown stop zeroes all positions when the portfolio falls more than 20% from its peak, preventing momentum crash events from destroying the strategy. The execution layer connects to Alpaca's free paper trading API, meaning the system runs end-to-end with no real capital at risk. The weight sensitivity analysis identifies which signals add genuine value — if removing a signal reduces Sharpe, it's contributing positively.

---

## Dependencies

```bash
pip install numpy pandas yfinance plotly streamlit requests
pip install alpaca-py   # optional — for live Alpaca paper trading
```

---

*Pranshu Mundada*