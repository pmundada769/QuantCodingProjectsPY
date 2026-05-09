# Time-Series Momentum (TSMOM) + Volatility Targeting

Replicates the core strategy from Moskowitz, Ooi & Pedersen (2012), *"Time Series Momentum"*, Journal of Financial Economics. Each asset's position is sized by the sign of its own 12-month return, scaled so the portfolio targets a fixed annualised volatility. Volatility is estimated using GARCH(1,1) via the `arch` library.

---

## How to run

```bash
cd Desktop/CODING/"Quant Projects"/TSMOM
source venv/bin/activate
pip install -r requirements.txt

streamlit run app.py

python cli.py --tickers SPY QQQ TLT GLD EEM VNQ --target-vol 15
python cli.py --tickers SPY TLT GLD --start 2008-01-01 --target-vol 10
```

---

## Files

| File | What it does |
|---|---|
| `tsmom.py` | Signal construction, GARCH vol estimation, vol-scaled positions, backtest |
| `charts.py` | Cumulative return, signal drill-down, drawdown, position heatmap, vol sweep |
| `app.py` | Streamlit dashboard |
| `cli.py` | Terminal interface with current long/short signals |
| `requirements.txt` | Dependencies |

---

## The strategy in plain English

1. For each asset, look at its return over the past 12 months
2. If positive → go long. If negative → go short. This is the signal (+1 or -1)
3. Scale the position so that `position = signal × (target_vol / realised_vol)`
4. If the asset has been calm (low vol), take a bigger position to hit target vol
5. If the asset has been volatile (high vol), take a smaller position
6. Repeat monthly across all assets, equal-weight the results

---

## Why GARCH over rolling std

Rolling standard deviation treats every day in the window equally — a spike 59 days ago counts the same as yesterday. GARCH(1,1) gives exponentially more weight to recent observations and also models variance persistence: `σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}`. The `α + β` sum (typically 0.97–0.99 for equities) measures how long volatility shocks persist. This is why GARCH forecasts are more responsive to sudden vol spikes — exactly what you want for position sizing during market stress.

---

## Dashboard tabs

- **Cumulative Return** — TSMOM vs equal-weight buy & hold of the same assets
- **Signal Drill-Down** — price, signal, vol, and position for any individual asset
- **Drawdown** — underwater curve
- **Position Heatmap** — monthly long/short/size per asset over time. Clusters show regime periods
- **Vol Target Sweep** — reruns at 5 different vol targets, shows Sharpe/return/drawdown trade-off
- **Monthly Returns** — calendar heatmap, green = good months, red = bad

---

## Key parameters

| Parameter | Default | What changing it does |
|---|---|---|
| Target Vol | 15% | Higher = more leverage = more return and more risk |
| Lookback | 252 days | Shorter = more reactive, more turnover. Standard = 252 (1 year) |
| Max Leverage | 2.0× | Caps position size in low-vol periods |
| GARCH vs rolling | GARCH | GARCH is more accurate but slower. Rolling is fast and transparent |

---

## What I'd say in an interview

I replicated the TSMOM strategy from Moskowitz, Ooi & Pedersen (2012). The signal is the sign of the trailing 12-month return on each asset — positive means long, negative means short. I scaled each position by target vol divided by realised vol, which keeps portfolio volatility approximately constant regardless of market regime. I used GARCH(1,1) for vol estimation rather than rolling std, because GARCH models variance persistence and is more responsive to regime changes. The position heatmap clearly shows the 2008 and 2020 crisis periods where the strategy correctly went short equities and long bonds.

---

## Dependencies

```bash
pip install numpy pandas scipy yfinance plotly streamlit arch
```

---

*Pranshu Mundada*