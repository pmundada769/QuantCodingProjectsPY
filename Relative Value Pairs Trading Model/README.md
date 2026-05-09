# Statistical Arbitrage — Pairs Trading

Tests pairs of stocks for cointegration using Engle-Granger and Johansen methods, constructs a mean-reverting spread, and generates entry/exit signals based on the z-score. Includes a full universe scanner to find the best cointegrated pairs.

---

## How to run

```bash
cd Desktop/CODING/"Quant Projects"/"Pairs Trading"
source venv/bin/activate
pip install -r requirements.txt

streamlit run app.py

python cli.py --pair XOM CVX
python cli.py --pair GS MS --start 2015-01-01 --entry 2.0 --exit 0.5
```

---

## Files

| File | What it does |
|---|---|
| `pairs.py` | Cointegration tests, hedge ratio, spread, z-score, signals, backtest |
| `charts.py` | Spread/signal chart, P&L, price comparison, universe scan scatter |
| `app.py` | Streamlit dashboard |
| `cli.py` | CLI showing current z-score and signal |
| `requirements.txt` | Dependencies |

---

## How it works

**Step 1 — Cointegration test (Engle-Granger)**
Two stocks are cointegrated if their prices share a long-run relationship — they can drift apart short-term but always revert to a fixed ratio. The Engle-Granger test regresses one price on the other and tests whether the residuals are stationary. p-value < 0.05 means cointegrated.

**Step 2 — Hedge ratio**
OLS regression of price A on price B gives the hedge ratio: how many shares of B to hold per share of A to make the spread stationary.

**Step 3 — Spread and z-score**
`spread = price_A − hedge_ratio × price_B`
`z-score = (spread − rolling_mean) / rolling_std`

**Step 4 — Trading rule**
- z < −2: spread is abnormally low → go long spread (buy A, short B)
- z > +2: spread is abnormally high → go short spread (short A, buy B)
- |z| < 0.5: mean reversion complete → exit

**Half-life** measures mean reversion speed via the Ornstein-Uhlenbeck model. Shorter = faster reversion = better pair. Anything under 30 days is excellent.

---

## Good pairs to test

| Sector | Pairs |
|---|---|
| Energy | XOM/CVX, XOM/COP |
| Banks | GS/MS, JPM/BAC |
| Beverages | KO/PEP |
| Fast food | MCD/YUM |
| Tech | MSFT/GOOGL |
| Retail | WMT/TGT |

---

## Dashboard tabs

- **Spread & Signal** — three-panel: spread, z-score with thresholds, position signal
- **P&L** — cumulative strategy return vs buy & hold of asset A
- **Price Comparison** — normalised prices of both assets showing co-movement
- **Universe Scan** — tests all pairs in the ticker list, plots p-value vs Sharpe
- **Stats** — full numeric table including half-life and hedge ratio

---

## What I'd say in an interview

I implemented a statistical arbitrage pairs trading model using Engle-Granger cointegration testing from statsmodels. The hedge ratio is estimated via OLS, the spread z-score is computed on a rolling 60-day window, and positions are entered when the z-score exceeds ±2σ and exited at ±0.5σ. I also implemented the Johansen test for multi-asset cointegration, which is more powerful for groups of stocks. The half-life of mean reversion is estimated via the Ornstein-Uhlenbeck equation — pairs with half-lives under 30 days are the most tradeable. The universe scanner evaluates all N×(N-1)/2 pairs and ranks them by both statistical significance and Sharpe ratio.

---

## Dependencies

```bash
pip install numpy pandas scipy statsmodels yfinance plotly streamlit
```

---

*Pranshu Mundada*