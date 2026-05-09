# alphalens Factor Analysis

Industry-standard factor evaluation framework — replicates the key Quantopian alphalens outputs using pandas and scipy, with an upgrade path to `alphalens-reloaded`. Computes IC, IC IR, quantile returns, factor decay, and turnover for any alpha signal.

---

## How to run

```bash
cd Desktop/CODING/"Quant Projects"/"Alphalens Factor"
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

alphalens is optional — the manual implementation produces equivalent results.

---

## Key metrics

**IC (Information Coefficient)**
Spearman rank correlation between today's factor values and forward stock returns. The only metric that matters for factor evaluation.

| IC | Interpretation |
|---|---|
| > 0.10 | Strong — institutional quality |
| 0.05–0.10 | Meaningful — worth trading |
| < 0.05 | Weak — may be noise |

**IC IR (Information Ratio)**
Mean IC divided by standard deviation of IC. Measures consistency of the signal. IC IR > 0.5 means the factor works reliably, not just on average.

**Quantile Returns**
Stocks ranked into quintiles by factor score. The top quintile should outperform the bottom. The spread (Q5 − Q1) is the factor return.

**IC Decay**
How IC changes at 1, 2, 3, 5, 10, 21 day horizons. A good factor decays gradually. Rapid decay means the signal is only useful for very short holding periods.

**Turnover**
Fraction of stocks that change quantile each month. High turnover = high transaction costs. Momentum has ~50-70% monthly turnover.

---

## Dashboard tabs

- **IC Time Series** — daily IC bars + 21-day rolling mean. Shows when the factor worked and when it did not.
- **Quantile Returns** — bar chart of mean return per quintile. Top should beat bottom.
- **Spread Return** — cumulative P&L of long Q5 / short Q1 portfolio.
- **IC Decay** — IC by forward horizon. Shows how long the signal lasts.
- **Turnover** — monthly turnover chart with average line.

---

## What to say in an interview

I evaluated my momentum factor using the alphalens framework — the industry standard at Two Sigma, D.E. Shaw, and Millennium for factor research. I computed the IC (Spearman rank correlation of factor vs forward returns) at 1, 5, and 21-day horizons, the IC IR to measure consistency, and quantile return spreads to verify that top-ranked stocks actually outperform. The IC decay curve shows how quickly the signal fades — momentum IC typically peaks at 5-10 days and decays to zero around 30 days, confirming it is a medium-frequency signal that requires monthly rebalancing.

---

## Dependencies

```bash
pip install numpy pandas scipy yfinance plotly streamlit
pip install alphalens-reloaded   # optional
```

---

*Pranshu Mundada*