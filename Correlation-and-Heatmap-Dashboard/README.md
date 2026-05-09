# Correlation & Heatmap Dashboard

Rolling correlation matrix between assets with regime shift detection.
Flags when correlations spike (crisis) or collapse (divergence).
Animated heatmap shows how correlation structure evolves over time.

---

## How to run

```bash
cd Desktop/CODING/"Quant Projects"/"Correlation Dashboard"
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## Files

| File | What it does |
|---|---|
| `correlation.py` | Fetches prices, computes rolling correlations, detects regime shifts, dispersion |
| `charts.py` | Static heatmap, rolling lines, avg correlation, animated heatmap, dispersion |
| `app.py` | Streamlit dashboard |
| `requirements.txt` | Dependencies |

---

## Dashboard tabs

- **Current Heatmap** — full correlation matrix using all available history. Red = assets move together, blue = opposite
- **Rolling Correlations** — select any pairs and see how their correlation changed over time
- **Avg Correlation** — single line showing average pairwise correlation. Spikes = crisis. Red zones auto-shaded above 0.6
- **Animated Heatmap** — press Play to watch the correlation matrix evolve across 6 time snapshots
- **Dispersion** — rolling cross-sectional volatility. Low = everything moving together. High = stock-specific movement
- **Regime Shifts** — table of every date where a pair's correlation jumped or dropped by more than the threshold

---

## Key concepts

**Average pairwise correlation** is the most important number.
- Normal markets: 0.2–0.4
- Elevated: 0.4–0.6
- Crisis regime (2008, March 2020, etc.): 0.6–0.9
When it spikes, diversification breaks down — your 10-stock portfolio suddenly behaves like 1 stock.

**Regime shift** = a correlation pair that changed by more than your threshold (default 0.30) within 20 days.
These often coincide with macro events: Fed announcements, earnings seasons, geopolitical shocks.

**Dispersion** is the inverse of correlation.
High dispersion = stocks moving differently = more opportunity for stock-picking.
Low dispersion = macro regime = everything trades as one.

---

## Presets

| Preset | Assets | What it shows |
|---|---|---|
| Risk Assets | SPY, QQQ, EEM, HYG, GLD | How risky assets correlate in stress |
| Multi-Asset | SPY, TLT, GLD, VNQ, EFA, HYG | Classic diversified portfolio cross-correlations |
| Tech Stocks | AAPL, MSFT, NVDA, META, GOOGL, AMZN | Intra-sector correlations |
| Sectors | XLK, XLF, XLE, XLV, XLI, XLP | How sectors rotate relative to each other |

---

## Data

- **Prices:** Live from Yahoo Finance, cached for 10 minutes per session
- **Correlations:** Computed in real time from downloaded prices — not pre-stored

---

*Pranshu Mundada*