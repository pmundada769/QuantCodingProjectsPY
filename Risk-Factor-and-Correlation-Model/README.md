# PCA Risk Factor Model

Applies Principal Component Analysis to a universe of stocks to extract latent risk factors — the underlying sources of co-movement driving returns across the entire universe. Includes rolling PCA to detect when factor structure changes (regime shifts).

---

## How to run

```bash
cd Desktop/CODING/"Quant Projects"/"PCA Factor"
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## What PCA does

PCA finds linear combinations of stocks (principal components) that explain the maximum amount of variance. Applied to stock returns:

- **PC1 ≈ Market factor** — all stocks load positively. Explains 30-60% of total variance. This is systematic risk.
- **PC2 ≈ Growth vs Value** — tech and growth stocks load one way, defensives and value load the other.
- **PC3+ ≈ Sector/style factors** — energy vs financials, momentum vs reversal, etc.

The scree plot shows how many factors you need to explain 80% of variance. Rolling PCA shows when that structure changes.

---

## Dashboard tabs

- **Scree Plot** — variance explained per PC, cumulative curve. The "elbow" shows how many factors matter.
- **Factor Loadings** — heatmap of each stock's sensitivity to each PC.
- **Factor Returns** — long-short portfolio for each PC (long high-loading, short low-loading stocks).
- **Rolling PC1** — how dominant PC1 is over time. Spikes = crisis regime where everything moves together.
- **Variance Attribution** — per-stock breakdown: how much is market-driven vs idiosyncratic.
- **Interpretation** — plain-English explanation of what each PC represents.

---

## Why rolling PCA matters

In normal markets, PC1 explains ~35% of variance — stocks have their own stories. In the March 2020 COVID crash, PC1 jumped to ~75% — everything fell together, sector bets became useless. Rolling PCA detects this shift in real time. When PC1 variance exceeds 60%, it's a signal to reduce factor-specific bets and focus on macro risk.

---

## What to say in an interview

I applied PCA to a 15-stock universe using sklearn, standardising returns before fitting to prevent high-vol stocks from dominating. PC1 consistently explains the most variance and represents systematic market risk. I built long-short portfolios for each factor by going long the highest-loading stocks and shorting the lowest-loading ones. Rolling PCA on a 126-day window shows regime changes — when PC1 variance spikes above 60%, correlations have broken down and the factor structure has shifted. This is the statistical foundation for risk decomposition used at multi-strategy funds.

---

## Dependencies

```bash
pip install numpy pandas scipy scikit-learn yfinance plotly streamlit
```

---

*Pranshu Mundada*