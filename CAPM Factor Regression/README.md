# CAPM / Fama-French 3-Factor Regression

Regresses stock returns against market, SMB, and HML factors to decompose return sources.
Outputs alpha, beta, R², t-statistics, rolling estimates, and a return attribution breakdown.

---

## How to run

```bash
cd Desktop/CODING/"Quant Projects"/"CAPM Factor"
source venv/bin/activate
pip install -r requirements.txt

streamlit run app.py

python cli.py --ticker AAPL
python cli.py --tickers AAPL MSFT NVDA META JPM --start 2015-01-01

python tests.py
```

---

## Files

| File | What it does |
|---|---|
| `regression.py` | Downloads FF3 factors, runs CAPM and FF3 OLS, rolling beta, decomposition |
| `charts.py` | Scatter plot, rolling beta/alpha, factor betas bar, waterfall, multi-ticker bubble |
| `app.py` | Streamlit dashboard |
| `cli.py` | Terminal interface |
| `tests.py` | Unit tests using synthetic data |

---

## What it computes

**CAPM:** `excess_return = α + β × (Mkt-RF) + ε`

**FF3:** `excess_return = α + β₁ × (Mkt-RF) + β₂ × SMB + β₃ × HML + ε`

| Output | Meaning |
|---|---|
| Alpha (α) | Return not explained by any factor — true skill or mispricing |
| Alpha t-stat | > 2 means alpha is statistically significant at 95% confidence |
| Beta Market | Sensitivity to market moves. 1.2 = moves 1.2% per 1% market move |
| Beta SMB | Positive = behaves like a small-cap stock |
| Beta HML | Positive = behaves like a value stock |
| R² | Fraction of return variance explained by the factors |

---

## Dashboard tabs

- **CAPM Scatter** — monthly returns plotted against market. Slope = beta, intercept = alpha
- **Rolling Beta** — 24-month sliding window beta and alpha. Flat = stable, rising = becoming more market-sensitive
- **Factor Betas** — CAPM vs FF3 side by side. If beta_market changes a lot, SMB/HML were correlated with market
- **Decomposition** — waterfall chart: how much of average return came from alpha vs each factor
- **Multi-Ticker** — bubble chart of all tickers: x=beta, y=annualised alpha, size=R²

---

## Data sources

- **Stock prices:** Yahoo Finance via yfinance (live, fetched on every run)
- **FF3 factors:** Ken French's Data Library at Dartmouth (monthly, downloaded live)

---

## What to say in an interview

I implemented CAPM and Fama-French 3-factor regressions from scratch using both scipy's linregress and raw matrix OLS `(X'X)⁻¹X'y`. I computed rolling 24-month estimates to detect alpha decay and beta drift. I built a return decomposition breaking average monthly return into its factor contributions. The multi-ticker view plots alpha vs beta with R² as bubble size — the standard way quants screen for genuine alpha versus leveraged market exposure.

---

*Pranshu Mundada*