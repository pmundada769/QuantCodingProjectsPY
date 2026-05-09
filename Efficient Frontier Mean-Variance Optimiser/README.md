# Efficient Frontier / Mean-Variance Optimizer

Implements Harry Markowitz's 1952 portfolio theory from scratch using scipy.
Finds the optimal portfolio weights that maximise Sharpe ratio (or minimise volatility),
plots the full efficient frontier, and shows the correlation structure of the asset universe.

---

## How to run it

```bash
cd Desktop/CODING/"Quant Projects"/"Efficient Frontier"
source venv/bin/activate
pip install -r requirements.txt

# Streamlit dashboard
streamlit run app.py

# CLI
python cli.py --tickers AAPL MSFT GOOGL NVDA JPM
python cli.py --tickers SPY TLT GLD VNQ EFA --start 2019-01-01 --rf 4.5

# Tests
python tests.py
```

---

## Files

| File | What it does |
|---|---|
| `optimizer.py` | Core maths — downloads data, builds frontier, finds Max Sharpe and Min Vol |
| `charts.py` | Six Plotly charts |
| `app.py` | Streamlit dashboard |
| `cli.py` | Terminal interface with weight bar charts |
| `tests.py` | 16 unit tests using synthetic data (no internet needed) |

---

## What the dashboard shows

**Frontier tab** — the classic Markowitz chart. X-axis = risk (vol), Y-axis = return.
Every dot is a possible portfolio. The blue line is the frontier — the set of portfolios with
maximum return for each level of risk. The star is Max Sharpe. The diamond is Min Vol.

**Weights tab** — exact allocation percentages for both optimal portfolios side by side.

**Allocation tab** — pie charts of both portfolios. Weights below 0.5% are hidden.

**Sharpe Curve** — how the Sharpe ratio changes along the frontier. Peak = Max Sharpe portfolio.

**Correlation** — heatmap of pairwise return correlations.
Low or negative correlation = diversification benefit.

**Distributions** — overlapping histograms of daily returns per asset.

---

## Key parameters

| Parameter | Where | What it does |
|---|---|---|
| Tickers | Sidebar text area | Add any Yahoo Finance ticker |
| Start Year | Sidebar | How far back to pull price history |
| Risk-Free Rate | Sidebar slider | Used in Sharpe calculation (use current 3m Treasury yield) |
| Show random cloud | Sidebar checkbox | Toggle the feasible set scatter behind the frontier |

---

## Quick presets

- **US Tech** — AAPL, MSFT, GOOGL, NVDA, META
- **Diversified** — SPY, TLT, GLD, VNQ, EFA (stocks + bonds + gold + real estate + international)
- **FAANG** — META, AAPL, AMZN, NFLX, GOOGL
- **Sectors** — XLK, XLF, XLE, XLV, XLI (technology, financials, energy, health, industrials)

---

## The maths in one paragraph

You have N assets, each with an expected return μᵢ and variance σᵢ². You also have a covariance
matrix Σ describing how they move together. A portfolio is a weight vector w (summing to 1).
Portfolio return = w·μ. Portfolio variance = wᵀΣw. The efficient frontier is the set of weight
vectors that minimise variance for each possible target return level. This is solved with
constrained quadratic optimisation (scipy SLSQP). The Max Sharpe portfolio maximises
(return − risk_free_rate) / volatility. The Min Vol portfolio just minimises volatility directly.

---

## What to say in an interview

I implemented Markowitz mean-variance optimisation from scratch using scipy's SLSQP solver.
I built the efficient frontier by sweeping target return levels and solving a constrained
minimum-variance problem at each point. I found the tangency portfolio by maximising the
Sharpe ratio, incorporated an Ito-corrected drift for consistency with my GBM simulator,
and visualised the Capital Market Line. I also computed the correlation matrix to show
diversification benefits across asset classes.

---

*Pranshu Mundada*