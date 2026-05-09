# Monte Carlo Portfolio Simulator

Simulates 10,000 portfolio paths using Geometric Brownian Motion. Outputs Value at Risk (VaR), Conditional VaR (CVaR), probability of ruin, and six interactive charts. Supports single-asset and multi-asset correlated portfolios.

---

## How to run it

```bash
# navigate to folder and activate venv
cd Desktop/CODING/"Quant Projects"/"Monte Carlo"
source venv/bin/activate

# install dependencies (first time only)
pip install -r requirements.txt

# launch Streamlit dashboard
streamlit run app.py

# or use the CLI
python cli.py --value 100000 --return 8 --vol 15 --days 252

# run tests
python tests.py
```

---

## CLI examples

```bash
# default: $100k, 8% return, 15% vol, 1 year, 10k paths
python cli.py

# aggressive equity portfolio
python cli.py --value 500000 --return 10 --vol 20 --days 504 --sims 20000

# change ruin threshold to 30% loss
python cli.py --value 100000 --return 8 --vol 15 --days 252 --ruin 30

# fixed seed for reproducible output
python cli.py --value 100000 --return 8 --vol 15 --days 252 --seed 42
```

---

## Files

| File | What it does |
|---|---|
| `simulator.py` | GBM engine — runs simulations, computes VaR/CVaR/ruin |
| `charts.py` | Six Plotly charts — paths, distribution, drawdowns, CDF, sensitivity |
| `app.py` | Streamlit dashboard |
| `cli.py` | Terminal interface |
| `tests.py` | 14 unit tests |

---

## Key parameters

| Parameter | What it means |
|---|---|
| Initial Value | Starting portfolio size in dollars |
| Annual Return | Expected yearly return (e.g. 8% for a diversified equity portfolio) |
| Annual Volatility | How much the portfolio jumps around (S&P 500 ≈ 15–20%) |
| Horizon | How far into the future to simulate (252 = 1 trading year) |
| Simulations | More = more accurate but slower. 10,000 is the standard |
| Ruin Threshold | What counts as catastrophic loss (e.g. 20% = losing >$20k on $100k) |

---

## Risk metrics explained

**VaR 95% (Value at Risk)**
The loss you will not exceed in 95% of scenarios. If VaR 95% = $12,000, then in 95 out of 100 simulated years, you lose less than $12,000.

**CVaR 95% (Conditional VaR / Expected Shortfall)**
The average loss in the worst 5% of scenarios. Always worse than VaR. This is what Basel III bank regulation uses. It answers: "when things go badly, how badly do they go on average?"

**Probability of Ruin**
Fraction of simulated paths where the loss exceeded your ruin threshold. If ruin = 20% and prob_ruin = 8%, then in 800 out of 10,000 simulations the portfolio fell more than 20%.

**Probability of Profit**
Fraction of paths that ended above the starting value.

---

## Presets (dashboard)

| Preset | Return | Vol | Based on |
|---|---|---|---|
| S&P 500 | 10% | 18% | US large-cap historical average |
| 60/40 | 7% | 10% | Classic balanced portfolio |
| Bonds | 4% | 5% | Investment-grade fixed income |
| EM Equity | 9% | 25% | Emerging markets equity |

---

## What to say about this in an interview

I built a Monte Carlo portfolio risk engine using Geometric Brownian Motion with an Ito-corrected drift term. I simulated 10,000 paths, computed VaR and CVaR at 95% and 99% confidence levels, and measured probability of ruin across different loss thresholds. I extended the model to multi-asset portfolios using Cholesky decomposition to inject correlated return shocks. I also built a volatility sensitivity analysis showing the non-linear relationship between vol and tail risk.

---

*Pranshu Mundada*