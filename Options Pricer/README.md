# Options Pricer

A Black-Scholes options pricing engine with a Streamlit dashboard and a command-line interface. Prices call and put options, computes all Greeks (first and second order), solves for implied volatility, and plots payoff diagrams for single options and multi-leg strategies.

---

## How to run it

```bash
# Step 1: Open VSCode terminal (Ctrl + `)

# Step 2: Navigate to the project folder and activate your virtual environment
cd Desktop/quant_projects/Options_Pricer
source venv/bin/activate          # Mac / Linux
# venv\Scripts\activate           # Windows

# Step 3a: Launch the Streamlit dashboard (recommended)
streamlit run app.py

# Step 3b: Or use the command-line interface instead
python cli.py --spot 100 --strike 100 --days 30 --vol 20 --rate 5

# Step 4: Run the test suite to verify everything works
python -m pytest tests.py -v
# or if pytest is not installed:
python tests.py
```

The Streamlit dashboard opens automatically in your browser at `http://localhost:8501`.

---

## What it produces

**Streamlit dashboard (app.py):**
- Live option price, intrinsic value, and time value
- Full Greeks table — Delta, Vega, Theta, Rho, Gamma, Vanna, Charm, Volga
- Payoff diagram: what the option is worth at different stock prices, both now and at expiry
- Greeks vs Spot: how each Greek changes as the stock moves
- Greeks vs Time: how Delta and Theta change as expiry approaches
- Volatility smile: how real market vol differs from flat Black-Scholes vol
- Strategy builder: payoff diagrams for Bull Call Spread, Iron Condor, Straddle, and five others
- Sensitivity table: option price across a grid of spots and volatilities

**CLI (cli.py):**
- All the same numbers printed in the terminal with colour formatting
- Implied volatility solver: give it a market price and it tells you what vol is implied

---

## Files

| File | What it does |
|---|---|
| `black_scholes.py` | Core maths — prices options and computes all Greeks |
| `charts.py` | All Plotly charts — payoff diagrams, Greeks plots, vol smile, strategy builder |
| `app.py` | Streamlit dashboard — the interactive browser UI |
| `cli.py` | Command-line interface — same output in the terminal |
| `tests.py` | Unit tests — verifies the maths is correct |

---

## CLI usage examples

```bash
# Price an ATM call option
python cli.py --spot 100 --strike 100 --days 30 --vol 20 --rate 5

# Price an ITM put option
python cli.py --spot 90 --strike 100 --days 60 --vol 30 --rate 4.5 --type put

# Price with a dividend yield
python cli.py --spot 150 --strike 145 --days 45 --vol 25 --rate 5 --div 1.5

# Back out implied volatility from a known market price
python cli.py --spot 100 --strike 100 --days 30 --vol 20 --rate 5 --market-price 3.50
```

---

## Parameters explained

| Flag | What it means | Example |
|---|---|---|
| `--spot` | Current stock price (S) | `100` |
| `--strike` | The option's strike price (K) | `105` |
| `--days` | Days until the option expires | `30` |
| `--vol` | Volatility in % (not decimal) | `20` means 20% |
| `--rate` | Risk-free interest rate in % | `5` means 5% |
| `--div` | Continuous dividend yield in % | `1.5` means 1.5% |
| `--type` | `call` or `put` | `put` |
| `--market-price` | Observed market price → solves for IV | `3.50` |

---

## Key parameters you can change in the dashboard

Everything is controlled by the sliders and inputs in the left sidebar:

| Control | What it does |
|---|---|
| Spot (S) | Current underlying stock price |
| Strike (K) | The price at which the option can be exercised |
| Days to Expiry | Drag to change time remaining — watch Theta accelerate near zero |
| Volatility (%) | Implied vol — drag up to see how expensive options become in a volatile market |
| Risk-free Rate (%) | Usually the 3-month US Treasury yield |
| Div Yield (%) | For stocks that pay dividends — reduces call value, increases put value |
| Market Price (for IV) | Type in a real price you saw on a broker — it calculates what vol that implies |

---

## Greeks quick reference

| Greek | Symbol | What it tells you |
|---|---|---|
| Delta | Δ | How much the option price moves per $1 move in the stock |
| Gamma | Γ | How much Delta itself changes per $1 move |
| Theta | Θ | How much value the option loses per day (always negative for long options) |
| Vega | ν | How much the price changes per 1% move in volatility |
| Rho | ρ | How much the price changes per 1% move in interest rates |
| Vanna | — | How Delta changes when volatility moves |
| Charm | — | How Delta changes as time passes (per day) |
| Volga | — | How Vega changes when volatility moves |

---

## How to evaluate results

**Is the price reasonable?**
An ATM option (spot = strike) with 30 days to expiry and 20% vol should cost roughly 2–3% of the spot price. If it prices at $3 on a $100 stock, that is 3% — normal.

**Does the Delta make sense?**
A call option at-the-money (spot = strike) always has Delta close to 0.50. Deep in-the-money calls approach 1.0. Deep out-of-the-money calls approach 0.0.

**Is Theta behaving correctly?**
Theta should always be negative for a long option — you lose a little value every day due to time decay. Theta accelerates (gets more negative) as expiry approaches.

**Implied volatility interpretation:**
If you see a $5 option on a $100 stock and the model says the implied vol is 35%, that means the market is pricing in large expected moves. Normal "calm market" vol for large caps is 15–25%. Earnings announcements, macro events, or small-cap stocks can push IV to 60–100%+.

---

## Dependencies

```bash
pip install numpy scipy streamlit plotly pandas
```

---

*Pranshu Mundada*