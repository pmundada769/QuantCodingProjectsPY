# OPTIONS PRICER — WALKTHROUGH
# Everything explained from scratch, assuming only CS50

---

## What this project actually does

An option is a financial contract that gives you the **right, but not the obligation**, to buy or sell a stock at a fixed price on or before a specific date.

You pay a premium upfront to own that right. The question is: **how much should that premium be?**

In 1973, Fischer Black and Myron Scholes published a formula that answers this question mathematically. It won the Nobel Prize. It is still the industry standard. This project implements it.

The project also computes the "Greeks" — measures of how sensitive the option's price is to changes in the stock price, time, volatility, and interest rates. Traders use Greeks to manage risk.

---

## The mental model before reading any code

An option's price depends on five inputs:

```
S  = current stock price
K  = strike price (the price at which you can buy/sell)
T  = time until expiry (in years)
r  = risk-free interest rate
σ  = volatility (how much the stock moves around)
```

The Black-Scholes formula takes these five numbers and outputs a fair price. The formula also has derivatives (in the calculus sense) — how much does the price change if S moves a little? If T shrinks a little? Those derivatives are the Greeks.

The pipeline is:

```
INPUTS (S, K, T, r, σ)  →  BLACK-SCHOLES FORMULA  →  PRICE + GREEKS  →  DASHBOARD / CLI
```

Each file in the project is one layer of that pipeline.

---

---

# black_scholes.py — The core maths

This is the most important file. Everything else just displays what this file computes.

---

## `OptionResult` — the data container

```python
@dataclass
class OptionResult:
    option_type: str
    price: float
    delta: float
    theta: float
    ...
```

**What is a `@dataclass`?** In CS50 you used classes with `__init__`. A dataclass does the same thing automatically — it creates a class where you just list the fields and Python generates the constructor for you. `OptionResult` is just a neat container to hold all the outputs together so you can write `res.delta` rather than returning a tuple of 15 numbers.

---

## `black_scholes` — the main function

```python
def black_scholes(S, K, T, r, sigma, q=0.0, option_type="call") -> OptionResult:
```

**What is `q`?** The continuous dividend yield. Some stocks pay dividends — regular cash payments to shareholders. Dividends reduce a call option's value (because when the stock pays out cash, its price drops by roughly that amount, hurting the call holder). `q=0.0` means no dividends, which is fine as a default.

**What is `Literal["call", "put"]`?** A type hint that says this argument can only be the string `"call"` or `"put"`. Python will not enforce this at runtime but your editor (VSCode) will warn you if you pass something else.

---

### The d1 and d2 calculations

```python
sqrt_T = np.sqrt(T)
d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
d2 = d1 - sigma * sqrt_T
```

**What is `np.log`?** Natural logarithm (base e, not base 10). In finance, log returns are used everywhere because they have nice mathematical properties — they are additive over time, and they prevent prices from going negative.

**What does `S / K` mean intuitively?** The ratio of current stock price to strike price. If `S / K > 1`, the option is in-the-money (the stock is already above the strike). `np.log(S/K)` measures how far in or out of the money the option is, on a log scale.

**What are d1 and d2 actually?** They are intermediate values that feed into the normal distribution. d1 roughly measures the probability-adjusted "distance to the money." d2 is d1 minus a volatility adjustment. You do not need to memorise the formula — just know they are inputs to the normal distribution function below.

---

### The normal distribution

```python
Nd1 = norm.cdf(d1)
Nd2 = norm.cdf(d2)
nd1 = norm.pdf(d1)
```

**What is `norm.cdf`?** The cumulative distribution function of the standard normal distribution. `norm.cdf(d1)` gives you the probability that a standard normal random variable is less than d1. In Black-Scholes, `N(d2)` is approximately the probability the option expires in the money (the stock ends above the strike). `N(d1)` is a risk-adjusted version of the same idea.

**What is `norm.pdf`?** The probability density function — the height of the normal bell curve at a given point. It appears in the Greek formulas (Gamma, Vega, Vanna etc.) because those are derivatives of the price formula, and the derivative of the cumulative normal is its density.

If you remember the bell curve from statistics — `cdf` gives you the area to the left of a point, `pdf` gives you the height of the curve at that point.

---

### The price formula

```python
# Call
price = S * disc_q * Nd1 - K * disc * Nd2

# Put
price = K * disc * Nnd2 - S * disc_q * Nnd1
```

**What is `disc`?** Short for discount factor. `np.exp(-r * T)` converts a future value to a present value. A $100 payment in 1 year is worth slightly less than $100 today, because you could have earned interest on that money. `np.exp(-r * T)` is the continuous-time version of `1 / (1 + r)^T`.

**What is `disc_q`?** The same thing but for dividends. It discounts the stock price to account for dividend payments that will reduce its value over time.

**Reading the call formula in English:**
`S * disc_q * Nd1` = the probability-weighted present value of receiving the stock
minus
`K * disc * Nd2` = the probability-weighted present value of paying the strike

If it is likely the option expires in-the-money (`Nd1` and `Nd2` close to 1), the call is worth approximately `S - K * disc`. If it is unlikely (deep out of the money), both `Nd1` and `Nd2` approach 0 and the option is nearly worthless.

---

### The Greeks

**Delta:**
```python
delta = disc_q * Nd1   # for calls
delta = -disc_q * Nnd1  # for puts
```
Delta is the first derivative of option price with respect to stock price (∂Price/∂S). It tells you: if the stock moves $1, the option price moves by Delta dollars. A call with Delta = 0.5 gains $0.50 when the stock rises $1. Delta is always between 0 and 1 for calls, and between -1 and 0 for puts.

**Gamma:**
```python
gamma = disc_q * nd1 / (S * sigma * sqrt_T)
```
Gamma is the second derivative of price with respect to stock price (∂²Price/∂S²). It tells you how fast Delta itself changes. High Gamma means Delta is very sensitive to stock moves — the option is acting more and more like the stock as it moves in-the-money. Gamma is always positive for long options (both calls and puts).

**Theta:**
```python
theta = (theta_base - r * K * disc * Nd2) / 365
```
Theta is the derivative of price with respect to time (∂Price/∂T). The `/365` converts it to a per-calendar-day number. It is almost always negative — your option loses value every day just by sitting there, even if nothing else changes. This is called "time decay." Theta accelerates as expiry approaches — the last few days before expiry see the fastest decay.

**Vega:**
```python
vega = S * disc_q * nd1 * sqrt_T / 100
```
Vega is the derivative of price with respect to volatility (∂Price/∂σ). The `/100` makes it a per-1%-point-of-vol number (so if vol goes from 20% to 21%, the price changes by Vega). Vega is always positive — higher volatility always makes options more valuable, because more movement means a higher chance of a big profitable move.

**Rho:**
```python
rho = K * T * disc * Nd2 / 100   # calls
```
Rho is the derivative of price with respect to the interest rate. For most retail options (short-dated, small rates), Rho is tiny and barely matters. It matters more for long-dated options or in high-rate environments.

---

### The second-order Greeks

**Vanna:**
```python
vanna = -disc_q * nd1 * d2 / sigma
```
How Delta changes when volatility changes (∂Delta/∂σ). Useful for hedging: if you are Delta-hedged but vol spikes, Vanna tells you how much your hedge drifts. Used by options market makers.

**Charm:**
```python
charm = (...) / 365
```
How Delta changes as time passes (∂Delta/∂T per day). Important for traders who Delta-hedge daily — your hedge ratio changes just from time passing, even if the stock does not move.

**Volga (Vomma):**
```python
volga = vega * d1 * d2 / sigma
```
How Vega changes when volatility changes (∂Vega/∂σ). Tells you how your volatility exposure accelerates. Important for volatility trading strategies.

---

## `implied_volatility` — solving backwards

```python
def implied_volatility(market_price, S, K, T, r, q=0.0, option_type="call"):
    sigma = 0.20  # initial guess
    for _ in range(max_iter):
        res  = black_scholes(S, K, T, r, sigma, q, option_type)
        diff = res.price - market_price
        vega_unit = res.vega * 100
        sigma_new = sigma - diff / vega_unit
        ...
```

**The problem:** You observe an option trading at $4.50 on the market. You know S, K, T, and r. What volatility does the market imply?

Black-Scholes cannot be inverted analytically for sigma — there is no algebraic formula for it. So you solve numerically.

**Newton-Raphson** is an algorithm from CS50-level maths. The idea: start with a guess, compute how wrong you are (`diff`), then use the slope of the function (`vega_unit`) to figure out which direction to adjust. Repeat until the answer is close enough.

In one sentence: `new_guess = old_guess - (how_wrong / how_fast_things_change)`. This converges in about 5–10 iterations.

**Why use Vega as the slope?** Vega is exactly ∂Price/∂sigma — the rate of change of price with respect to volatility. That is precisely what Newton-Raphson needs.

**What is `np.nan`?** "Not a Number" — a special float value representing a missing or undefined result. The function returns `np.nan` if it fails to converge, so the calling code can check for it.

---

---

# charts.py — All the visualisations

This file turns Black-Scholes results into interactive Plotly charts. You do not need to understand every line — just understand what each function produces and what it is for.

---

## `payoff_diagram`

```python
expiry_pnl = np.maximum(spot_range - K, 0) - premium   # call at expiry
current_pnl = [black_scholes(s, ...).price - premium for s in spot_range]
```

**What is `np.maximum(spot_range - K, 0)`?** At expiry, a call option is worth `max(S - K, 0)`. If the stock is above the strike, you exercise and make `S - K`. If it is below, the option expires worthless (worth 0). `np.maximum` applies this max across every value in `spot_range` at once — this is called vectorisation, and it is much faster than a Python for-loop.

**Why two lines on the chart?** The solid line shows what the option will be worth at expiry. The dotted line shows what it is worth right now (today, before expiry). The gap between them is time value — the premium you paid for the possibility that the stock moves further in your favour before expiry. As expiry approaches, these two lines converge.

**What is the break-even line?** For a call, you break even when the stock price at expiry equals strike + premium paid. If you paid $3 for a $100 call, you need the stock to be above $103 at expiry to make money.

---

## `greeks_vs_spot`

Loops through 300 stock prices from 50% to 150% of current price, calls `black_scholes()` at each one, and plots Delta, Gamma, Theta, Vega.

**Why does Delta follow an S-curve?** A deep out-of-the-money call has Delta near 0 (the stock is far from the strike, unlikely to be exercised). A deep in-the-money call has Delta near 1 (almost certain to be exercised — behaves like owning the stock). The transition between 0 and 1 follows a smooth S-shaped curve centred on the strike.

**Why does Gamma peak at the strike?** Gamma measures how fast Delta is changing. Delta changes fastest right at the strike — that is where the option is transitioning between "likely to expire worthless" and "likely to be exercised." Away from the strike in either direction, Delta barely changes (it is already close to 0 or 1), so Gamma falls toward 0.

---

## `greeks_vs_time`

Shows how Delta and Theta evolve as expiry approaches, for the current stock price.

**Why does Theta accelerate near expiry?** Time value cannot go below zero. As expiry approaches, all remaining time value has to disappear in less and less time — so the daily decay rate gets larger. An option with 30 days left loses much more per day than the same option with 180 days left.

---

## `vol_smile`

```python
smile_vol = sigma + skew * moneyness + curvature * moneyness**2
```

**What is a volatility smile?** Black-Scholes assumes volatility is constant across all strikes. In reality, if you look at market prices for options at different strikes and back out the implied volatility, you get a curve — not a flat line. This curve is called the "smile" (or "smirk" when it is skewed).

**Why does the smile exist?** Because Black-Scholes is wrong in its assumption of constant vol. Real stocks have fat tails — large moves happen more often than a normal distribution predicts. Out-of-the-money puts (crash protection) are bid up by investors who fear large drops, pushing their implied vol higher.

**The chart here is synthetic** — it illustrates the shape using a formula, not real market data. In a real trading system, you would use actual market prices to build the smile.

---

## `strategy_payoff`

```python
STRATEGIES = {
    "Bull Call Spread": [
        {"type": "call", "sign": +1, "offset": -0.05},
        {"type": "call", "sign": -1, "offset": +0.05},
    ],
    ...
}
```

**What is a multi-leg strategy?** Combining multiple options into one position. For example, a Bull Call Spread: you buy a call at a lower strike and sell a call at a higher strike. The sold call partially offsets the cost of the bought call, so you pay less upfront — but you also cap your maximum profit.

**What does `"sign": +1` mean?** Long (you bought it). `"sign": -1` means short (you sold it). The payoff of a sold option is the mirror image of a bought option.

**What does `"offset": -0.05` mean?** The strike is set relative to the current spot price. `-0.05` means the strike is 5% below spot. `+0.05` means 5% above. This keeps the strategy sensible regardless of what spot price you input.

**The loop builds the net payoff:** For each leg, it calculates the expiry P&L of that leg individually. Then it adds them all together to get the net P&L curve shown in amber.

---

---

# app.py — The Streamlit dashboard

Streamlit is a Python library that turns a Python script into an interactive web app, automatically. You do not write HTML or JavaScript — you write Python and Streamlit renders it in the browser.

---

## How Streamlit works

Every time a user moves a slider or changes an input, Streamlit re-runs the entire `app.py` script from top to bottom with the new values. This is different from how normal programs work (where state persists). It is simple but powerful for data dashboards.

```python
S = st.number_input("Spot (S)", value=100.0, ...)
K = st.number_input("Strike (K)", value=100.0, ...)
```

These lines create input widgets in the browser sidebar. The values the user types or drags become the Python variables `S` and `K` in the script. Every time the user changes them, the whole script re-runs and all the charts update.

---

## The sidebar

```python
with st.sidebar:
    ...
```

Everything inside this block appears in the left panel. All your input controls — spot, strike, expiry, vol, rate, dividend — live here.

---

## The metrics row

```python
c1.metric("Option Price", f"${res.price:.4f}")
```

`st.metric` creates a nice card with a label and a number. The `:.4f` format code means "4 decimal places." These six cards across the top give you the price and key auxiliary values at a glance.

---

## The tabs

```python
tab1, tab2, tab3, tab4, tab5 = st.tabs([...])

with tab1:
    fig = payoff_diagram(...)
    st.plotly_chart(fig, use_container_width=True)
```

Each tab shows one chart. `st.plotly_chart` renders an interactive Plotly figure in the browser — you can zoom, pan, hover for exact values.

---

## The sensitivity table

```python
for v in vol_range:
    row = []
    for s in spot_range:
        p = black_scholes(s, K, T, r, v, q, option_type).price
        row.append(round(p, 3))
    table.append(row)
```

A nested loop: for each combination of volatility and spot price, compute the option price and store it. The result is a 7×7 grid — rows are volatility levels, columns are spot prices. This lets you see at a glance how sensitive the price is to both inputs simultaneously.

`df.style.background_gradient(cmap="YlOrRd")` colours the cells from yellow (low) to red (high), making the pattern obvious visually.

---

---

# cli.py — The command-line interface

The CLI produces the same numbers as the dashboard but in the terminal, with colour formatting. Useful when you want to run a quick calculation without opening a browser.

```python
import argparse
parser.add_argument("--spot", type=float, required=True, ...)
args = parser.parse_args()
```

**What is `argparse`?** A standard Python library for building command-line tools. It reads the flags you type after `python cli.py` (like `--spot 100`) and converts them into Python variables automatically. `required=True` means the script will print an error message if you forget to supply that argument.

The colour codes like `"\033[96m"` are ANSI escape sequences — special codes that tell the terminal to change text colour. `\033[0m` resets back to default. These are the same codes that make your terminal show green text for git additions and red for deletions.

---

---

# tests.py — Making sure the maths is correct

```python
def test_put_call_parity():
    call = black_scholes(S, K, T, r, sigma, q, "call")
    put  = black_scholes(S, K, T, r, sigma, q, "put")
    lhs  = call.price - put.price
    rhs  = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert abs(lhs - rhs) < 1e-8
```

**What is put-call parity?** A fundamental no-arbitrage relationship in options theory: the price of a call minus the price of a put (same strike, same expiry) must equal the discounted forward price of the stock minus the discounted strike. If this is violated, you could make risk-free money by trading the two options against each other. The test verifies the formula respects this identity to 8 decimal places.

**What is `assert`?** You know this from CS50. If the condition is False, Python raises an `AssertionError` and the test fails. Here, `abs(lhs - rhs) < 1e-8` checks that the two sides are equal to within a tiny floating-point tolerance (0.00000001).

**Why test at all?** The Black-Scholes formula has a lot of moving parts. It is easy to introduce a sign error or forget a factor of 100 somewhere. The tests catch these. The test suite checks: ATM call price matches a known benchmark, Delta is between 0 and 1, Theta is negative, Vega is positive, put-call parity holds, and the IV solver recovers the original vol.

---

---

## How to use this in an interview

You built a full Black-Scholes options pricing engine from scratch. You implemented the generalised Merton model with continuous dividend yield, computing the option price and eight Greeks: Delta, Gamma, Theta, Vega, Rho, Vanna, Charm, and Volga. You also implemented a Newton-Raphson implied volatility solver that converges to six decimal places. The front end is a Streamlit dashboard with interactive sliders, five chart tabs including a strategy builder for eight multi-leg strategies, and a spot-vol sensitivity table. You also built a colourised CLI and a unit test suite including put-call parity verification.

That is an honest, complete answer.

---

## What to experiment with

**Change spot vs strike and watch Delta move:**
In the dashboard, set spot to $80 and strike to $100 (deep OTM call). Delta will be near 0. Slide spot up to $120 (deep ITM). Delta approaches 1. This is the most intuitive way to understand what Delta means.

**Watch Theta accelerate:**
Set days to expiry to 180. Note the Theta. Now drag it down to 7. Theta gets much more negative — the option is losing value faster as expiry approaches.

**Understand the vol smile:**
On the Vol Smile tab, your flat vol (dotted line) is what Black-Scholes assumes. The curved line is what the market actually prices. The left side (low strikes, out-of-the-money puts) has higher implied vol — that is the "skew." Traders pay more for downside protection.

**Try the Iron Condor strategy:**
In the Strategy Builder tab, select Iron Condor. The payoff chart shows a flat profit zone in the middle and losses on both extremes. This is a bet that the stock stays within a range. The strategy makes money from time decay as long as the stock does not move much.

---

Here's the tldr:

What it is: A calculator for options — financial contracts that let you bet on whether a stock will go up or down, without buying the stock itself.

The sidebar — your inputs:

Spot (S) — current stock price
Strike (K) — the price you're betting the stock will reach
Days to Expiry — how long until the contract expires
Volatility — how much the stock jumps around (higher = more expensive option)
Rate — interest rate (use current US rate ~5%)
Div Yield — dividends the stock pays (0 for most tech stocks)


The numbers at the top:

Option Price — what this contract should fairly cost right now
Intrinsic — how much it's worth if you exercised it today
Time Value — the extra premium you're paying for time remaining


The Greeks tables — risk dials:

Delta — if the stock moves $1, your option moves by this much. 0.53 = moves 53 cents per $1
Theta — how much value you lose per day just by doing nothing. Always negative — time kills options
Vega — how much the price changes if volatility spikes. Big news event = vol spike = option gets more expensive
Gamma — how fast Delta is changing. High near expiry


The tabs:

Payoff Diagram — shows exactly how much you make or lose at different stock prices at expiry
Greeks vs Spot — shows how your risk changes as the stock moves
Greeks vs Time — shows how fast you're losing value as days pass
Vol Smile — shows how real markets price vol differently at different strikes
Strategy Builder — combines multiple options into one position (see below)


Strategies — the most useful tab:

Bull Call Spread — you think stock goes up, but want to pay less. Cap your upside, cap your cost
Bear Put Spread — you think stock goes down
Long Straddle — you think something big happens but don't know which direction. Profits from big moves either way
Long Strangle — same as straddle but cheaper, needs an even bigger move to profit
Covered Call — you own the stock already, sell a call to collect premium income
Protective Put — you own the stock, buy a put as insurance against a crash
Iron Condor — you think the stock goes nowhere. Profit from time decay as long as it stays in a range
Long Butterfly — very precise bet the stock lands at exactly one price at expiry


Sensitivity Table — the grid at the bottom. Shows you the option price at every combination of stock price and volatility. Red = expensive, yellow = cheap. Useful for seeing how much vol matters vs stock price.

What it works for:
It prices any underlying — stocks, indices (S&P 500), FX (EUR/USD), commodities. Just change the inputs. For FX, the "dividend yield" input becomes the foreign interest rate. For bonds it's less applicable — bonds use different models entirely.

How a trader actually uses this:

Look up the real market price of an option on a broker (IBKR, Robinhood etc.)
Type it into the IV Calculator at the bottom of the sidebar
It tells you the implied vol — what vol the market is pricing in
If you think actual vol will be higher than that, the option is cheap → buy it
If you think actual vol will be lower, the option is expensive → sell it

---

*Pranshu Mundada*
