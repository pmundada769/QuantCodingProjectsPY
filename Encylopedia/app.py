#app.py — Quantitative Finance Encyclopedia & Calculator
# University-level reference: formulas, definitions, Greek alphabet, indicators, risk metrics
# Run with: streamlit run app.py

import streamlit as st # type: ignore
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(page_title="Quant Encyclopedia", page_icon="📚", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&family=Source+Serif+4:ital,wght@0,300;0,400;0,600;1,400&display=swap');
html,body,[class*="css"]{background:#06080A;color:#D0D8E0;font-family:'Source Serif 4',serif;}
h1{font-family:'Fira Code',monospace!important;color:#FFD700!important;font-size:1.8rem!important;}
h2{font-family:'Fira Code',monospace!important;color:#00CCFF!important;font-size:1.1rem!important;}
h3{font-family:'Fira Code',monospace!important;color:#00FF88!important;font-size:0.9rem!important;}
code,pre{font-family:'Fira Code',monospace!important;background:#0A1218;padding:2px 6px;border-radius:3px;color:#00FF88;}
[data-testid="stSidebar"]{background:#040608;border-right:1px solid #0E1820;}
.stTabs [data-baseweb="tab"]{font-family:'Fira Code',monospace;font-size:.65rem;background:#0A1018;border:1px solid #0E2030;color:#1A3040;border-radius:3px;}
.stTabs [aria-selected="true"]{background:#0E2030!important;border-color:#FFD700!important;color:#FFD700!important;}
hr{border-color:#0E1820!important;}
[data-testid="metric-container"]{background:#0A1018;border:1px solid #0E2030;border-radius:3px;padding:10px 14px;}
[data-testid="metric-container"] [data-testid="metric-value"]{font-family:'Fira Code',monospace!important;color:#FFD700!important;}
table{width:100%;}
th{color:#FFD700!important;font-family:'Fira Code',monospace;}
</style>
""", unsafe_allow_html=True)

GOLD="#FFD700"; CYAN="#00CCFF"; GREEN="#00FF88"; RED="#FF4466"
BG="#06080A"; GRID="#0E1820"; TEXT="#D0D8E0"

st.markdown("# 📚 Quantitative Finance Encyclopedia")
st.markdown("`University-level reference — formulas, definitions, proofs, calculators`")
st.markdown("---")

tabs = st.tabs([
    "α β Greek Alphabet",
    "📐 Technical Indicators",
    "🌀 Ichimoku System",
    "🎯 Options & Greeks",
    "📊 Risk & Performance",
    "🔢 Quant Formulas",
    "🏦 Fundamental Analysis",
    "🧮 Calculators",
    "📖 Finance Concepts",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — GREEK ALPHABET
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("## Greek Alphabet — Finance & Statistics Usage")
    st.markdown("Every Greek letter used in quantitative finance, statistics, and mathematics.")

    greek_data = [
        ("Α α", "Alpha",   "α",
         "Excess return above benchmark (Jensen's alpha). Significance level in hypothesis testing. "
         "Intercept in regression. Type I error probability. Euler–Mascheroni constant.",
         "CAPM: R_i = α + β·R_m + ε  |  H₀ rejection: p < α"),
        ("Β β", "Beta",    "β",
         "Systematic risk — sensitivity of asset return to market return. Type II error probability. "
         "Regression coefficient. Beta distribution parameter.",
         "β = Cov(R_i, R_m) / Var(R_m)  |  β>1 aggressive, β<1 defensive"),
        ("Γ γ", "Gamma",   "Γ/γ",
         "Rate of change of Delta with respect to underlying price (options). "
         "Gamma function Γ(n) = (n−1)!. Incomplete gamma function. Shape parameter in distributions.",
         "Γ = ∂²V/∂S²  |  Γ(n) = ∫₀^∞ t^(n-1) e^(-t) dt"),
        ("Δ δ", "Delta",   "Δ/δ",
         "Change / difference operator. Option price sensitivity to underlying (∂V/∂S). "
         "Dirac delta function δ(x). Kronecker delta δᵢⱼ.",
         "Call Δ = N(d₁)  |  Put Δ = N(d₁) − 1  |  Range: [−1, +1]"),
        ("Ε ε", "Epsilon", "ε",
         "Residual / error term in regression. Small positive number (epsilon-neighbourhood). "
         "Elasticity. Machine epsilon in numerical computing.",
         "OLS: y = Xβ + ε  |  E[ε] = 0, Var(ε) = σ²I"),
        ("Ζ ζ", "Zeta",    "ζ",
         "Riemann zeta function ζ(s). Damping ratio in control theory. "
         "Risk measure (Zeta model for bankruptcy). Yield measure.",
         "ζ(s) = Σₙ₌₁^∞ 1/nˢ  (Re(s) > 1)"),
        ("Η η", "Eta",     "η",
         "Price elasticity of demand. Learning rate in machine learning / gradient descent. "
         "Efficiency parameter. Dirichlet eta function.",
         "η = (∂Q/∂P)·(P/Q)  |  SGD: θ ← θ − η·∇L(θ)"),
        ("Θ θ", "Theta",   "Θ/θ",
         "Option time decay — rate of change of option value with respect to time. "
         "Parameter vector in statistical models. Angle in polar coordinates.",
         "Θ = ∂V/∂t  (negative for long options — daily cost of holding)"),
        ("Ι ι", "Iota",    "ι",
         "Imaginary unit in some notations. Rarely used in finance directly.",
         "—"),
        ("Κ κ", "Kappa",   "κ",
         "Mean reversion speed in Ornstein–Uhlenbeck / Vasicek model. "
         "Condition number of a matrix. Kurtosis (excess) sometimes denoted κ.",
         "dX = κ(μ−X)dt + σdW  (mean reversion to μ at speed κ)"),
        ("Λ λ", "Lambda",  "Λ/λ",
         "Lagrange multiplier. Intensity parameter of Poisson process. "
         "Eigenvalue. Hazard rate / default intensity in credit models. Jump intensity in Merton model.",
         "Lagrangian: L = f(x) − λ·g(x)  |  Poisson: P(N=k) = e^(-λ)λᵏ/k!"),
        ("Μ μ", "Mu",      "μ",
         "Population mean / expected value. Drift term in stochastic processes. "
         "Mean of a normal distribution N(μ, σ²). Risk-neutral drift.",
         "GBM: dS = μS·dt + σS·dW  |  E[X] = μ"),
        ("Ν ν", "Nu",      "ν",
         "Degrees of freedom in t and chi-squared distributions. "
         "Frequency (physics). Option vega sometimes ν.",
         "t-dist: E[X]=0, Var=ν/(ν−2)  |  χ²(ν): E[X]=ν"),
        ("Ξ ξ", "Xi",      "ξ",
         "Random variable in some notations. Extreme value index parameter. "
         "Generalised extreme value (GEV) shape parameter.",
         "GEV: F(x) = exp[−(1+ξ·(x−μ)/σ)^(−1/ξ)]"),
        ("Ο ο", "Omicron", "ο",
         "Little-o notation for asymptotic analysis: f = o(g) means f/g → 0. "
         "Rarely used as a variable.",
         "f(n) = o(g(n)) ⟺ lim f(n)/g(n) = 0  as n→∞"),
        ("Π π", "Pi",      "Π/π",
         "Product operator Π. Mathematical constant π ≈ 3.14159. "
         "Portfolio value/P&L in many texts. Inflation rate π.",
         "Π = Πᵢ xᵢ  |  Fisher equation: (1+r) = (1+ρ)(1+π)"),
        ("Ρ ρ", "Rho",     "ρ",
         "Correlation coefficient. Option sensitivity to interest rate (Rho Greek). "
         "Spearman rank correlation. Discount factor ρ = e^(−rt).",
         "ρ = Cov(X,Y)/[σ(X)·σ(Y)]  |  Rho = ∂V/∂r"),
        ("Σ σ", "Sigma",   "Σ/σ",
         "Summation operator Σ. Standard deviation σ. Volatility in finance. "
         "Covariance matrix Σ. Sigma algebra. Normal distribution N(μ, σ²).",
         "σ = √[Σ(xᵢ−μ)²/N]  |  GBM volatility: σ√dt term"),
        ("Τ τ", "Tau",     "τ",
         "Time to expiry/maturity. Kendall's tau rank correlation. "
         "Torque. Optical depth. Characteristic time of mean reversion.",
         "τ = T − t  (time remaining to option expiry)"),
        ("Υ υ", "Upsilon", "Υ/υ",
         "Rarely used in mainstream finance. Occasionally used for speed of mean reversion "
         "or as a placeholder variable.",
         "—"),
        ("Φ φ", "Phi",     "Φ/φ",
         "Cumulative standard normal distribution Φ(x) = N(d). "
         "Golden ratio φ = (1+√5)/2 ≈ 1.618 (Fibonacci). "
         "Standard normal PDF φ(x) = (1/√2π)e^(−x²/2).",
         "Black-Scholes: Call = S·Φ(d₁) − K·e^(-rT)·Φ(d₂)"),
        ("Χ χ", "Chi",     "χ",
         "Chi-squared distribution χ²(ν). "
         "Used in goodness-of-fit tests, variance tests, Jarque-Bera normality test. "
         "Chi-squared statistic: Σ(O−E)²/E.",
         "JB = n/6·[S² + (K−3)²/4] ~ χ²(2)  under normality"),
        ("Ψ ψ", "Psi",     "ψ",
         "Digamma function ψ(x) = d/dx ln Γ(x). "
         "Wave function in quantum mechanics. "
         "Sometimes used for cash flow or portfolio function.",
         "ψ(n) = −γ + Σₖ₌₀^(n-1) 1/(k+1)  where γ is Euler–Mascheroni"),
        ("Ω ω", "Omega",   "Ω/ω",
         "Omega ratio (performance measure). Big-Omega notation Ω(g) in complexity. "
         "Covariance matrix sometimes Ω. Angular frequency ω = 2πf.",
         "Omega = [∫_r^∞(1−F(x))dx] / [∫_{-∞}^r F(x)dx]  (r = threshold return)"),
    ]

    for symbol, name, notation, description, formula in greek_data:
        with st.expander(f"**{symbol}** — {name}  `{notation}`"):
            st.markdown(f"**Used for:** {description}")
            if formula != "—":
                st.code(formula)

    st.markdown("---")
    st.markdown("### Big-O / Asymptotic Notation Summary")
    st.markdown("""
| Notation | Meaning | Example |
|---|---|---|
| O(g) | Upper bound — grows no faster than g | O(n²) |
| o(g) | Strictly slower than g | o(n²) means grows slower than n² |
| Ω(g) | Lower bound — grows at least as fast | Ω(n log n) |
| ω(g) | Strictly faster than g | ω(n) |
| Θ(g) | Tight bound — grows at same rate | Θ(n log n) |
""")


# ══════════════════════════════════════════════════════════════
# TAB 2 — TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("## Technical Indicators — Mathematical Definitions")

    with st.expander("Moving Averages — SMA, EMA, WMA, VWAP", expanded=False):
        st.markdown("""
### Simple Moving Average (SMA)
```
SMA(n, t) = (1/n) · Σᵢ₌₀ⁿ⁻¹ Cₜ₋ᵢ
```
Equal weight to all n periods. Lags price. Used as dynamic support/resistance.

### Exponential Moving Average (EMA)
```
EMA(t) = α · Cₜ + (1−α) · EMA(t−1)
α = 2 / (n+1)      (smoothing factor)
```
Exponentially decaying weights — recent prices weighted more heavily. Reacts faster than SMA.

### Weighted Moving Average (WMA)
```
WMA(n, t) = Σᵢ₌₁ⁿ i·Cₜ₋ₙ₊ᵢ / Σᵢ₌₁ⁿ i = Σᵢ₌₁ⁿ i·Cₜ₋ₙ₊ᵢ / [n(n+1)/2]
```
Linear weights — most recent bar gets weight n, oldest gets weight 1.

### VWAP — Volume Weighted Average Price
```
VWAP(t) = Σᵢ₌₁ᵗ (TPᵢ · Vᵢ) / Σᵢ₌₁ᵗ Vᵢ
TPᵢ = (Hᵢ + Lᵢ + Cᵢ) / 3       (typical price)
```
Resets each session. Institutional benchmark — large orders try to beat VWAP.
""")

    with st.expander("RSI — Relative Strength Index (Wilder, 1978)", expanded=False):
        st.markdown("""
### Formula
```
ΔCₜ = Cₜ − Cₜ₋₁

Gain(t) = max(ΔCₜ, 0)
Loss(t) = max(−ΔCₜ, 0)

Using Wilder's smoothing (equivalent to EMA with α=1/n):
AvgGain(t) = [(n−1)·AvgGain(t−1) + Gain(t)] / n
AvgLoss(t) = [(n−1)·AvgLoss(t−1) + Loss(t)] / n

RS  = AvgGain / AvgLoss
RSI = 100 − 100/(1+RS)
```

### Properties
- Bounded: RSI ∈ [0, 100]
- RSI = 50 when AvgGain = AvgLoss (balanced market)
- RSI → 100 as AvgLoss → 0 (uninterrupted gains)
- RSI → 0 as AvgGain → 0 (uninterrupted losses)

### Standard thresholds
| Threshold | Interpretation |
|---|---|
| RSI > 70 | Overbought — potential reversal down |
| RSI < 30 | Oversold — potential reversal up |
| RSI crossing 50 | Trend change confirmation |
| RSI divergence | Price vs RSI disagree — weakening trend |

**Divergence:** Price makes new high but RSI makes lower high = bearish divergence.
Price makes new low but RSI makes higher low = bullish divergence.

**Parameters:** n=14 default. n=8 more sensitive (short-term). n=2 extremely sensitive.
""")

    with st.expander("MACD — Moving Average Convergence Divergence (Appel, 1979)", expanded=False):
        st.markdown("""
### Formula
```
MACD Line   = EMA(12) − EMA(26)
Signal Line = EMA(9) of MACD Line
Histogram   = MACD Line − Signal Line
```

### Signals
- MACD crosses above Signal → bullish momentum
- MACD crosses below Signal → bearish momentum
- Histogram sign change → early warning of crossover
- Histogram divergence with price → trend weakening

### Zero-line crossovers
- MACD crosses above 0: short-term average > long-term = bullish regime
- MACD crosses below 0: bearish regime

**Default: (12, 26, 9)**
""")

    with st.expander("Bollinger Bands (John Bollinger, 1983)", expanded=False):
        st.markdown("""
### Formula
```
Middle Band = SMA(n)
σ(n)        = rolling standard deviation over n periods
Upper Band  = SMA(n) + k·σ(n)
Lower Band  = SMA(n) − k·σ(n)
```
**Default:** n=20, k=2

### Statistical interpretation
With k=2 and normally distributed returns, price lies within the bands ~95% of the time.
With k=3: ~99.7% of the time.

### Applications
- **Squeeze:** bands contract (low σ) → anticipate breakout
- **Walk the band:** price hugging upper/lower band = strong trend
- **%B:** %B = (Price − Lower) / (Upper − Lower)  (position within bands)
- **Bandwidth:** BW = (Upper − Lower) / Middle  (normalised volatility)
""")

    with st.expander("CCI — Commodity Channel Index (Lambert, 1980)", expanded=False):
        st.markdown("""
### Formula
```
TP(t)  = [H(t) + L(t) + C(t)] / 3         (typical price)
MA_TP  = SMA(TP, n)                         (rolling mean)
MAD    = (1/n) · Σᵢ₌₀ⁿ⁻¹ |TP(t−i) − MA_TP|  (mean absolute deviation)

CCI    = (TP − MA_TP) / (0.015 · MAD)
```

### The 0.015 constant
Chosen by Lambert so that ~70-80% of values fall within ±100, making ±100 meaningful thresholds.
Note: this assumes a roughly normal distribution of typical prices.

### Interpretation
| CCI | Signal |
|---|---|
| Crossing above +100 | Entering overbought — momentum long |
| Crossing below −100 | Entering oversold — momentum short |
| Crossing back inside ±100 | Potential reversal |
| Zero-line cross | Trend direction change |
""")

    with st.expander("ATR — Average True Range (Wilder, 1978)", expanded=False):
        st.markdown("""
### Formula
```
TR(t) = max(H(t)−L(t),  |H(t)−C(t−1)|,  |L(t)−C(t−1)|)

ATR(t) = [(n−1)·ATR(t−1) + TR(t)] / n     (Wilder's smoothing)
```

### Why three terms?
The three TR candidates handle different scenarios:
1. **H−L**: normal intra-bar range
2. **|H−PrevClose|**: gap up — yesterday's close was below today's low
3. **|L−PrevClose|**: gap down — yesterday's close was above today's high

TR takes the max of all three so gaps are correctly captured.

### Uses
- **Dynamic stop loss:** SL = Entry ± ATR × multiplier
- **Position sizing:** smaller position when ATR is large
- **Volatility filter:** avoid trading when ATR below rolling average (consolidation)
- **Chandelier exit:** Exit = Highest High − ATR × 3
""")

    with st.expander("Stochastic Oscillator (George Lane, 1950s)", expanded=False):
        st.markdown("""
### Formula
```
%K = 100 · (C − LL(n)) / (HH(n) − LL(n))

where:
  LL(n) = lowest low over n periods
  HH(n) = highest high over n periods

%D = SMA(%K, m)      (signal line, typically m=3)
```
**Default:** n=14, m=3

### Interpretation
- %K > 80: overbought zone
- %K < 20: oversold zone
- %K crossing above %D: bullish signal
- %K crossing below %D: bearish signal
- Divergence between price and stochastic: trend exhaustion

### Fast vs Slow vs Full
- **Fast:** raw %K and %D
- **Slow:** smoothed — first %D becomes new %K, then re-smoothed → reduces noise
- **Full:** user-controlled smoothing on both lines
""")

    with st.expander("ADX — Average Directional Index (Wilder, 1978)", expanded=False):
        st.markdown("""
### Formula
```
+DM = H(t) − H(t−1)  if positive and > |L(t) − L(t−1)|, else 0
−DM = L(t−1) − L(t)  if positive and > |H(t) − H(t−1)|, else 0

ATR = Wilder's Average True Range (14)

+DI = 100 · Wilder_Avg(+DM) / ATR
−DI = 100 · Wilder_Avg(−DM) / ATR

DX  = 100 · |+DI − −DI| / (+DI + −DI)
ADX = Wilder_Avg(DX, 14)
```

### Interpretation
| ADX | Trend Strength |
|---|---|
| < 20 | No trend / weak trend |
| 20–25 | Emerging trend |
| 25–50 | Strong trend |
| > 50 | Very strong trend |

ADX measures **trend strength**, NOT direction. Use +DI and −DI for direction.
+DI > −DI = bullish trend. −DI > +DI = bearish trend.
""")


# ══════════════════════════════════════════════════════════════
# TAB 3 — ICHIMOKU
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("## Ichimoku Kinko Hyo")
    st.markdown("*One-look equilibrium chart. Developed by Goichi Hosoda (1935–1969), published after 30 years of development.*")

    with st.expander("The five components — formulas and meaning", expanded=True):
        st.markdown("""
### Mathematical Definitions
```
Tenkan-sen  (Conversion Line, T)   = [HH(9)  + LL(9)]  / 2
Kijun-sen   (Base Line, K)         = [HH(26) + LL(26)] / 2
Senkou Span A (Leading Span A, SA) = [(T + K) / 2]  shifted +26 periods forward
Senkou Span B (Leading Span B, SB) = [HH(52) + LL(52)] / 2  shifted +26 periods forward
Chikou Span  (Lagging Span, Ch)    = Close  shifted −26 periods backward
```
where HH(n) = highest high over n periods, LL(n) = lowest low over n periods.

### Why midpoints, not averages?
Each line uses (HH + LL)/2 — the midpoint of the trading range — rather than a price average.
This represents "equilibrium" — the price where buyers and sellers are balanced over that lookback.

### The Cloud (Kumo) = area between SA and SB
- **Green cloud** (SA > SB): bullish — recent equilibrium above historical equilibrium
- **Red cloud** (SA < SB): bearish — opposite
- **Thickness**: proxy for support/resistance strength and volatility
- **Thin cloud**: weak support/resistance, high uncertainty — avoid trading
- **Cloud twist** (SA crosses SB): future trend change signal

### Chikou Span interpretation
Chikou is plotted 26 periods in the past. A "clean" Chikou (no price structure around it) confirms the current trend has historical momentum.
- Chikou above price from 26 bars ago: bullish
- Chikou below price from 26 bars ago: bearish
- Chikou in cloud: uncertain / avoid

### Standard parameters
| Parameter | Default | Alternative |
|---|---|---|
| Tenkan | 9 | 7 (crypto, 24h markets) |
| Kijun | 26 | 22 (crypto) |
| Senkou B | 52 | 44 (crypto) |
| Displacement | 26 | 22 (crypto) |

The original 9/26/52 parameters were designed for the Tokyo Stock Exchange (6 trading days per week at the time).
For 5-day-week markets, some traders use 7/22/44.
""")

    with st.expander("Signal hierarchy and trade rules", expanded=False):
        st.markdown("""
### Signal strength hierarchy (strongest to weakest)

**1. Perfect Ichimoku Signal (all conditions met)**
- Price above/below cloud
- TK cross (Tenkan crosses Kijun) above/below cloud
- Chikou in free space (no price structure around it)
- Cloud ahead is bullish/bearish (future cloud)

**2. Cloud breakout**
- Price exits cloud after being inside it
- Entry: on close of first candle outside cloud
- Stop: at near edge of cloud
- Target: extreme of previous cloud in trend direction

**3. TK Cross (Tenkan/Kijun intersection)**
- Golden cross: Tenkan crosses above Kijun = bullish
- Dead cross: Tenkan crosses below Kijun = bearish
- Strength depends on WHERE the cross occurs:
  - Above cloud = strong signal
  - Inside cloud = neutral
  - Below cloud = weak signal

**4. Chikou confirmation only**
Weakest signal — only use as confirmation, not standalone entry.

### TP/SL framework
```
Bullish cloud breakout:
  Entry = close of first candle above cloud top
  SL    = cloud bottom (current)
  TP    = highest point of previous cloud above price

Risk = Entry − SL
RRR  = (TP − Entry) / Risk
```
""")


# ══════════════════════════════════════════════════════════════
# TAB 4 — OPTIONS & GREEKS
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("## Options Pricing & The Greeks")

    with st.expander("Black-Scholes-Merton Model (1973)", expanded=True):
        st.markdown(r"""
### The PDE (Black-Scholes Equation)
```
∂V/∂t + (1/2)σ²S²(∂²V/∂S²) + rS(∂V/∂S) − rV = 0
```
This partial differential equation governs the value V of any derivative on S.
Derived by constructing a risk-free portfolio: long the option, short ∂V/∂S shares (delta hedging).

### Assumptions
1. Log-normal price dynamics: dS = μS·dt + σS·dW
2. Constant volatility σ (the main real-world limitation)
3. Continuous trading possible (no transaction costs)
4. Risk-free rate r constant
5. No dividends (BSM), or continuous yield q (Merton extension)
6. European exercise only

### Analytic solution
```
d₁ = [ln(S/K) + (r − q + σ²/2)·T] / (σ·√T)
d₂ = d₁ − σ·√T

Call = S·e^(−qT)·N(d₁) − K·e^(−rT)·N(d₂)
Put  = K·e^(−rT)·N(−d₂) − S·e^(−qT)·N(−d₁)

N(·) = cumulative standard normal distribution
```

### Put-Call Parity
```
C − P = S·e^(−qT) − K·e^(−rT)
```
Arbitrage-free relationship between European call and put prices.
Derivable without any model assumptions.
""")

    with st.expander("The Greeks — complete reference", expanded=False):
        st.markdown("""
### First-order Greeks

**Delta (Δ)** — sensitivity to underlying price
```
Δ_call = e^(−qT) · N(d₁)          range: [0, +1]
Δ_put  = e^(−qT) · (N(d₁) − 1)    range: [−1, 0]
```
Interpretation: Δ = 0.6 means option gains $0.60 per $1 rise in stock.
Also interpreted as approximate probability of expiring in-the-money.

**Vega (ν)** — sensitivity to implied volatility
```
ν = S·e^(−qT)·N'(d₁)·√T
```
Same for calls and puts. Units: $ change per 1% change in IV.
Long options always have positive Vega — you benefit from higher volatility.

**Theta (Θ)** — time decay
```
Θ_call = −[S·N'(d₁)·σ·e^(−qT)/(2√T)] − r·K·e^(−rT)·N(d₂) + q·S·e^(−qT)·N(d₁)
```
Negative for long options (daily cost of holding). Accelerates near expiry, especially ATM.

**Rho (ρ)** — sensitivity to risk-free rate
```
ρ_call = K·T·e^(−rT)·N(d₂)
ρ_put  = −K·T·e^(−rT)·N(−d₂)
```
Small for short-dated options. More relevant for long-dated options (LEAPS).

### Second-order Greeks

**Gamma (Γ)** — rate of change of Delta
```
Γ = N'(d₁)·e^(−qT) / (S·σ·√T)
```
Same for calls and puts. Highest ATM near expiry. Long options = long Gamma = convex payoff.
**Gamma scalping:** if you are long Gamma, you profit by delta-hedging frequently.

**Vanna** — dΔ/dσ (cross-derivative, price × vol)
```
Vanna = −N'(d₁)·d₂ / σ
```
Important for vol trading desks — tells you how your Delta exposure changes as vol moves.

**Charm** — dΔ/dt (time decay of Delta)
```
Charm_call = −e^(−qT)·N'(d₁)·[2(r−q)T − d₂·σ·√T] / (2T·σ·√T)
```
Critical for options approaching expiry — Delta changes rapidly.

**Volga (Vomma)** — dVega/dσ (vol convexity)
```
Volga = Vega · d₁·d₂ / σ
```
Second derivative of price with respect to volatility. Used in volatility arbitrage.

### Greeks summary table
| Greek | Formula | Long option sign | Largest when |
|---|---|---|---|
| Δ | ∂V/∂S | + (call), − (put) | Deep ITM |
| Γ | ∂²V/∂S² | + | ATM, near expiry |
| Θ | ∂V/∂t | − | ATM, near expiry |
| ν (Vega) | ∂V/∂σ | + | ATM, long-dated |
| ρ | ∂V/∂r | + (call), − (put) | Long-dated, ITM |
| Vanna | ∂Δ/∂σ | sign varies | OTM |
| Charm | ∂Δ/∂t | sign varies | Near expiry |
| Volga | ∂ν/∂σ | + | OTM |
""")

    with st.expander("Implied Volatility & Volatility Surface", expanded=False):
        st.markdown("""
### Implied Volatility
BSM gives a closed-form price given σ. Implied volatility inverts this:

```
Market Price = BSM(S, K, T, r, q, σ_implied)
→ solve for σ_implied numerically (Newton-Raphson)
```

### Newton-Raphson IV Solver
```
σ_{n+1} = σ_n − (BSM(σ_n) − Market Price) / Vega(σ_n)
```
Converges in ~5 iterations from a reasonable starting guess.

### Volatility smile / skew
BSM assumes constant σ. In reality, IV varies by strike and maturity.

**Put skew:** OTM puts have higher IV than OTM calls.
Reason: demand for downside protection (portfolio hedging with puts).
This creates a **volatility skew** (IV higher for low strikes).

**Volatility surface:** the 3D surface of IV(K, T).

**Term structure:** IV at different maturities.
Normal: longer-dated IV > short-dated IV (contango)
Inverted: short-dated IV > long-dated IV (backwardation = near-term fear)
""")

    with st.expander("Live Black-Scholes Calculator", expanded=False):
        c1, c2 = st.columns(2)
        S  = c1.number_input("Stock Price (S)", value=100.0, key="bs_S")
        K  = c1.number_input("Strike Price (K)", value=100.0, key="bs_K")
        T  = c1.number_input("Time to Expiry (years)", value=0.25, step=0.05, key="bs_T")
        r  = c1.slider("Risk-free Rate (%)", 0.0, 15.0, 4.5, key="bs_r") / 100
        iv = c2.slider("Implied Volatility (%)", 1.0, 150.0, 25.0, key="bs_iv") / 100
        q  = c2.slider("Dividend Yield (%)", 0.0, 10.0, 0.0, key="bs_q") / 100

        if T > 0:
            d1 = (np.log(S/K) + (r - q + iv**2/2)*T) / (iv*np.sqrt(T))
            d2 = d1 - iv*np.sqrt(T)
            call_px = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            put_px  = K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
            delta_c = np.exp(-q*T)*norm.cdf(d1)
            delta_p = np.exp(-q*T)*(norm.cdf(d1)-1)
            gamma   = np.exp(-q*T)*norm.pdf(d1)/(S*iv*np.sqrt(T))
            vega    = S*np.exp(-q*T)*norm.pdf(d1)*np.sqrt(T)/100
            theta_c = (-(S*norm.pdf(d1)*iv*np.exp(-q*T))/(2*np.sqrt(T))
                       - r*K*np.exp(-r*T)*norm.cdf(d2) + q*S*np.exp(-q*T)*norm.cdf(d1))/365

            ca, cb = st.columns(2)
            ca.markdown("**CALL**")
            ca.metric("Price",  f"${call_px:.4f}")
            ca.metric("Delta",  f"{delta_c:.4f}")
            ca.metric("Gamma",  f"{gamma:.6f}")
            ca.metric("Theta",  f"${theta_c:.4f}/day")
            ca.metric("Vega",   f"${vega:.4f}/1%vol")
            cb.markdown("**PUT**")
            cb.metric("Price",  f"${put_px:.4f}")
            cb.metric("Delta",  f"{delta_p:.4f}")
            cb.metric("Gamma",  f"{gamma:.6f}")
            cb.metric("Theta",  f"${theta_c:.4f}/day")
            cb.metric("Vega",   f"${vega:.4f}/1%vol")
            st.markdown(f"**d₁** = {d1:.4f}  ·  **d₂** = {d2:.4f}  ·  **Put-Call Parity check:** C−P = ${call_px-put_px:.4f}, S·e^(-qT)−K·e^(-rT) = ${S*np.exp(-q*T)-K*np.exp(-r*T):.4f}")


# ══════════════════════════════════════════════════════════════
# TAB 5 — RISK & PERFORMANCE
# ══════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("## Risk & Performance Measurement")

    with st.expander("Return measures", expanded=False):
        st.markdown("""
### Arithmetic vs Geometric Returns
```
Arithmetic: r = (P_t − P_{t−1}) / P_{t−1}
Geometric:  r = ln(P_t / P_{t−1})
```
Geometric returns are additive over time. Arithmetic returns have better statistical properties.

### CAGR (Compound Annual Growth Rate)
```
CAGR = (V_final / V_initial)^(1/n) − 1
```
where n = number of years. The single annual rate that produces the same final value.

### Log-returns and their properties
```
r_log = ln(P_t / P_{t−1})
Multi-period: R_{0,T} = Σᵢ r_{log,i}   (additive)
```
Log-returns are approximately normal (by Central Limit Theorem for many small returns).
""")

    with st.expander("Volatility measures", expanded=False):
        st.markdown("""
### Historical Volatility
```
σ_hist = √[1/(n−1) · Σᵢ(rᵢ − r̄)²] · √252   (annualised from daily returns)
```

### EWMA Volatility (RiskMetrics)
```
σ²_t = λ·σ²_{t−1} + (1−λ)·r²_{t−1}
```
λ = 0.94 (daily, RiskMetrics standard). No mean subtraction — assumes mean ≈ 0 at daily frequency.

### GARCH(1,1) (Bollerslev, 1986)
```
σ²_t = ω + α·ε²_{t−1} + β·σ²_{t−1}
```
Parameters: ω (constant), α (ARCH effect — impact of shocks), β (GARCH effect — persistence)
Long-run variance: σ²_LR = ω / (1 − α − β)
Stationarity requires: α + β < 1
""")

    with st.expander("Sharpe, Sortino, Calmar, Information Ratio", expanded=False):
        st.markdown("""
### Sharpe Ratio (Sharpe, 1966)
```
SR = (R_p − R_f) / σ_p
```
Annualised: multiply numerator by 252, denominator by √252 (for daily returns).
Interpretation: return earned per unit of total risk.

### Sortino Ratio
```
Sortino = (R_p − R_f) / σ_downside
σ_downside = √[1/n · Σᵢ min(rᵢ − MAR, 0)²]
```
MAR = minimum acceptable return (often 0 or R_f).
Only penalises downside deviation — more appropriate for asymmetric return distributions.

### Calmar Ratio
```
Calmar = Annualised Return / |Maximum Drawdown|
```

### Information Ratio
```
IR = (R_p − R_b) / TE
TE = tracking error = σ(R_p − R_b)
```
Measures active return per unit of active risk.
IR > 0.5 is considered good. IR > 1.0 is excellent.

### Maximum Drawdown
```
DD(t) = (V(t) − max_{s≤t} V(s)) / max_{s≤t} V(s)
MDD   = min_t DD(t)
```
Largest peak-to-trough decline. Used in Calmar ratio.
""")

    with st.expander("VaR and CVaR / Expected Shortfall", expanded=False):
        st.markdown("""
### Value at Risk (VaR)
```
P(R < −VaR_α) = 1 − α

Parametric (normal): VaR_α = μ − σ·Φ⁻¹(α)   e.g. 95% VaR = μ − 1.645σ
Historical:          VaR_α = −quantile(returns, 1−α)
Monte Carlo:         simulate paths, take (1−α) percentile of terminal values
```
Limitation: VaR is not sub-additive — combining portfolios can increase VaR (Basel 3 moved to ES).

### Expected Shortfall (ES / CVaR)
```
ES_α = E[−R | R < −VaR_α] = −(1/(1−α)) · ∫_{−∞}^{VaR_α} r·f(r)dr
```
Average loss in the worst (1−α)% of scenarios.
**Sub-additive** — diversification always reduces ES. Used in Basel 3/4.

### Coherent Risk Measures
A risk measure ρ is coherent if it satisfies:
1. **Monotonicity**: if X ≤ Y always, then ρ(X) ≥ ρ(Y)
2. **Sub-additivity**: ρ(X+Y) ≤ ρ(X) + ρ(Y)
3. **Positive homogeneity**: ρ(λX) = λρ(X) for λ > 0
4. **Translation invariance**: ρ(X+a) = ρ(X) − a
VaR fails sub-additivity. ES satisfies all four.
""")

    with st.expander("Information Coefficient (IC) — alpha evaluation", expanded=False):
        st.markdown("""
### IC — Spearman Rank Correlation of Signal vs Forward Return
```
IC = ρ_Spearman(signal_t, return_{t+h})
   = 1 − 6·Σdᵢ² / [n(n²−1)]
```
where dᵢ = difference in ranks between signal and return for asset i.

### IC IR — Information Ratio of the IC
```
IC_IR = mean(IC) / std(IC)
```
Measures consistency of the signal, not just average strength.

### Benchmarks
| Metric | Threshold | Interpretation |
|---|---|---|
| IC > 0.10 | Strong | Institutional-quality alpha |
| IC > 0.05 | Meaningful | Worth trading with risk management |
| IC < 0.05 | Weak | Likely noise |
| IC IR > 0.5 | Consistent | Signal works reliably, not just on average |

### Fundamental Law of Active Management (Grinold, 1989)
```
IR = IC · √BR
```
IR = Information Ratio, IC = Information Coefficient, BR = breadth (number of independent bets per year).
Higher IC or more independent bets = better IR.
""")


# ══════════════════════════════════════════════════════════════
# TAB 6 — QUANT FORMULAS
# ══════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("## Quantitative Finance — Core Formulas")

    with st.expander("Stochastic Calculus & Ito's Lemma", expanded=False):
        st.markdown(r"""
### Brownian Motion (Wiener Process)
A stochastic process W(t) is standard Brownian motion if:
1. W(0) = 0
2. Increments W(t) − W(s) ~ N(0, t−s) for t > s
3. Increments on non-overlapping intervals are independent
4. Sample paths are continuous (but nowhere differentiable)

Key property: E[dW²] = dt (quadratic variation is deterministic)

### Geometric Brownian Motion (GBM)
```
dS = μS·dt + σS·dW

Solution:
S(T) = S(0) · exp[(μ − σ²/2)T + σ·W(T)]

E[S(T)] = S(0)·e^(μT)
Var[S(T)] = S(0)²·e^(2μT)·(e^(σ²T) − 1)
```
The (μ − σ²/2) term is the Ito correction.

### Ito's Lemma
For a function f(S, t) where dS = a·dt + b·dW:
```
df = (∂f/∂t + a·∂f/∂S + (1/2)b²·∂²f/∂S²)dt + b·∂f/∂S·dW
```
This is the stochastic chain rule. The extra (1/2)b²·∂²f/∂S² term arises because (dW)² = dt.

### Applying Ito to derive log-return dynamics
Let f = ln(S):
```
∂f/∂S = 1/S,  ∂²f/∂S² = −1/S²,  ∂f/∂t = 0

d(ln S) = (μ − σ²/2)dt + σ·dW
```
Log-returns are normally distributed with drift (μ − σ²/2) and variance σ².
""")

    with st.expander("Ornstein-Uhlenbeck (Mean Reversion)", expanded=False):
        st.markdown("""
### OU Process (Vasicek model for interest rates)
```
dX = κ(θ − X)dt + σ·dW
```
Parameters:
- κ = speed of mean reversion
- θ = long-run mean (equilibrium)
- σ = volatility

### Analytical solution
```
X(t) = θ + (X(0) − θ)·e^(−κt) + σ·∫₀ᵗ e^(−κ(t−s)) dW(s)

E[X(t)] = θ + (X(0)−θ)·e^(−κt)    → θ as t→∞
Var[X(t)] = σ²/(2κ)·(1−e^(−2κt)) → σ²/(2κ) as t→∞
```

### Half-life
```
t_{1/2} = ln(2) / κ
```
Time for deviation from mean to decay by half on average. Used in pairs trading.

### Estimating κ from data (OLS)
```
ΔX_t = κ(θ − X_{t−1})Δt + σ·ΔW
→ regress ΔX_t on X_{t-1} to estimate κ and θ
```
""")

    with st.expander("CAPM and Fama-French Factor Models", expanded=False):
        st.markdown("""
### CAPM (Sharpe 1964, Lintner 1965, Mossin 1966)
```
E[R_i] = R_f + β_i·(E[R_m] − R_f)

β_i = Cov(R_i, R_m) / Var(R_m)
```
Security Market Line (SML): all assets plot on this line in equilibrium.
Alpha = intercept = excess return above CAPM prediction.

### Fama-French Three-Factor Model (1993)
```
R_i − R_f = α + β₁·MKT + β₂·SMB + β₃·HML + ε

MKT = R_m − R_f               (market excess return)
SMB = Small Minus Big          (size factor: small cap − large cap)
HML = High Minus Low           (value factor: high B/M − low B/M)
```

### Carhart Four-Factor Model (1997)
Adds momentum:
```
R_i − R_f = α + β₁·MKT + β₂·SMB + β₃·HML + β₄·MOM + ε

MOM = momentum factor (past 12-1 month winners − losers)
```

### Fama-French Five-Factor Model (2015)
Adds profitability (RMW) and investment (CMA):
```
R_i − R_f = α + β₁·MKT + β₂·SMB + β₃·HML + β₄·RMW + β₅·CMA + ε

RMW = Robust Minus Weak  (operating profitability)
CMA = Conservative Minus Aggressive  (investment)
```

### OLS Estimation
```
β = (X'X)⁻¹X'y
Standard errors: se(β) = √[σ²·diag((X'X)⁻¹)]
t-statistic: t = β / se(β)  ~ t(n−k)
R² = 1 − SSR/SST = 1 − Σeᵢ²/Σ(yᵢ−ȳ)²
```
""")

    with st.expander("Portfolio Theory — Markowitz (1952)", expanded=False):
        st.markdown("""
### Portfolio Return and Variance
For n assets with weights w = [w₁,...,wₙ], returns μ = [μ₁,...,μₙ], covariance matrix Σ:
```
E[R_p] = w'μ
Var(R_p) = w'Σw
```

### Minimum Variance Portfolio
```
min_w  w'Σw
s.t.   w'1 = 1,  w'μ = target return  (if any)
```
Solved with Lagrange multipliers or SLSQP.

### Efficient Frontier
Set of portfolios with maximum return for a given risk. Parameterised by target return.
Capital Market Line (CML): tangent line from risk-free rate to efficient frontier.

### Tangency (Maximum Sharpe) Portfolio
```
w* = (Σ⁻¹(μ−r_f·1)) / (1'Σ⁻¹(μ−r_f·1))
```

### Diversification — intuition
```
Var(R_p) = (1/n)·σ̄² + (1−1/n)·C̄ov
```
As n→∞, idiosyncratic risk vanishes. Only systematic (correlated) risk remains.
""")

    with st.expander("Fixed Income — Duration, Convexity, Yield", expanded=False):
        st.markdown("""
### Bond Pricing
```
P = Σₜ C/(1+y)ᵗ + F/(1+y)ᵀ

C = coupon payment, F = face value, y = yield to maturity, T = maturity
```

### Macaulay Duration
```
D = (1/P) · Σₜ t · PV(CF_t)  =  Σₜ [t · CF_t/(1+y)ᵗ] / P
```
Weighted average time to receive cash flows. Measures interest rate sensitivity in years.

### Modified Duration
```
D_mod = D / (1+y)
ΔP/P ≈ −D_mod · Δy
```

### Convexity
```
Cx = (1/P) · d²P/dy² = Σₜ [t(t+1) · CF_t/(1+y)^(t+2)] / P
ΔP/P ≈ −D_mod·Δy + (1/2)·Cx·(Δy)²
```
Convexity is always positive for plain bonds → price falls less than duration predicts when yields rise.

### Yield Curve shapes
- **Normal (upward sloping):** long-term rates > short-term → growth expected
- **Inverted:** short-term > long-term → recession signal
- **Flat:** transitioning between states
- **Humped:** medium-term rates highest (uncommon)
""")

    with st.expander("Piotroski F-Score — all 9 signals with formulas", expanded=False):
        st.markdown("""
### Piotroski F-Score (Joseph Piotroski, 2000)
Published in *Journal of Accounting Research*. Predicts 1-year ahead returns using accounting signals.

**Profitability (4 signals)**
```
F1: ROA_t > 0               where ROA = Net Income / Total Assets
F2: OCF_t > 0               where OCF = Operating Cash Flow
F3: ΔROA > 0                ROA_t > ROA_{t-1}
F4: Accrual < 0             (OCF/Assets) > ROA  ← cash quality signal
```

**Leverage / Liquidity (3 signals)**
```
F5: ΔLeverage < 0          Long-term Debt / Assets decreased
F6: ΔCurrent Ratio > 0     (Current Assets / Current Liabilities) improved
F7: No dilution             Shares_t ≤ Shares_{t-1}
```

**Operating Efficiency (2 signals)**
```
F8: ΔGross Margin > 0      Gross Profit / Revenue improved
F9: ΔAsset Turnover > 0    Revenue / Total Assets improved
```

**Score interpretation:**
8-9 = long signal, 0-2 = short signal.
Original paper: long high-F, short low-F → +23% annual return 1976-1996 (US).
""")

    with st.expander("Altman Z-Score — derivation and components", expanded=False):
        st.markdown("""
### Altman Z-Score (Edward Altman, 1968)
Multiple discriminant analysis on financial ratios to predict bankruptcy within 2 years.
Original sample: 66 US manufacturing firms (33 bankrupt, 33 healthy).

**Model:**
```
Z = 1.2·X₁ + 1.4·X₂ + 3.3·X₃ + 0.6·X₄ + 1.0·X₅
```

**Variables:**
```
X₁ = Working Capital / Total Assets           (short-term liquidity)
X₂ = Retained Earnings / Total Assets         (reinvestment and profitability history)
X₃ = EBIT / Total Assets                      (operating profitability, pre-leverage)
X₄ = Market Value of Equity / Total Liabilities (leverage — market-based)
X₅ = Revenue / Total Assets                   (asset utilisation efficiency)
```

**Zones:**
```
Z > 2.99    Safe zone     — low bankruptcy probability
1.81-2.99   Grey zone     — uncertain, monitor closely
Z < 1.81    Distress zone — high bankruptcy probability
```

**Coefficients:** derived from linear discriminant analysis — chosen to maximise separation between bankrupt and non-bankrupt firms.

**Limitations:** calibrated on 1960s US manufacturing. Variants exist for private firms (Z'), non-manufacturing (Z''), and emerging markets.
""")


# ══════════════════════════════════════════════════════════════
# TAB 7 — FUNDAMENTAL ANALYSIS
# ══════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("## Fundamental Analysis — Valuation & Accounting")

    with st.expander("Valuation Multiples — formulas and benchmarks", expanded=False):
        st.markdown("""
### Price/Earnings (P/E) Ratio
```
P/E = Market Price per Share / Earnings Per Share (EPS)
    = Market Capitalisation / Net Income
```
- TTM (trailing): last 12 months earnings
- Forward: next 12 months estimated earnings
- **Earnings yield** = 1/P/E (compare to bond yields)

### Price/Sales (P/S)
```
P/S = Market Cap / Revenue
```
Useful for unprofitable companies. Less manipulable than earnings.

### Price/Book (P/B)
```
P/B = Market Cap / Book Value of Equity
    = Market Price / (Total Assets − Intangibles − Liabilities)
```
P/B < 1 means market values company below accounting net worth.

### EV/EBITDA
```
EV = Market Cap + Total Debt − Cash and Equivalents
EV/EBITDA = Enterprise Value / EBITDA
```
Capital-structure neutral. More appropriate than P/E for comparing leveraged companies.

### PEG Ratio
```
PEG = P/E / Annual EPS Growth Rate
```
PEG < 1: potentially undervalued relative to growth.
PEG > 2: potentially overvalued.

### Dividend Discount Model (DDM)
```
P = D₁ / (r − g)     (Gordon Growth Model)
```
D₁ = next year's dividend, r = required return, g = sustainable growth rate.

### Return metrics
```
ROE = Net Income / Shareholders' Equity
ROA = Net Income / Total Assets
ROIC = NOPAT / Invested Capital    (most comprehensive)
```
DuPont decomposition:
```
ROE = (Net Income/Revenue) · (Revenue/Assets) · (Assets/Equity)
    = Net Margin · Asset Turnover · Financial Leverage
```
""")

    with st.expander("DCF Valuation — theory and formula", expanded=False):
        st.markdown("""
### Discounted Cash Flow (DCF)
The intrinsic value of a business = present value of all future free cash flows.

```
Value = Σₜ₌₁ᵀ FCF_t / (1+WACC)ᵗ + Terminal Value / (1+WACC)ᵀ
```

### Free Cash Flow
```
FCFF (to firm)   = EBIT·(1−tax) + D&A − ΔWorking Capital − Capex
FCFE (to equity) = Net Income + D&A − ΔWorking Capital − Capex + Net Borrowing
```

### Terminal Value
```
Gordon Growth: TV = FCF_T · (1+g) / (WACC − g)
Exit Multiple: TV = EBITDA_T · EV/EBITDA_multiple
```

### WACC
```
WACC = (E/V)·r_e + (D/V)·r_d·(1−tax)

r_e = R_f + β·(R_m − R_f)    (from CAPM)
E/V = equity / (equity + debt)
```

### Sensitivity analysis
DCF is highly sensitive to WACC and terminal growth rate.
Always present as a range (sensitivity table) not a single number.
""")

    with st.expander("Financial Statements — structure and relationships", expanded=False):
        st.markdown("""
### Income Statement
```
Revenue
− Cost of Goods Sold (COGS)
= Gross Profit
− Operating Expenses (SG&A, R&D)
= EBIT (Operating Income)
− Interest Expense
= EBT (Pre-tax Income)
− Taxes
= Net Income

EPS = Net Income / Diluted Shares Outstanding
EBITDA = EBIT + Depreciation + Amortisation
```

### Balance Sheet
```
Assets = Liabilities + Shareholders' Equity

Assets:     Current (cash, receivables, inventory) + Non-current (PP&E, intangibles)
Liabilities: Current (AP, short-term debt) + Long-term (bonds, deferred tax)
Equity:     Common stock + Retained Earnings + AOCI
```

### Cash Flow Statement
```
Operating CF  = Net Income + non-cash items (D&A) − ΔWorking Capital
Investing CF  = −Capex + asset sales + acquisitions
Financing CF  = Debt raised − repaid + equity raised − dividends − buybacks

Net Change in Cash = OCF + ICF + FCF
```

### The three statements link
```
Net Income (IS) → Retained Earnings (BS)
Net Income (IS) → Starting point of Operating CF (CFS)
Capex (CFS)     → PP&E increase (BS)
D&A (CFS addback) → PP&E decrease (BS)
```
""")


# ══════════════════════════════════════════════════════════════
# TAB 8 — CALCULATORS
# ══════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown("## Calculators")

    calc = st.selectbox("Choose calculator", [
        "Position Size & Risk",
        "Kelly Criterion",
        "RRR → Breakeven Win Rate",
        "CAGR & Compound Growth",
        "Bond Duration & Price",
        "Sharpe / Sortino from returns",
        "DCF — Quick Valuation",
        "Pip Value (FX)",
    ])

    if calc == "Position Size & Risk":
        c1,c2 = st.columns(2)
        acc   = c1.number_input("Account ($)", value=10_000.0)
        risk  = c1.slider("Risk per trade (%)", 0.25, 5.0, 1.0, step=0.25)
        entry = c2.number_input("Entry price", value=100.0, format="%.5f")
        sl    = c2.number_input("Stop Loss price", value=98.0, format="%.5f")
        pip_v = c2.number_input("$ per unit (pip value or 1 share)", value=1.0)
        risk_amt = acc*risk/100
        sl_dist  = abs(entry - sl)
        units    = risk_amt / sl_dist if sl_dist > 0 else 0
        c1.metric("Risk Amount",  f"${risk_amt:,.2f}")
        c1.metric("SL Distance",  f"{sl_dist:.5g}")
        c2.metric("Max Units",    f"{units:,.1f}")
        c2.metric("Kelly-adj (55% WR, 2:1 RRR)", f"{max(0,0.55-0.45/2.0)*100:.1f}% of capital")

    elif calc == "Kelly Criterion":
        st.markdown("Optimal fraction of capital to risk per trade.")
        wr = st.slider("Win Rate (%)", 30, 80, 55) / 100
        b  = st.slider("Reward:Risk", 0.5, 5.0, 2.0, step=0.25)
        f  = wr - (1-wr)/b
        st.metric("Full Kelly", f"{f*100:.1f}%")
        st.metric("Half Kelly (recommended)", f"{f/2*100:.1f}%")
        st.metric("Expected value per $1 risked", f"{f*b-(1-f)*1:.3f}")
        if f <= 0:
            st.error("Negative Kelly — this bet has negative expected value.")

    elif calc == "RRR → Breakeven Win Rate":
        rrr = st.slider("Reward:Risk Ratio", 0.5, 5.0, 2.0, step=0.25)
        be  = 1/(1+rrr)
        st.metric("Breakeven Win Rate", f"{be*100:.1f}%")
        wrs = np.arange(0.3, 0.75, 0.05)
        ev  = [w*rrr - (1-w) for w in wrs]
        fig = go.Figure(go.Bar(x=[f"{w*100:.0f}%" for w in wrs], y=ev,
            marker_color=[GREEN if v>0 else RED for v in ev],
            text=[f"{v:+.2f}R" for v in ev], textposition="outside"))
        fig.add_hline(y=0)
        fig.update_layout(plot_bgcolor=BG, paper_bgcolor=BG,
            font=dict(color=TEXT), height=300, title=f"Expected Value at {rrr}:1 RRR")
        st.plotly_chart(fig, use_container_width=True)

    elif calc == "CAGR & Compound Growth":
        c1,c2 = st.columns(2)
        start  = c1.number_input("Starting capital ($)", value=10_000.0)
        ret    = c1.slider("Annual return (%)", -20, 100, 20)
        yrs    = c2.slider("Years", 1, 40, 10)
        monthly= c2.number_input("Monthly contribution ($)", value=0.0)
        r = ret/100
        vals = [start]
        for y in range(yrs):
            vals.append(vals[-1]*(1+r) + monthly*12)
        st.metric("Final Value",  f"${vals[-1]:,.0f}")
        st.metric("Total Return", f"{(vals[-1]/start-1)*100:.1f}%")
        fig2 = go.Figure(go.Scatter(y=vals, mode="lines+markers",
            line=dict(color=CYAN, width=2)))
        fig2.update_layout(plot_bgcolor=BG, paper_bgcolor=BG,
            font=dict(color=TEXT), height=280, title="Portfolio Growth")
        st.plotly_chart(fig2, use_container_width=True)

    elif calc == "Bond Duration & Price":
        c1,c2 = st.columns(2)
        face   = c1.number_input("Face Value ($)", value=1000.0)
        coupon = c1.slider("Coupon Rate (%)", 0.0, 15.0, 5.0, step=0.25) / 100
        ytm    = c2.slider("YTM (%)", 0.5, 20.0, 5.0, step=0.25) / 100
        T      = c2.slider("Maturity (years)", 1, 30, 10)
        c_pay  = face * coupon
        times  = np.arange(1, T+1)
        pvs    = c_pay / (1+ytm)**times
        pvs[-1] += face / (1+ytm)**T
        price  = pvs.sum()
        mac_d  = (pvs * times).sum() / price
        mod_d  = mac_d / (1+ytm)
        convex = ((pvs * times * (times+1)).sum()) / (price * (1+ytm)**2)
        c1.metric("Bond Price",       f"${price:,.2f}")
        c1.metric("Macaulay Duration",f"{mac_d:.3f} years")
        c2.metric("Modified Duration",f"{mod_d:.3f}")
        c2.metric("Convexity",        f"{convex:.3f}")
        shock = st.slider("Rate shock (bps)", -200, 200, -50)
        dy = shock/10000
        dp_lin = -mod_d * dy * price
        dp_conv= dp_lin + 0.5 * convex * dy**2 * price
        st.markdown(f"**{shock:+}bps shock:** Linear ΔP = ${dp_lin:+.2f} · With convexity = ${dp_conv:+.2f}")

    elif calc == "Sharpe / Sortino from returns":
        st.caption("Paste comma-separated daily returns (e.g. 0.01,-0.005,0.02)")
        raw = st.text_area("Daily returns", value="0.01,-0.005,0.02,0.008,-0.003,0.015,-0.001,0.012")
        rf  = st.slider("Risk-free rate (annual %)", 0.0, 10.0, 4.5) / 100 / 252
        try:
            rets = np.array([float(x.strip()) for x in raw.split(",")])
            excess = rets - rf
            sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
            down   = rets[rets < 0]
            sortino= excess.mean() / down.std() * np.sqrt(252) if len(down)>1 and down.std()>0 else 0
            c1,c2,c3 = st.columns(3)
            c1.metric("Sharpe (ann)",  f"{sharpe:.3f}")
            c2.metric("Sortino (ann)", f"{sortino:.3f}")
            c3.metric("Ann. Return",   f"{rets.mean()*252*100:.2f}%")
        except: st.error("Invalid input — use comma-separated numbers")

    elif calc == "DCF — Quick Valuation":
        c1,c2 = st.columns(2)
        rev    = c1.number_input("Latest Revenue ($M)", value=50_000.0)
        g1     = c1.slider("Growth yr 1-5 (%)", 0, 50, 15) / 100
        g2     = c1.slider("Growth yr 6-10 (%)", 0, 30, 8) / 100
        fm     = c2.slider("FCF Margin (%)", 1, 50, 20) / 100
        w      = c2.slider("WACC (%)", 5, 20, 10) / 100
        gt     = c2.slider("Terminal Growth (%)", 0, 5, 3) / 100
        shares = c1.number_input("Shares Outstanding (M)", value=1_000.0)
        r = rev
        pv = 0
        for yr in range(1, 11):
            r *= (1+g1) if yr <= 5 else (1+g2)
            pv += r*fm / (1+w)**yr
        tv = r*fm*(1+gt)/(w-gt) / (1+w)**10 if w > gt else 0
        iv = (pv + tv) / shares
        c1.metric("Intrinsic Value / Share", f"${iv:.2f}")
        c2.metric("Enterprise Value", f"${(pv+tv)/1e3:.1f}B")

    elif calc == "Pip Value (FX)":
        pair  = st.selectbox("Pair", ["EURUSD","GBPUSD","USDJPY","XAUUSD","GBPJPY","AUDUSD"])
        lots  = st.number_input("Lot size", value=0.1, step=0.01)
        pip_m = {"EURUSD":0.0001,"GBPUSD":0.0001,"USDJPY":0.01,"XAUUSD":0.1,"GBPJPY":0.01,"AUDUSD":0.0001}
        pip_v = {"EURUSD":10,"GBPUSD":10,"USDJPY":9.2,"XAUUSD":10,"GBPJPY":9.0,"AUDUSD":10}
        pv = pip_v.get(pair, 10)
        st.metric(f"Pip Value ({pair})", f"${pv*lots:.2f} per pip")
        st.markdown(f"1 pip = {pip_m.get(pair,0.0001)}")


# ══════════════════════════════════════════════════════════════
# TAB 9 — FINANCE CONCEPTS
# ══════════════════════════════════════════════════════════════
with tabs[8]:
    st.markdown("## Finance Concepts — Explained")

    with st.expander("Efficient Market Hypothesis (EMH)", expanded=False):
        st.markdown("""
### Three forms (Fama, 1970)

**Weak form:** Prices reflect all past price information.
Technical analysis cannot generate consistent alpha.
Supported by: random walk tests, autocorrelation tests.

**Semi-strong form:** Prices reflect all publicly available information.
Fundamental analysis cannot generate consistent alpha.
Supported by: event studies (prices react immediately to news).

**Strong form:** Prices reflect all information, including private.
Even insiders cannot generate alpha.
Not supported: insider trading laws exist because insiders DO profit.

### Implications
Under weak-form EMH:
- Future prices are not predictable from past prices
- Price changes are approximately random walks

### Counter-evidence (market anomalies)
- Momentum factor (Jegadeesh & Titman, 1993)
- Value premium (Fama & French, 1992)
- Size premium
- Post-earnings announcement drift (PEAD)
- Calendar effects (January effect)
""")

    with st.expander("Arbitrage Pricing Theory (APT)", expanded=False):
        st.markdown("""
### Ross (1976) — Alternative to CAPM
Expected return is a linear function of multiple systematic risk factors:
```
E[R_i] = R_f + β_{i,1}·λ₁ + β_{i,2}·λ₂ + ... + β_{i,k}·λₖ
```
λ_j = risk premium for factor j
β_{i,j} = sensitivity of asset i to factor j

### Key insight
No-arbitrage condition → expected returns must be linear in factor exposures.

### Compared to CAPM
CAPM: one factor (market). APT: multiple factors, unspecified.
APT is more general but doesn't specify what the factors are.
Fama-French models implement APT with specified empirical factors.
""")

    with st.expander("Market Microstructure — bid-ask, slippage, liquidity", expanded=False):
        st.markdown("""
### Bid-Ask Spread
```
Spread = Ask − Bid
Mid    = (Ask + Bid) / 2
```
Market maker earns the spread. Buyer pays Ask, seller receives Bid.
Effective spread accounts for price impact.

### Components of the spread
1. **Inventory cost**: market maker holds unwanted inventory
2. **Order processing**: administrative costs
3. **Adverse selection**: fear of trading with informed traders

### Price impact (Almgren-Chriss model)
```
ΔP = η·X + γ·v
η = temporary impact coefficient (per unit traded)
γ = permanent impact (information)
v = trading velocity (shares per unit time)
X = trade size
```
Large orders move the market against you.

### Transaction costs in backtesting
```
Net Return = Gross Return − TC
TC = (Spread/2 + Commission) × Turnover + Market Impact
```
Ignoring TC makes strategies look better than they are.
""")

    with st.expander("Risk Premia — why factors have positive expected returns", expanded=False):
        st.markdown("""
### Why do systematic factors earn positive returns?
Three explanations for each factor:

**Market premium (CAPM)**
- Risk-based: undiversifiable systematic risk
- Investors require compensation for bearing market risk

**Value premium (HML)**
- Risk-based: value stocks are distressed, have higher fundamental risk
- Behavioural: investors overpay for glamour/growth stocks

**Size premium (SMB)**
- Risk-based: small caps are less liquid, harder to monitor, higher distress risk
- Behavioural: neglect of small caps by institutional investors

**Momentum (MOM)**
- Behavioural: investor under-reaction to news → slow drift → momentum
- Not risk-based (crashes during recessions)

### Factor zoo
As of 2020, over 400 factors have been published in academic literature.
Harvey, Liu, Zhu (2016): most factors are likely false discoveries (multiple testing problem).
Requires t-stat > 3.0 (not 2.0) to account for data mining.
""")

    with st.expander("Volatility Arbitrage — vol surface trading", expanded=False):
        st.markdown("""
### The basic idea
If realised volatility will differ from implied volatility, you can profit by:
1. Buying options (long vol) if you expect realised > implied
2. Selling options (short vol) if you expect realised < implied
Then delta-hedge to isolate the vol exposure.

### P&L of a delta-hedged option position
```
P&L ≈ (1/2) · Γ · S² · (σ²_realised − σ²_implied) · dt
```
Gamma (Γ) is always positive for long options.
If realised_vol > implied_vol: long option P&L > 0.
If realised_vol < implied_vol: short option P&L > 0.

### Vol surface arbitrage
- Calendar spread: buy near-term vol, sell far-term vol (or vice versa)
- Skew trading: buy cheap OTM options, sell expensive ones
- Dispersion trading: sell index vol, buy single-stock vol (index vol > component vol on average)

### VIX and variance swaps
```
Variance swap payoff = N · (σ²_realised − K_var)
K_var ≈ VIX² / 100²     (VIX = expected √(annual variance))
```
Pure play on realised vs implied variance.
""")