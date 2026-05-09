#app.py

# Encyclopedia & Calculator
# Every formula, indicator, concept from all projects explained from scratch.
# Run with: streamlit run app.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math

st.set_page_config(page_title="Encyclopedia", page_icon="📚", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&family=Source+Serif+4:wght@300;400;600&display=swap');
html, body, [class*="css"] { background-color: #06080A; color: #D0D8E0; font-family: 'Source Serif 4', serif; }
h1 { font-family: 'Fira Code', monospace !important; color: #FFD700 !important; font-size: 1.8rem !important; }
h2 { font-family: 'Fira Code', monospace !important; color: #00CCFF !important; font-size: 1.1rem !important; }
h3 { font-family: 'Fira Code', monospace !important; color: #00FF88 !important; font-size: 0.9rem !important; }
code { font-family: 'Fira Code', monospace !important; background: #0A1218; padding: 2px 6px; border-radius: 3px; color: #00FF88; }
[data-testid="stSidebar"] { background: #040608; border-right: 1px solid #0E1820; }
.stTabs [data-baseweb="tab"] { font-family: 'Fira Code', monospace; font-size: 0.65rem; background: #0A1018; border-radius: 3px; border: 1px solid #0E2030; color: #1A3040; }
.stTabs [aria-selected="true"] { background: #0E2030 !important; border-color: #FFD700 !important; color: #FFD700 !important; }
hr { border-color: #0E1820 !important; }
[data-testid="metric-container"] { background: #0A1018; border: 1px solid #0E2030; border-radius: 3px; padding: 10px 14px; }
[data-testid="metric-container"] [data-testid="metric-value"] { font-family: 'Fira Code', monospace !important; color: #FFD700 !important; }
</style>
""", unsafe_allow_html=True)

GOLD="#FFD700"; CYAN="#00CCFF"; GREEN="#00FF88"; RED="#FF4466"
BG="#06080A"; GRID="#0E1820"; TEXT="#D0D8E0"

st.markdown("# 📚 Encyclopedia & Calculator")
st.markdown("`Every formula, indicator, and concept explained from scratch.`")
st.markdown("---")

tab_list = [
    "📐  Indicators",
    "📈  Ichimoku",
    "🧮  Calculators",
    "🎯  Options & Greeks",
    "📊  Risk Metrics",
    "🔢  Quant Formulas",
    "📡  Your Strategies",
    "💡  Concepts",
]
tabs = st.tabs(tab_list)

# ── TAB 1: Indicators ─────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("## Technical Indicators — From Scratch")

    with st.expander("RSI — Relative Strength Index", expanded=False):
        st.markdown("""
### RSI (Welles Wilder, 1978)
Measures the speed and magnitude of recent price changes. Tells you if a market is overbought or oversold.

**Formula:**
```
RS  = Average Gain over N periods / Average Loss over N periods
RSI = 100 - (100 / (1 + RS))
```

**Parameters:** N = 14 (default), 8 (Mark I — more sensitive), 2 (very short-term)

**Interpretation:**
- RSI > 70: overbought — price may reverse down
- RSI < 30: oversold — price may reverse up
- RSI crossing 50: trend change confirmation
- RSI divergence: price makes new high but RSI doesn't → weakness signal

**Your Mark I rule:** RSI8 should be on same side of 50 as the trend direction. On PSAR flip, RSI must cross 50 OR be in extreme zone (>70 or <30).
""")

    with st.expander("CCI — Commodity Channel Index", expanded=False):
        st.markdown("""
### CCI (Donald Lambert, 1980)
Measures how far price is from its average. Originally for commodities, used everywhere now.

**Formula:**
```
Typical Price (TP) = (High + Low + Close) / 3
SMA_TP             = Simple Moving Average of TP over N periods
Mean Deviation     = Average of |TP - SMA_TP| over N periods
CCI                = (TP - SMA_TP) / (0.015 × Mean Deviation)
```

**Parameters:** N = 14 (ICH+CCI note), 20, 25 (you like 25), 50, 100

**Interpretation:**
- CCI > +100: strong uptrend or overbought
- CCI < -100: strong downtrend or oversold
- CCI crossing +100 from below: bullish breakout signal
- CCI crossing -100 from above: bearish breakdown signal
- CCI crossing 0 (midline): trend direction change

**Your ICH+CCI rule:** CCI crosses ±100 limit lines in the direction of the Ichimoku cloud breakout = strong entry signal.
""")

    with st.expander("Williams %R (WPR)", expanded=False):
        st.markdown("""
### Williams %R (Larry Williams, 1973)
Oscillator showing where price is relative to the highest high over N periods.

**Formula:**
```
%R = -100 × (Highest High over N - Close) / (Highest High over N - Lowest Low over N)
```

**Parameters:** N = 14 (default)

**Range:** 0 to -100 (note: inverted — higher = weaker)

**Interpretation:**
- %R from -20 to 0: overbought zone
- %R from -80 to -100: oversold zone
- Entering/exiting extreme zones = signals

**Synergy with CCI (your Mark II):** Both entering extreme zone in same direction = high conviction.
""")

    with st.expander("PSAR — Parabolic SAR", expanded=False):
        st.markdown("""
### Parabolic SAR (Welles Wilder, 1978)
A trailing stop-and-reverse system. Dots appear above or below price to show trend direction.

**Formula:**
```
Uptrend:
  SAR(t) = SAR(t-1) + AF × (EP - SAR(t-1))
  EP = highest high since uptrend started
  AF starts at 0.02, increases by 0.02 each new high, max 0.2

Downtrend: mirror of above using lowest low
```

**Your setting:** AF step = 0.04, maximum = 0.2

**Mark I rule:**
1. PSAR dots flip (change from above to below price, or vice versa)
2. RSI is in extreme zone (>70 or <30) OR just crossed 50
3. Previous candle colour matches new PSAR direction
4. Previous candle is NOT a doji or hammer
→ Take the trade
""")

    with st.expander("MACD — Moving Average Convergence Divergence", expanded=False):
        st.markdown("""
### MACD (Gerald Appel, 1979)
Shows the relationship between two EMAs. Measures momentum and direction.

**Formula:**
```
MACD Line   = EMA(12) - EMA(26)
Signal Line = EMA(9) of MACD Line
Histogram   = MACD Line - Signal Line
```

**Interpretation:**
- MACD crosses above Signal: bullish
- MACD crosses below Signal: bearish
- Histogram growing: momentum increasing
- Divergence: price makes new high, MACD doesn't = weakness

**Default:** 12, 26, 9
""")

    with st.expander("Bollinger Bands", expanded=False):
        st.markdown("""
### Bollinger Bands (John Bollinger, 1980s)
Adaptive bands that expand during volatility and contract during calm.

**Formula:**
```
Middle Band = SMA(20)
Upper Band  = SMA(20) + 2 × StdDev(20)
Lower Band  = SMA(20) - 2 × StdDev(20)
```

**Your use:** BB cloud (50, 2) or (20, 3). Price outside bands + reversal candle = entry. STR stop at BB midline.

**Squeeze:** when bands are very narrow = low volatility = big move coming
""")

    with st.expander("ATR — Average True Range", expanded=False):
        st.markdown("""
### ATR (Welles Wilder, 1978)
Measures market volatility. Does NOT tell you direction — only magnitude of moves.

**Formula:**
```
True Range (TR) = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
ATR = Rolling mean of TR over 14 periods
```

**Uses:**
- Setting stop losses: SL = Entry ± (ATR × multiplier)
- Position sizing: smaller position when ATR is large
- Volatility filter: avoid trading when ATR is unusually low (consolidation)

**TP/SL in this dashboard:** SL = ATR × 1.5, TP = SL × RRR
""")


# ── TAB 2: Ichimoku ───────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("## Ichimoku Kinko Hyo — 'One Look Equilibrium Chart'")
    st.markdown("*Created by Goichi Hosoda, Japanese journalist, published 1969 after 30 years of development.*")

    with st.expander("The five lines explained", expanded=True):
        st.markdown("""
### The 5 Components

**Tenkan-sen (Conversion Line / CL)**
```
Tenkan = (Highest High + Lowest Low) / 2   over 9 periods
```
Short-term momentum line. Think of it as a faster moving average.
Acts as support/resistance on pullbacks.

**Kijun-sen (Base Line / BL)**
```
Kijun = (Highest High + Lowest Low) / 2   over 26 periods
```
Medium-term momentum. Strong support/resistance.
Price returning to Kijun = pullback to fair value.

**Senkou Span A (Leading Span A)**
```
Span A = (Tenkan + Kijun) / 2   shifted 26 periods FORWARD
```

**Senkou Span B (Leading Span B)**
```
Span B = (Highest High + Lowest Low) / 2   over 52 periods, shifted 26 FORWARD
```

**The Cloud (Kumo)** = area between Span A and Span B
- Green cloud (Span A > Span B) = bullish
- Red cloud (Span A < Span B) = bearish
- Thickness = volatility proxy (your note: thick = volatile)

**Chikou Span (Lagging Span / LAGs)**
```
Chikou = Current Close shifted 26 periods BACKWARD
```
Chikou above price = bullish long-term trend
Chikou below price = bearish long-term trend
Your note: LAGs determines trend direction

**Default settings:** 9, 26, 52, 26
""")

    with st.expander("Your ICH+CCI rules decoded", expanded=True):
        st.markdown("""
### ICH + CCI v1.0 (your notebook)

**The setup:**
1. **ICHc** — Ichimoku cloud breakout (price exits cloud)
2. **CL/BL** — Tenkan/Kijun intersection (optional confirmation)
3. **LAGs** — Chikou span confirms trend direction
4. **CCI** — CCI crosses ±100 limit lines

**The rules:**
```
LONG entry when ALL of:
  ✓ Price breaks above cloud (bull breakout)
  ✓ CCI crosses above -100 (leaves oversold)  
  ✓ Chikou is above current price (bullish trend)
  ✓ Cloud is NOT thin (<0.5% of price)
  Optional: Tenkan crosses above Kijun

SHORT entry: mirror of above
```

**Thin cloud = high uncertainty** (your note)
→ Cloud width < ~0.5% of price = skip the trade
→ Thick cloud after consolidation = strong breakout signal

**ICHc as S&R at H/L** — cloud top/bottom become support/resistance
**Clouds as TLN** — treat cloud edges like trendlines
**Thickness = VOLA** — wide cloud = high volatility regime

**TP rule (Mark II):** Take profit at the extreme (highest/lowest point) of the previous cloud on the same trend side.
**SL rule:** Stop at the extreme of the current cloud.
""")


# ── TAB 3: Calculators ────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("## Calculators")

    calc_type = st.selectbox("Calculator", [
        "Position Size & Risk",
        "Kelly Criterion",
        "Pip Value (FX)",
        "RRR & Breakeven Win Rate",
        "Compound Growth (CAGR)",
    ])

    if calc_type == "Position Size & Risk":
        st.markdown("### Position Size Calculator")
        c1, c2 = st.columns(2)
        acc   = c1.number_input("Account Size ($)", value=10_000.0)
        risk  = c1.slider("Risk per trade (%)", 0.25, 5.0, 1.0, step=0.25)
        entry = c2.number_input("Entry Price", value=1.1000, format="%.5f")
        sl    = c2.number_input("Stop Loss Price", value=1.0950, format="%.5f")
        pip_v = c2.number_input("Pip Value ($ per standard lot)", value=10.0)

        risk_amt    = acc * risk / 100
        sl_dist     = abs(entry - sl)
        sl_pips     = sl_dist * 10000 if entry < 100 else sl_dist  # FX vs other
        lot_size    = risk_amt / (sl_pips * pip_v / sl_pips) if sl_pips > 0 else 0
        units       = lot_size * 100_000

        st.markdown(f"""
| | Value |
|---|---|
| Risk Amount | **${risk_amt:,.2f}** |
| SL Distance | **{sl_dist:.5f}** ({sl_pips:.1f} pips) |
| Lot Size | **{lot_size:.3f} lots** |
| Units | **{units:,.0f}** |
""")

    elif calc_type == "Kelly Criterion":
        st.markdown("### Kelly Criterion — Optimal Position Sizing")
        st.markdown("*Tells you what fraction of your capital to risk per trade.*")
        win_rate = st.slider("Win Rate (%)", 30, 80, 55) / 100
        rrr      = st.slider("Reward:Risk Ratio", 0.5, 5.0, 2.0, step=0.25)

        kelly_full = win_rate - (1 - win_rate) / rrr
        kelly_half = kelly_full / 2

        st.metric("Full Kelly", f"{kelly_full*100:.1f}% of capital")
        st.metric("Half Kelly (recommended)", f"{kelly_half*100:.1f}% of capital")
        st.caption("Full Kelly maximises long-run growth but causes large drawdowns. Half Kelly is the practical choice.")

        if kelly_full <= 0:
            st.warning(f"Negative Kelly ({kelly_full*100:.1f}%) — this trade has negative expected value. Don't take it.")

    elif calc_type == "Pip Value (FX)":
        st.markdown("### FX Pip Value Calculator")
        pair      = st.selectbox("Pair", ["EURUSD","GBPUSD","USDJPY","XAUUSD","AUDUSD"])
        lot_size  = st.number_input("Lot Size", value=0.1, step=0.01)
        pip_sizes = {"EURUSD":0.0001,"GBPUSD":0.0001,"USDJPY":0.01,"XAUUSD":0.1,"AUDUSD":0.0001}
        pip_vals  = {"EURUSD":10,"GBPUSD":10,"USDJPY":9.2,"XAUUSD":10,"AUDUSD":10}
        ps = pip_sizes.get(pair, 0.0001)
        pv = pip_vals.get(pair, 10)
        st.metric(f"Pip Value ({pair})", f"${pv * lot_size:.2f} per pip per {lot_size} lots")

    elif calc_type == "RRR & Breakeven Win Rate":
        st.markdown("### RRR → Minimum Win Rate to Be Profitable")
        rrr_val = st.slider("Reward:Risk Ratio", 0.5, 5.0, 2.0, step=0.25)
        breakeven = 1 / (1 + rrr_val)
        st.metric("Breakeven Win Rate", f"{breakeven*100:.1f}%")
        st.markdown(f"At {rrr_val}:1 RRR, you only need to win **{breakeven*100:.1f}%** of trades to break even. Above that = profitable.")

        wr_range = np.arange(0.3, 0.75, 0.05)
        ev = [wr * rrr_val - (1-wr) for wr in wr_range]
        fig = go.Figure(go.Bar(
            x=[f"{w*100:.0f}%" for w in wr_range],
            y=ev,
            marker_color=[GREEN if v>0 else RED for v in ev],
            text=[f"{v:+.2f}R" for v in ev],
            textposition="outside",
        ))
        fig.add_hline(y=0)
        fig.update_layout(
            plot_bgcolor=BG, paper_bgcolor=BG,
            font=dict(color=TEXT), margin=dict(t=40,b=40),
            title=f"Expected Value (R) at {rrr_val}:1 RRR by Win Rate",
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    elif calc_type == "Compound Growth (CAGR)":
        st.markdown("### CAGR & Compound Growth")
        start_val  = st.number_input("Starting Capital ($)", value=10_000.0)
        ann_return = st.slider("Annual Return (%)", -20, 100, 20)
        years_     = st.slider("Years", 1, 30, 10)
        monthly_add= st.number_input("Monthly Contribution ($)", value=0.0)

        r = ann_return / 100
        values = []
        val = start_val
        for y in range(years_+1):
            values.append(val)
            val = val * (1+r) + monthly_add * 12

        st.metric("Final Value", f"${values[-1]:,.0f}")
        st.metric("Total Return", f"{(values[-1]/start_val - 1)*100:.1f}%")


# ── TAB 4: Options & Greeks ───────────────────────────────────────────────────
with tabs[3]:
    st.markdown("## Options Pricing & Greeks")

    with st.expander("Black-Scholes-Merton Formula", expanded=True):
        st.markdown("""
### Black-Scholes-Merton (1973)
Prices European options assuming log-normal returns.

**Inputs:**
- S = current stock price
- K = strike price
- T = time to expiry (years)
- r = risk-free rate (annualised)
- σ = implied volatility (annualised)
- q = continuous dividend yield

**Formula:**
```
d1 = [ln(S/K) + (r - q + σ²/2) × T] / (σ × √T)
d2 = d1 - σ × √T

Call = S × e^(-qT) × N(d1) - K × e^(-rT) × N(d2)
Put  = K × e^(-rT) × N(-d2) - S × e^(-qT) × N(-d1)

N() = cumulative standard normal distribution
```

**Assumptions:**
- Log-normal price distribution
- No dividends (BSM) or continuous dividend (with q)
- Constant volatility (the big limitation — vol smile breaks this)
- European exercise only
""")

    with st.expander("The Greeks — what each measures", expanded=True):
        st.markdown("""
### The 8 Greeks

| Greek | Symbol | What it measures |
|---|---|---|
| **Delta** | Δ | Change in option price per $1 change in stock price. Call: 0 to 1, Put: -1 to 0 |
| **Gamma** | Γ | Rate of change of Delta. High near expiry and ATM |
| **Theta** | Θ | Time decay — option loses this much per day (always negative for long options) |
| **Vega** | ν | Change in option price per 1% change in implied vol. High for long-dated options |
| **Rho** | ρ | Sensitivity to interest rate changes. Small for short-dated options |
| **Vanna** | | dΔ/dσ — how Delta changes with vol. Useful for vol trading |
| **Charm** | | dΔ/dt — how Delta changes over time. Important near expiry |
| **Volga** | | d²price/dσ² — convexity of option to vol changes. Used in vol arb |

**Key rules:**
- Long calls: positive Delta, positive Gamma, negative Theta, positive Vega
- Long puts: negative Delta, positive Gamma, negative Theta, positive Vega
- Short options: opposite signs — you sell time decay
- Gamma scalping: delta-hedge frequently when long Gamma to extract volatility
""")

    with st.expander("Live options calculator", expanded=False):
        c1, c2 = st.columns(2)
        S  = c1.number_input("Stock Price (S)", value=100.0)
        K  = c1.number_input("Strike Price (K)", value=100.0)
        T  = c1.number_input("Time to Expiry (years)", value=0.25, step=0.05)
        r  = c1.slider("Risk-free Rate (%)", 0.0, 10.0, 4.5) / 100
        iv = c2.slider("Implied Volatility (%)", 1.0, 100.0, 20.0) / 100
        q  = c2.slider("Dividend Yield (%)", 0.0, 10.0, 0.0) / 100

        from scipy.stats import norm
        try:
            d1 = (np.log(S/K) + (r - q + iv**2/2) * T) / (iv * np.sqrt(T))
            d2 = d1 - iv * np.sqrt(T)
            call_px = S * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
            put_px  = K * np.exp(-r*T) * norm.cdf(-d2) - S * np.exp(-q*T) * norm.cdf(-d1)
            delta_c = np.exp(-q*T) * norm.cdf(d1)
            delta_p = np.exp(-q*T) * (norm.cdf(d1) - 1)
            gamma   = np.exp(-q*T) * norm.pdf(d1) / (S * iv * np.sqrt(T))
            theta_c = (-(S * norm.pdf(d1) * iv * np.exp(-q*T)) / (2*np.sqrt(T))
                       - r*K*np.exp(-r*T)*norm.cdf(d2) + q*S*np.exp(-q*T)*norm.cdf(d1)) / 365
            vega    = S * np.exp(-q*T) * norm.pdf(d1) * np.sqrt(T) / 100

            col_c, col_p = st.columns(2)
            col_c.markdown("**CALL**")
            col_c.metric("Price",  f"${call_px:.4f}")
            col_c.metric("Delta",  f"{delta_c:.4f}")
            col_c.metric("Gamma",  f"{gamma:.4f}")
            col_c.metric("Theta",  f"${theta_c:.4f}/day")
            col_c.metric("Vega",   f"${vega:.4f}/1%")

            col_p.markdown("**PUT**")
            col_p.metric("Price",  f"${put_px:.4f}")
            col_p.metric("Delta",  f"{delta_p:.4f}")
            col_p.metric("Gamma",  f"{gamma:.4f}")
            col_p.metric("Theta",  f"${theta_c:.4f}/day")
            col_p.metric("Vega",   f"${vega:.4f}/1%")
        except Exception as e:
            st.error(f"Calculation error: {e}. Check T > 0.")


# ── TAB 5: Risk Metrics ───────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("## Risk & Performance Metrics")

    with st.expander("Sharpe, Sortino, Calmar", expanded=True):
        st.markdown("""
### Sharpe Ratio (William Sharpe, 1966)
Risk-adjusted return relative to total volatility.
```
Sharpe = (Portfolio Return - Risk-Free Rate) / Portfolio Std Dev
       (annualised: multiply by √252 for daily returns)
```
- > 1.0: good
- > 2.0: very good
- > 3.0: excellent (often too good to be true)

### Sortino Ratio
Like Sharpe but only penalises downside volatility.
```
Sortino = (Portfolio Return - Risk-Free Rate) / Downside Std Dev
Downside Std Dev = Std Dev of negative returns only × √252
```
Better than Sharpe for strategies with positive skew.

### Calmar Ratio
Return per unit of maximum drawdown.
```
Calmar = Annualised Return / |Max Drawdown|
```
- > 1.0: good
- > 3.0: excellent

### Maximum Drawdown
Largest peak-to-trough decline.
```
DD(t) = (Portfolio Value(t) - Peak Value) / Peak Value
Max DD = min(DD(t)) over all t
```
""")

    with st.expander("VaR and CVaR", expanded=False):
        st.markdown("""
### Value at Risk (VaR)
Maximum expected loss at a given confidence level.
```
VaR(95%) = 5th percentile of return distribution
           (i.e., on 95% of days, loss will be less than this)
```

### CVaR (Conditional VaR / Expected Shortfall)
Average loss in the worst X% of scenarios.
```
CVaR(95%) = Average return of the worst 5% of days
```
CVaR is a better risk measure than VaR because it captures tail losses.

### Why this matters
A strategy with high VaR but low frequency of those events is different from one where large losses are common. CVaR captures both.
""")

    with st.expander("Information Coefficient (IC)", expanded=False):
        st.markdown("""
### IC — Information Coefficient
Spearman rank correlation between factor score and forward return.
```
IC = Spearman(factor_today, return_tomorrow)
```
- IC > 0.10: strong alpha signal (institutional quality)
- IC > 0.05: meaningful
- IC < 0.05: likely noise

### IC IR — Information Ratio
Consistency of the IC signal.
```
IC IR = Mean IC / Std Dev of IC
```
- IC IR > 0.5: signal works consistently, not just on average
""")


# ── TAB 6: Quant Formulas ─────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("## Quantitative Finance Formulas")

    with st.expander("Geometric Brownian Motion (GBM) & Ito's Lemma", expanded=False):
        st.markdown(r"""
### GBM — The Foundation of Option Pricing

Stock prices follow GBM under the risk-neutral measure:
```
dS = μS dt + σS dW
```
- S = stock price
- μ = expected return (drift)
- σ = volatility
- dW = Wiener process (random noise)

In discrete time (used in Monte Carlo simulation):
```
S(t+1) = S(t) × exp[(μ - σ²/2)Δt + σ√Δt × Z]
Z ~ N(0,1)  (standard normal random number)
```
The (μ - σ²/2) term is the Ito correction — it adjusts for the fact that log-returns have different statistics than returns.

### Ito's Lemma
If f is a function of a stochastic process S:
```
df = (∂f/∂t + μS∂f/∂S + ½σ²S²∂²f/∂S²)dt + σS∂f/∂S dW
```
This is the stochastic chain rule. It's how we derive the Black-Scholes equation.
""")

    with st.expander("Piotroski F-Score — all 9 signals", expanded=False):
        st.markdown("""
### Piotroski F-Score (Joseph Piotroski, 2000)
A 9-point checklist of fundamental quality. One point for each passing signal.

**Profitability (4 points)**
```
F1: ROA > 0           (Return on Assets = Net Income / Total Assets)
F2: OCF > 0           (Operating Cash Flow > 0)
F3: ΔROA > 0          (ROA improved year-over-year)
F4: OCF/Assets > ROA  (Cash earnings > accounting earnings = quality)
```

**Leverage / Liquidity (3 points)**
```
F5: ΔLeverage < 0     (Long-term debt ratio decreased)
F6: ΔCurrent Ratio > 0 (Current assets/liabilities ratio improved)
F7: Shares not diluted (no new shares issued)
```

**Operating Efficiency (2 points)**
```
F8: ΔGross Margin > 0  (Gross profit / Revenue improved)
F9: ΔAsset Turnover > 0 (Revenue / Total Assets improved)
```

Score 8-9: Strong — likely outperformer
Score 0-2: Weak — potential short
Score 3-7: Average — monitor
""")

    with st.expander("Altman Z-Score", expanded=False):
        st.markdown("""
### Altman Z-Score (Edward Altman, 1968)
Bankruptcy prediction model. Uses 5 financial ratios.

```
Z = 1.2×X1 + 1.4×X2 + 3.3×X3 + 0.6×X4 + 1.0×X5

X1 = Working Capital / Total Assets          (liquidity)
X2 = Retained Earnings / Total Assets        (reinvestment efficiency)
X3 = EBIT / Total Assets                     (profitability)
X4 = Market Cap / Total Liabilities          (leverage buffer)
X5 = Revenue / Total Assets                  (asset efficiency)
```

**Zones:**
- Z > 2.99: Safe zone
- 1.81 < Z < 2.99: Grey zone
- Z < 1.81: Distress zone

*Originally calibrated for US manufacturing firms 1946-1965. Still widely used.*
""")

    with st.expander("Kelly Criterion derivation", expanded=False):
        st.markdown(r"""
### Kelly Criterion (John L. Kelly Jr., 1956)
Optimal fraction of capital to bet given known edge.

```
f* = (p × b - q) / b

p  = probability of winning
q  = 1 - p = probability of losing
b  = net odds (RRR — amount won per $1 risked)
f* = fraction of capital to bet
```

**Example:** 55% win rate, 2:1 RRR
```
f* = (0.55 × 2 - 0.45) / 2 = (1.10 - 0.45) / 2 = 0.325 = 32.5%
```
This is the full Kelly — in practice use half Kelly (16.25%) to reduce volatility.

**Why not always use full Kelly?** Because our edge estimate is uncertain. If you overestimate p, full Kelly will cause massive drawdowns. Half Kelly gives 75% of the return with much less risk.
""")


# ── TAB 7: Your Strategies ────────────────────────────────────────────────────
with tabs[6]:
    st.markdown("## Your Strategies — Decoded & Explained")

    with st.expander("Mark I: PSAR + SMA50 + RSI8", expanded=True):
        st.markdown("""
### What it is
A trend-following + reversal confirmation system.

### When to trade
```
1. PSAR flips direction (dots switch side)
2. RSI8 crosses 50 OR enters extreme zone (>70 or <30) in trend direction
3. Previous candle colour matches new PSAR direction
4. Previous candle is NOT a doji or hammer
5. Preferably trade with the SMA50 trend direction
```

### Candle rules
- Candle colour match: if PSAR flips bullish, previous candle should be green
- Avoid doji (open ≈ close, body < 10% of range) — means indecision
- Avoid hammer (long lower wick ≥ 2× body) — potential reversal already happened

### SMA50 context
- Price well above SMA50 + PSAR flip = strong signal
- Price near SMA50: PSAR flip + RSI touching 50 = expect SMA to act as support/resistance
- Trading against SMA50 trend: only if reversal is confirmed and RSI in extreme zone
""")

    with st.expander("ICH + CCI v1.0: Your Notebook Rules", expanded=True):
        st.markdown("""
### The strongest signal in your rulebook

**All components:**
- **ICHc** = Ichimoku cloud breakout
- **CL/BL** = Tenkan/Kijun cross (optional confirmation)
- **LAGs** = Chikou span trend confirmation
- **CCI** = crosses ±100 lines

**Entry rules:**
```
LONG:
  ✓ Price breaks above cloud top (bullish BO)
  ✓ CCI crosses above -100 (came from oversold)
  ✓ Chikou above price 26 bars ago (confirms uptrend)
  ✓ Cloud is thick (thickness > 0.5% of price)
  Optional: Tenkan crosses above Kijun

SHORT: exact mirror
```

**Skip if:**
- Cloud is thin → high uncertainty (as your note says)
- Chikou is inside the cloud → no clear trend
- Breakout happens too fast (> 3-4 candles to exit cloud)

**TP/SL:**
- TP: highest/lowest point of the previous cloud in the trend direction
- SL: extreme of the cloud currently being broken out from
""")

    with st.expander("3-Candle Sniper Entry", expanded=True):
        st.markdown("""
### The Pattern
2-3 small candles in the same direction, followed by a reversal candle that closes past the open of the first small candle.

```
Example (bearish sniper → long entry):
  Candle 1: small bearish  (open > close)
  Candle 2: small bearish
  Candle 3: small bearish
  Candle 4: BULLISH candle that closes ABOVE open of Candle 1 → LONG entry

"Small" = body < 70% of 20-period average body size
```

### Why it works
The small candles represent weak selling pressure — sellers can't push price hard. The reversal candle shows buyers overwhelming sellers. The fact that it closes past C1's open means sellers have been fully rejected.

### Best used when
- Near a key EMA (Mark IV — near EMA21)
- At cloud support/resistance
- CCI near ±100 crossover
- After a trend pullback (not in the middle of a trend)
""")

    with st.expander("CCI + WPR Synergy (your favourite)", expanded=True):
        st.markdown("""
### The Setup
Both CCI and WPR enter/exit extreme zones in the same direction within 1-2 candles.

**Bullish signal:**
```
CCI crosses above -100 (exits oversold) AND
WPR crosses above -80 (exits oversold)
→ Within 1-2 candles of each other
→ LONG
```

**Bearish signal:**
```
CCI crosses below +100 (exits overbought) AND  
WPR crosses below -20 (exits overbought)
→ Within 1-2 candles of each other
→ SHORT
```

**Why synergy matters:** Each indicator measures something slightly different (CCI = deviation from average, WPR = position relative to recent range). When both agree = higher probability.

**Can also use:** When one enters extreme zone as the other exits → momentum building/fading
""")


# ── TAB 8: Concepts ───────────────────────────────────────────────────────────
with tabs[7]:
    st.markdown("## Concepts & Market Structure")

    with st.expander("VIX Term Structure — why shape matters more than level", expanded=True):
        st.markdown("""
### What is the VIX?
The VIX (CBOE Volatility Index) measures the market's expectation of 30-day forward volatility, derived from S&P 500 option prices.

**VIX9D** = implied vol over next 9 days
**VIX Spot** = implied vol over next 30 days
**VIX3M** = implied vol over next 3 months

### Contango (normal)
```
VIX9D < VIX Spot < VIX3M
```
Near-term fear is low. Markets expect calm now but uncertainty increases further out. This is the normal state.

### Backwardation (stress)
```
VIX9D > VIX Spot > VIX3M
```
Near-term fear is extreme — markets expect something bad RIGHT NOW. This often happens at market bottoms and turns.

**Why shape > level:** A VIX at 25 in contango = orderly selloff. A VIX at 25 in backwardation = panic. Very different.
""")

    with st.expander("Credit Spreads as Leading Equity Indicator", expanded=False):
        st.markdown("""
### The Logic
Credit markets are dominated by institutional investors with more information and fewer emotional biases than equity markets. When credit deteriorates while equity holds, credit is usually right.

**HYG** (iShares HYG ETF) = high-yield (junk) bond prices
**LQD** = investment-grade bond prices

When HYG falls:
- Companies are finding it harder to borrow
- Defaults are expected to rise
- Credit is pricing in economic weakness

When this happens while SPY holds firm → divergence → equity will likely follow credit down.

**Historical examples:** HYG led SPY down in Nov 2007, Oct 2018, Feb 2020.
""")

    with st.expander("DXY — The Global Liquidity Gauge", expanded=False):
        st.markdown("""
### What is the DXY?
The US Dollar Index — a weighted geometric mean of the USD against 6 currencies:
EUR (57.6%), JPY (13.6%), GBP (11.9%), CAD (9.1%), SEK (4.2%), CHF (3.6%)

### Why it matters for everything
Most global trade and commodities are priced in USD. When DXY rises:
- USD is strong → commodities (gold, oil) usually fall
- EM currencies weaken → capital flows from EM to US
- Global liquidity tightens (less cheap USD to borrow)
- Multinational US company earnings get hurt (foreign revenues worth less in USD)

### DXY rising = less liquidity globally
When DXY falls aggressively = dollar weakening = global liquidity expansion = positive for risk assets everywhere.
""")

    with st.expander("Real Yields (not the rate, but the rate too)", expanded=False):
        st.markdown("""
### Nominal vs Real Yields

**Nominal yield** = what the bond pays (e.g., 4.5% on 10Y Treasury)

**Real yield** = nominal - inflation expectations
```
Real Yield ≈ TIPS yield (Treasury Inflation-Protected Securities)
           = 10Y Nominal - 10Y Breakeven Inflation
```

### Why real yields matter more
A 5% nominal yield when inflation is 6% = -1% real yield = you're losing purchasing power.
A 5% nominal yield when inflation is 2% = +3% real yield = a great return.

**Rising real yields** = discount rate for equities is rising = equity valuations fall
**Negative real yields** = financial repression = investors forced into risky assets (good for gold, equities)

This is why the Fed's real policy is more about real rates than nominal ones.
""")