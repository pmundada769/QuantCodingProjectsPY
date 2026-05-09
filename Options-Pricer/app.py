"""
Options Pricer — Streamlit Dashboard
Run with:  streamlit run app.py
"""

import streamlit as st # type: ignore
import numpy as np
import pandas as pd
from black_scholes import black_scholes, implied_volatility
from charts import (
    payoff_diagram, greeks_vs_spot, greeks_vs_time,
    vol_smile, strategy_payoff, STRATEGIES,
)

# ─────────────────────────── Page config ──────────────────────────────────
st.set_page_config(
    page_title="Options Pricer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── Custom CSS ───────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0D1117;
    color: #E8EEF4;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #131C27;
    border: 1px solid #1E2A38;
    border-radius: 8px;
    padding: 12px 16px;
}
[data-testid="metric-container"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.08em;
    color: #6B7E95 !important;
    text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.3rem !important;
    font-weight: 700;
    color: #E8EEF4 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0A0F16;
    border-right: 1px solid #1E2A38;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: transparent; gap: 8px; }
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
    background: #131C27;
    border-radius: 6px;
    border: 1px solid #1E2A38;
    color: #6B7E95;
}
.stTabs [aria-selected="true"] {
    background: #1A2940 !important;
    border-color: #00C4B4 !important;
    color: #00C4B4 !important;
}

/* Header */
h1 { font-family: 'JetBrains Mono', monospace !important; color: #E8EEF4 !important; }
h2, h3 { font-family: 'Inter', sans-serif !important; color: #B0BEC5 !important; font-weight: 500 !important; }

/* Dividers */
hr { border-color: #1E2A38 !important; }

/* DataFrame */
[data-testid="stDataFrame"] { border: 1px solid #1E2A38; border-radius: 8px; }

/* Number inputs */
[data-baseweb="input"] { background: #131C27 !important; border-color: #1E2A38 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── Sidebar inputs ───────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Parameters")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        option_type = st.radio("Type", ["call", "put"], index=0,
                               format_func=str.upper)
    with col2:
        exercise = st.radio("Style", ["European"], index=0)

    st.markdown("#### Underlying")
    S     = st.number_input("Spot (S)",      value=100.0, min_value=0.01, step=1.0, format="%.2f")
    K     = st.number_input("Strike (K)",    value=100.0, min_value=0.01, step=1.0, format="%.2f")

    st.markdown("#### Market")
    T_days = st.slider("Days to Expiry",     min_value=1, max_value=730, value=30)
    T      = T_days / 365.0
    r      = st.slider("Risk-free Rate (%)", min_value=0.0, max_value=15.0, value=5.0, step=0.1) / 100
    sigma  = st.slider("Volatility (%)",     min_value=1.0, max_value=150.0, value=20.0, step=0.5) / 100
    q      = st.slider("Div Yield (%)",      min_value=0.0, max_value=15.0, value=0.0, step=0.1) / 100

    st.markdown("---")
    st.markdown("#### IV Calculator")
    market_price = st.number_input("Market Price (for IV)", value=0.0, min_value=0.0, format="%.4f")
    if market_price > 0:
        iv = implied_volatility(market_price, S, K, T, r, q, option_type)
        if np.isnan(iv):
            st.error("IV did not converge")
        else:
            st.success(f"**Implied Vol: {iv*100:.2f}%**")

    st.markdown("---")
    st.caption("Black-Scholes (Generalized Merton) model\nDividends modelled as continuous yield")

# ─────────────────────────── Compute ──────────────────────────────────────
res = black_scholes(S, K, T, r, sigma, q, option_type)

# Moneyness label
moneyness_pct = (S - K) / K * 100
if abs(moneyness_pct) < 1:
    money_label = "ATM"
elif option_type == "call":
    money_label = "ITM" if S > K else "OTM"
else:
    money_label = "ITM" if S < K else "OTM"

# ─────────────────────────── Header row ───────────────────────────────────
st.markdown("# 📈 Options Pricer")
st.markdown(f"`{option_type.upper()}` · `{money_label}` · `{T_days}d` · `σ={sigma*100:.1f}%` · `r={r*100:.1f}%`")
st.markdown("---")

# Price + key Greeks
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Option Price",   f"${res.price:.4f}")
c2.metric("Intrinsic",      f"${res.intrinsic:.4f}")
c3.metric("Time Value",     f"${res.time_value:.4f}")
c4.metric("d₁",             f"{res.d1:.4f}")
c5.metric("d₂",             f"{res.d2:.4f}")
c6.metric("N(d₁)",          f"{res.nd1:.4f}")

st.markdown("---")

# Full Greeks table
col_g, col_2g = st.columns([1, 1])

with col_g:
    st.markdown("### First-Order Greeks")
    g1 = {
        "Greek": ["Δ Delta", "ν Vega", "Θ Theta", "ρ Rho"],
        "Value": [
            f"{res.delta:+.6f}",
            f"{res.vega:+.6f}",
            f"{res.theta:+.6f}",
            f"{res.rho:+.6f}",
        ],
        "Meaning": [
            "Price change per $1 move in S",
            "Price change per 1% move in σ",
            "Price change per calendar day",
            "Price change per 1% move in r",
        ],
    }
    st.dataframe(pd.DataFrame(g1), use_container_width=True, hide_index=True)

with col_2g:
    st.markdown("### Second-Order Greeks")
    g2 = {
        "Greek": ["Γ Gamma", "Vanna", "Charm", "Volga"],
        "Value": [
            f"{res.gamma:+.6f}",
            f"{res.vanna:+.6f}",
            f"{res.charm:+.6f}",
            f"{res.volga:+.6f}",
        ],
        "Meaning": [
            "Δ change per $1 move in S",
            "Δ change per 1% move in σ",
            "Δ change per calendar day",
            "Vega change per 1% move in σ",
        ],
    }
    st.dataframe(pd.DataFrame(g2), use_container_width=True, hide_index=True)

# ─────────────────────────── Tabs ─────────────────────────────────────────
st.markdown("---")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Payoff Diagram",
    "⚡  Greeks vs Spot",
    "⏱️  Greeks vs Time",
    "〰️  Vol Smile",
    "🧩  Strategy Builder",
])

with tab1:
    show_be = st.checkbox("Show break-even line", value=True, key="be")
    fig = payoff_diagram(S, K, T, r, sigma, q, option_type, show_be)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig2 = greeks_vs_spot(S, K, T, r, sigma, q, option_type)
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    fig3 = greeks_vs_time(S, K, T, r, sigma, q, option_type)
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    st.caption("Illustrative vol smile: shows how real market IVs deviate from flat BS vol across strikes.")
    fig4 = vol_smile(S, K, T, r, sigma, q, option_type)
    st.plotly_chart(fig4, use_container_width=True)

with tab5:
    strat = st.selectbox("Select strategy", list(STRATEGIES.keys()))
    fig5 = strategy_payoff(S, T, r, sigma, q, strat)
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("Legs are constructed relative to the current spot price.")

# ─────────────────────────── Sensitivity table ────────────────────────────
with st.expander("📐 Sensitivity Table (Spot × Vol)"):
    st.markdown("Option price across a grid of spot prices and volatilities.")
    spot_range = np.linspace(S * 0.80, S * 1.20, 7)
    vol_range  = np.linspace(max(sigma * 0.5, 0.05), sigma * 1.5, 7)

    table = []
    for v in vol_range:
        row = []
        for s in spot_range:
            p = black_scholes(s, K, T, r, v, q, option_type).price
            row.append(round(p, 3))
        table.append(row)

    df = pd.DataFrame(
        table,
        index=[f"σ={v*100:.0f}%" for v in vol_range],
        columns=[f"S={s:.1f}" for s in spot_range],
    )
    st.dataframe(df.style.background_gradient(cmap="YlOrRd", axis=None), use_container_width=True)