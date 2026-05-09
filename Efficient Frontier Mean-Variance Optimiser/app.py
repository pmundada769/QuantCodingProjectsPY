#app.py

# Efficient Frontier / Mean-Variance Optimizer — Streamlit Dashboard
# Run with: streamlit run app.py

import streamlit as st # type: ignore
import numpy as np
import pandas as pd
from optimizer import compute_frontier, get_price_data, correlation_matrix
from charts import (
    frontier_chart, weights_chart, correlation_heatmap,
    sharpe_curve, weights_pie, return_distributions,
)

# page config
st.set_page_config(
    page_title = "Efficient Frontier",
    page_icon  = "📐",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# custom CSS — Syne Mono, electric blue on deep navy, refined minimalist
# different aesthetic from Options Pricer (teal/dark) and Monte Carlo (amber/black)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne+Mono&family=Syne:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #090F1A;
    color: #C8D8E8;
}
h1 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    color: #C8D8E8 !important;
    letter-spacing: -0.03em;
}
h2, h3 {
    font-family: 'Syne', sans-serif !important;
    color: #5A6A80 !important;
    font-weight: 500 !important;
}
[data-testid="metric-container"] {
    background: #0E1826;
    border: 1px solid #162030;
    border-radius: 4px;
    padding: 14px 18px;
}
[data-testid="metric-container"] label {
    font-family: 'Syne Mono', monospace !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.12em;
    color: #3A4A60 !important;
    text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    font-family: 'Syne Mono', monospace !important;
    font-size: 1.2rem !important;
    font-weight: 400;
    color: #4F8EF7 !important;
}
[data-testid="stSidebar"] {
    background: #060C14;
    border-right: 1px solid #162030;
}
.stTabs [data-baseweb="tab-list"] { background: transparent; gap: 4px; }
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne Mono', monospace;
    font-size: 0.72rem;
    background: #0E1826;
    border-radius: 3px;
    border: 1px solid #162030;
    color: #3A4A60;
}
.stTabs [aria-selected="true"] {
    background: #162030 !important;
    border-color: #4F8EF7 !important;
    color: #4F8EF7 !important;
}
hr { border-color: #162030 !important; }
</style>
""", unsafe_allow_html=True)

# sidebar inputs
with st.sidebar:
    st.markdown("## 📐 Parameters")
    st.markdown("---")

    # ticker input
    st.markdown("#### Universe")
    ticker_input = st.text_area(
        "Tickers (one per line or comma-separated)",
        value="AAPL\nMSFT\nGOOGL\nAMZN\nNVDA\nJPM\nBRK-B\nVZ",
        height=160,
    )

    # parse tickers from text area
    raw     = ticker_input.replace(",", "\n").replace(" ", "\n")
    tickers = [t.strip().upper() for t in raw.split("\n") if t.strip()]
    tickers = list(dict.fromkeys(tickers))   # deduplicate while preserving order

    st.markdown("#### History")
    start_year = st.selectbox("Start Year", options=[2015, 2016, 2017, 2018, 2019, 2020, 2021], index=3)
    start_date = f"{start_year}-01-01"

    st.markdown("#### Optimiser")
    risk_free = st.slider("Risk-Free Rate (%)", min_value=0.0, max_value=8.0, value=4.0, step=0.1) / 100
    show_random = st.checkbox("Show random portfolio cloud", value=True)

    st.markdown("---")

    # quick preset universes
    st.markdown("#### Quick Presets")
    col1, col2 = st.columns(2)
    if col1.button("US Tech"):
        tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "META"]
    if col2.button("Diversified"):
        tickers = ["SPY", "TLT", "GLD", "VNQ", "EFA"]
    col3, col4 = st.columns(2)
    if col3.button("FAANG"):
        tickers = ["META", "AAPL", "AMZN", "NFLX", "GOOGL"]
    if col4.button("Sectors"):
        tickers = ["XLK", "XLF", "XLE", "XLV", "XLI"]

    st.markdown("---")
    run_btn = st.button("▶  Optimise", use_container_width=True)
    st.caption("Markowitz (1952) Mean-Variance\nLong-only, fully invested")


# main computation — wrapped in cache so it doesn't re-run on every widget interaction
@st.cache_data(ttl=600)    # cache for 10 minutes
def cached_run(tickers_tuple, start_date, risk_free):
    prices = get_price_data(list(tickers_tuple), start=start_date)
    result = compute_frontier(prices, risk_free_rate=risk_free)
    return prices, result

# run and handle errors cleanly
try:
    with st.spinner("Downloading prices and computing frontier..."):
        prices, result = cached_run(tuple(tickers), start_date, risk_free)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.info("Check that all tickers are valid Yahoo Finance symbols.")
    st.stop()

ms = result.max_sharpe
mv = result.min_vol

# header
st.markdown("# 📐 Efficient Frontier")
st.markdown(
    f"`{len(result.tickers)} assets` · "
    f"`{start_date} → today` · "
    f"`rf = {risk_free*100:.1f}%`"
)
st.markdown("---")

# max sharpe metrics
st.markdown("### ★ Max Sharpe Portfolio")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Sharpe Ratio",      f"{ms.sharpe:.3f}")
c2.metric("Expected Return",   f"{ms.expected_return*100:.2f}%")
c3.metric("Volatility",        f"{ms.volatility*100:.2f}%")
top_asset = max(ms.weights, key=ms.weights.get)
c4.metric("Largest Holding",   f"{top_asset}")
c5.metric("Top Weight",        f"{ms.weights[top_asset]*100:.1f}%")

st.markdown("---")

# min vol metrics
st.markdown("### ◆ Min Volatility Portfolio")
d1, d2, d3, d4, d5 = st.columns(5)
d1.metric("Sharpe Ratio",      f"{mv.sharpe:.3f}")
d2.metric("Expected Return",   f"{mv.expected_return*100:.2f}%")
d3.metric("Volatility",        f"{mv.volatility*100:.2f}%")
top_asset_mv = max(mv.weights, key=mv.weights.get)
d4.metric("Largest Holding",   f"{top_asset_mv}")
d5.metric("Top Weight",        f"{mv.weights[top_asset_mv]*100:.1f}%")

st.markdown("---")

# tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈  Frontier",
    "⚖️  Weights",
    "🥧  Allocation",
    "📉  Sharpe Curve",
    "🌡️  Correlation",
    "📊  Distributions",
])

with tab1:
    fig1 = frontier_chart(result, show_random=show_random)
    st.plotly_chart(fig1, use_container_width=True)
    st.caption(
        "Blue dots = random feasible portfolios coloured by Sharpe ratio. "
        "Blue line = efficient frontier. ★ = Max Sharpe (tangency portfolio). "
        "◆ = Min Volatility. Dashed line = Capital Market Line."
    )

with tab2:
    fig2 = weights_chart(result)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Exact Weights")
    col_ms, col_mv = st.columns(2)
    with col_ms:
        st.markdown("**★ Max Sharpe**")
        ms_df = pd.DataFrame(
            {"Asset": list(ms.weights.keys()),
             "Weight": [f"{w*100:.2f}%" for w in ms.weights.values()]}
        ).sort_values("Weight", ascending=False)
        st.dataframe(ms_df, hide_index=True, use_container_width=True)
    with col_mv:
        st.markdown("**◆ Min Volatility**")
        mv_df = pd.DataFrame(
            {"Asset": list(mv.weights.keys()),
             "Weight": [f"{w*100:.2f}%" for w in mv.weights.values()]}
        ).sort_values("Weight", ascending=False)
        st.dataframe(mv_df, hide_index=True, use_container_width=True)

with tab3:
    col_a, col_b = st.columns(2)
    with col_a:
        fig3a = weights_pie(result, "max_sharpe")
        st.plotly_chart(fig3a, use_container_width=True)
    with col_b:
        fig3b = weights_pie(result, "min_vol")
        st.plotly_chart(fig3b, use_container_width=True)

with tab4:
    fig4 = sharpe_curve(result)
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("Shows how Sharpe ratio changes along the frontier. The peak is the Max Sharpe portfolio.")

with tab5:
    fig5 = correlation_heatmap(prices)
    st.plotly_chart(fig5, use_container_width=True)
    st.caption(
        "Pearson correlation of daily returns. "
        "Red = positive correlation (move together). "
        "Blue = negative correlation (move opposite — best for diversification)."
    )

with tab6:
    fig6 = return_distributions(prices)
    st.plotly_chart(fig6, use_container_width=True)
    st.caption("Overlapping histograms of daily returns. Wider = more volatile.")

# full frontier table
with st.expander("📋  Full Frontier Data Table"):
    rows = []
    for p in result.frontier_portfolios:
        row = {"Return (%)": round(p.expected_return * 100, 3),
               "Vol (%)":    round(p.volatility      * 100, 3),
               "Sharpe":     round(p.sharpe,               3)}
        for t in result.tickers:
            row[t] = f"{p.weights.get(t, 0)*100:.1f}%"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)