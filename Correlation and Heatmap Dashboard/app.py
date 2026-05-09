#app.py

# Correlation & Heatmap Dashboard — Streamlit
# Run with: streamlit run app.py

import streamlit as st # type: ignore
import numpy as np
import pandas as pd
from correlation import (
    fetch_prices, daily_returns, rolling_correlation,
    correlation_at_date, average_correlation,
    detect_regime_shifts, dispersion,
)
from charts import (
    static_heatmap, rolling_correlation_lines,
    average_correlation_chart, correlation_heatmap_animated,
    dispersion_chart,
)

st.set_page_config(page_title="Correlation Dashboard", page_icon="🌡️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Lato:wght@300;400;700&display=swap');
html, body, [class*="css"] { font-family: 'Lato', sans-serif; background-color: #0E0A06; color: #EAD8C0; }
h1 { font-family: 'JetBrains Mono', monospace !important; color: #FF6B35 !important; letter-spacing: -0.02em; }
h2, h3 { font-family: 'Lato', sans-serif !important; color: #7A5A3A !important; font-weight: 300 !important; }
[data-testid="metric-container"] { background: #18110A; border: 1px solid #241A0E; border-radius: 4px; padding: 14px 18px; }
[data-testid="metric-container"] label { font-family: 'JetBrains Mono', monospace !important; font-size: 0.62rem !important; letter-spacing: 0.1em; color: #4A3020 !important; text-transform: uppercase; }
[data-testid="metric-container"] [data-testid="metric-value"] { font-family: 'JetBrains Mono', monospace !important; font-size: 1.15rem !important; color: #FF6B35 !important; }
[data-testid="stSidebar"] { background: #0A0704; border-right: 1px solid #241A0E; }
.stTabs [data-baseweb="tab"] { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; background: #18110A; border-radius: 3px; border: 1px solid #241A0E; color: #4A3020; }
.stTabs [aria-selected="true"] { background: #241A0E !important; border-color: #FF6B35 !important; color: #FF6B35 !important; }
hr { border-color: #241A0E !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🌡️ Parameters")
    st.markdown("---")

    ticker_input = st.text_area("Tickers (one per line)", value="SPY\nQQQ\nTLT\nGLD\nEFA\nEEM\nVNQ\nHYG", height=180)
    tickers      = [t.strip().upper() for t in ticker_input.split("\n") if t.strip()]
    tickers      = list(dict.fromkeys(tickers))

    start_year  = st.selectbox("Start Year", [2015, 2016, 2017, 2018, 2019, 2020], index=2)
    roll_window = st.slider("Rolling window (days)", 20, 120, 60, step=10)
    shift_threshold = st.slider("Regime shift threshold", 0.15, 0.60, 0.30, step=0.05)

    st.markdown("#### Presets")
    col1, col2 = st.columns(2)
    if col1.button("Risk Assets"):
        tickers = ["SPY", "QQQ", "EEM", "HYG", "GLD"]
    if col2.button("Multi-Asset"):
        tickers = ["SPY", "TLT", "GLD", "VNQ", "EFA", "HYG"]
    col3, col4 = st.columns(2)
    if col3.button("Tech Stocks"):
        tickers = ["AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN"]
    if col4.button("Sectors"):
        tickers = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP"]

    st.markdown("---")
    st.caption("Rolling correlation + regime detection")

start = f"{start_year}-01-01"

@st.cache_data(ttl=600)
def load_data(tickers_tuple, start, window):
    prices  = fetch_prices(list(tickers_tuple), start=start)
    returns = daily_returns(prices)
    roll    = rolling_correlation(returns, window=window)
    avg_c   = average_correlation(returns, window=window)
    disp    = dispersion(returns, window=20)
    current_corr = correlation_at_date(returns)
    return prices, returns, roll, avg_c, disp, current_corr

with st.spinner("Computing rolling correlations..."):
    try:
        prices, returns, roll_corr, avg_c, disp, current_corr = load_data(
            tuple(tickers), start, roll_window
        )
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

# detect regime shifts
shifts = detect_regime_shifts(roll_corr, threshold=shift_threshold)

# summary metrics
avg_now   = float(avg_c.iloc[-1]) if len(avg_c) > 0 else 0
avg_1y    = float(avg_c.tail(252).mean()) if len(avg_c) > 0 else 0
avg_all   = float(avg_c.mean()) if len(avg_c) > 0 else 0
regime    = "🔴 CRISIS" if avg_now > 0.6 else ("🟡 ELEVATED" if avg_now > 0.4 else "🟢 NORMAL")

st.markdown("# 🌡️ Correlation & Regime Dashboard")
st.markdown(f"`{len(tickers)} assets` · `{start} → today` · `{roll_window}d rolling window`")
st.markdown("---")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Current Avg Corr",  f"{avg_now:.3f}")
c2.metric("1Y Avg Corr",       f"{avg_1y:.3f}")
c3.metric("Full-Period Avg",   f"{avg_all:.3f}")
c4.metric("Regime",            regime)
c5.metric("Regime Shifts",     str(len(shifts)))

st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌡️  Current Heatmap",
    "📈  Rolling Correlations",
    "📡  Avg Correlation",
    "🎬  Animated Heatmap",
    "📊  Dispersion",
    "⚠️  Regime Shifts",
])

with tab1:
    fig1 = static_heatmap(current_corr, title="Current Correlation Matrix (full history)")
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("Blue = negative (good for diversification). Red = positive (assets moving together).")

with tab2:
    all_pairs = list(roll_corr.keys())
    pair_labels = [f"{a}–{b}" for a, b in all_pairs]
    selected    = st.multiselect("Select pairs to display", pair_labels, default=pair_labels[:5])
    selected_pairs = [all_pairs[pair_labels.index(s)] for s in selected if s in pair_labels]

    fig2 = rolling_correlation_lines(roll_corr, selected_pairs)
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    fig3 = average_correlation_chart(avg_c, shifts)
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Red zones = average correlation above 0.6 (crisis regime — diversification breaks down).")

with tab4:
    st.caption("Correlation matrix computed on trailing 63 days at 6 equally-spaced snapshots. Press Play.")
    fig4 = correlation_heatmap_animated(returns)
    st.plotly_chart(fig4, use_container_width=True)

with tab5:
    fig5 = dispersion_chart(disp)
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("Low dispersion = assets all moving together (crisis or macro event). High = more stock-specific movement.")

with tab6:
    st.markdown("### Detected Regime Shifts")
    st.caption(f"Correlation changes of ≥ {shift_threshold:.2f} over any {20}-day window.")
    if shifts:
        shift_df = pd.DataFrame([{
            "Date": s.date, "Pair": s.pair,
            "Before": f"{s.old_corr:.3f}", "After": f"{s.new_corr:.3f}",
            "Change": f"{s.delta:+.3f}", "Direction": s.direction,
        } for s in shifts[:50]])
        st.dataframe(shift_df, hide_index=True, use_container_width=True)
    else:
        st.info("No regime shifts detected at the current threshold.")