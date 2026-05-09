#app.py

# Time-Series Momentum (TSMOM) + Volatility Targeting — Streamlit Dashboard
# Run with: streamlit run app.py

import streamlit as st # type: ignore
import numpy as np
import pandas as pd
import yfinance as yf
from tsmom import run_tsmom, compare_vol_targets, ARCH_AVAILABLE
from charts import cumulative_return_chart, signal_chart, vol_target_comparison, drawdown_chart, position_heatmap

st.set_page_config(page_title="TSMOM", page_icon="📡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Courier+Prime:wght@400;700&family=Source+Sans+3:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; background-color: #0C0808; color: #EAD5D5; }
h1 { font-family: 'Courier Prime', monospace !important; color: #C0392B !important; }
h2, h3 { color: #7F8C8D !important; font-weight: 400 !important; }
[data-testid="metric-container"] { background: #150E0E; border: 1px solid #1E1010; border-radius: 4px; padding: 14px 18px; }
[data-testid="metric-container"] label { font-family: 'Courier Prime', monospace !important; font-size: 0.62rem !important; letter-spacing: 0.1em; color: #4A2020 !important; text-transform: uppercase; }
[data-testid="metric-container"] [data-testid="metric-value"] { font-family: 'Courier Prime', monospace !important; font-size: 1.15rem !important; color: #C0392B !important; }
[data-testid="stSidebar"] { background: #090505; border-right: 1px solid #1E1010; }
.stTabs [data-baseweb="tab"] { font-family: 'Courier Prime', monospace; font-size: 0.72rem; background: #150E0E; border-radius: 3px; border: 1px solid #1E1010; color: #4A2020; }
.stTabs [aria-selected="true"] { background: #1E1010 !important; border-color: #C0392B !important; color: #C0392B !important; }
hr { border-color: #1E1010 !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 📡 TSMOM Parameters")
    st.markdown("---")

    ticker_input = st.text_area("Tickers (one per line)", value="SPY\nQQQ\nTLT\nGLD\nEEM\nVNQ\nDBC\nHYG", height=180)
    tickers = [t.strip().upper() for t in ticker_input.split("\n") if t.strip()]
    tickers = list(dict.fromkeys(tickers))

    start_year  = st.selectbox("Start Year", [2005, 2008, 2010, 2012, 2015], index=1)
    target_vol  = st.slider("Target Volatility (%)", 5, 30, 15, step=1) / 100
    lookback    = st.selectbox("Momentum Lookback (days)", [126, 252, 378], index=1,
                               help="252 = 12 months (standard MOP2012)")
    use_garch   = st.checkbox("Use GARCH vol (vs rolling std)", value=ARCH_AVAILABLE,
                               disabled=not ARCH_AVAILABLE)
    max_lev     = st.slider("Max Leverage", 1.0, 4.0, 2.0, step=0.5)

    st.markdown("---")
    if not ARCH_AVAILABLE:
        st.warning("Install `arch` for GARCH vol:\n`pip install arch`")
    st.caption("Moskowitz, Ooi & Pedersen (2012)\nJournal of Financial Economics")

start = f"{start_year}-01-01"

@st.cache_data(ttl=600)
def load_result(tickers_tuple, start, target_vol, lookback, use_garch, max_lev):
    return run_tsmom(list(tickers_tuple), start=start, target_vol=target_vol,
                     lookback=lookback, use_garch=use_garch, max_leverage=max_lev)

@st.cache_data(ttl=600)
def load_benchmark(tickers_tuple, start):
    raw  = yf.download(list(tickers_tuple), start=start, auto_adjust=True, threads=False, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame()
    rets  = raw.pct_change().dropna()
    port  = rets.mean(axis=1)
    return (1 + port).cumprod()

with st.spinner("Running TSMOM simulation..."):
    try:
        result = load_result(tuple(tickers), start, target_vol, lookback, use_garch, max_lev)
        bench  = load_benchmark(tuple(tickers), start)
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

# header
st.markdown("# 📡 Time-Series Momentum (TSMOM)")
st.markdown(
    f"`{len(result.tickers)} assets` · "
    f"`target vol = {target_vol*100:.0f}%` · "
    f"`lookback = {lookback}d` · "
    f"`{'GARCH' if use_garch and ARCH_AVAILABLE else 'rolling std'}`"
)
st.caption("Moskowitz, Ooi & Pedersen (2012) — *Time Series Momentum*, Journal of Financial Economics")
st.markdown("---")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Ann. Return",   f"{result.ann_return*100:.2f}%")
c2.metric("Ann. Vol",      f"{result.ann_vol*100:.2f}%")
c3.metric("Sharpe Ratio",  f"{result.sharpe:.3f}")
c4.metric("Sortino Ratio", f"{result.sortino:.3f}")
c5.metric("Max Drawdown",  f"{result.max_drawdown*100:.2f}%")
c6.metric("Hit Rate",      f"{result.hit_rate*100:.1f}%")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈  Cumulative Return",
    "🎛️  Signal Drill-Down",
    "📉  Drawdown",
    "🗓️  Position Heatmap",
    "🎯  Vol Target Sweep",
    "📋  Monthly Returns",
])

with tab1:
    fig1 = cumulative_return_chart(result, bench)
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("Solid = TSMOM vol-scaled strategy. Dotted = equal-weight buy & hold of the same assets.")

with tab2:
    ticker_sel = st.selectbox("Select asset", result.tickers)
    asset_obj  = next((a for a in result.asset_signals if a.ticker == ticker_sel), None)
    if asset_obj:
        fig2 = signal_chart(asset_obj)
        st.plotly_chart(fig2, use_container_width=True)
    st.caption("Green shading = long. No shading = short. Position size scales inversely with vol.")

with tab3:
    fig3 = drawdown_chart(result)
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    fig4 = position_heatmap(result)
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("Green = long position. Red = short. Intensity = leverage. Shows regime clustering across assets.")

with tab5:
    st.caption("Reruns at 5 vol targets — shows the performance/risk trade-off of the vol target parameter.")
    with st.spinner("Running vol target sweep..."):
        vt_results = compare_vol_targets(result.tickers, start=start)
    fig5 = vol_target_comparison(vt_results)
    st.plotly_chart(fig5, use_container_width=True)

with tab6:
    monthly = result.portfolio_returns.resample("ME").apply(lambda x: (1+x).prod()-1)
    monthly_pct = monthly * 100
    yr_mo = pd.DataFrame({
        "Year":   monthly.index.year,
        "Month":  monthly.index.strftime("%b"),
        "Return": monthly_pct.round(3),
    })
    pivot = yr_mo.pivot(index="Year", columns="Month", values="Return")
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot = pivot[[m for m in month_order if m in pivot.columns]]
    st.dataframe(
        pivot.style.background_gradient(cmap="RdYlGn", axis=None, vmin=-10, vmax=10)
                   .format("{:.2f}%"),
        use_container_width=True,
    )