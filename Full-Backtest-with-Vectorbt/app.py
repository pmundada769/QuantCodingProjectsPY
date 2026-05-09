#app.py

# Professional Momentum Backtest — Streamlit Dashboard
# Run with: streamlit run app.py

import streamlit as st # type: ignore
import numpy as np
import pandas as pd
import plotly.graph_objects as go # type: ignore
import plotly.subplots as sp # type: ignore
from vbt_backtest import run_vbt_backtest, parameter_sweep, VBT_AVAILABLE

st.set_page_config(page_title="vectorbt Backtest", page_icon="⚡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Exo+2:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'Exo 2', sans-serif; background-color: #050C08; color: #A8D8B8; }
h1 { font-family: 'Orbitron', monospace !important; color: #00E676 !important; font-size: 1.6rem !important; }
h2, h3 { color: #2A5A38 !important; font-weight: 400 !important; }
[data-testid="metric-container"] { background: #0A180C; border: 1px solid #0E2814; border-radius: 4px; padding: 14px 18px; }
[data-testid="metric-container"] label { font-family: 'Orbitron', monospace !important; font-size: 0.58rem !important; letter-spacing: 0.12em; color: #1A3820 !important; text-transform: uppercase; }
[data-testid="metric-container"] [data-testid="metric-value"] { font-family: 'Orbitron', monospace !important; font-size: 1.0rem !important; color: #00E676 !important; }
[data-testid="stSidebar"] { background: #030A05; border-right: 1px solid #0E2814; }
.stTabs [data-baseweb="tab"] { font-family: 'Orbitron', monospace; font-size: 0.65rem; background: #0A180C; border-radius: 3px; border: 1px solid #0E2814; color: #1A3820; }
.stTabs [aria-selected="true"] { background: #0E2814 !important; border-color: #00E676 !important; color: #00E676 !important; }
hr { border-color: #0E2814 !important; }
</style>
""", unsafe_allow_html=True)

GREEN="#00E676"; LIME="#76FF03"; CORAL="#FF5252"; GOLD="#FFD740"; MUTED="#2A5A38"
BG="#050C08"; GRID="#0E2814"; TEXT="#A8D8B8"

_base = dict(
    plot_bgcolor=BG, paper_bgcolor=BG,
    font=dict(color=TEXT, size=11),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    margin=dict(l=60, r=30, t=60, b=60),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID),
)

with st.sidebar:
    st.markdown("## ⚡ vectorbt Backtest")
    st.markdown("---")
    if not VBT_AVAILABLE:
        st.warning("vectorbt not installed:\n```\npip install vectorbt\n```\nUsing manual fallback — all features work.")

    ticker_input = st.text_area("Universe", height=160,
        value="AAPL\nMSFT\nGOOGL\nAMZN\nNVDA\nMETA\nJPM\nBAC\nXOM\nJNJ")
    tickers = [t.strip().upper() for t in ticker_input.split("\n") if t.strip()]
    tickers = list(dict.fromkeys(tickers))

    start_year = st.selectbox("Start Year", [2010, 2015, 2018], index=1)
    lookback   = st.selectbox("Lookback (days)", [126, 189, 252, 378], index=2)
    cost_bps   = st.slider("Transaction Cost (bps)", 0, 50, 10, step=5)
    slip_bps   = st.slider("Slippage (bps)", 0, 30, 5, step=5)
    sizing     = st.radio("Position Sizing", ["equal", "vol_weighted"], index=0)
    st.caption("vectorbt: fast vectorised backtesting\nFallback: manual numpy/pandas")

start = f"{start_year}-01-01"

@st.cache_data(ttl=600)
def load_backtest(tickers_t, start, lb, cost, slip, sizing):
    return run_vbt_backtest(list(tickers_t), start=start, lookback=lb,
                            cost_bps=cost, slippage_bps=slip, sizing=sizing)

with st.spinner("Running backtest..."):
    try:
        result = load_backtest(tuple(tickers), start, lookback, cost_bps, slip_bps, sizing)
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

st.markdown("# ⚡ VECTORBT MOMENTUM BACKTEST")
st.markdown(f"`{len(result.tickers)} stocks` · `lookback={lookback}d` · `costs={cost_bps+slip_bps}bps` · `{sizing} sizing`")
if not VBT_AVAILABLE:
    st.info("vectorbt not installed — using manual numpy/pandas implementation. Results are identical.")
st.markdown("---")

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Ann. Return",   f"{result.ann_return*100:.2f}%")
c2.metric("Ann. Vol",      f"{result.ann_vol*100:.2f}%")
c3.metric("Sharpe",        f"{result.sharpe:.3f}")
c4.metric("Sortino",       f"{result.sortino:.3f}")
c5.metric("Max Drawdown",  f"{result.max_drawdown*100:.2f}%")
c6.metric("Total Trades",  f"{result.total_trades:,}")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈  Equity Curve",
    "📉  Drawdown",
    "📊  Returns Distribution",
    "📋  Monthly Calendar",
    "🔬  Parameter Sweep",
])

with tab1:
    cum = result.cumulative
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum.index, y=(cum-1)*100, mode="lines",
        name="Strategy", line=dict(color=GREEN, width=2.5),
        fill="tozeroy", fillcolor="rgba(0,230,118,0.06)"))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_layout(**{**_base, "title": "Cumulative Return (net of costs)"})
    fig.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig, use_container_width=True)

    # rolling 252-day Sharpe
    r   = result.returns
    rs  = (r.rolling(252).mean() * 252) / (r.rolling(252).std() * np.sqrt(252))
    fig_rs = go.Figure()
    fig_rs.add_trace(go.Scatter(x=rs.index, y=rs, mode="lines",
        line=dict(color=LIME, width=2), showlegend=False))
    fig_rs.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig_rs.add_hline(y=1, line=dict(color=GOLD, dash="dot", width=1),
                     annotation_text="Sharpe=1")
    fig_rs.update_layout(**{**_base, "title": "Rolling 252-Day Sharpe Ratio", "height": 300})
    st.plotly_chart(fig_rs, use_container_width=True)

with tab2:
    cum  = result.cumulative
    peak = cum.cummax()
    dd   = (cum - peak) / peak * 100
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=dd.index, y=dd, mode="lines",
        line=dict(color=CORAL, width=1.5),
        fill="tozeroy", fillcolor="rgba(255,82,82,0.12)"))
    fig2.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig2.update_layout(**{**_base, "title": "Underwater Curve (Drawdown)"})
    fig2.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    r   = result.returns * 100
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(x=r, nbinsx=80,
        marker=dict(color=GREEN, opacity=0.7)))
    fig3.add_vline(x=r.mean(), line=dict(color=GOLD, dash="dash", width=2),
                   annotation_text=f"Mean: {r.mean():.3f}%", annotation_font_color=GOLD)
    fig3.add_vline(x=0, line=dict(color=MUTED, width=1))
    fig3.update_layout(**{**_base, "title": "Daily Returns Distribution"})
    fig3.update_xaxes(ticksuffix="%")
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    monthly = result.returns.resample("ME").apply(lambda x: (1+x).prod()-1) * 100
    if len(monthly) > 0:
        yr_mo = pd.DataFrame({
            "Year":   monthly.index.year,
            "Month":  monthly.index.strftime("%b"),
            "Return": monthly.round(2).values,
        })
        pivot = yr_mo.pivot_table(index="Year", columns="Month", values="Return", aggfunc="first")
        month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        pivot = pivot[[m for m in month_order if m in pivot.columns]]
        st.dataframe(
            pivot.style.background_gradient(cmap="RdYlGn", axis=None, vmin=-10, vmax=10)
                       .format("{:.2f}%"),
            use_container_width=True,
        )

with tab5:
    st.caption("Runs 16 backtest combinations across 4 lookbacks × 4 cost levels.")
    if st.button("▶  Run Parameter Sweep (takes ~60s)"):
        with st.spinner("Running 16 backtests..."):
            sweep_df = parameter_sweep(tickers, start=start)
        st.dataframe(sweep_df.style.background_gradient(subset=["Sharpe"], cmap="RdYlGn"),
                     hide_index=True, use_container_width=True)

        # heatmap of Sharpe by lookback × cost
        if len(sweep_df) > 0:
            pivot_sharpe = sweep_df.pivot(index="Lookback (days)", columns="Cost (bps)", values="Sharpe")
            fig5 = go.Figure(go.Heatmap(
                z=pivot_sharpe.values, x=[str(c) for c in pivot_sharpe.columns],
                y=[str(l) for l in pivot_sharpe.index],
                colorscale="RdYlGn", zmid=0,
                text=np.round(pivot_sharpe.values, 3), texttemplate="%{text}",
                textfont=dict(size=11, color="white"),
                colorbar=dict(title=dict(text="Sharpe", font=dict(color=TEXT)),
                              tickfont=dict(color=TEXT), thickness=12),
            ))
            fig5.update_layout(**{**_base, "title": "Sharpe Ratio: Lookback × Transaction Cost",
                                   "height": 380})
            fig5.update_xaxes(title_text="Transaction Cost (bps)")
            fig5.update_yaxes(title_text="Lookback (days)")
            st.plotly_chart(fig5, use_container_width=True)