#app.py

# alphalens-Style Factor Analysis — Streamlit Dashboard
# Run with: streamlit run app.py

import streamlit as st # type: ignore
import numpy as np
import pandas as pd
import plotly.graph_objects as go # type: ignore
import plotly.subplots as sp # type: ignore
from alpha_analysis import run_factor_analysis, ALPHALENS_AVAILABLE

st.set_page_config(page_title="Factor Analysis", page_icon="🔭", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #0A060E; color: #D0C0E8; }
h1 { font-family: 'DM Mono', monospace !important; color: #B388FF !important; }
h2, h3 { color: #4A3A60 !important; font-weight: 400 !important; }
[data-testid="metric-container"] { background: #12081C; border: 1px solid #1E1030; border-radius: 4px; padding: 14px 18px; }
[data-testid="metric-container"] label { font-family: 'DM Mono', monospace !important; font-size: 0.62rem !important; letter-spacing: 0.1em; color: #2A1840 !important; text-transform: uppercase; }
[data-testid="metric-container"] [data-testid="metric-value"] { font-family: 'DM Mono', monospace !important; font-size: 1.15rem !important; color: #B388FF !important; }
[data-testid="stSidebar"] { background: #070410; border-right: 1px solid #1E1030; }
.stTabs [data-baseweb="tab"] { font-family: 'DM Mono', monospace; font-size: 0.72rem; background: #12081C; border-radius: 3px; border: 1px solid #1E1030; color: #2A1840; }
.stTabs [aria-selected="true"] { background: #1E1030 !important; border-color: #B388FF !important; color: #B388FF !important; }
hr { border-color: #1E1030 !important; }
</style>
""", unsafe_allow_html=True)

PURPLE="#B388FF"; TEAL="#4FC3F7"; CORAL="#FF7043"; GOLD="#FFD54F"; MUTED="#4A3A60"
BG="#0A060E"; GRID="#1E1030"; TEXT="#D0C0E8"

_base = dict(
    plot_bgcolor=BG, paper_bgcolor=BG,
    font=dict(family="DM Mono, monospace", color=TEXT, size=11),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    margin=dict(l=60, r=30, t=60, b=60),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID),
)

with st.sidebar:
    st.markdown("## 🔭 Factor Analysis")
    st.markdown("---")
    if not ALPHALENS_AVAILABLE:
        st.info("Using manual IC implementation.\nFor full alphalens:\n`pip install alphalens-reloaded`")

    ticker_input = st.text_area("Universe", height=180,
        value="AAPL\nMSFT\nGOOGL\nAMZN\nNVDA\nMETA\nJPM\nBAC\nJNJ\nXOM\nKO\nPG\nHD\nDIS\nV")
    tickers = [t.strip().upper() for t in ticker_input.split("\n") if t.strip()]
    tickers = list(dict.fromkeys(tickers))

    start_year   = st.selectbox("Start Year", [2015, 2018, 2020], index=0)
    lookback     = st.selectbox("Momentum Lookback (days)", [126, 189, 252, 378], index=2)
    n_quantiles  = st.slider("Quantiles", 3, 10, 5)
    factor_name  = st.text_input("Factor Name", value="Momentum (12-1)")
    st.caption("Industry standard: Quantopian alphalens\nIC = Spearman rank correlation")

start = f"{start_year}-01-01"

@st.cache_data(ttl=600)
def load_analysis(tickers_t, start, lb, nq, fname):
    return run_factor_analysis(list(tickers_t), start=start, lookback=lb,
                               n_quantiles=nq, factor_name=fname)

with st.spinner(f"Running factor analysis for {factor_name}..."):
    try:
        fa = load_analysis(tuple(tickers), start, lookback, n_quantiles, factor_name)
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

st.markdown("# 🔭 alphalens Factor Analysis")
st.markdown(f"`{factor_name}` · `{len(tickers)} stocks` · `{start} → today`")
st.markdown("---")

ic_rating = "Strong" if abs(fa.mean_ic) > 0.10 else ("Meaningful" if abs(fa.mean_ic) > 0.05 else "Weak")
c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Mean IC (1d)",  f"{fa.mean_ic:.4f}")
c2.metric("IC IR",         f"{fa.ic_ir:.3f}")
c3.metric("Mean IC (5d)",  f"{fa.mean_ic_5d:.4f}")
c4.metric("Mean IC (21d)", f"{fa.mean_ic_21d:.4f}")
c5.metric("IC Rating",     ic_rating)
c6.metric("Avg Turnover",  f"{fa.avg_turnover*100:.1f}%")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📡  IC Time Series",
    "📊  Quantile Returns",
    "📈  Spread Return",
    "⏱️  IC Decay",
    "🔄  Turnover",
])

with tab1:
    ic = fa.ic_series.dropna()
    rolling_ic = ic.rolling(21).mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=ic.index, y=ic.values,
        marker_color=[TEAL if v > 0 else CORAL for v in ic.values],
        name="Daily IC", opacity=0.5))
    fig.add_trace(go.Scatter(x=rolling_ic.index, y=rolling_ic.values,
        mode="lines", line=dict(color=PURPLE, width=2.5),
        name="21-day rolling IC"))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.add_hline(y=0.05,  line=dict(color=TEAL,  dash="dot", width=1),
                  annotation_text="IC=0.05 (meaningful)")
    fig.add_hline(y=-0.05, line=dict(color=CORAL, dash="dot", width=1))
    fig.update_layout(**{**_base, "title": f"{factor_name} — IC Time Series"})
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Mean IC = {fa.mean_ic:.4f} · IC IR = {fa.ic_ir:.3f} · Rating: {ic_rating}")
    st.caption("IC > 0.05 is considered meaningful. IC IR > 0.5 means the signal is consistent.")

with tab2:
    if len(fa.quantile_returns) > 0:
        q_means = fa.quantile_returns.mean() * 100
        colours = [CORAL if i == 0 else (TEAL if i == len(q_means)-1 else MUTED)
                   for i in range(len(q_means))]
        fig2 = go.Figure(go.Bar(
            x=q_means.index, y=q_means.values,
            marker_color=colours,
            text=[f"{v:.3f}%" for v in q_means.values],
            textposition="outside", textfont=dict(color=TEXT),
        ))
        fig2.add_hline(y=0, line=dict(color=MUTED, width=1))
        fig2.update_layout(**{**_base, "title": f"Mean Return by Quantile (5-day forward)"})
        fig2.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Top quantile should outperform bottom — this is what a working factor looks like.")

        # heatmap of quantile returns over time
        fig2b = go.Figure(go.Heatmap(
            z=fa.quantile_returns.T.values,
            x=fa.quantile_returns.index,
            y=fa.quantile_returns.columns.tolist(),
            colorscale="RdYlGn", zmid=0,
            colorbar=dict(title=dict(text="Return", font=dict(color=TEXT)),
                          tickfont=dict(color=TEXT), thickness=12),
        ))
        fig2b.update_layout(**{**_base, "title": "Quantile Returns Heatmap", "height": 300})
        st.plotly_chart(fig2b, use_container_width=True)

with tab3:
    if len(fa.spread_return) > 0:
        spread_cum = (1 + fa.spread_return).cumprod()
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=spread_cum.index, y=(spread_cum-1)*100,
            mode="lines", line=dict(color=PURPLE, width=2.5),
            fill="tozeroy", fillcolor="rgba(179,136,255,0.07)"))
        fig3.add_hline(y=0, line=dict(color=MUTED, width=1))
        fig3.update_layout(**{**_base, "title": "Top Minus Bottom Quantile Spread Return"})
        fig3.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig3, use_container_width=True)

with tab4:
    decay = fa.ic_by_horizon
    fig4  = go.Figure()
    fig4.add_trace(go.Bar(
        x=decay["Horizon (days)"].astype(str), y=decay["Mean IC"],
        marker_color=[TEAL if v > 0 else CORAL for v in decay["Mean IC"]],
        text=[f"{v:.4f}" for v in decay["Mean IC"]],
        textposition="outside", textfont=dict(color=TEXT),
    ))
    fig4.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig4.update_layout(**{**_base, "title": f"{factor_name} — IC by Forward Return Horizon (Decay)"})
    fig4.update_xaxes(title_text="Forward Horizon (days)")
    fig4.update_yaxes(title_text="Mean IC")
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("A good factor has positive IC that decays gradually. Rapid decay = signal is short-lived.")

with tab5:
    if len(fa.monthly_turnover) > 0:
        fig5 = go.Figure()
        fig5.add_trace(go.Bar(x=fa.monthly_turnover.index,
            y=fa.monthly_turnover.values * 100,
            marker_color=GOLD, name="Monthly Turnover"))
        fig5.add_hline(y=fa.avg_turnover*100, line=dict(color=PURPLE, dash="dash", width=2),
                       annotation_text=f"Avg: {fa.avg_turnover*100:.1f}%",
                       annotation_font_color=PURPLE)
        fig5.update_layout(**{**_base, "title": "Monthly Factor Turnover (%)"})
        fig5.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig5, use_container_width=True)
        st.caption("High turnover = higher transaction costs. Momentum naturally has ~50-70% monthly turnover.")