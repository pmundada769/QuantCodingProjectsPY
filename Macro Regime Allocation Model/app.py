#app.py

# Macro Regime Allocation Model — Streamlit Dashboard
# Run with: streamlit run app.py

import streamlit as st # type: ignore
import numpy as np
import pandas as pd
import plotly.graph_objects as go # type: ignore
import plotly.subplots as sp # type: ignore
from macro_regime import fetch_macro_data, build_regime_series, REGIME_ALLOCATIONS

st.set_page_config(page_title="Macro Regime", page_icon="🌍", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Lato:wght@300;400;700&display=swap');
html, body, [class*="css"] { font-family: 'Lato', sans-serif; background-color: #0A0C08; color: #D4CCA8; }
h1 { font-family: 'Playfair Display', serif !important; color: #C8A84B !important; }
h2, h3 { color: #5A5A3A !important; font-weight: 400 !important; }
[data-testid="metric-container"] { background: #121408; border: 1px solid #1E2010; border-radius: 4px; padding: 14px 18px; }
[data-testid="metric-container"] label { font-family: 'Lato', sans-serif !important; font-size: 0.62rem !important; letter-spacing: 0.12em; color: #3A3820 !important; text-transform: uppercase; }
[data-testid="metric-container"] [data-testid="metric-value"] { font-family: 'Playfair Display', serif !important; font-size: 1.15rem !important; color: #C8A84B !important; }
[data-testid="stSidebar"] { background: #080A06; border-right: 1px solid #1E2010; }
.stTabs [data-baseweb="tab"] { font-size: 0.72rem; background: #121408; border-radius: 3px; border: 1px solid #1E2010; color: #3A3820; }
.stTabs [aria-selected="true"] { background: #1E2010 !important; border-color: #C8A84B !important; color: #C8A84B !important; }
hr { border-color: #1E2010 !important; }
</style>
""", unsafe_allow_html=True)

GOLD="#C8A84B"; TEAL="#2ECC71"; CORAL="#E74C3C"; BLUE="#3498DB"; MUTED="#5A5A3A"
BG="#0A0C08"; GRID="#1E2010"; TEXT="#D4CCA8"

REGIME_COLOURS = {
    "Growth↑ Inflation↑": "#E67E22",
    "Growth↑ Inflation↓": "#2ECC71",
    "Growth↓ Inflation↓": "#3498DB",
    "Growth↓ Inflation↑": "#E74C3C",
}

_base = dict(
    plot_bgcolor=BG, paper_bgcolor=BG,
    font=dict(color=TEXT, size=11),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    margin=dict(l=60, r=30, t=60, b=60),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID),
)

with st.sidebar:
    st.markdown("## 🌍 Macro Regime")
    st.markdown("---")
    start_year = st.selectbox("Start Year", [2000, 2005, 2010, 2015], index=0)
    growth_threshold = st.slider("PMI Growth threshold", 45.0, 55.0, 50.0, step=0.5)
    inflation_threshold = st.slider("Inflation threshold (%)", 1.0, 4.0, 2.5, step=0.25)
    st.markdown("---")
    st.caption("Data: FRED API (free)\nGrowth: ISM PMI\nInflation: CPI YoY\nYield: 10Y-2Y spread")
    st.caption("Inspired by Bridgewater All Weather")

start = f"{start_year}-01-01"

@st.cache_data(ttl=3600)
def load_regime(start, gt, it):
    macro = fetch_macro_data(start=start)
    return build_regime_series(macro, start=start)

with st.spinner("Fetching FRED macro data..."):
    try:
        result = load_regime(start, growth_threshold, inflation_threshold)
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

st.markdown("# 🌍 Macro Regime Allocation Model")
st.markdown(f"`{start} → today` · Growth: PMI · Inflation: CPI YoY · Yield: 10Y-2Y")
st.markdown("---")

regime_icon = {"Growth↑ Inflation↑":"🔴","Growth↑ Inflation↓":"🟢",
               "Growth↓ Inflation↓":"🔵","Growth↓ Inflation↑":"🟠"}
cur = result.current_regime
alloc = result.allocation

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Current Regime", f"{regime_icon.get(cur,'⚪')} {cur}")
c2.metric("PMI (latest)",   f"{result.growth_signal.dropna().iloc[-1]:.1f}" if len(result.growth_signal.dropna()) > 0 else "N/A")
c3.metric("CPI YoY",        f"{result.inflation_signal.dropna().iloc[-1]:.2f}%" if len(result.inflation_signal.dropna()) > 0 else "N/A")
c4.metric("Yield Curve",    f"{result.yield_curve.dropna().iloc[-1]:.2f}%" if len(result.yield_curve.dropna()) > 0 else "N/A")
if len(result.portfolio_returns) > 0:
    ann_r = result.portfolio_returns.mean() * 12
    c5.metric("Strategy Ann. Return", f"{ann_r*100:.1f}%")

st.markdown("---")
st.markdown(f"### Current Allocation: **{cur}**")
alloc_cols = st.columns(len(alloc))
for col, (ticker, weight) in zip(alloc_cols, alloc.items()):
    col.metric(ticker, f"{weight*100:.0f}%")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📡  Regime Timeline",
    "📊  Growth & Inflation",
    "📈  Strategy P&L",
    "🏦  Allocation by Regime",
    "〰️  Yield Curve",
])

with tab1:
    rh = result.regime_history
    fig = go.Figure()

    for regime, colour in REGIME_COLOURS.items():
        mask = rh["regime"] == regime
        dates_in = rh.index[mask]
        if len(dates_in) > 0:
            fig.add_trace(go.Scatter(
                x=dates_in, y=[regime]*len(dates_in),
                mode="markers", name=regime,
                marker=dict(color=colour, size=8, symbol="square"),
            ))

    fig.update_layout(**{**_base, "title": "Macro Regime Classification — Monthly",
                         "height": 350})
    fig.update_yaxes(title_text="Regime")
    st.plotly_chart(fig, use_container_width=True)

    # regime duration table
    regime_counts = result.regime.value_counts()
    regime_df = pd.DataFrame({
        "Regime": regime_counts.index,
        "Months": regime_counts.values,
        "Fraction": [f"{v/len(result.regime)*100:.1f}%" for v in regime_counts.values],
    })
    st.dataframe(regime_df, hide_index=True, use_container_width=True)

with tab2:
    fig2 = sp.make_subplots(rows=2, cols=1,
        subplot_titles=["PMI (Manufacturing) — > 50 = Expansion",
                        "CPI Year-over-Year (%) — > 2.5% = High Inflation"],
        vertical_spacing=0.12)

    fig2.add_trace(go.Scatter(x=result.growth_signal.index,
        y=result.growth_signal.values, mode="lines",
        line=dict(color=TEAL, width=2), showlegend=False), row=1, col=1)
    fig2.add_hline(y=50, line=dict(color=MUTED, dash="dash", width=1.5),
                   annotation_text="Expansion/Contraction (50)", row=1, col=1)

    fig2.add_trace(go.Scatter(x=result.inflation_signal.index,
        y=result.inflation_signal.values, mode="lines",
        line=dict(color=GOLD, width=2), showlegend=False), row=2, col=1)
    fig2.add_hline(y=2.5, line=dict(color=MUTED, dash="dash", width=1.5),
                   annotation_text="Fed target (2.5%)", row=2, col=1)

    for r in [1, 2]:
        fig2.update_xaxes(gridcolor=GRID, row=r, col=1)
        fig2.update_yaxes(gridcolor=GRID, row=r, col=1)

    fig2.update_layout(**{**_base, "height": 520, "title": "Macro Indicators"})
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    if len(result.portfolio_returns) > 0:
        cum = (1 + result.portfolio_returns).cumprod()
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=cum.index, y=(cum-1)*100, mode="lines",
            line=dict(color=GOLD, width=2.5),
            fill="tozeroy", fillcolor="rgba(200,168,75,0.08)"))
        fig3.add_hline(y=0, line=dict(color=MUTED, width=1))
        fig3.update_layout(**{**_base, "title": "Regime-Based Allocation — Cumulative Return"})
        fig3.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig3, use_container_width=True)

        ann_ret = result.portfolio_returns.mean() * 12
        ann_vol = result.portfolio_returns.std() * np.sqrt(12)
        sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0
        peak    = cum.cummax()
        max_dd  = ((cum - peak) / peak).min()
        st.markdown(f"Ann. Return: **{ann_ret*100:.2f}%** · Sharpe: **{sharpe:.3f}** · Max DD: **{max_dd*100:.2f}%**")
    else:
        st.info("Not enough data for backtest — try an earlier start year.")

with tab4:
    alloc_rows = []
    for regime, alloc_dict in REGIME_ALLOCATIONS.items():
        for ticker, weight in alloc_dict.items():
            alloc_rows.append({"Regime": regime, "Asset": ticker, "Weight": weight*100})
    alloc_df = pd.DataFrame(alloc_rows)

    fig4 = go.Figure()
    assets = alloc_df["Asset"].unique().tolist()
    colours = [TEAL, GOLD, CORAL, BLUE, "#9B59B6", "#E67E22", "#1ABC9C", "#F39C12"]
    for i, asset in enumerate(assets):
        sub = alloc_df[alloc_df["Asset"] == asset]
        fig4.add_trace(go.Bar(
            name=asset, x=sub["Regime"], y=sub["Weight"],
            marker_color=colours[i % len(colours)],
        ))
    fig4.update_layout(**{**_base, "title": "Asset Allocation by Regime",
                          "barmode": "stack", "height": 420})
    fig4.update_yaxes(title_text="Weight (%)", ticksuffix="%")
    st.plotly_chart(fig4, use_container_width=True)

with tab5:
    if len(result.yield_curve.dropna()) > 0:
        yc = result.yield_curve.dropna()
        fig5 = go.Figure()
        colours_yc = [TEAL if v > 0 else CORAL for v in yc.values]
        fig5.add_trace(go.Bar(x=yc.index, y=yc.values, marker_color=colours_yc))
        fig5.add_hline(y=0, line=dict(color=MUTED, width=1.5),
                       annotation_text="Inversion = recession warning")
        fig5.update_layout(**{**_base, "title": "10Y-2Y Yield Curve Spread (%)",
                               "height": 380})
        fig5.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig5, use_container_width=True)
        st.caption("When the spread goes negative (yield curve inverts), it has historically preceded recessions by 12-18 months.")