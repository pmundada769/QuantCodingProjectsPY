#app.py

# Time-Series Momentum (TSMOM) + Volatility Targeting — Streamlit Dashboard
# Run with: streamlit run app.py

import streamlit as st # type: ignore
import numpy as np
import pandas as pd
import plotly.graph_objects as go # type: ignore
import plotly.subplots as sp # type: ignore
import yfinance as yf
from tsmom import run_tsmom, compare_vol_targets, ARCH_AVAILABLE

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

CRIMSON="#C0392B"; EMBER="#E74C3C"; SAND="#F39C12"; TEAL="#1ABC9C"; MUTED="#7F8C8D"
BG="#0C0808"; GRID="#1E1010"; TEXT="#EAD5D5"

_base = dict(
    plot_bgcolor=BG, paper_bgcolor=BG,
    font=dict(family="Courier New, monospace", color=TEXT, size=11),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    margin=dict(l=60, r=30, t=60, b=60),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID),
)

with st.sidebar:
    st.markdown("## 📡 TSMOM Parameters")
    st.markdown("---")
    ticker_input = st.text_area("Tickers (one per line)",
        value="SPY\nQQQ\nTLT\nGLD\nEEM\nVNQ\nDBC\nHYG", height=180)
    tickers = [t.strip().upper() for t in ticker_input.split("\n") if t.strip()]
    tickers = list(dict.fromkeys(tickers))
    start_year = st.selectbox("Start Year", [2005, 2008, 2010, 2012, 2015], index=1)
    target_vol = st.slider("Target Volatility (%)", 5, 30, 15, step=1) / 100
    lookback   = st.selectbox("Momentum Lookback (days)", [126, 252, 378], index=1)
    use_garch  = st.checkbox("Use GARCH vol", value=ARCH_AVAILABLE, disabled=not ARCH_AVAILABLE)
    max_lev    = st.slider("Max Leverage", 1.0, 4.0, 2.0, step=0.5)
    st.markdown("---")
    if not ARCH_AVAILABLE:
        st.warning("Install `arch` for GARCH:\n`pip install arch`")
    st.caption("Moskowitz, Ooi & Pedersen (2012)")

start = f"{start_year}-01-01"

@st.cache_data(ttl=600)
def load_result(tickers_tuple, start, tv, lb, ug, ml):
    return run_tsmom(list(tickers_tuple), start=start, target_vol=tv,
                     lookback=lb, use_garch=ug, max_leverage=ml)

@st.cache_data(ttl=600)
def load_benchmark(tickers_tuple, start):
    raw = yf.download(list(tickers_tuple), start=start, auto_adjust=True,
                      threads=False, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame()
    return (1 + raw.pct_change().dropna().mean(axis=1)).cumprod()

with st.spinner("Running TSMOM simulation..."):
    try:
        result = load_result(tuple(tickers), start, target_vol, lookback, use_garch, max_lev)
        bench  = load_benchmark(tuple(tickers), start)
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

st.markdown("# 📡 Time-Series Momentum (TSMOM)")
st.markdown(f"`{len(result.tickers)} assets` · `target vol={target_vol*100:.0f}%` · `lookback={lookback}d`")
st.caption("Moskowitz, Ooi & Pedersen (2012) — *Time Series Momentum*, JFE")
st.markdown("---")

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Ann. Return",   f"{result.ann_return*100:.2f}%")
c2.metric("Ann. Vol",      f"{result.ann_vol*100:.2f}%")
c3.metric("Sharpe Ratio",  f"{result.sharpe:.3f}")
c4.metric("Sortino Ratio", f"{result.sortino:.3f}")
c5.metric("Max Drawdown",  f"{result.max_drawdown*100:.2f}%")
c6.metric("Hit Rate",      f"{result.hit_rate*100:.1f}%")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈  Cumulative Return", "🎛️  Signal Drill-Down",
    "📉  Drawdown", "🗓️  Position Heatmap",
    "🎯  Vol Target Sweep", "📋  Monthly Returns",
])

with tab1:
    cum = result.portfolio_cumret
    b   = bench.reindex(cum.index).ffill()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum.index, y=(cum-1)*100, mode="lines",
        name="TSMOM", line=dict(color=CRIMSON, width=2.5),
        fill="tozeroy", fillcolor="rgba(192,57,43,0.08)"))
    fig.add_trace(go.Scatter(x=b.index, y=(b-1)*100, mode="lines",
        name="Buy & Hold", line=dict(color=MUTED, width=1.5, dash="dot")))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_layout(**{**_base, "title": "TSMOM — Cumulative Return"})
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Return (%)", ticksuffix="%")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    ticker_sel = st.selectbox("Select asset", result.tickers)
    asset_obj  = next((a for a in result.asset_signals if a.ticker == ticker_sel), None)

    if asset_obj is not None:
        raw_ret  = asset_obj.returns.dropna()
        raw_sig  = asset_obj.raw_signal.dropna()
        raw_vol  = asset_obj.realised_vol.dropna()
        raw_pos  = asset_obj.scaled_position.dropna()
        cum_p    = (1 + raw_ret).cumprod()

        fig2 = sp.make_subplots(rows=3, cols=1,
            subplot_titles=[f"{ticker_sel} Price", "Signal (+1 Long / -1 Short)",
                            "Realised Vol (%) + Scaled Position"],
            vertical_spacing=0.12)

        fig2.add_trace(go.Scatter(x=cum_p.index, y=cum_p.values,
            mode="lines", line=dict(color=TEAL, width=1.8), showlegend=False), row=1, col=1)

        # colour background by signal
        long_dates = raw_sig[raw_sig == 1].index
        if len(long_dates) > 1:
            for i in range(1, min(len(long_dates), 200)):  # cap for performance
                fig2.add_vrect(x0=long_dates[i-1], x1=long_dates[i],
                    fillcolor="rgba(26,188,156,0.10)", layer="below", line_width=0,
                    row=1, col=1)

        fig2.add_trace(go.Scatter(x=raw_sig.index, y=raw_sig.values,
            mode="lines", line=dict(color=SAND, width=1.5), showlegend=False), row=2, col=1)
        fig2.add_trace(go.Scatter(x=raw_vol.index, y=raw_vol.values*100,
            mode="lines", line=dict(color=CRIMSON, width=1.8), showlegend=False), row=3, col=1)
        fig2.add_trace(go.Scatter(x=raw_pos.index, y=raw_pos.values,
            mode="lines", line=dict(color=SAND, width=1.5, dash="dot"), showlegend=False), row=3, col=1)

        for r in range(1, 4):
            fig2.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, row=r, col=1)
            fig2.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, row=r, col=1)

        fig2.update_layout(**{**_base, "height": 700, "title": f"{ticker_sel} — Signal Decomposition"})
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning(f"No signal data for {ticker_sel}")

with tab3:
    cum  = result.portfolio_cumret
    peak = cum.cummax()
    dd   = (cum - peak) / peak * 100
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines",
        line=dict(color=EMBER, width=1.5),
        fill="tozeroy", fillcolor="rgba(231,76,60,0.12)"))
    fig3.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig3.update_layout(**{**_base, "title": "TSMOM Portfolio Drawdown"})
    fig3.update_xaxes(title_text="Date")
    fig3.update_yaxes(title_text="Drawdown (%)", ticksuffix="%")
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    pos_data = {}
    for asset in result.asset_signals:
        s = asset.scaled_position.dropna()
        if len(s) > 0:
            pos_data[asset.ticker] = s.resample("ME").last()

    if pos_data:
        pos_df = pd.DataFrame(pos_data).dropna(how="all")
        fig4   = go.Figure(go.Heatmap(
            z=pos_df.T.values, x=pos_df.index, y=pos_df.columns.tolist(),
            colorscale="RdYlGn", zmid=0, zmin=-2, zmax=2,
            colorbar=dict(title=dict(text="Position", font=dict(color=TEXT)),
                          tickfont=dict(color=TEXT), thickness=12),
        ))
        fig4.update_layout(**{**_base,
            "title": "Monthly Position Heatmap (Green=Long, Red=Short)",
            "height": max(350, len(result.tickers) * 45)})
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning("No position data — try a longer date range.")

with tab5:
    st.caption("Reruns at 5 vol targets — shows how Sharpe/return/drawdown scale with target vol.")
    if st.button("▶  Run Vol Target Sweep"):
        vol_targets = [0.05, 0.10, 0.15, 0.20, 0.25]
        sharpes, ann_rets, max_dds = [], [], []
        prog = st.progress(0)
        for i, vt in enumerate(vol_targets):
            r = run_tsmom(result.tickers, start=start, target_vol=vt, use_garch=False)
            sharpes.append(r.sharpe)
            ann_rets.append(r.ann_return * 100)
            max_dds.append(r.max_drawdown * 100)
            prog.progress((i+1)/len(vol_targets))

        vt_pct = [v*100 for v in vol_targets]
        fig5   = sp.make_subplots(rows=1, cols=3,
            subplot_titles=["Sharpe Ratio", "Ann. Return (%)", "Max Drawdown (%)"])
        for vals, ci, col in [(sharpes,1,CRIMSON),(ann_rets,2,SAND),(max_dds,3,TEAL)]:
            fig5.add_trace(go.Scatter(x=vt_pct, y=vals, mode="lines+markers",
                line=dict(color=col, width=2.5), marker=dict(size=8), showlegend=False),
                row=1, col=ci)
            fig5.update_xaxes(gridcolor=GRID, ticksuffix="%", title_text="Target Vol %", row=1, col=ci)
            fig5.update_yaxes(gridcolor=GRID, row=1, col=ci)
        fig5.update_layout(**{**_base, "title": "Performance vs Vol Target", "height": 380})
        st.plotly_chart(fig5, use_container_width=True)

with tab6:
    monthly = result.portfolio_returns.resample("ME").apply(lambda x: (1+x).prod()-1)
    if len(monthly) > 0:
        yr_mo = pd.DataFrame({
            "Year":   monthly.index.year,
            "Month":  monthly.index.strftime("%b"),
            "Return": (monthly * 100).round(3).values,
        })
        pivot = yr_mo.pivot_table(index="Year", columns="Month",
                                   values="Return", aggfunc="first")
        month_order = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        pivot = pivot[[m for m in month_order if m in pivot.columns]]
        st.dataframe(
            pivot.style.background_gradient(cmap="RdYlGn", axis=None, vmin=-10, vmax=10)
                       .format("{:.2f}%"),
            use_container_width=True,
        )
    else:
        st.warning("Not enough data for monthly table — try an earlier start year.")